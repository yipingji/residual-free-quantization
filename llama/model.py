from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int = 192
    intermediate_size: int = 768
    num_hidden_layers: int = 24
    num_attention_heads: int = 3
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    qkv_bias: bool = False
    qk_norm: bool = False
    hidden_act: str = "gelumlp"
    qk_depth_gamma: float = 1.0
    skipless: bool = False
    ortho_init: float = 1.0
    skip_ortho_init: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_size < 1 or self.num_attention_heads < 1:
            raise ValueError("hidden_size and num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("rotary embeddings require an even attention head dimension")
        if self.qk_depth_gamma <= 0:
            raise ValueError("qk_depth_gamma must be positive")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads < 1 or self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by positive num_key_value_heads")
        if self.num_key_value_heads != self.num_attention_heads and (
            (self.skipless and self.ortho_init > 0) or self.skip_ortho_init
        ):
            raise ValueError("paired orthogonal initialization requires one K/V head per query head")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        cos = self.cos_cached[:, :seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch_size, seq_len, kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(batch_size, seq_len, kv_heads, repeats, head_dim)
    return x.reshape(batch_size, seq_len, kv_heads * repeats, head_dim)


def get_ortho(dim: int, device=None):
    matrix = torch.randn(dim, dim, device=device)
    u, _, _ = torch.linalg.svd(matrix)
    return u


def suo_initialize(rows: int, cols: int, device=None):
    x = torch.randn(min(rows, cols), max(rows, cols), device=device)
    gram = x @ x.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    inv_sqrt = eigenvalues.clamp(min=1e-8).rsqrt()
    result = (eigenvectors @ torch.diag(inv_sqrt) @ eigenvectors.T) @ x
    return result if rows <= cols else result.T


@torch.no_grad()
def apply_skipless_ortho_qk_init(model, scale: float = 1.0):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    device = model.embed_tokens.weight.device

    for layer in model.layers:
        for head_idx in range(num_heads):
            ortho = get_ortho(hidden_size, device=device)
            qk = scale * ortho[:, :head_dim]
            start = head_idx * head_dim
            end = start + head_dim
            layer.self_attn.q_proj.weight[start:end].copy_(qk.T.contiguous())
            layer.self_attn.k_proj.weight[start:end].copy_(qk.T.contiguous())


@torch.no_grad()
def apply_skipless_ortho_vo_init(model, scale: float = 1.0):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    device = model.embed_tokens.weight.device

    for layer in model.layers:
        for head_idx in range(num_heads):
            ortho = get_ortho(hidden_size, device=device)
            vo = scale * ortho[:, :head_dim]
            start = head_idx * head_dim
            end = start + head_dim
            layer.self_attn.v_proj.weight[start:end].copy_(vo.T.contiguous())
            layer.self_attn.o_proj.weight[start:end].copy_(vo.T.contiguous())


@torch.no_grad()
def apply_skipless_suo_init(model):
    device = model.embed_tokens.weight.device
    for layer in model.layers:
        up_rows, up_cols = layer.mlp.up_proj.weight.shape
        down_rows, down_cols = layer.mlp.down_proj.weight.shape
        layer.mlp.up_proj.weight.copy_(suo_initialize(up_rows, up_cols, device=device))
        layer.mlp.down_proj.weight.copy_(suo_initialize(down_rows, down_cols, device=device))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.qk_depth_gamma = config.qk_depth_gamma
        self.layer_idx = 0

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.qk_depth_gamma ** (-self.layer_idx) if self.qk_depth_gamma != 1.0 else 1.0
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale * (self.head_dim**-0.5)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.hidden_act not in ("gelumlp", "swishmlp", "gelu", "swish"):
            raise ValueError(f"Unsupported hidden_act: {config.hidden_act}")
        self.act = F.gelu if "gelu" in config.hidden_act else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.skipless = config.skipless
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.self_attn.layer_idx = layer_idx
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(self.input_layernorm(x))
        x = attn if self.skipless else x + attn
        mlp = self.mlp(self.post_attention_layernorm(x))
        return mlp if self.skipless else x + mlp


class LlamaForCausalLM(nn.Module):
    # Cross-entropy already averages over the valid tokens in each microbatch.
    accepts_loss_kwargs = False

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        if (config.skipless and config.ortho_init > 0) or config.skip_ortho_init:
            apply_skipless_ortho_qk_init(self, scale=config.ortho_init)
            apply_skipless_ortho_vo_init(self, scale=config.ortho_init)
            apply_skipless_suo_init(self)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.config.tie_word_embeddings:
            state.pop("lm_head.weight", None)
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        embedding_key = prefix + "embed_tokens.weight"
        head_key = prefix + "lm_head.weight"
        if self.config.tie_word_embeddings and embedding_key in state_dict:
            state_dict.setdefault(head_key, state_dict[embedding_key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return {"loss": loss}

        return {"loss": None, "logits": logits}

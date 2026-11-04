import argparse
import json
import math
import os
from dataclasses import asdict

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from llama import LlamaConfig, LlamaForCausalLM
from optim import KLShampoo


class ConstantLengthDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        token_buffer = []

        for row in self.dataset:
            text = row["text"].strip()
            if not text:
                continue

            ids = self.tokenizer(text + self.tokenizer.eos_token, verbose=False)["input_ids"]
            token_buffer.extend(ids)

            while len(token_buffer) >= self.block_size:
                input_ids = token_buffer[: self.block_size]
                token_buffer = token_buffer[self.block_size :]
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.copy(),
                }


class BinTokenDataset(IterableDataset):
    def __init__(self, bin_path, block_size, seed=42):
        self.bin_path = bin_path
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        tokens = np.memmap(self.bin_path, dtype=np.uint32, mode="r")
        if len(tokens) < self.block_size:
            raise ValueError("train_bin has fewer tokens than block_size")

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(
            self.seed + rank * 10007 + worker_id * (10007 * world_size)
        )
        max_start = len(tokens) - self.block_size

        # Trainer controls the number of optimizer steps across all ranks.
        while True:
            start = int(rng.integers(0, max_start + 1))
            input_ids = torch.from_numpy(tokens[start : start + self.block_size].astype(np.int64))
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            }


class PretrainingTrainer(Trainer):
    def __init__(
        self,
        *args,
        grad_norm_steps=50,
        optimizer_name="adamw",
        kl_shampoo_kwargs=None,
        lr_scaling=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_norm_steps = grad_norm_steps
        self.optimizer_name = optimizer_name
        self.kl_shampoo_kwargs = kl_shampoo_kwargs or {}
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.train_log_path = os.path.join(self.args.output_dir, "train_log.jsonl")
        self.grad_norm_path = os.path.join(self.args.output_dir, "grad_norm_by_block.jsonl")
        self.lr_scaling = lr_scaling

    def create_optimizer(self):
        if self.optimizer_name != "klshampoo":
            return super().create_optimizer()
        if self.optimizer is not None:
            return self.optimizer

        self.optimizer = KLShampoo(
            self._build_klshampoo_param_groups(),
            lr=self.args.learning_rate,
            **self.kl_shampoo_kwargs,
        )
        return self.optimizer

    def _build_klshampoo_param_groups(self):
        model = self._unwrap_model(self.model)
        num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Could not infer num_hidden_layers from model.config")
        if num_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_layers}")

        base_lr = self.args.learning_rate
        if self.lr_scaling:
            qk_lr = base_lr / math.sqrt(num_layers)
            ov_mlp_lr = base_lr / num_layers
        else:
            qk_lr = base_lr
            ov_mlp_lr = base_lr

        param_groups = {
            "attn_qk_decay": {
                "params": [],
                "lr": qk_lr,
                "weight_decay": self.args.weight_decay,
            },
            "attn_ov_mlp_decay": {
                "params": [],
                "lr": ov_mlp_lr,
                "weight_decay": self.args.weight_decay,
            },
            "other_decay": {
                "params": [],
                "lr": base_lr,
                "weight_decay": self.args.weight_decay,
            },
            "no_decay": {
                "params": [],
                "lr": base_lr,
                "weight_decay": 0.0,
            },
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim < 2:
                param_groups["no_decay"]["params"].append(param)
            elif self._is_attn_qk_weight(name):
                param_groups["attn_qk_decay"]["params"].append(param)
            elif self._is_attn_ov_or_mlp_weight(name):
                param_groups["attn_ov_mlp_decay"]["params"].append(param)
            else:
                param_groups["other_decay"]["params"].append(param)

        return [group for group in param_groups.values() if group["params"]]

    @staticmethod
    def _is_attn_qk_weight(name):
        return name.endswith("self_attn.q_proj.weight") or name.endswith("self_attn.k_proj.weight")

    @staticmethod
    def _is_attn_ov_or_mlp_weight(name):
        return (
            name.endswith("self_attn.o_proj.weight")
            or name.endswith("self_attn.v_proj.weight")
            or name.endswith("mlp.up_proj.weight")
            or name.endswith("mlp.down_proj.weight")
        )

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)

        if self.grad_norm_steps > 0 and self.is_world_process_zero() and self.accelerator.sync_gradients:
            next_step = self.state.global_step + 1
            if next_step % self.grad_norm_steps == 0:
                self._write_block_grad_norms(next_step)

        return loss

    def log(self, logs, *args, **kwargs):
        if self.is_world_process_zero():
            record = {"step": self.state.global_step, **self._jsonable(logs)}
            with open(self.train_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return super().log(logs, *args, **kwargs)

    def _write_block_grad_norms(self, step):
        model = self._unwrap_model(self.model)
        blocks = {}
        weights = {}

        for idx, block in enumerate(model.layers):
            blocks[str(idx)] = self._module_grad_norm(block)

        for name, param in model.named_parameters():
            if param.grad is not None:
                weights[name] = self._tensor_grad_norm(param)

        record = {"step": step, "blocks": blocks, "weights": weights}
        with open(self.grad_norm_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _module_grad_norm(self, module):
        total = 0.0
        for param in module.parameters():
            if param.grad is not None:
                total += self._tensor_grad_norm(param) ** 2
        return math.sqrt(total)

    def _tensor_grad_norm(self, param):
        grad = param.grad.detach().float()
        return math.sqrt(grad.pow(2).sum().item())

    def _unwrap_model(self, model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)


class DynamicCheckpointCallback(TrainerCallback):
    def __init__(self, warmup_steps, warmup_save_steps, post_warmup_save_steps):
        self.warmup_steps = warmup_steps
        self.warmup_save_steps = warmup_save_steps
        self.post_warmup_save_steps = post_warmup_save_steps

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        max_steps = state.max_steps if state.max_steps is not None else args.max_steps

        if self._should_save(step, max_steps):
            control.should_save = True

        return control

    def _should_save(self, step, max_steps):
        if step <= 0:
            return False
        if max_steps > 0 and step >= max_steps:
            return True
        if step <= self.warmup_steps:
            return step % self.warmup_save_steps == 0
        return (step - self.warmup_steps) % self.post_warmup_save_steps == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal C4 pretraining with a local LLaMA model.")
    parser.add_argument(
        "--tokenizer_name",
        default="HuggingFaceTB/SmolLM2-360M",
    )
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_attention_heads", type=int, default=3)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--hidden_act", default="gelumlp")
    parser.add_argument("--skipless", action="store_true")
    parser.add_argument("--ortho_init", type=float, default=1.0)
    parser.add_argument("--skip_ortho_init", type=float, default=0.0)
    parser.add_argument("--dataset_name", default="allenai/c4")
    parser.add_argument("--dataset_config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_bin", default=None)
    parser.add_argument("--output_dir", default="outputs/c4-pretrain")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "klshampoo"], default="klshampoo")
    parser.add_argument("--kl_beta1", type=float, default=0.9)
    parser.add_argument("--kl_beta2", type=float, default=0.98)
    parser.add_argument("--kl_shampoo_beta", type=float, default=-1.0)
    parser.add_argument("--kl_eps", type=float, default=1e-8)
    parser.add_argument("--kl_precondition_frequency", type=int, default=10)
    parser.add_argument("--kl_max_precondition_dim", type=int, default=2048)
    parser.add_argument("--kl_init_factor", type=float, default=0.1)
    parser.add_argument("--kl_normalize_grads", action="store_true")
    parser.add_argument("--warmup_save_steps", type=int, default=200)
    parser.add_argument("--post_warmup_save_steps", type=int, default=4000)
    parser.add_argument("--logging_steps", type=int, default=32)
    parser.add_argument("--grad_norm_steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--learning_rate_scaling", action="store_true", help="scale KL-Shampoo learning rates with model depth")
    parser.add_argument("--qk_depth_gamma", type=float, default=1.0,
                        help="multiply attention scores by gamma^(-layer_idx); 1.0 disables")

    args = parser.parse_args()
    if args.block_size < 2:
        parser.error("--block_size must be at least 2 for next-token prediction")
    if min(args.warmup_save_steps, args.post_warmup_save_steps, args.kl_precondition_frequency) < 1:
        parser.error("checkpoint and preconditioner intervals must be positive")
    return args


def main():
    args = parse_args()
    if args.warmup_steps < 0:
        args.warmup_steps = int(args.max_steps * args.warmup_ratio)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.train_bin:
        train_dataset = BinTokenDataset(
            bin_path=args.train_bin,
            seed=args.seed,
            block_size=args.block_size,
        )
    else:
        c4 = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.split,
            streaming=True,
        )
        c4 = c4.shuffle(seed=args.seed, buffer_size=10_000)

        train_dataset = ConstantLengthDataset(
            dataset=c4,
            tokenizer=tokenizer,
            block_size=args.block_size,
        )

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.block_size,
        rope_theta=args.rope_theta,
        qk_norm=args.qk_norm,
        hidden_act=args.hidden_act,
        skipless=args.skipless,
        ortho_init=args.ortho_init,
        skip_ortho_init=args.skip_ortho_init,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        qk_depth_gamma=args.qk_depth_gamma,
    )
    model = LlamaForCausalLM(config)

    param_count = sum(param.numel() for param in model.parameters())
    print(f"Local LLaMA parameters: {param_count / 1_000_000:.2f}M")

    os.makedirs(args.output_dir, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        run_config = {
            "args": vars(args),
            "model_config": asdict(config),
            "param_count": param_count,
            "visible_gpus": torch.cuda.device_count(),
        }
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.max_steps + 1,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        grad_norm_steps=args.grad_norm_steps,
        optimizer_name=args.optimizer,
        kl_shampoo_kwargs={
            "betas": (args.kl_beta1, args.kl_beta2),
            "shampoo_beta": args.kl_shampoo_beta,
            "eps": args.kl_eps,
            "precondition_frequency": args.kl_precondition_frequency,
            "max_precondition_dim": args.kl_max_precondition_dim,
            "init_factor": args.kl_init_factor,
            "normalize_grads": args.kl_normalize_grads,
        },
        callbacks=[
            DynamicCheckpointCallback(
                warmup_steps=args.warmup_steps,
                warmup_save_steps=args.warmup_save_steps,
                post_warmup_save_steps=args.post_warmup_save_steps,
            )
        ],
        lr_scaling=args.learning_rate_scaling,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

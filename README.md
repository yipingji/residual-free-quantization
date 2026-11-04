# The Quantization Benefits of Residual-Free Transformers

**NeurIPS 2026** · [Paper on arXiv](https://arxiv.org/abs/2605.25880) · [OpenReview](https://openreview.net/forum?id=18XtiYe9Gk)

Yiping Ji\*, Mahalakshmi Sabanayagam\*, Peyman Moghadam, Hemanth Saratchandran, Simon Lucey

\* Equal contribution

This repository contains the C4 language-model pretraining code from the paper's supplementary material. It trains LLaMA-style decoders with and without residual connections. The residual-free configuration combines paired orthogonal attention initialization, semi-orthogonal MLP initialization, and KL-Shampoo. The paper's low-bit evaluation scripts and pretrained checkpoints are outside the scope of this release.

## Setup

Use Python 3.11. Create an environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install a [PyTorch build for your hardware](https://pytorch.org/get-started/locally/) in that environment, then install the training dependencies.

```bash
python -m pip install -r requirements.txt
```

The training launcher uses CUDA and bfloat16. The small example below runs in float32. The first data preparation run needs access to Hugging Face for English C4 and the SmolLM2 tokenizer.

## Run a small pretraining example

From the repository root, write one million tokens to a local binary file.

```bash
python prepare_c4_bin.py --max_tokens 1000000 --output_bin data/c4_1m.bin
```

Train a two-layer residual-free model for three optimizer steps.

```bash
python train.py --train_bin data/c4_1m.bin --skipless --optimizer klshampoo \
  --hidden_size 64 --intermediate_size 128 --num_layers 2 --num_attention_heads 2 \
  --block_size 64 --batch_size 2 --max_steps 3 --logging_steps 1 \
  --output_dir outputs/smoke
```

## Compare the training configurations

Prepare the token stream used by the launcher. The binary contains contiguous uint32 token IDs, with preprocessing metadata in the adjacent `.meta.json` file. Use the same tokenizer for preprocessing and training.

```bash
python prepare_c4_bin.py --max_tokens 1000000000 --output_bin data/c4_en_1b.bin
```

| Configuration | Command |
| --- | --- |
| Residual-free, paired orthogonal initialization, KL-Shampoo | `bash scripts/run_skipless_klshampoo.sh` |
| Residual, standard initialization, KL-Shampoo | `USE_SKIPLESS=0 bash scripts/run_skipless_klshampoo.sh` |
| Residual, standard initialization, AdamW | `USE_SKIPLESS=0 OPTIMIZER=adamw bash scripts/run_skipless_klshampoo.sh` |

The supplementary launcher uses 24 layers, width 192, 3 heads, MLP width 768, sequence length 128, and 5,000 optimizer steps. It selects GPUs 0 and 1 and uses 128 sequences per GPU per step. Set `CUDA_VISIBLE_DEVICES=0` for a single GPU. Set `TRAIN_BIN` and `OUTPUT_DIR` to use a different dataset or isolate runs with different settings.

| Option | Effect |
| --- | --- |
| `ORTHO_INIT` | Scales the paired Q/K and V/O attention initialization in the residual-free recipe. Set to 0 to use standard initialization throughout the model. |
| `QK_DEPTH_GAMMA` | Multiplies attention logits in zero-indexed layer *l* by γ⁻ˡ. The launcher default of 1 leaves attention temperatures constant across layers. |
| `USE_LR_SCALING` | For KL-Shampoo, scales the Q/K learning rate by 1/√L and the V/O and MLP rates by 1/L, where L is the number of layers. Enabled by default. |
| `BATCH_SIZE`, `GRAD_ACCUM` | Per-GPU microbatch size and number of gradient accumulation steps. The effective batch size is their product times the GPU count. |

## Code and outputs

| Path | Contents |
| --- | --- |
| [llama/model.py](llama/model.py) | Decoder, residual switch, initialization, and attention temperature |
| [optim/kl_shampoo.py](optim/kl_shampoo.py) | KL-Shampoo updates and AdamW fallback for vectors and oversized tensors |
| [prepare_c4_bin.py](prepare_c4_bin.py) | Streams, shuffles, and tokenizes C4 |
| [train.py](train.py) | Pretraining, parameter groups, loss and gradient-norm logging, and checkpoints |
| [scripts/run_skipless_klshampoo.sh](scripts/run_skipless_klshampoo.sh) | Single-node training launcher |

Each run saves its arguments and model configuration in `config.json`, metrics in `train_log.jsonl`, and a final model and tokenizer in the output directory. Per-layer and per-parameter gradient norms go to `grad_norm_by_block.jsonl` at `--grad_norm_steps` intervals. Intermediate checkpoints use `checkpoint-<step>/`. The release logs Trainer's loss directly. Older logs from the submission code applied an additional division by the accumulation count and visible GPU count.

## Citation

```bibtex
@inproceedings{ji2026quantization,
  title     = {The Quantization Benefits of Residual-Free Transformers},
  author    = {Ji, Yiping and Sabanayagam, Mahalakshmi and Moghadam, Peyman and Saratchandran, Hemanth and Lucey, Simon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.25880}
}
```

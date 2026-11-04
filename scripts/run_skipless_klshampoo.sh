#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

TRAIN_BIN="${TRAIN_BIN:-data/c4_en_1b.bin}"
NUM_LAYERS="${NUM_LAYERS:-24}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
OPTIMIZER="${OPTIMIZER:-klshampoo}"
USE_SKIPLESS="${USE_SKIPLESS:-1}"
ORTHO_INIT="${ORTHO_INIT:-1.0}"
SKIP_ORTHO_INIT="${SKIP_ORTHO_INIT:-0.0}"
QK_DEPTH_GAMMA="${QK_DEPTH_GAMMA:-1.0}"
USE_LR_SCALING="${USE_LR_SCALING:-1}"

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"
TOTAL_BS="$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"

MODEL_ARGS=(--optimizer "$OPTIMIZER")
if [[ "$USE_SKIPLESS" == "1" ]]; then
  SKIPLESS_NAME="_skipless_ortho${ORTHO_INIT}_qkgamma${QK_DEPTH_GAMMA}"
  MODEL_ARGS+=(--skipless)
else
  SKIPLESS_NAME="_ortho${SKIP_ORTHO_INIT}"
fi

if [[ "$USE_LR_SCALING" == "1" ]]; then
  MODEL_ARGS+=(--learning_rate_scaling)
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/l${NUM_LAYERS}_lr${LEARNING_RATE}_bs${TOTAL_BS}${SKIPLESS_NAME}_steps${MAX_STEPS}_${OPTIMIZER}}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="$NUM_GPUS")
else
  LAUNCHER=(python)
fi

"${LAUNCHER[@]}" train.py \
  --train_bin "$TRAIN_BIN" \
  --output_dir "$OUTPUT_DIR" \
  --bf16 \
  "${MODEL_ARGS[@]}" \
  --hidden_size 192 \
  --num_layers "$NUM_LAYERS" \
  --num_attention_heads 3 \
  --intermediate_size 768 \
  --hidden_act gelumlp \
  --block_size 128 \
  --ortho_init "$ORTHO_INIT" \
  --skip_ortho_init "$SKIP_ORTHO_INIT" \
  --qk_depth_gamma "$QK_DEPTH_GAMMA" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.01 \
  --max_steps "$MAX_STEPS" \
  --warmup_ratio 0.1 \
  --warmup_save_steps 200 \
  --post_warmup_save_steps 4000 \
  --logging_steps 10 \
  --grad_norm_steps 50 \
  --kl_beta1 0.9 \
  --kl_beta2 0.98 \
  --kl_precondition_frequency 10 \
  --kl_max_precondition_dim 2048 \
  --kl_init_factor 0.1

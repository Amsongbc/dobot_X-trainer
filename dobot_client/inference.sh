#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TASK="${TASK:-pour}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-32155}"
DEVICE="${DEVICE:-}"
WAN_PATH="${WAN_PATH:-$ROOT_DIR/pretrained_models/Wan2.2-TI2V-5B}"
VLM_PATH="${VLM_PATH:-$ROOT_DIR/pretrained_models/Qwen3-VL-2B-Instruct}"
T5_EMBEDDINGS_DIR="${T5_EMBEDDINGS_DIR:-}"

case "$TASK" in
  cook)
    MODEL_CONFIG_DEFAULT="$ROOT_DIR/configs/dobot_c.yaml"
    DATASET_NAME_DEFAULT="dobot_cook_vegetable"
    DEFAULT_INSTRUCTION_FILE_DEFAULT="$ROOT_DIR/dobot/cook_instruction.txt"
    DEFAULT_T5_EMBEDDINGS_PATH_DEFAULT="$ROOT_DIR/full_cook_vegetable.pt"
    ;;
  pour)
    MODEL_CONFIG_DEFAULT="$ROOT_DIR/configs/dobot_p.yaml"
    DATASET_NAME_DEFAULT="dobot_pour_water"
    DEFAULT_INSTRUCTION_FILE_DEFAULT="$ROOT_DIR/dobot/pour_instruction.txt"
    DEFAULT_T5_EMBEDDINGS_PATH_DEFAULT="$ROOT_DIR/full_pour_water.pt"
    ;;
  *)
    echo "Unsupported TASK='$TASK'. Use TASK=cook or TASK=pour." >&2
    exit 1
    ;;
esac

MODEL_CONFIG="${MODEL_CONFIG:-$MODEL_CONFIG_DEFAULT}"
DATASET_NAME="${DATASET_NAME:-$DATASET_NAME_DEFAULT}"
DEFAULT_INSTRUCTION_FILE="${DEFAULT_INSTRUCTION_FILE:-$DEFAULT_INSTRUCTION_FILE_DEFAULT}"
DEFAULT_T5_EMBEDDINGS_PATH="${DEFAULT_T5_EMBEDDINGS_PATH:-$DEFAULT_T5_EMBEDDINGS_PATH_DEFAULT}"

: "${CKPT_DIR:?Set CKPT_DIR to the Motus checkpoint directory before running this script.}"

if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "MODEL_CONFIG not found: $MODEL_CONFIG" >&2
  exit 1
fi

if [[ ! -d "$WAN_PATH" ]]; then
  echo "WAN_PATH not found: $WAN_PATH" >&2
  exit 1
fi

if [[ ! -d "$VLM_PATH" ]]; then
  echo "VLM_PATH not found: $VLM_PATH" >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_INSTRUCTION_FILE" ]]; then
  echo "DEFAULT_INSTRUCTION_FILE not found: $DEFAULT_INSTRUCTION_FILE" >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_T5_EMBEDDINGS_PATH" && -z "$T5_EMBEDDINGS_DIR" ]]; then
  echo "DEFAULT_T5_EMBEDDINGS_PATH not found: $DEFAULT_T5_EMBEDDINGS_PATH" >&2
  echo "Either provide DEFAULT_T5_EMBEDDINGS_PATH or set T5_EMBEDDINGS_DIR." >&2
  exit 1
fi

CMD=(
  python dobot/server_vlm_mask.py
  --model_config "$MODEL_CONFIG"
  --ckpt_dir "$CKPT_DIR"
  --wan_path "$WAN_PATH"
  --vlm_path "$VLM_PATH"
  --dataset_name "$DATASET_NAME"
  --default_instruction_file "$DEFAULT_INSTRUCTION_FILE"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi

if [[ -n "$T5_EMBEDDINGS_DIR" ]]; then
  CMD+=(--t5_embeddings_dir "$T5_EMBEDDINGS_DIR")
fi

if [[ -f "$DEFAULT_T5_EMBEDDINGS_PATH" ]]; then
  CMD+=(--default_t5_embeddings_path "$DEFAULT_T5_EMBEDDINGS_PATH")
fi

if [[ -n "${ACTION_MIN:-}" ]]; then
  read -r -a ACTION_MIN_ARR <<< "$ACTION_MIN"
  CMD+=(--action_min "${ACTION_MIN_ARR[@]}")
fi

if [[ -n "${ACTION_MAX:-}" ]]; then
  read -r -a ACTION_MAX_ARR <<< "$ACTION_MAX"
  CMD+=(--action_max "${ACTION_MAX_ARR[@]}")
fi

echo "Launching Dobot server with:"
printf '  %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"

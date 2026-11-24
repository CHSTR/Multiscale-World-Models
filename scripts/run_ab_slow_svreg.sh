#!/usr/bin/env bash
# Portable A/B launcher for SIGReg vs Slow SVReg on Cube (ogb) and MultiScaleDial (msdial).
# Zero hardcoded local paths — configure via environment variables or CLI flags.
#
# Example:
#   STABLEWM_HOME=./swm_home LOCAL_DATASET_DIR=./swm_home \
#     bash scripts/run_ab_slow_svreg.sh --dataset msdial --model lewm --loss slow_svreg --epochs 10
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="msdial"          # ogb | msdial
MODEL="lewm"              # lewm | dinov3_lora | convnext_ms | convnext_ms_scratch
LOSS_MODE="slow_svreg"    # sigreg | slow_svreg
REPLAY="disabled"         # disabled | per_is
EPOCHS="10"
SEED="0"
BATCH_SIZE=""
LAM="30.0"
SLOW_WEIGHT="null"        # null -> beta = 1 / lam
NORM="GroupNorm1d"        # GroupNorm1d | LayerNorm (for slow_svreg)
NUM_EVAL="50"
PROBE_FRAMES="50000"
RUN_NAME=""
SKIP_EVAL=0
SKIP_PROBE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --loss) LOSS_MODE="$2"; shift 2 ;;
    --replay) REPLAY="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lam) LAM="$2"; shift 2 ;;
    --slow-weight) SLOW_WEIGHT="$2"; shift 2 ;;
    --norm) NORM="$2"; shift 2 ;;
    --num-eval) NUM_EVAL="$2"; shift 2 ;;
    --probe-frames) PROBE_FRAMES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --skip-probe) SKIP_PROBE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$BATCH_SIZE" ]]; then
  if [[ "$MODEL" == convnext_ms* ]]; then
    BATCH_SIZE="32"
  else
    BATCH_SIZE="64"
  fi
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="ab_${DATASET}_${MODEL}_${LOSS_MODE}_r${REPLAY}_s${SEED}"
fi

TRAIN_ARGS=(
  train.py
  "data=${DATASET}"
  "model=${MODEL}"
  "replay=${REPLAY}"
  "seed=${SEED}"
  "trainer.max_epochs=${EPOCHS}"
  "loader.batch_size=${BATCH_SIZE}"
  "output_model_name=${RUN_NAME}"
)

if [[ "$LOSS_MODE" == "slow_svreg" ]]; then
  if [[ "$NORM" == "LayerNorm" ]]; then
    NORM_TARGET="torch.nn.LayerNorm"
  else
    NORM_TARGET="module.GroupNorm1d"
  fi
  TRAIN_ARGS+=(
    "+experiment=slow_svreg"
    "loss.svreg.weight=${LAM}"
    "loss.slow.weight=${SLOW_WEIGHT}"
    "model.projector.norm_fn._target_=${NORM_TARGET}"
  )
else
  TRAIN_ARGS+=("loss=sigreg")
fi

POLICY_REL="${RUN_NAME}/weights_epoch_${EPOCHS}.pt"

if [[ "$DATASET" == "ogb" ]]; then
  EVAL_CFG="cube"
elif [[ "$DATASET" == "msdial" ]]; then
  EVAL_CFG="msdial"
else
  echo "Unsupported --dataset: $DATASET (expected ogb|msdial)" >&2
  exit 1
fi

EVAL_ARGS=(
  eval.py
  "--config-name=${EVAL_CFG}"
  "policy=${POLICY_REL}"
  "eval.num_eval=${NUM_EVAL}"
  "seed=${SEED}"
)
if [[ -n "${STABLEWM_HOME:-}" ]]; then
  EVAL_ARGS+=("cache_dir=${STABLEWM_HOME}")
fi

run_cmd() {
  echo "+ $*"
  if (( ! DRY_RUN )); then
    "$@"
  fi
}

run_cmd "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

if (( ! SKIP_EVAL )); then
  run_cmd "$PYTHON_BIN" "${EVAL_ARGS[@]}"
fi

if (( ! SKIP_PROBE )); then
  SWM_ROOT="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
  DATA_ROOT="${LOCAL_DATASET_DIR:-$SWM_ROOT}"
  if [[ "$DATASET" == "ogb" ]]; then
    PROBE_ARGS=(
      probe_train.py
      --ckpt "${SWM_ROOT}/checkpoints/${POLICY_REL}"
      --source emb
      --n-frames "${PROBE_FRAMES}"
      --cache-dir "${DATA_ROOT}"
      --out-dir "${SWM_ROOT}/probe_outputs"
      --seed "${SEED}"
    )
    run_cmd "$PYTHON_BIN" "${PROBE_ARGS[@]}"
  else
    PROBE_ARGS=(
      swm_msdial/probe.py
      --policy "${POLICY_REL}"
      --data "${DATA_ROOT}/datasets/msdial/msdial_val_k2_balanced.h5"
      --out "${SWM_ROOT}/checkpoints/${RUN_NAME}/probe_msdial.json"
    )
    run_cmd "$PYTHON_BIN" "${PROBE_ARGS[@]}"
  fi
fi

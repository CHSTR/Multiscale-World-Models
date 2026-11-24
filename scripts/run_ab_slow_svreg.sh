#!/usr/bin/env bash
# Portable A/B launcher for SIGReg vs Slow SVReg on:
#   - MultiScaleDial (msdial)
#   - Cube (ogb)
#   - PushT online via stable-envs (pusht_online)
#   - TwoRoom online via stable-envs (tworoom_online)
# Zero hardcoded local paths — configure via environment variables or CLI flags.
#
# Examples:
#   STABLEWM_HOME=./swm_home LOCAL_DATASET_DIR=./swm_home \
#     bash scripts/run_ab_slow_svreg.sh --dataset pusht_online --data-name pusht_state.h5 \
#       --model convnext_ms --loss slow_svreg --epochs 10
#   STABLEWM_HOME=./swm_home LOCAL_DATASET_DIR=./swm_home \
#     bash scripts/run_ab_slow_svreg.sh --dataset tworoom_online --model lewm --loss slow_svreg
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="msdial"          # ogb | msdial | pusht_online | tworoom_online
DATA_NAME=""              # optional override for data.dataset.name (e.g. pusht_state.h5)
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
CEM_SAMPLES="300"
CEM_STEPS="30"
CEM_TOPK="30"
EVAL_BUDGET=""
GOAL_OFFSET=""
PROBE_FRAMES="50000"
RUN_NAME=""
SKIP_EVAL=0
SKIP_PROBE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-name) DATA_NAME="$2"; shift 2 ;;
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
    --cem-samples) CEM_SAMPLES="$2"; shift 2 ;;
    --cem-steps) CEM_STEPS="$2"; shift 2 ;;
    --cem-topk) CEM_TOPK="$2"; shift 2 ;;
    --eval-budget) EVAL_BUDGET="$2"; shift 2 ;;
    --goal-offset) GOAL_OFFSET="$2"; shift 2 ;;
    --probe-frames) PROBE_FRAMES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --skip-probe) SKIP_PROBE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,15p' "$0"
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

SWM_ROOT="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
DATA_ROOT="${LOCAL_DATASET_DIR:-$SWM_ROOT}"

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

if [[ -n "$DATA_NAME" ]]; then
  TRAIN_ARGS+=("data.dataset.name=${DATA_NAME}")
fi

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

case "$DATASET" in
  ogb)
    EVAL_MODE="swm"
    EVAL_CFG="cube"
    ;;
  msdial)
    EVAL_MODE="swm"
    EVAL_CFG="msdial"
    ;;
  pusht_online)
    EVAL_MODE="snv"
    SNV_ENV="pusht"
    : "${EVAL_BUDGET:=50}"
    : "${GOAL_OFFSET:=25}"
    ;;
  tworoom_online)
    EVAL_MODE="snv"
    SNV_ENV="tworoom"
    : "${EVAL_BUDGET:=100}"
    : "${GOAL_OFFSET:=50}"
    ;;
  *)
    echo "Unsupported --dataset: $DATASET (expected ogb|msdial|pusht_online|tworoom_online)" >&2
    exit 1
    ;;
esac

if [[ "$EVAL_MODE" == "swm" ]]; then
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
else
  EVAL_ARGS=(
    eval_snv.py
    --env "${SNV_ENV}"
    --policy "${POLICY_REL}"
    --num-eval "${NUM_EVAL}"
    --goal-offset "${GOAL_OFFSET}"
    --budget "${EVAL_BUDGET}"
    --samples "${CEM_SAMPLES}"
    --steps "${CEM_STEPS}"
    --topk "${CEM_TOPK}"
    --seed "${SEED}"
    --cache-dir "${SWM_ROOT}"
    --out "${SWM_ROOT}/checkpoints/${RUN_NAME}/results.jsonl"
  )
  if [[ -n "$DATA_NAME" ]]; then
    EVAL_ARGS+=(--data "${DATA_NAME}")
  fi
fi

run_cmd() {
  echo "+ $*"
  if (( ! DRY_RUN )); then
    "$@"
  fi
}

run_cmd "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

if (( ! SKIP_EVAL )); then
  if [[ "$EVAL_MODE" == "snv" ]]; then
    run_cmd env "PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}." "$PYTHON_BIN" "${EVAL_ARGS[@]}"
  else
    run_cmd "$PYTHON_BIN" "${EVAL_ARGS[@]}"
  fi
fi

if (( ! SKIP_PROBE )); then
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
  elif [[ "$DATASET" == "msdial" ]]; then
    PROBE_ARGS=(
      swm_msdial/probe.py
      --policy "${POLICY_REL}"
      --data "${DATA_ROOT}/datasets/msdial/msdial_val_k2_balanced.h5"
      --out "${SWM_ROOT}/checkpoints/${RUN_NAME}/probe_msdial.json"
    )
    run_cmd "$PYTHON_BIN" "${PROBE_ARGS[@]}"
  else
    echo "# Skipping probe for ${DATASET} (no dedicated physical probe script; CEM metrics saved to results.jsonl)"
  fi
fi

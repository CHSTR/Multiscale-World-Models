#!/bin/bash
# Barrido ancho 12 h MultiScaleDial (1 seed, conclusiones direccionales).
# Espera a que la cadena del piloto termine el move+symlink y el entrenamiento
# de k1 (checkpoint) y después corre:
#   Fase A: k2 x 12 brazos x s0  (cada run: train 2ep -> eval fine 30eps -> probe)
#   Fase B: k1,k3 x {lewm, starlet, convnext_ms} x s0
set -e
P=/home/chr/le-wm/.venv/bin/python
LOCAL=$HOME/.stable_worldmodel/datasets/msdial
CKPTS=$HOME/.stable_worldmodel/checkpoints
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- esperar move+symlink del piloto
while [ ! -L "$LOCAL" ]; do sleep 20; done
echo "[sweep] datasets en externo $(date)"
# --- esperar a que el piloto termine el training (checkpoint k1)
PILOT_CK=$CKPTS/msdial_k1_balanced__lewm__s0/weights_epoch_2.pt
for i in $(seq 1 180); do
  [ -f "$PILOT_CK" ] && break
  sleep 20
done
echo "[sweep] arrancando barrido $(date)"

ARM_ORDER=(lewm starlet swt wave_vit convnext_ms lewm_s8 lewm_rgbrep \
           dinov2_lora dinov2_starlet_lora dinov3_lora dinov3_starlet_lora \
           dinov3_convnext_tiny)

run() {
  local arm=$1 k=$2 style=$3 seed=$4 modes=${5:-fine}
  local kk=${k#k}
  local run=msdial_${k}_${style}__${arm}__s${seed}
  local ck=$CKPTS/$run/weights_epoch_2.pt
  if [ ! -f "$ck" ]; then
    echo "[sweep] train $run $(date +%H:%M)"
    $P train.py data=msdial model=$arm \
      data.dataset.name=msdial/msdial_train_${k}_${style}.h5 \
      trainer.max_epochs=2 loader.batch_size=64 \
      output_model_name=$run seed=$seed \
      wandb.enabled=true wandb.config.entity=chstr wandb.config.project=lewm \
      > logs/train_${run}.log 2>&1 || { echo "[sweep] TRAIN FAIL $run"; return 0; }
  fi
  for mode in $modes; do
    echo "[sweep] eval $run $mode $(date +%H:%M)"
    $P viz/eval_reward.py policy=$ck \
      eval.dataset_name=msdial/msdial_eval_${k}_${style}_${mode}.h5 \
      world.levels="[$kk]" world.dial_style=$style eval.num_eval=30 \
      --config-name=msdial > logs/eval_${run}_${mode}.log 2>&1 \
      || { echo "[sweep] EVAL FAIL $run $mode"; continue; }
  done
  $P swm_msdial/probe.py --policy $ck \
    --data $LOCAL/msdial_val_${k}_${style}.h5 \
    --out reports/probe_${run}.json > logs/probe_${run}.log 2>&1 \
    || echo "[sweep] PROBE FAIL $run"
}

# --- Fase A: k=2, todos los brazos, seed 0
for arm in "${ARM_ORDER[@]}"; do run "$arm" k2 balanced 0 fine; done
echo "[sweep] FASE A done $(date)"

# --- Fase B: tendencia k=1 y k=3 en los 3 exponentes
for k in k1 k3; do
  for arm in lewm starlet convnext_ms; do run "$arm" $k balanced 0 fine; done
done
echo "[sweep] FASE B done $(date)"
echo "[sweep] ALL DONE $(date)"

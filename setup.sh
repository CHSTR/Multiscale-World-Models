#!/usr/bin/env bash
# setup.sh — réplica rápida del entorno de entrenamiento en cualquier máquina.
#
# Basado en el repo original https://github.com/lucas-maes/le-wm :
#   uv venv --python=3.10 ; uv pip install "stable-worldmodel[train,env]"
#   datos HDF5 de https://huggingface.co/collections/quentinll/lewm
#   tar --zstd -xvf archive.tar.zst -> $STABLEWM_HOME (~/.stable-wm/)
#
# Uso:
#   ./setup.sh [--python 3.10] [--torch auto|cu128|cu126|cu124|cu121|cu118|cpu]
#              [--data tworoom|pusht|all|none] [--dino3-ckpt PATH-O-URL]
#              [--train none|smoke|baseline|dino|all] [--wandb offline|none]
#              [--recreate] [--force]
#
# Todo es idempotente: re-ejecutar no reinstala ni re-descarga salvo --recreate/--force.
# Log completo en setup.log ; pines finales en requirements.lock
set -euo pipefail

PYTHON="3.10"
TORCH="auto"
DATA="all"
DINO3_CKPT="${DINOV3_CKPT:-./checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
TRAIN="smoke"
WANDB_MODE_ARG="offline"
RECREATE=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2;;
    --torch) TORCH="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --dino3-ckpt) DINO3_CKPT="$2"; shift 2;;
    --train) TRAIN="$2"; shift 2;;
    --wandb) WANDB_MODE_ARG="$2"; shift 2;;
    --recreate) RECREATE=1; shift;;
    --force) FORCE=1; shift;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "flag desconocida: $1 (usa --help)"; exit 1;;
  esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== setup.sh $(date -u +%FT%TZ) en $ROOT ==="

# 1. Detección dinámica de CUDA (solo lectura, nunca instala drivers).
detect_torch_index() {
  local ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)"
  fi
  if [[ -z "$ver" ]] && command -v nvcc >/dev/null 2>&1; then
    ver="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)"
  fi
  if [[ -z "$ver" ]] && [[ -f /usr/local/cuda/version.txt ]]; then
    ver="$(grep -oE '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt | head -n1 || true)"
  fi
  if [[ -z "$ver" ]]; then echo "cpu"; return; fi
  local maj="${ver%%.*}" min="${ver##*.}"
  if (( maj < 11 )); then echo "cpu"; return; fi
  if (( maj == 11 )); then echo "cu118"; return; fi
  # maj == 12 (o superior: se usa el build más cercano por debajo)
  if (( min >= 8 )); then echo "cu128";
  elif (( min >= 6 )); then echo "cu126";
  elif (( min >= 4 )); then echo "cu124";
  elif (( min >= 1 )); then echo "cu121";
  else echo "cu118"; fi
}

if [[ "$TORCH" == "auto" ]]; then
  TORCH="$(detect_torch_index)"
  echo "CUDA detectada -> TORCH_INDEX=$TORCH"
else
  echo "TORCH_INDEX manual: $TORCH"
fi
case "$TORCH" in
  cu128|cu126|cu124|cu121|cu118) TORCH_URL="https://download.pytorch.org/whl/$TORCH";;
  cpu) TORCH_URL="https://download.pytorch.org/whl/cpu";;
  *) echo "TORCH inválido: $TORCH"; exit 1;;
esac

# 2. Deps del sistema (solo lectura si ya están).
#    box2d-py (vía gymnasium[all] <- stable-worldmodel[env]) compila con swig.
if ! command -v swig >/dev/null 2>&1; then
  echo "instalando swig del sistema (requerido por box2d-py)..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y swig build-essential
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y swig gcc gcc-c++ make
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --noconfirm swig base-devel
  else
    echo "AVISO: instala 'swig' a mano y re-ejecuta (sin swig falla box2d-py)."
  fi
fi

# 3. uv + venv (Python 3.10 como el original).
if ! command -v uv >/dev/null 2>&1; then
  echo "instalando uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if (( RECREATE )) && [[ -d "$ROOT/.venv" ]]; then rm -rf "$ROOT/.venv"; fi
if [[ ! -d "$ROOT/.venv" ]]; then
  uv venv --python="$PYTHON" "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# 3. Dependencias (resto de PyPI; el par torch/torchvision se fija después).
#    stable-worldmodel[train,env] arrastra stable-pretraining, lightning, hydra.
if [[ ! -f "$ROOT/.venv/.setup_done" ]] || (( FORCE )); then
  uv pip install "stable-worldmodel[train,env]" einops wandb huggingface_hub hdf5plugin h5py pyarrow-hotfix
  touch "$ROOT/.venv/.setup_done"
else
  echo "deps ya instaladas (usa --force para reinstalar)"
fi
# Deps que stable_worldmodel importa pero no declara: se verifican siempre
# (auto-repara venvs creados antes de añadirlas, sin --force).
python -c "import hdf5plugin, h5py" 2>/dev/null || uv pip install hdf5plugin h5py
# datasets necesita pa.PyExtensionType (eliminado en pyarrow nuevo);
# pyarrow-hotfix lo restaura vía .pth al arrancar el intérprete.
python -c "import pyarrow as pa; assert hasattr(pa, 'PyExtensionType')" 2>/dev/null || uv pip install pyarrow-hotfix
# Parche crítico (se verifica SIEMPRE, auto-repara venvs rotos sin --force):
# torch y torchvision deben ser matched pair del MISMO índice. Si torchvision
# viene de PyPI y torch del índice CUDA (o viceversa), `import torchvision`
# falla con "operator torchvision::nms does not exist".
if ! python -c "import torch, torchvision; torchvision.ops.nms(torch.rand(2,4), torch.rand(2), 0.5)" >/dev/null 2>&1; then
  echo "reparando pair torch/torchvision desde $TORCH_URL ..."
  uv pip install --index-url "$TORCH_URL" --force-reinstall --no-deps torch torchvision
fi
python -c "import torch, torchvision, torchmetrics, lightning; print('stack OK torch', torch.__version__, 'cuda_ok=', torch.cuda.is_available())"

# 4. Datos (flujo original: HF collection + tar --zstd -> $STABLEWM_HOME).
#    Nombres literales de este fork, tal cual están en config/train/data/:
#      tworoom.h5 , pusht_expert_train.lance (se dejan como están).
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-$STABLEWM_HOME}"
mkdir -p "$STABLEWM_HOME"
echo "STABLEWM_HOME=$STABLEWM_HOME LOCAL_DATASET_DIR=$LOCAL_DATASET_DIR"

declare -A REPOS=( [tworoom]="quentinll/lewm-tworooms" [pusht]="quentinll/lewm-pusht"
                   [cube]="quentinll/lewm-cube" [reacher]="quentinll/lewm-reacher" )
want=()
case "$DATA" in
  tworoom) want=(tworoom);; pusht) want=(pusht);;
  all) want=(tworoom pusht cube reacher);; none) want=();;
  *) echo "DATA inválido: $DATA"; exit 1;;
esac
for d in "${want[@]:-}"; do
  repo="${REPOS[$d]}"
  echo "--- dataset $d <- $repo ---"
  if (( FORCE )) || ! ls "$STABLEWM_HOME" | grep -qi "$d"; then
    hf download "$repo" --repo-type dataset --local-dir "$STABLEWM_HOME/$d" 2>/dev/null \
      || huggingface-cli download "$repo" --repo-type dataset --local-dir "$STABLEWM_HOME/$d"
    # archives estilo original: tar --zstd -xvf archive.tar.zst
    for a in "$STABLEWM_HOME/$d"/*.tar.zst "$STABLEWM_HOME"/*.tar.zst; do
      [[ -f "$a" ]] || continue
      echo "extrayendo $a ..."
      tar --zstd -xvf "$a" -C "$STABLEWM_HOME"
    done
    # aplanar si hf creó subcarpeta por repo
    shopt -s nullglob
    for f in "$STABLEWM_HOME/$d"/*.h5 "$STABLEWM_HOME/$d"/*.lance; do
      [[ -f "$STABLEWM_HOME/$(basename "$f")" ]] || cp -r "$f" "$STABLEWM_HOME/"
    done
    shopt -u nullglob
  else
    echo "dataset $d ya presente (usa --force para re-descargar)"
  fi
done
echo "contenido STABLEWM_HOME:"; ls "$STABLEWM_HOME" || true
# verificación literal (nombres tal cual, sin normalizar)
[[ "$DATA" == "none" ]] || true
if [[ " ${want[*]} " == *"tworoom"* ]]; then
  [[ -f "$STABLEWM_HOME/tworoom.h5" ]] && echo "OK tworoom.h5" || echo "AVISO: falta tworoom.h5 en $STABLEWM_HOME"
fi
if [[ " ${want[*]} " == *"pusht"* ]]; then
  [[ -e "$STABLEWM_HOME/pusht_expert_train.lance" ]] && echo "OK pusht_expert_train.lance" || echo "AVISO: falta pusht_expert_train.lance en $STABLEWM_HOME"
fi

# 5. Pesos DINO (v2 por hub = auto ; v3 vía DINOV3_CKPT, sin tocar yamls).
export DINOV3_CKPT="$DINO3_CKPT"
if [[ "$DINO3_CKPT" =~ ^https?:// ]]; then
  mkdir -p ./checkpoints
  out="./checkpoints/$(basename "$DINO3_CKPT")"
  [[ -f "$out" ]] && (( ! FORCE )) || curl -L "$DINO3_CKPT" -o "$out"
  export DINOV3_CKPT="$out"
elif [[ "$DINO3_CKPT" =~ ^file:// ]]; then
  export DINOV3_CKPT="${DINO3_CKPT#file://}"
fi
# resolver rutas relativas a absolutas (el smoke/train heredan el env)
if [[ -n "${DINOV3_CKPT:-}" && "$DINOV3_CKPT" != /* && "$DINOV3_CKPT" =~ ^\.?/ ]]; then
  DINOV3_CKPT="$(realpath -m "$ROOT/$DINOV3_CKPT" 2>/dev/null || echo "$ROOT/$DINOV3_CKPT")"
  export DINOV3_CKPT
fi
if [[ -f "$DINOV3_CKPT" ]]; then
  echo "OK DINOV3_CKPT=$DINOV3_CKPT"
else
  # compat con la ruta original de tu máquina principal
  if [[ -f "/home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" ]]; then
    export DINOV3_CKPT="/home/chr/dinov3_wm/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    echo "OK DINOV3_CKPT (legacy)=$DINOV3_CKPT"
  else
    echo "AVISO: no existe DINOV3_CKPT=$DINOV3_CKPT (pásalo con --dino3-ckpt o \$DINOV3_CKPT). DINOv2 seguirá funcionando por hub."
  fi
fi
echo "DINOv2: descarga automática por torch.hub en el primer uso (requiere internet)."

# 6. Verificación (sin entrenar).
python smoke_dino_encoders.py

# 7. Entrenamiento (matriz rápida).
if [[ "$WANDB_MODE_ARG" == "offline" ]]; then export WANDB_MODE=offline; fi
run() { echo "+ $*"; "$@"; }
case "$TRAIN" in
  none) echo "train omitido";;
  smoke) run python train.py dataset=tworoom model=lewm trainer.max_epochs=1;;
  baseline) run python train.py dataset=tworoom model=lewm;;
  dino)
    run python train.py dataset=tworoom model=dinov2_lora "model.encoder.ckpt_path=null"
    run python train.py dataset=tworoom model=dinov3_lora "model.encoder.ckpt_path=$DINOV3_CKPT"
    ;;
  all)
    run python train.py dataset=tworoom model=lewm
    run python train.py dataset=tworoom model=dinov2_lora "model.encoder.ckpt_path=null"
    run python train.py dataset=tworoom model=dinov3_lora "model.encoder.ckpt_path=$DINOV3_CKPT"
    ;;
  *) echo "TRAIN inválido: $TRAIN"; exit 1;;
esac

# 8. Réplica: pines finales.
uv pip freeze > "$ROOT/requirements.lock"
echo "=== setup OK. lock en requirements.lock, log en setup.log ==="

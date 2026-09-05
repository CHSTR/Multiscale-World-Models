#!/usr/bin/env bash
# setup.sh — réplica rápida del entorno de entrenamiento en cualquier máquina.
#
# Basado en el repo original https://github.com/lucas-maes/le-wm , con el fix
# de su upstream (galilai-group/stable-worldmodel desde fuente):
#   uv venv --python=3.10
#   git clone https://github.com/galilai-group/stable-worldmodel
#   uv pip install -e ".[all]"   (= train,env,format,data: trae hdf5plugin,
#                                  stack lance, gymnasium[all] y pines compatibles)
#   datos de https://huggingface.co/collections/quentinll/lewm
#   (tar --zstd si aplica) -> $STABLEWM_HOME/datasets/ (layout que exige el loader)
#
# Uso:
#   ./setup.sh [--python 3.10] [--torch auto|cu128|cu126|cu124|cu121|cu118|cpu]
#              [--data tworoom|pusht|cube|reacher|all|none] [--dino3-ckpt PATH-O-URL]
#              [--swm-src DIR] [--train none|smoke|baseline|dino|all]
#              [--wandb offline|none] [--recreate] [--force]
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
    --swm-src) SWM_SRC="$2"; shift 2;;
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
SWM_SRC="${SWM_SRC:-$ROOT/../stable-worldmodel}"
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

# 3. stable-worldmodel desde fuente (editable, extras [all]).
#    El sdist de PyPI deja fuera pines que el repo sí resuelve; instalar desde
#    git con [all] (= train,env,format,data) trae además hdf5plugin/h5py,
#    torchcodec, el stack lance (lancedb/pylance/pyarrow) y gymnasium[all].
#    Equivale al fix manual: git clone stable-worldmodel + uv pip install -e ".[all]".
if [[ ! -d "$SWM_SRC/.git" ]]; then
  echo "clonando stable-worldmodel en $SWM_SRC ..."
  git clone https://github.com/galilai-group/stable-worldmodel "$SWM_SRC"
elif (( FORCE )); then
  echo "actualizando stable-worldmodel en $SWM_SRC ..."
  git -C "$SWM_SRC" pull --ff-only || echo "AVISO: no se pudo actualizar $SWM_SRC, se sigue con lo que hay"
else
  echo "stable-worldmodel ya clonado en $SWM_SRC (usa --force para actualizar)"
fi
if [[ ! -f "$ROOT/.venv/.setup_done_v2" ]] || (( FORCE )); then
  uv pip install -e "$SWM_SRC[all]" einops wandb huggingface_hub pyarrow-hotfix
  touch "$ROOT/.venv/.setup_done_v2"
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
# falla con "operator torchvision::nms does not exist". Además, si hay GPU
# pero torch no la ve (wheel CPU), se reinstala del índice CUDA detectado.
NEED_TORCH_FIX=0
python -c "import torch, torchvision; torchvision.ops.nms(torch.rand(2,4), torch.rand(2), 0.5)" >/dev/null 2>&1 || NEED_TORCH_FIX=1
if (( ! NEED_TORCH_FIX )) && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1 || NEED_TORCH_FIX=1
fi
if (( NEED_TORCH_FIX )); then
  echo "reparando pair torch/torchvision desde $TORCH_URL ..."
  uv pip install --index-url "$TORCH_URL" --force-reinstall --no-deps torch torchvision
fi
python -c "import torch, torchvision, torchmetrics, lightning; print('stack OK torch', torch.__version__, 'cuda_ok=', torch.cuda.is_available())"

# 4. Datos (HF collection quentinll/lewm).
#    El loader actual (stable_worldmodel.data.load_dataset) resuelve nombres
#    contra <cache>/datasets/ (cache = $LOCAL_DATASET_DIR o $STABLEWM_HOME):
#    el nombre debe existir ahí como archivo o directorio. Los yamls de este
#    fork usan nombres literales con extensión (tworoom.h5,
#    pusht_expert_train.lance), así que se colocan TAL CUAL bajo datasets/.
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-$STABLEWM_HOME}"
DS_DIR="$LOCAL_DATASET_DIR/datasets"
mkdir -p "$DS_DIR"
echo "STABLEWM_HOME=$STABLEWM_HOME LOCAL_DATASET_DIR=$LOCAL_DATASET_DIR DS_DIR=$DS_DIR"

place_dataset_files() { # $1 = dir descargado -> copia *.h5/*.lance a $DS_DIR
  local src="$1" f
  shopt -s nullglob
  for pat in "$src"/*.h5 "$src"/*.hdf5 "$src"/*/*.h5 "$src"/*/*.hdf5; do
    [[ -f "$pat" ]] || continue
    cp -f "$pat" "$DS_DIR/" && echo "dataset: $DS_DIR/$(basename "$pat")"
  done
  for pat in "$src"/*.lance "$src"/*/*.lance; do
    [[ -e "$pat" ]] || continue
    rm -rf "$DS_DIR/$(basename "$pat")"
    cp -r "$pat" "$DS_DIR/" && echo "dataset: $DS_DIR/$(basename "$pat")"
  done
  shopt -u nullglob
}

declare -A REPOS=( [tworoom]="quentinll/lewm-tworooms" [pusht]="quentinll/lewm-pusht"
                   [cube]="quentinll/lewm-cube" [reacher]="quentinll/lewm-reacher" )
want=()
case "$DATA" in
  tworoom) want=(tworoom);; pusht) want=(pusht);; cube) want=(cube);; reacher) want=(reacher);;
  all) want=(tworoom pusht cube reacher);; none) want=();;
  *) echo "DATA inválido: $DATA"; exit 1;;
esac
for d in "${want[@]:-}"; do
  repo="${REPOS[$d]}"
  echo "--- dataset $d <- $repo ---"
  dl="$STABLEWM_HOME/_dl_$d"
  if (( FORCE )) || ! ls "$DS_DIR" | grep -qi "$d"; then
    mkdir -p "$dl"
    hf download "$repo" --repo-type dataset --local-dir "$dl" 2>/dev/null \
      || huggingface-cli download "$repo" --repo-type dataset --local-dir "$dl"
    # archives estilo original: tar --zstd -xvf archive.tar.zst (contenedor).
    # pusht viene como pusht_expert_train.h5.zst: un stream zstd PLANO (no tar),
    # se descomprime con zstd -d y queda el .h5 al lado.
    for a in "$dl"/*.tar.zst; do
      [[ -f "$a" ]] || continue
      echo "extrayendo (tar) $a ..."
      tar --zstd -xvf "$a" -C "$dl"
    done
    for a in "$dl"/*.h5.zst; do
      [[ -f "$a" ]] || continue
      echo "descomprimiendo (zstd) $a ..."
      zstd -d -f "$a"
    done
    place_dataset_files "$dl"
    # cube: name: ogbench/cube_single_expert.h5 -> resuelve a <DS_DIR>/ogbench/...
    if [[ "$d" == "cube" ]]; then
      mkdir -p "$DS_DIR/ogbench"
      for pat in "$dl"/*.h5 "$dl"/*/*.h5; do
        [[ -f "$pat" ]] || continue
        [[ "$(basename "$pat")" == "cube_single_expert.h5" ]] || continue
        cp -f "$pat" "$DS_DIR/ogbench/cube_single_expert.h5"
        echo "dataset: $DS_DIR/ogbench/cube_single_expert.h5"
      done
    fi
  else
    echo "dataset $d ya presente en $DS_DIR (usa --force para re-descargar)"
  fi
done
echo "contenido DS_DIR:"; ls "$DS_DIR" || true
# verificación: los nombres que resuelve el loader (literales de los yamls)
if [[ " ${want[*]} " == *"tworoom"* ]]; then
  [[ -f "$DS_DIR/tworoom.h5" ]] && echo "OK datasets/tworoom.h5" || echo "AVISO: falta datasets/tworoom.h5 (name: tworoom.h5 no resolverá)"
fi
if [[ " ${want[*]} " == *"pusht"* ]]; then
  [[ -f "$DS_DIR/pusht_expert_train.h5" ]] && echo "OK datasets/pusht_expert_train.h5" || echo "AVISO: falta datasets/pusht_expert_train.h5 (name: pusht_expert_train.h5 no resolverá)"
fi
if [[ " ${want[*]} " == *"cube"* ]]; then
  [[ -f "$DS_DIR/ogbench/cube_single_expert.h5" ]] && echo "OK datasets/ogbench/cube_single_expert.h5" || echo "AVISO: falta datasets/ogbench/cube_single_expert.h5 (name: ogbench/cube_single_expert.h5 no resolverá)"
fi
if [[ " ${want[*]} " == *"reacher"* ]]; then
  [[ -f "$DS_DIR/reacher.h5" ]] && echo "OK datasets/reacher.h5" || echo "AVISO: falta datasets/reacher.h5 (name: reacher.h5 no resolverá)"
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
  smoke) run python train.py data=tworoom model=lewm trainer.max_epochs=1;;
  baseline) run python train.py data=tworoom model=lewm;;
  dino)
    run python train.py data=tworoom model=dinov2_lora "model.encoder.ckpt_path=null"
    run python train.py data=tworoom model=dinov3_lora "model.encoder.ckpt_path=$DINOV3_CKPT"
    ;;
  all)
    run python train.py data=tworoom model=lewm
    run python train.py data=tworoom model=dinov2_lora "model.encoder.ckpt_path=null"
    run python train.py data=tworoom model=dinov3_lora "model.encoder.ckpt_path=$DINOV3_CKPT"
    ;;
  *) echo "TRAIN inválido: $TRAIN"; exit 1;;
esac

# 8. Réplica: pines finales.
uv pip freeze > "$ROOT/requirements.lock"
echo "=== setup OK. lock en requirements.lock, log en setup.log ==="

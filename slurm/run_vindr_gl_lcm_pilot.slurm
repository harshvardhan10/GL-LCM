#!/bin/bash -l
#SBATCH --job-name=gl-lcm-vindr-pilot
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/gl_lcm_vindr_pilot_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/gl_lcm_vindr_pilot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

# ---------------------------------------------------------------------------
# Compute-node internet access
# ---------------------------------------------------------------------------
PROXY="http://proxy.nhr.fau.de:80"

export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"

unset all_proxy ALL_PROXY no_proxy NO_PROXY

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT="/home/vault/iwi5/iwi5362h/data/bone_suppression/gl_lcm"

REPO_DIR="${ROOT}"
REPO_CODES="${ROOT}/codes"

CHECKPOINT_DIR="${ROOT}/checkpoints"
ENV_DIR="${ROOT}/gl_lcm_inference_env"
SCRIPT_DIR="${ROOT}/scripts"

VINDR_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr"
VINDR_TRAIN_IMAGES="${VINDR_ROOT}/train"

PILOT_DIR="${ROOT}/vindr_train_pilot"
SZCH_OUTPUT="${PILOT_DIR}/outputs_szch"
JSRT_OUTPUT="${PILOT_DIR}/outputs_jsrt"

export TORCH_HOME="${ROOT}/torch_cache"
export XDG_CACHE_HOME="${ROOT}/cache"
export MPLCONFIGDIR="${ROOT}/matplotlib_cache"

mkdir -p \
    "${PILOT_DIR}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${MPLCONFIGDIR}" \
    /home/vault/iwi5/iwi5362h/logs

# ---------------------------------------------------------------------------
# Load Python/Conda
# ---------------------------------------------------------------------------
module load python

if ! command -v conda >/dev/null 2>&1; then
    echo "[Error] Conda is unavailable after 'module load python'."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

ANNOTATIONS_CSV="${VINDR_ROOT}/annotations/annotations_train.csv"

# ---------------------------------------------------------------------------
# Validate all required paths
# ---------------------------------------------------------------------------
required_paths=(
    "${ENV_DIR}/bin/python"
    "${REPO_CODES}/model.py"
    "${REPO_CODES}/config.py"
    "${REPO_CODES}/modules/unet.py"
    "${SCRIPT_DIR}/prepare_vindr_train_pilot.py"
    "${SCRIPT_DIR}/infer_gl_lcm_external.py"
    "${SCRIPT_DIR}/make_gl_lcm_pilot_montage.py"
    "${CHECKPOINT_DIR}/vqgan_szch.pth"
    "${CHECKPOINT_DIR}/unet_szch.pth"
    "${CHECKPOINT_DIR}/vqgan_jsrt.pth"
    "${CHECKPOINT_DIR}/unet_jsrt.pth"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[Error] Required path missing: ${path}"
        exit 1
    fi
done

if [[ ! -d "${VINDR_TRAIN_IMAGES}" ]]; then
    echo "[Error] VinDr train image directory missing:"
    echo "        ${VINDR_TRAIN_IMAGES}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Runtime verification
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "VinDr train GL-LCM inference pilot"
echo "Host:              $(hostname)"
echo "Started:           $(date --iso-8601=seconds)"
echo "Repository:        ${REPO_DIR}"
echo "Scripts:           ${SCRIPT_DIR}"
echo "VinDr train:       ${VINDR_TRAIN_IMAGES}"
echo "Annotations:       ${ANNOTATIONS_CSV:-not found; random sampling will be used}"
echo "Pilot directory:   ${PILOT_DIR}"
echo "Conda prefix:      ${CONDA_PREFIX}"
echo "Python:            $(which python)"
echo "======================================================================"

python --version

echo
echo "[GPU]"
nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

echo
echo "[Verify] Python environment"

python - <<'PY'
import sys

import cv2
import diffusers
import generative
import monai
import torch
import torchvision

import lungs_segmentation
from lungs_segmentation.pre_trained_models import create_model

print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the allocated A40 job")

print("GPU:", torch.cuda.get_device_name(0))
print("OpenCV:", cv2.__version__)
print("Diffusers:", diffusers.__version__)
print("MONAI:", monai.__version__)
print("Core imports successful")
PY

# ---------------------------------------------------------------------------
# 1. Select ten VinDr train images and prepare lung masks
# ---------------------------------------------------------------------------
PREPARE_ARGS=(
    --images_root "${VINDR_TRAIN_IMAGES}"
    --output_dir "${PILOT_DIR}"
    --num_images 10
    --seed 42
    --image_size 1024
    --segmentation_model resnet34
    --mask_threshold 0.20
)

if [[ -n "${ANNOTATIONS_CSV}" ]]; then
    PREPARE_ARGS+=(--annotations_csv "${ANNOTATIONS_CSV}")
fi

echo
echo "[Stage 1/4] Preparing VinDr images and lung masks"

python "${SCRIPT_DIR}/prepare_vindr_train_pilot.py" \
    "${PREPARE_ARGS[@]}"

# ---------------------------------------------------------------------------
# 2. Run SZCH checkpoint pair
# ---------------------------------------------------------------------------
echo
echo "[Stage 2/4] Running SZCH checkpoint pair"

cd "${REPO_CODES}"

python "${SCRIPT_DIR}/infer_gl_lcm_external.py" \
    --repo_codes_dir "${REPO_CODES}" \
    --input_dir "${PILOT_DIR}/CXR" \
    --mask_dir "${PILOT_DIR}/Mask" \
    --masked_input_dir "${PILOT_DIR}/Masked_CXR" \
    --file_list "${PILOT_DIR}/pilot_files.txt" \
    --output_dir "${SZCH_OUTPUT}" \
    --vqgan_checkpoint "${CHECKPOINT_DIR}/vqgan_szch.pth" \
    --unet_checkpoint "${CHECKPOINT_DIR}/unet_szch.pth" \
    --domain szch \
    --num_infer_steps 50 \
    --seed 42 \
    --device cuda

# Each inference command is a separate Python process, so its GPU memory is
# released when it exits. This is only a lightweight sanity check.
python - <<'PY'
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(
    "CUDA memory currently allocated:",
    torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
)
PY

# ---------------------------------------------------------------------------
# 3. Run JSRT checkpoint pair
# ---------------------------------------------------------------------------
echo
echo "[Stage 3/4] Running JSRT checkpoint pair"

python "${SCRIPT_DIR}/infer_gl_lcm_external.py" \
    --repo_codes_dir "${REPO_CODES}" \
    --input_dir "${PILOT_DIR}/CXR" \
    --mask_dir "${PILOT_DIR}/Mask" \
    --masked_input_dir "${PILOT_DIR}/Masked_CXR" \
    --file_list "${PILOT_DIR}/pilot_files.txt" \
    --output_dir "${JSRT_OUTPUT}" \
    --vqgan_checkpoint "${CHECKPOINT_DIR}/vqgan_jsrt.pth" \
    --unet_checkpoint "${CHECKPOINT_DIR}/unet_jsrt.pth" \
    --domain jsrt \
    --num_infer_steps 50 \
    --seed 42 \
    --device cuda

# ---------------------------------------------------------------------------
# 4. Produce comparison montage
# ---------------------------------------------------------------------------
echo
echo "[Stage 4/4] Creating comparison montage"

python "${SCRIPT_DIR}/make_gl_lcm_pilot_montage.py" \
    --pilot_dir "${PILOT_DIR}" \
    --szch_dir "${SZCH_OUTPUT}" \
    --jsrt_dir "${JSRT_OUTPUT}" \
    --output "${PILOT_DIR}/vindr_train_gl_lcm_comparison.png"

echo
echo "======================================================================"
echo "Pilot complete"
echo "======================================================================"

echo
echo "Manifest:"

if command -v column >/dev/null 2>&1; then
    column -s, -t < "${PILOT_DIR}/pilot_manifest.csv"
else
    cat "${PILOT_DIR}/pilot_manifest.csv"
fi

echo
echo "Generated files:"
find "${PILOT_DIR}" -maxdepth 3 -type f | sort

echo
echo "Pilot storage usage:"
du -sh "${PILOT_DIR}"

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"
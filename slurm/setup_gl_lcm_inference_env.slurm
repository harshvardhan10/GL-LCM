#!/bin/bash -l
#SBATCH --job-name=gl-lcm-env
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/gl_lcm_env_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/gl_lcm_env_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

PROXY="http://proxy.nhr.fau.de:80"
export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"
unset all_proxy ALL_PROXY no_proxy NO_PROXY

ROOT="/home/vault/iwi5/iwi5362h/data/bone_suppression/gl_lcm"
ENV_DIR="${ROOT}/gl_lcm_inference_env"
CONDA_PKGS_DIRS="${ROOT}/conda_package_cache"
PIP_CACHE_DIR="${ROOT}/pip_cache"
TORCH_HOME="${ROOT}/torch_cache"
XDG_CACHE_HOME="${ROOT}/cache"

export CONDA_PKGS_DIRS PIP_CACHE_DIR TORCH_HOME XDG_CACHE_HOME
export PIP_NO_CACHE_DIR=1

mkdir -p \
    "${ROOT}" \
    "${CONDA_PKGS_DIRS}" \
    "${PIP_CACHE_DIR}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    /home/woody/iwi5/iwi5362h/logs

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    local candidates=(
        "${HOME}/miniconda3/bin/conda"
        "${HOME}/anaconda3/bin/conda"
        "${HOME}/miniforge3/bin/conda"
        "${HOME}/mambaforge/bin/conda"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

CONDA_EXE="$(find_conda || true)"
if [[ -z "${CONDA_EXE}" ]]; then
    echo "[Error] Conda was not found."
    module avail 2>&1 | grep -Ei "conda|anaconda|miniconda|miniforge" || true
    exit 1
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[Error] Conda initialization script not found: ${CONDA_SH}"
    exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"

echo "======================================================================"
echo "GL-LCM inference environment setup"
echo "Host:        $(hostname)"
echo "Started:     $(date --iso-8601=seconds)"
echo "Environment: ${ENV_DIR}"
echo "Conda:       $(command -v conda)"
echo "======================================================================"

if [[ -d "${ENV_DIR}" && ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "[Conda] Removing incomplete environment: ${ENV_DIR}"
    rm -rf "${ENV_DIR}"
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    conda create \
        --yes \
        --prefix "${ENV_DIR}" \
        python=3.10 \
        pip
else
    echo "[Conda] Existing environment found: ${ENV_DIR}"
fi

# The repository was developed with PyTorch 2.0.1 + CUDA 11.7.
conda install \
    --yes \
    --prefix "${ENV_DIR}" \
    -c pytorch \
    -c nvidia \
    pytorch=2.0.1 \
    torchvision=0.15.2 \
    torchaudio=2.0.2 \
    pytorch-cuda=11.7

conda run \
    --no-capture-output \
    --prefix "${ENV_DIR}" \
    python -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

# Install the non-PyTorch dependencies explicitly. huggingface-hub is pinned
# because diffusers 0.27.2 is incompatible with recent hub releases that
# removed cached_download.
conda run \
    --no-capture-output \
    --prefix "${ENV_DIR}" \
    python -m pip install \
    --no-cache-dir \
    "diffusers==0.27.2" \
    "huggingface-hub==0.20.3" \
    "monai==1.2.0" \
    "monai-generative==0.2.2" \
    "numpy==1.26.4" \
    "opencv-python-headless==4.8.1.78" \
    "pandas==2.0.3" \
    "Pillow==9.3.0" \
    "scikit-image==0.22.0" \
    "scikit-learn==1.3.1" \
    "scipy==1.11.3" \
    "tqdm==4.66.1" \
    "torch-ema==0.3" \
    "torchsummary==1.5.1" \
    "matplotlib==3.7.2" \
    "albumentations==1.3.1" \
    "segmentation-models-pytorch==0.2.1" \
    "pretrainedmodels==0.7.4" \
    "efficientnet-pytorch==0.7.1" \
    "timm==0.6.13"

# The official GL-LCM notebook uses this package for lung masks.
# --no-deps avoids it replacing the controlled PyTorch stack above.
conda run \
    --no-capture-output \
    --prefix "${ENV_DIR}" \
    python -m pip install \
    --no-cache-dir \
    --no-deps \
    "lungs-segmentation==1.1.1"

echo
echo "[Verify] Core imports and versions"
conda run \
    --no-capture-output \
    --prefix "${ENV_DIR}" \
    python - <<'PY'
import sys
import torch
import torchvision
import cv2
import diffusers
import monai
import generative
import numpy
import pandas
import lungs_segmentation
from lungs_segmentation.pre_trained_models import create_model

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA runtime visible to Torch:", torch.version.cuda)
print("OpenCV:", cv2.__version__)
print("Diffusers:", diffusers.__version__)
print("MONAI:", monai.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("lungs_segmentation import: OK")
print("create_model import: OK")
PY

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"

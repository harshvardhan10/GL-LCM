#!/bin/bash -l
#SBATCH --job-name=gl-lcm-env
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/gl_lcm_env_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/gl_lcm_env_%j.err
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
ENV_DIR="${ROOT}/gl_lcm_inference_env"

export CONDA_PKGS_DIRS="${ROOT}/conda_package_cache"
export PIP_CACHE_DIR="${ROOT}/pip_cache"
export TORCH_HOME="${ROOT}/torch_cache"
export XDG_CACHE_HOME="${ROOT}/cache"

# Packages should not remain duplicated in the pip cache.
export PIP_NO_CACHE_DIR=1

mkdir -p \
    "${ROOT}" \
    "${CONDA_PKGS_DIRS}" \
    "${PIP_CACHE_DIR}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    /home/vault/iwi5/iwi5362h/logs

# ---------------------------------------------------------------------------
# Load Conda from the cluster module
# ---------------------------------------------------------------------------
module load python

if ! command -v conda >/dev/null 2>&1; then
    echo "[Error] Conda is unavailable after 'module load python'."
    exit 1
fi

# Makes `conda activate` available in a non-interactive Slurm shell.
eval "$(conda shell.bash hook)"

echo "======================================================================"
echo "GL-LCM inference environment setup"
echo "Host:        $(hostname)"
echo "Started:     $(date --iso-8601=seconds)"
echo "Environment: ${ENV_DIR}"
echo "Conda:       $(command -v conda)"
echo "Version:     $(conda --version)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Create or reuse the environment
# ---------------------------------------------------------------------------
if [[ -d "${ENV_DIR}" && ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "[Conda] Removing incomplete environment: ${ENV_DIR}"
    rm -rf "${ENV_DIR}"
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "[Conda] Creating environment."

    conda create \
        --yes \
        --prefix "${ENV_DIR}" \
        python=3.10 \
        pip
else
    echo "[Conda] Existing environment found."
fi

conda activate "${ENV_DIR}"

echo "[Conda] Active prefix: ${CONDA_PREFIX}"
echo "[Conda] Python:        $(which python)"
python --version

# ---------------------------------------------------------------------------
# Install PyTorch and CUDA runtime
# ---------------------------------------------------------------------------
conda install \
    --yes \
    -c pytorch \
    -c nvidia \
    pytorch=2.0.1 \
    torchvision=0.15.2 \
    torchaudio=2.0.2 \
    pytorch-cuda=11.7

python -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

# ---------------------------------------------------------------------------
# Install GL-LCM dependencies
# ---------------------------------------------------------------------------
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

# The official GL-LCM notebook uses this package for lung segmentation.
# --no-deps prevents it from modifying the controlled PyTorch environment.
python -m pip install \
    --no-cache-dir \
    --no-deps \
    "lungs-segmentation==1.1.1"

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
echo
echo "[Verify] Core imports and versions"

python - <<'PY'
import sys

import cv2
import diffusers
import generative
import monai
import numpy
import pandas
import torch
import torchvision

import lungs_segmentation
from lungs_segmentation.pre_trained_models import create_model

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA runtime packaged with Torch:", torch.version.cuda)
print("CUDA available on this node:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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
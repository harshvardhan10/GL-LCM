#!/bin/bash -l
#SBATCH --job-name=gl-lcm-vindr-pilot
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/woody/iwi5/iwi5362h/logs/gl_lcm_vindr_pilot_%j.out
#SBATCH --error=/home/woody/iwi5/iwi5362h/logs/gl_lcm_vindr_pilot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

# ---------------------------------------------------------------------------
# Internet access is needed the first time lungs-segmentation downloads weights.
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
REPO_DIR="${ROOT}/GL-LCM"
REPO_CODES="${REPO_DIR}/codes"
CHECKPOINT_DIR="${ROOT}/checkpoints"
ENV_DIR="${ROOT}/gl_lcm_inference_env"
SCRIPT_DIR="${ROOT}/pilot_scripts"

VINDR_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr"
VINDR_TRAIN_IMAGES="${VINDR_ROOT}/train"

PILOT_DIR="${ROOT}/vindr_train_pilot"
SZCH_OUTPUT="${PILOT_DIR}/outputs_szch"
JSRT_OUTPUT="${PILOT_DIR}/outputs_jsrt"

TORCH_HOME="${ROOT}/torch_cache"
XDG_CACHE_HOME="${ROOT}/cache"
MPLCONFIGDIR="${ROOT}/matplotlib_cache"
export TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR

mkdir -p \
    "${PILOT_DIR}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${MPLCONFIGDIR}" \
    /home/woody/iwi5/iwi5362h/logs

# ---------------------------------------------------------------------------
# Locate VinDr train annotations. The pilot still works without annotations,
# but label-aware sampling is preferred.
# ---------------------------------------------------------------------------
ANNOTATION_CANDIDATES=(
    "${VINDR_ROOT}/annotations_train.csv"
    "${VINDR_ROOT}/annotation_train.csv"
    "${VINDR_ROOT}/train.csv"
    "${VINDR_ROOT}/annotations/annotations_train.csv"
)

ANNOTATIONS_CSV=""
for candidate in "${ANNOTATION_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
        ANNOTATIONS_CSV="${candidate}"
        break
    fi
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
required_paths=(
    "${ENV_DIR}/bin/python"
    "${REPO_CODES}/model.py"
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
    echo "[Error] VinDr train image directory missing: ${VINDR_TRAIN_IMAGES}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate Conda
# ---------------------------------------------------------------------------
CONDA_EXE="$(command -v conda || true)"
if [[ -z "${CONDA_EXE}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/bin/conda" \
        "${HOME}/anaconda3/bin/conda" \
        "${HOME}/miniforge3/bin/conda" \
        "${HOME}/mambaforge/bin/conda"; do
        if [[ -x "${candidate}" ]]; then
            CONDA_EXE="${candidate}"
            break
        fi
    done
fi

if [[ -z "${CONDA_EXE}" ]]; then
    echo "[Error] Conda not found"
    exit 1
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

echo "======================================================================"
echo "VinDr train GL-LCM inference pilot"
echo "Host:              $(hostname)"
echo "Started:           $(date --iso-8601=seconds)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Python:            $(which python)"
python --version
echo "Repository:        ${REPO_DIR}"
echo "VinDr train:       ${VINDR_TRAIN_IMAGES}"
echo "Annotations:       ${ANNOTATIONS_CSV:-not found; random sampling will be used}"
echo "Pilot directory:   ${PILOT_DIR}"
echo "======================================================================"

# ---------------------------------------------------------------------------
# 1. Select ten VinDr train images and prepare lung masks.
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

python "${SCRIPT_DIR}/prepare_vindr_train_pilot.py" "${PREPARE_ARGS[@]}"

# ---------------------------------------------------------------------------
# 2. Run the SZCH checkpoint pair.
# ---------------------------------------------------------------------------
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

# Free Python/CUDA process memory before loading the second checkpoint pair.
python - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

# ---------------------------------------------------------------------------
# 3. Run the JSRT checkpoint pair with the same selected images and noise seeds.
# ---------------------------------------------------------------------------
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
# 4. Produce a contact sheet for visual model selection.
# ---------------------------------------------------------------------------
python "${SCRIPT_DIR}/make_gl_lcm_pilot_montage.py" \
    --pilot_dir "${PILOT_DIR}" \
    --szch_dir "${SZCH_OUTPUT}" \
    --jsrt_dir "${JSRT_OUTPUT}" \
    --output "${PILOT_DIR}/vindr_train_gl_lcm_comparison.png"

echo
echo "======================================================================"
echo "Pilot complete"
echo "Manifest:"
column -s, -t < "${PILOT_DIR}/pilot_manifest.csv" || cat "${PILOT_DIR}/pilot_manifest.csv"
echo
echo "Outputs:"
find "${PILOT_DIR}" -maxdepth 3 -type f | sort
echo
du -sh "${PILOT_DIR}"
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"

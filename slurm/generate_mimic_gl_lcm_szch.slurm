#!/bin/bash -l
#SBATCH --job-name=gl-lcm-mimic-szch
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-430%8
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/gl_lcm_mimic_szch_%A_%a.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/gl_lcm_mimic_szch_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT="/home/vault/iwi5/iwi5362h/data/bone_suppression/gl_lcm"
REPO_CODES="${ROOT}/codes"
CHECKPOINT_DIR="${ROOT}/checkpoints"
ENV_DIR="${ROOT}/gl_lcm_inference_env"

SCRIPT="${ROOT}/scripts/generate_mimic_gl_lcm_szch.py"

MANIFEST="/home/woody/iwi5/iwi5362h/ALBEF/data/chexmask_manifests/mimic_cxr_chexmask_common_valid_lung_maskcache.json"
MASK_ROOT="/home/woody/iwi5/iwi5362h/data/chexmask/mask_cache/common/lung"

OUTPUT_ROOT="/home/woody/iwi5/iwi5362h/data/bone_suppressed_szch_256"
STATUS_DIR="${OUTPUT_ROOT}/_status"

LOG_DIR="/home/vault/iwi5/iwi5362h/logs"

mkdir -p "${OUTPUT_ROOT}" "${STATUS_DIR}" "${LOG_DIR}"

# Keep framework caches out of the home quota.
export TORCH_HOME="${ROOT}/torch_cache"
export XDG_CACHE_HOME="${ROOT}/cache"
export MPLCONFIGDIR="${ROOT}/matplotlib_cache"
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python

CONDA_EXE="$(command -v conda || true)"
if [[ -z "${CONDA_EXE}" ]]; then
    echo "[ERROR] conda was not found after 'module load python'" >&2
    exit 1
fi

CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
for path in \
    "${SCRIPT}" \
    "${MANIFEST}" \
    "${MASK_ROOT}" \
    "${REPO_CODES}/model.py" \
    "${CHECKPOINT_DIR}/vqgan_szch.pth" \
    "${CHECKPOINT_DIR}/unet_szch.pth"
do
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] Required path missing: ${path}" >&2
        exit 2
    fi
done

echo "================================================================================"
echo "MIMIC-CXR SZCH GL-LCM production generation"
echo "Host:                 $(hostname)"
echo "Started:              $(date --iso-8601=seconds)"
echo "Array job/task:       ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Python:               $(which python)"
echo "Manifest:             ${MANIFEST}"
echo "Mask root:            ${MASK_ROOT}"
echo "Output root:          ${OUTPUT_ROOT}"
echo "Items per task:       500"
echo "Maximum concurrency:  8"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "================================================================================"

cd "${REPO_CODES}"

python -u "${SCRIPT}" \
    --repo_codes_dir "${REPO_CODES}" \
    --manifest "${MANIFEST}" \
    --mask_root "${MASK_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --status_dir "${STATUS_DIR}" \
    --vqgan_checkpoint "${CHECKPOINT_DIR}/vqgan_szch.pth" \
    --unet_checkpoint "${CHECKPOINT_DIR}/unet_szch.pth" \
    --shard_index "${SLURM_ARRAY_TASK_ID}" \
    --items_per_shard 500 \
    --gl_lcm_size 1024 \
    --output_size 256 \
    --num_infer_steps 50 \
    --num_train_timesteps 1000 \
    --alpha 3.0 \
    --base_seed 42 \
    --min_mask_component_area 100 \
    --png_compression 6 \
    --max_errors 25 \
    --device cuda

echo "Finished: $(date --iso-8601=seconds)"

#!/bin/bash -l
#SBATCH --job-name=gl-lcm-finalize
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/gl_lcm_finalize_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/gl_lcm_finalize_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

ROOT="/home/vault/iwi5/iwi5362h/data/bone_suppression/gl_lcm"
ENV_DIR="${ROOT}/gl_lcm_inference_env"
SCRIPT="${ROOT}/scripts/finalize_mimic_gl_lcm_szch.py"

SOURCE_MANIFEST="/home/woody/iwi5/iwi5362h/ALBEF/data/chexmask_manifests/mimic_cxr_chexmask_common_valid_lung_maskcache.json"
OUTPUT_ROOT="/home/woody/iwi5/iwi5362h/data/bone_suppressed_szch_256"

OUTPUT_MANIFEST="/home/woody/iwi5/iwi5362h/ALBEF/data/chexmask_manifests/mimic_cxr_bone_suppressed_szch_256.json"
MISSING_CSV="${OUTPUT_ROOT}/_status/missing_or_invalid_outputs.csv"

module load python

CONDA_EXE="$(command -v conda)"
CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

for path in "${SCRIPT}" "${SOURCE_MANIFEST}" "${OUTPUT_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] Required path missing: ${path}" >&2
        exit 2
    fi
done

python -u "${SCRIPT}" \
    --source_manifest "${SOURCE_MANIFEST}" \
    --output_root "${OUTPUT_ROOT}" \
    --output_manifest "${OUTPUT_MANIFEST}" \
    --missing_csv "${MISSING_CSV}" \
    --expected_size 256

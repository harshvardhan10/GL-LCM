#!/bin/bash -l
#SBATCH --job-name=vindr-bbox-review
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/vault/iwi5/iwi5362h/logs/vindr_bbox_review_%j.out
#SBATCH --error=/home/vault/iwi5/iwi5362h/logs/vindr_bbox_review_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harshvardhan.agarwal@fau.de

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT="/home/vault/iwi5/iwi5362h/data/bone_suppression/gl_lcm"
ENV_DIR="${ROOT}/gl_lcm_inference_env"
SCRIPT="${ROOT}/scripts/make_vindr_gl_lcm_bbox_comparison.py"

PILOT_DIR="${ROOT}/vindr_train_pilot"
OUTPUT_DIR="${PILOT_DIR}/bbox_review"

VINDR_ROOT="/home/woody/iwi5/iwi5362h/data/vindr_cxr"

# ---------------------------------------------------------------------------
# Load Python and activate the path-based Conda environment
# ---------------------------------------------------------------------------
module load python

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

echo "======================================================================"
echo "VinDr GL-LCM pathology-box review"
echo "Host:         $(hostname)"
echo "Started:      $(date --iso-8601=seconds)"
echo "Python:       $(which python)"
echo "Pilot dir:    ${PILOT_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "======================================================================"

ANNOTATIONS_CSV="${VINDR_ROOT}/annotations/annotations_train.csv"
META_CSV="${VINDR_ROOT}/train_meta.csv"

if [[ ! -f "${ANNOTATIONS_CSV}" ]]; then
    echo "[Error] Annotation CSV missing: ${ANNOTATIONS_CSV}"
    exit 1
fi

if [[ ! -f "${META_CSV}" ]]; then
    echo "[Error] Metadata CSV missing: ${META_CSV}"
    echo "Available metadata candidates:"
    find "${VINDR_ROOT}" -maxdepth 3 -type f -iname '*meta*.csv'
    exit 1
fi

echo "Annotations: ${ANNOTATIONS_CSV}"
echo "Metadata:    ${META_CSV}"

# ---------------------------------------------------------------------------
# Validate required files
# ---------------------------------------------------------------------------
required_paths=(
    "${ENV_DIR}/bin/python"
    "${SCRIPT}"
    "${PILOT_DIR}/CXR"
    "${PILOT_DIR}/Mask"
    "${PILOT_DIR}/outputs_szch/Fusion_BS"
    "${PILOT_DIR}/outputs_jsrt/Fusion_BS"
    "${PILOT_DIR}/pilot_files.txt"
    "${PILOT_DIR}/pilot_manifest.csv"
    "${ANNOTATIONS_CSV}"
    "${META_CSV}"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[Error] Required path missing: ${path}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Generate full-image box overlays and enlarged pathology crops
# ---------------------------------------------------------------------------
python "${SCRIPT}" \
    --pilot_dir "${PILOT_DIR}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --meta_csv "${META_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --szch_dir "${PILOT_DIR}/outputs_szch/Fusion_BS" \
    --jsrt_dir "${PILOT_DIR}/outputs_jsrt/Fusion_BS" \
    --nms_iou 0.50 \
    --crop_context 0.30 \
    --display_size 384 \
    --dpi 160

# ---------------------------------------------------------------------------
# Summarize outputs
# ---------------------------------------------------------------------------
echo
echo "======================================================================"
echo "Bounding-box review complete"
echo "======================================================================"

echo
echo "Main comparison montage:"
echo "${OUTPUT_DIR}/vindr_train_gl_lcm_bbox_comparison.png"

echo
echo "Generated files:"
find "${OUTPUT_DIR}" -maxdepth 2 -type f | sort

echo
echo "Number of full-image comparisons:"
find "${OUTPUT_DIR}/full" \
    -maxdepth 1 \
    -type f \
    -name "*.png" \
    2>/dev/null | wc -l

echo
echo "Number of pathology crops:"
find "${OUTPUT_DIR}/crops" \
    -maxdepth 1 \
    -type f \
    -name "*.png" \
    2>/dev/null | wc -l

echo
echo "Output size:"
du -sh "${OUTPUT_DIR}"

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "======================================================================"
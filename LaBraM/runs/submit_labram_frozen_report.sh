#!/bin/bash
# Submit a CPU job to regenerate the aggregated frozen-backbone report.
#
# Usage:
#   bash LaBraM/runs/submit_labram_frozen_report.sh
#   bash LaBraM/runs/submit_labram_frozen_report.sh --dependency afterok:12345,12346

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
REPORT_DATE="${REPORT_DATE:-$(date +%Y-%m-%d)}"
OUT_DIR="${EEG_ROOT}/reports/final_logs_frozen_report_${REPORT_DATE}"

mkdir -p "${EEG_ROOT}/final_logs_frozen/slurm" "${OUT_DIR}"

DEP_ARGS=()
if [[ "${1:-}" == "--dependency" && -n "${2:-}" ]]; then
  DEP_ARGS=(--dependency="${2}")
fi

sbatch \
  --account="${SLURM_ACCOUNT:-aip-aghodsib}" \
  --time=00:30:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --job-name=labram_frozen_report \
  --output="${EEG_ROOT}/final_logs_frozen/slurm/report-%j.out" \
  --error="${EEG_ROOT}/final_logs_frozen/slurm/report-%j.err" \
  "${DEP_ARGS[@]}" \
  --export=ALL,EEG_ROOT="${EEG_ROOT}",OUT_DIR="${OUT_DIR}" \
  --wrap="set -euo pipefail; cd '${EEG_ROOT}'; module load python/3.10 2>/dev/null || true; source ~/.bashrc 2>/dev/null || true; eval \"\$(conda shell.bash hook)\" 2>/dev/null || true; conda activate eeg_env 2>/dev/null || true; python reports/generate_final_logs_report.py --final-logs-dir '${EEG_ROOT}/final_logs_frozen' --out-dir '${OUT_DIR}'; echo Report written to ${OUT_DIR}"

echo "Report job submitted -> ${OUT_DIR}"

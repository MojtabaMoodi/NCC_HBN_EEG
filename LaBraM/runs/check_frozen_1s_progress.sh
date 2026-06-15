#!/bin/bash
# Report (and optionally update note.txt) with last completed epoch for each 1s frozen run.
#
# Usage:
#   bash LaBraM/runs/check_frozen_1s_progress.sh
#   bash LaBraM/runs/check_frozen_1s_progress.sh --update-note

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
NOTE_FILE="${EEG_ROOT}/final_logs_frozen/note.txt"
STATUS_FILE="${EEG_ROOT}/final_logs_frozen/1s_epoch_status.txt"

UPDATE_NOTE=0
if [[ "${1:-}" == "--update-note" ]]; then
  UPDATE_NOTE=1
fi

run_dir_for() {
  local task="$1"
  if [[ "${task}" == "regression" ]]; then
    echo "${EEG_ROOT}/final_logs_frozen/LaBraM/labram_age_regression_frozen_1s"
  else
    echo "${EEG_ROOT}/final_logs_frozen/LaBraM/labram_${task}_frozen_1s"
  fi
}

label_for() {
  case "$1" in
    age) echo "Age classification" ;;
    gender) echo "Gender classification" ;;
    regression) echo "Age regression" ;;
  esac
}

read_epoch_info() {
  local log_file="$1"
  python3 - "${log_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("-1 -1 0")
    raise SystemExit(0)

last_epoch = None
n_lines = 0
with path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        last_epoch = json.loads(line)["epoch"]

if last_epoch is None:
    print("-1 -1 0")
else:
  print(f"{last_epoch} {last_epoch + 1} {n_lines}")
PY
}

timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
report_lines=()
report_lines+=("Updated: ${timestamp}")
report_lines+=("")
report_lines+=("| Task | Run directory | Last completed epoch | Next resume epoch | Epochs in log.txt | Status |")
report_lines+=("|------|---------------|----------------------|-------------------|-------------------|--------|")

for TASK in age gender regression; do
  RUN_DIR="$(run_dir_for "${TASK}")"
  LOG_FILE="${RUN_DIR}/output/log.txt"
  LABEL="$(label_for "${TASK}")"
  RUN_NAME="$(basename "${RUN_DIR}")"

  read -r LAST_EPOCH NEXT_EPOCH N_LINES <<< "$(read_epoch_info "${LOG_FILE}")"

  if [[ "${LAST_EPOCH}" -lt 0 ]]; then
    STATUS="no metrics yet"
    LAST_DISP="—"
    NEXT_DISP="—"
  elif [[ "${LAST_EPOCH}" -ge 49 ]]; then
    STATUS="complete (50/50)"
    LAST_DISP="${LAST_EPOCH}"
    NEXT_DISP="—"
  else
    STATUS="in progress (${LAST_EPOCH}/49 done)"
    LAST_DISP="${LAST_EPOCH}"
    NEXT_DISP="${NEXT_EPOCH}"
  fi

  report_lines+=("| ${LABEL} | \`${RUN_NAME}\` | ${LAST_DISP} | ${NEXT_DISP} | ${N_LINES} | ${STATUS} |")
done

report_lines+=("")
report_lines+=("Resume command (auto-detects \`START_EPOCH\` from \`output/log.txt\`):")
report_lines+=("\`bash LaBraM/runs/submit_labram_frozen_resume_1s.sh\`")

printf '%s\n' "${report_lines[@]}"
printf '%s\n' "${report_lines[@]}" > "${STATUS_FILE}"

if [[ "${UPDATE_NOTE}" -eq 1 && -f "${NOTE_FILE}" ]]; then
  python3 - "${NOTE_FILE}" "${STATUS_FILE}" <<'PY'
import sys
from pathlib import Path

note_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
status = status_path.read_text(encoding="utf-8").rstrip() + "\n"

text = note_path.read_text(encoding="utf-8")
start = "<!-- FROZEN_1S_STATUS_START -->"
end = "<!-- FROZEN_1S_STATUS_END -->"
block = f"{start}\n{status}{end}"

if start in text and end in text:
    before = text.split(start, 1)[0]
    after = text.split(end, 1)[1]
    note_path.write_text(before + block + after, encoding="utf-8")
else:
    marker = "### 1s job progress (auto-updated)\n\n"
    if marker in text:
        note_path.write_text(text.replace(marker, marker + block + "\n"), encoding="utf-8")
    else:
        note_path.write_text(text.rstrip() + "\n\n### 1s job progress (auto-updated)\n\n" + block + "\n", encoding="utf-8")
PY
  echo ""
  echo "Updated ${NOTE_FILE} and ${STATUS_FILE}"
fi

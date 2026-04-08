#!/usr/bin/env bash
# ==================================================================
# Master script -- runs every cat_*.py experiment sequentially.
#
# For each script it:
#   1. Creates a per-script log file  (logs/cat_N.log)
#   2. Streams stdout+stderr to both the terminal AND the log file
#   3. Records wall-clock time
#   4. Appends a one-line summary to logs/summary.log
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh            # run all 11 experiments
#   ./run_all.sh 3 5 9      # run only cat_3, cat_5, cat_9
# ==================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/summary.log"

# Header for the summary file (overwrite each full run)
if [[ $# -eq 0 ]]; then
    printf "%-12s  %-10s  %-19s  %s\n" \
        "Script" "Status" "Start Time" "Elapsed" > "${SUMMARY}"
    printf '%.0s-' {1..60} >> "${SUMMARY}"
    echo "" >> "${SUMMARY}"
fi

# Which scripts to run: all 1..11 by default, or user-specified numbers
if [[ $# -gt 0 ]]; then
    NUMS=("$@")
else
    NUMS=(1 2 3 4 5 6 7 8 9 10 11)
fi

for N in "${NUMS[@]}"; do
    SCRIPT="cat_${N}.py"
    LOGFILE="${LOG_DIR}/cat_${N}.log"

    if [[ ! -f "${SCRIPT_DIR}/${SCRIPT}" ]]; then
        echo "WARNING: ${SCRIPT} not found -- skipping."
        continue
    fi

    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    START_SEC=$(date +%s)

    echo "=========================================="
    echo "  Running ${SCRIPT}  (${START_TS})"
    echo "  Log -> ${LOGFILE}"
    echo "=========================================="

    # Run the script; tee output to both console and log file
    set +e
    python3 "${SCRIPT_DIR}/${SCRIPT}" 2>&1 | tee "${LOGFILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    END_SEC=$(date +%s)
    ELAPSED=$(( END_SEC - START_SEC ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED(${EXIT_CODE})"
    fi

    printf "%-12s  %-10s  %-19s  %dm %02ds\n" \
        "${SCRIPT}" "${STATUS}" "${START_TS}" "${MINS}" "${SECS}" \
        >> "${SUMMARY}"

    echo ""
    echo "  Done: ${SCRIPT}  [${STATUS}]  ${MINS}m ${SECS}s"
    echo ""
done

echo "=========================================="
echo "  All requested experiments complete."
echo "  Summary -> ${SUMMARY}"
echo "=========================================="
cat "${SUMMARY}"

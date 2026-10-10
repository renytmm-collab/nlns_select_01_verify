#!/usr/bin/env bash
# Execute as a separate process, never source into an SSH shell.
set -e
if [ "$#" -ne 3 ] || [ "$1" != "--start" ] || [ "$2" != "--cap-total" ] || [ "$3" != "1021440" ]; then
    echo "Usage: bash tools/run_confirmation192.sh --start --cap-total 1021440"
    exit 2
fi
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_PY="${CONFIRM_PYTHON:-$HOME/projects/test_02_try/.venv/bin/python}"
test -x "$TASK_PY"
test ! -e runs/classifier-confirm-v1-192
stage() {
    echo "===== START: $1 ====="
    if "$TASK_PY" -u -B tools/confirm_point.py "$@" --start; then
        echo "===== COMPLETE: $1 ====="
    else
        rc=$?
        echo "===== FAILED: $1 EXIT_CODE=$rc ====="
        exit "$rc"
    fi
}
"$TASK_PY" -u -B tools/confirm_point.py check
stage freeze-plan
stage test-states --cap 38400
stage test-candidates
stage test-features
stage freeze-test-predictions
stage collect-test-a --cap 196608
stage freeze-test-reference
stage collect-test-b --cap 786432
"$TASK_PY" -u -B tools/confirm_point.py audit
"$TASK_PY" -u -B tools/confirm_point.py analyze --start --output runs/classifier-confirm-v1-192/report.json
echo "===== CONFIRMATION_COMPLETE: EXIT_CODE=0 ====="

#!/usr/bin/env bash
# The generators in their order, after the ranker's repointing. Stops at the first failure.
export PATH="/c/Windows:/c/Program Files/nodejs:/c/Program Files/Git/usr/bin:/c/Program Files/Git/bin:$PATH"
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8 UOL_FINAL_LABELS=corrected UOL_FINAL_SYSTEMS=experiments/runs/footing_20261101/systems_audit.json UOL_FINAL_THREADS=4 OMP_NUM_THREADS=4 PYTHONHASHSEED=0
run() { echo "=== $(date +%T) $*"; "$@" 2>&1 | grep -v Warning | tail -${TAIL:-4} | cut -c1-400; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FAILED: $*"; exit 1; }; }
run py -3.12 scripts/support_share.py
run py -3.12 scripts/gate_dev.py
run py -3.12 scripts/final_results.py
run py -3.12 scripts/t6_tables.py
PYTHONPATH=src run py -3.12 -m uol.thesis_figures.ch6_d191
PYTHONPATH=src run py -3.12 -m uol.thesis_figures.v21_figs
run py -3.12 scripts/final_results.py
run py -3.12 scripts/t6_tables.py
PYTHONPATH=src run py -3.12 -m uol.thesis_figures.ch6_figs --final --only F6_1,F6_11
run py -3.12 scripts/t5_output.py
echo "=== $(date +%T) done"

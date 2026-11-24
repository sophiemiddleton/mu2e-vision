#!/bin/bash
# Vision review of the 41-run TranAD comparison set against reference 125026, then the
# run-level comparison with TranAD. Safe to rerun: saved model responses are reused.
# Start detached so it survives a lost connection:
#   setsid nohup scripts/run_tranad_ref125026.sh > results/real-006-001-tranad-ref125026-opus55.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/.."
source setup_env.sh > /dev/null 2>&1
export LITELLM_API_KEY=$(cat ~/.litellm_key)
python -u -m histcompare.benchmark --data real/006-001-tranad \
    --instructions instructions/crv_real_draft.yaml \
    --out results/real-006-001-tranad-ref125026-opus55 \
    --base-url https://litellm.fnal.gov/v1 --model azure/claude-opus-5-5
python -m histcompare.compare_tranad \
    --vision results/real-006-001-tranad-ref125026-opus55/results.csv \
    --groups real/groups_006-001.json --out results/real-006-001-tranad-ref125026-compare \
    --tranad tranad=../../results/crvreco_006-001
echo "DONE $(date)"

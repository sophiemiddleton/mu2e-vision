# Environment for mu2e-vision on a Mu2e gpvm: the cvmfs "ana" env (uproot, hist, scipy,
# matplotlib, yaml) plus the Mu2e/DQM python package (github.com/Mu2e/DQM).
#
#   export DQM_DIR=/path/to/DQM            # optional; otherwise ../DQM or ../../DQM is used
#   source mu2e-vision/setup_env.sh
#   cd mu2e-vision && python -m histcompare.synthetic --out synthetic

_HC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin:$PATH"
export PYTHONPATH="$_HC${PYTHONPATH:+:$PYTHONPATH}"

_DQM=""
for _d in "${DQM_DIR:-}" "$_HC/../DQM" "$_HC/../../DQM"; do
    if [ -n "$_d" ] && [ -d "$_d/python/dqm" ]; then _DQM="$(cd "$_d" && pwd)"; break; fi
done
if [ -n "$_DQM" ]; then
    export PYTHONPATH="$_DQM/python:$PYTHONPATH"
else
    echo "WARNING: Mu2e/DQM not found; clone github.com/Mu2e/DQM and set DQM_DIR"
fi
export MPLBACKEND=Agg
unset _d _DQM

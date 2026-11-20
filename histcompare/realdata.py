"""
Real Mu2e CRV DQM runs, summed from their subrun files, in the layout the benchmark reads

The cosmics_crv DQM datasets hold one ROOT file per subrun, named
dqm.mu2e.cosmics_crv.<dataset>.<run>_<subrun>[...].root. For each selected run this sums
the chosen histograms over all of its subrun files and writes one file per run,
dqm.mu2e.cosmics_crv.run<run>.root, plus runs.json (reference run, run -> run type, and
per-run info). There is no truth.json: nobody has labelled these runs.

  survey  one row per run: subrun files, events, live channels, active FEBs
  build   sum the selected runs into a benchmark data directory

    python -m histcompare.realdata survey --out real/survey.csv
    python -m histcompare.realdata build --out real/006-001 --reference 120747 \
        --runs 120747 120825 ... [--instructions instructions/crv_real_draft.yaml]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import uproot
import yaml

DATASETS = Path("/pnfs/mu2e/persistent/datasets/phy-ntd/dqm/mu2e/cosmics_crv")
DEFAULT_DATASET = "crvreco-006-001-file"
NAME = re.compile(r"\.(\d{6})_(\d+)")


def run_files(dataset: str = DEFAULT_DATASET) -> Dict[str, List[Path]]:
    """Run -> its subrun files, sorted."""
    runs = defaultdict(list)
    for p in (DATASETS / dataset).rglob("*.root"):
        m = NAME.search(p.name)
        if m:
            runs[m.group(1)].append(p)
    return {r: sorted(f) for r, f in sorted(runs.items())}


def _survey_file(path: Path):
    with uproot.open(path) as f:
        ev = float(f["crvDigiDQM/nEvents"].values().sum()) if "crvDigiDQM/nEvents" in f else 0.0
        ch = f["crvDigiDQM/crvDigisPerChannel"].values() if "crvDigiDQM/crvDigisPerChannel" in f else None
        feb = f["crvStatusDQM/activeFebCount"].to_numpy() if "crvStatusDQM/activeFebCount" in f else None
    return ev, ch, feb


def survey(dataset: str, out: Path, workers: int = 16) -> None:
    rows = []
    files = run_files(dataset)
    with ThreadPoolExecutor(workers) as pool:
        for run, paths in files.items():
            res = list(pool.map(_survey_file, paths))
            events = sum(r[0] for r in res)
            chans = [r[1] for r in res if r[1] is not None]
            ch = np.sum(chans, axis=0) if chans else np.zeros(1)
            febs = [r[2] for r in res if r[2] is not None]
            if febs:
                counts = np.sum([c for c, _ in febs], axis=0)
                edges = febs[0][1]
                centers = 0.5 * (edges[1:] + edges[:-1])
                feb_mode = float(centers[np.argmax(counts)]) if counts.sum() else float("nan")
            else:
                feb_mode = float("nan")
            rows.append({"run": run, "files": len(paths), "events": int(events),
                         "digis": int(ch.sum()), "live_channels": int((ch > 0).sum()),
                         "active_febs_mode": feb_mode})
            print(f"  {run}: {len(paths)} files, {int(events)} events", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"✓ {len(rows)} runs surveyed; see {out}")


def _sum_run(paths: List[Path], hists: List[str], mean=frozenset()):
    """Each histogram summed over the run's subrun files; those in `mean` averaged instead
    (per-channel value maps such as rates, where a sum has no meaning)."""
    total, n = {}, defaultdict(int)
    for p in paths:
        with uproot.open(p) as f:
            for h in hists:
                if h not in f:
                    continue
                v = f[h].to_hist()
                total[h] = v if h not in total else total[h] + v
                n[h] += 1
    for h in mean:
        if h in total:
            total[h] = total[h] * (1.0 / n[h])
    return total


def build(dataset: str, out: Path, runs: List[str], reference: str, hists: List[str],
          run_type: str = "cosmics", workers: int = 8, reference_for: Dict[str, str] = None,
          mean=frozenset()) -> None:
    """`reference_for` (run -> reference run) overrides `reference` per run, e.g. one
    reference per detector configuration period."""
    files = run_files(dataset)
    reference_for = reference_for or {}
    refs = list(dict.fromkeys([reference, *reference_for.values()]))
    missing = [r for r in [*refs, *runs] if r not in files]
    if missing:
        raise SystemExit(f"runs not in {dataset}: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    todo = list(dict.fromkeys([*refs, *runs]))
    info = {}
    with ThreadPoolExecutor(workers) as pool:
        for run, total in zip(todo, pool.map(lambda r: _sum_run(files[r], hists, mean), todo)):
            with uproot.recreate(out / f"dqm.mu2e.cosmics_crv.run{run}.root") as f:
                for h, v in total.items():
                    f[h] = v
            info[run] = {"files": len(files[run]), "dataset": dataset,
                         "missing": [h for h in hists if h not in total]}
            print(f"  {run}: {len(files[run])} files combined", flush=True)
    meta = {"reference": {run_type: int(reference)}, "runs": {r: run_type for r in todo}, "info": info}
    if reference_for:
        meta["reference_for"] = {r: int(ref) for r, ref in reference_for.items()}
        meta["references"] = [int(r) for r in refs]
    (out / "runs.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ {len(todo)} runs written to {out}")


def histograms_from(instructions: str) -> List[str]:
    plots = yaml.safe_load(open(instructions))["plots"]
    return [f"{p.get('folder', '')}/{name}".lstrip("/") for name, p in plots.items()]


def mean_histograms_from(instructions: str) -> frozenset:
    """Plots marked `combine: mean` in the instructions (value maps, averaged over subruns)."""
    plots = yaml.safe_load(open(instructions))["plots"]
    return frozenset(f"{p.get('folder', '')}/{name}".lstrip("/")
                     for name, p in plots.items() if p.get("combine") == "mean")


def survey_collector(dataset: str, out: Path, workers: int = 16) -> None:
    """Survey for the older layout (crvreco-005-001 and before: everything under
    CrvDQMcollector): subrun files, live channels (most in any subrun), mean digi rate summed
    over channels, and coincidence-cluster entries per run."""
    def one(p):
        with uproot.open(p) as f:
            r = f["CrvDQMcollector/crvDigiRates"].values()
            cc = float(f["CrvDQMcollector/crvCoincidencesClusters"].values().sum())
        return int((r > 0).sum()), float(r.sum()), cc
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for run, paths in run_files(dataset).items():
            res = list(pool.map(one, paths))
            rows.append({"run": run, "files": len(paths),
                         "live_channels": max(x[0] for x in res),
                         "rate_sum_mean": round(float(np.mean([x[1] for x in res])), 4),
                         "coincidence_entries": int(sum(x[2] for x in res))})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"✓ {len(rows)} runs surveyed; see {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--dataset", default=DEFAULT_DATASET)
    s.add_argument("--out", required=True)
    s.add_argument("--layout", choices=["status", "collector"], default="status",
                   help="collector: the older CrvDQMcollector files (005-001 and before)")
    b = sub.add_parser("build")
    b.add_argument("--dataset", default=DEFAULT_DATASET)
    b.add_argument("--out", required=True)
    b.add_argument("--reference", required=True)
    b.add_argument("--runs", nargs="+", required=True)
    b.add_argument("--reference-map", help="JSON file: run -> reference run (overrides --reference)")
    b.add_argument("--instructions", default=str(Path(__file__).parents[1] / "instructions/crv_real_draft.yaml"),
                   help="the histograms to keep are the plots listed here")
    args = ap.parse_args(argv)
    if args.cmd == "survey":
        (survey_collector if args.layout == "collector" else survey)(args.dataset, Path(args.out))
    else:
        ref_map = json.loads(Path(args.reference_map).read_text()) if args.reference_map else None
        build(args.dataset, Path(args.out), args.runs, args.reference, histograms_from(args.instructions),
              reference_for=ref_map, mean=mean_histograms_from(args.instructions))


if __name__ == "__main__":
    main()

"""
Run-level comparison of TranAD variants with the vision LLM on the 005-001 add-on sample

A run counts as TranAD-flagged if any of its subruns is flagged; "no CRV data" rows (set
aside by --drop-empty) are counted separately. The vision verdict for a run is BAD if any of
its histograms is BAD.

    python -m histcompare.compare_tranad --vision results/real-005-001-opus55/results.csv \
        --groups real/groups_005-001.json --out results/real-005-001-compare \
        --tranad old=../../results/crvreco_005-001_model fixA=... fixB=... fixC=...
"""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


def _flag(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "1.0")


def tranad_runs(d: Path):
    """run -> dict(subruns, flagged, drift, max_score, empty) for one pipeline output dir."""
    report = [p for p in sorted(glob.glob(str(d / "anomaly_report_*.csv"))) if "anomalies_only" not in p][-1]
    out = defaultdict(lambda: {"subruns": 0, "flagged": 0, "drift": 0, "max_score": 0.0, "empty": 0})
    for r in csv.DictReader(open(report)):
        o = out[r["run"]]
        o["subruns"] += 1
        o["flagged"] += _flag(r["tranad_anomaly"])
        o["drift"] += _flag(r.get("training_drift", 0))
        o["max_score"] = max(o["max_score"], float(r["tranad_reconstruction_error"] or 0))
    empty = d / "no_crv_data.csv"
    if empty.exists():
        for r in csv.DictReader(open(empty)):
            out[r["run"]]["empty"] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vision", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--tranad", nargs="+", required=True, help="label=output_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    groups = json.load(open(a.groups))
    vis = defaultdict(list)
    for r in csv.DictReader(open(a.vision)):
        if r["ml"] == "BAD":
            vis[r["run"]].append(r["histogram"])
    variants = {lab: tranad_runs(Path(p)) for lab, p in (x.split("=", 1) for x in a.tranad)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(groups):
        row = {"run": run, "group": groups[run], "vision": "BAD" if vis.get(run) else "GOOD",
               "vision_bad_histograms": " ".join(vis.get(run, []))}
        for lab, t in variants.items():
            o = t.get(run)
            if o is None:
                row[lab] = "-"
            elif o["flagged"]:
                row[lab] = f"FLAG {o['flagged']}/{o['subruns']}"
            elif o["empty"] and not o["subruns"]:
                row[lab] = "NO-DATA"
            else:
                row[lab] = "pass" + (f" (+{o['empty']} empty)" if o["empty"] else "")
            row[lab + "_drift"] = o["drift"] if o else ""
            row[lab + "_max"] = round(o["max_score"], 4) if o else ""
        rows.append(row)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    labs = list(variants)
    lines = ["| Run | Group | Vision | " + " | ".join(labs) + " |", "|" + "---|" * (3 + len(labs))]
    for r in rows:
        lines.append(f"| {r['run']} | {r['group']} | {r['vision']} | " + " | ".join(r[l] for l in labs) + " |")
    lines += ["", "| | " + " | ".join(labs) + " |", "|" + "---|" * (1 + len(labs))]
    for what, test in [("runs flagged", lambda v: v.startswith("FLAG")),
                       ("flagged and vision BAD", None), ("flagged and vision GOOD", None),
                       ("vision BAD, not flagged", None)]:
        cells = []
        for l in labs:
            fl = [r for r in rows if r[l].startswith("FLAG")]
            if what == "runs flagged":
                n = len(fl)
            elif what == "flagged and vision BAD":
                n = sum(r["vision"] == "BAD" for r in fl)
            elif what == "flagged and vision GOOD":
                n = sum(r["vision"] == "GOOD" for r in fl)
            else:
                n = sum(r["vision"] == "BAD" and not r[l].startswith("FLAG") for r in rows)
            cells.append(str(n))
        lines.append(f"| {what} | " + " | ".join(cells) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

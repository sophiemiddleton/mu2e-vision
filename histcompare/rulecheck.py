"""
Rule-based check of the real-data vision verdicts: the draft shifter criteria
(instructions/crv_real_draft.yaml) coded as numeric tests on the bin contents.

This is an independent reading of the same written criteria, not truth: it checks whether a
vision BAD (or GOOD) is backed by the numbers. Tolerances follow the YAML; where the YAML is
qualitative ("visible on a linear scale", "clearly different shape") the cut chosen is noted
next to the test. Tests on a range with too few expected entries are skipped, so a very short
run can only fail on the tests its statistics allow.

    python -m histcompare.rulecheck --data real/006-001-tranad \
        --vision results/real-006-001-tranad-ref125026-opus55/results.csv \
        --out results/real-006-001-tranad-ref125026-rulecheck
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import uproot

PATHS = {
    "PEr": "crvRecoDQM/PEr", "logchi2": "crvRecoDQM/logchi2",
    "crvDigisPerChannel": "crvDigiDQM/crvDigisPerChannel", "h1_peakAdc": "crvDigiDQM/h1_peakAdc",
    "errorBits": "crvStatusDQM/errorBits", "activeFebCount": "crvStatusDQM/activeFebCount",
}
MIN_EXPECTED = 50      # skip a rate test on a range expected to hold fewer entries than this


def load(path: Path):
    f = uproot.open(path)
    return {h: (f[p].values().astype(float), f[p].axis().edges()) for h, p in PATHS.items()}


def _centres(e):
    return 0.5 * (e[1:] + e[:-1])


def check_channels(v, r):
    """Per FEB (64 channels): share of all digis vs the reference share, factor 1.5; dead < 0.1."""
    n = v.sum()
    if n == 0:
        return ["empty"]
    issues = []
    for b in range(len(v) // 64):
        sl = slice(64 * b, 64 * b + 64)
        rs = r[sl].sum() / r.sum()
        exp = rs * n
        if exp < MIN_EXPECTED:
            continue
        ratio = v[sl].sum() / exp
        if ratio < 0.1:
            issues.append(f"ch{64*b}-{64*b+63} empty")
        elif ratio < 1 / 1.5 or ratio > 1.5:
            issues.append(f"ch{64*b}-{64*b+63} x{ratio:.2f}")
    return issues


def _median_above(v, e, lo):
    c = _centres(e)
    m = c > lo
    cs = np.cumsum(v[m])
    return c[m][np.searchsorted(cs, cs[-1] / 2)] if cs[-1] > 0 else np.nan


def _peak_to_shoulder(h, c):
    sm = np.convolve(h, np.ones(3) / 3, "same")
    return h[(c > 4) & (c < 14)].max() / max(sm[(c > 18) & (c < 120)].max(), 1e-9)


def check_per(v, e, r):
    """Shoulder position (median PE above 15 PE, within 4 PE) and strength: the low-PE peak
    to shoulder height ratio may vary by a factor of two (known issue), not more."""
    c = _centres(e)
    if v.sum() == 0:
        return ["empty"]
    if v[c > 15].sum() < MIN_EXPECTED:
        return []
    issues = []
    a, b = _median_above(v, e, 15), _median_above(r, e, 15)
    if abs(a - b) > 4:
        issues.append(f"shoulder median {a:.0f} vs {b:.0f} PE")
    pv, pr = _peak_to_shoulder(v, c), _peak_to_shoulder(r, c)
    if not (0.5 < pv / pr < 2):
        issues.append(f"peak/shoulder {pv:.1f} vs {pr:.1f}")
    return issues


def check_peakadc(v, e, r):
    c = _centres(e)
    if v.sum() == 0:
        return ["empty"]
    issues = []
    pk = lambda h: c[np.argmax(np.where(c > 2050, h, 0))]
    if abs(pk(v) - pk(r)) > 20:
        issues.append(f"peak {pk(v):.0f} vs {pk(r):.0f}")
    sat = lambda h: h[-1] / h.sum()
    if v.sum() * sat(r) >= 10 and sat(v) > 3 * sat(r):                # "several times larger"
        issues.append(f"saturation {100*sat(v):.2f}% vs {100*sat(r):.2f}%")
    low = lambda h: h[c < 2050].sum() / h.sum()
    if low(v) > max(10 * low(r), 1e-3):                                # "well below baseline"
        issues.append(f"below baseline {100*low(v):.2f}%")
    sh = lambda h: h[(c > 2300) & (c < 2700)].mean() / h.max()          # shoulder height / peak
    if v.sum() >= 1000 and not (1 / 1.5 < sh(v) / sh(r) < 1.5):        # "clearly different shape"
        issues.append(f"shoulder/peak x{sh(v)/sh(r):.2f}")
    return issues


def check_logchi2(v, e, r):
    c = _centres(e)
    if v.sum() == 0:
        return ["empty"]
    issues = []
    pk = lambda h: c[np.argmax(h)]
    if abs(pk(v) - pk(r)) > 0.2:
        issues.append(f"peak {pk(v):.2f} vs {pk(r):.2f}")
    hi = v[c > 3.5].sum() / v.sum()
    if hi > 0.02 and v.sum() * 0.02 >= 10:                             # v0.2: "more than about 2%"
        issues.append(f"{100*hi:.2f}% above 3.5")
    return issues


def check_errorbits(v, headers):
    """Bits 1-7 on at least 1% of ROC headers ("visible on a linear scale")."""
    if headers <= 0:
        return []
    return [f"bit{i} {100*v[i]/headers:.1f}%" for i in range(1, 8) if v[i] / headers >= 0.01]


def check_febcount(v):
    n = v.sum()
    if n <= 0:
        return ["empty"]
    issues = []
    other = n - v[6] - v[24]
    if other / n > 1e-3:
        bins = [f"{i}:{100*v[i]/n:.0f}%" for i in range(len(v)) if i not in (6, 24) and v[i] / n > 1e-3]
        issues.append("other counts " + " ".join(bins))
    if max(v[6], v[24]) > 0 and abs(v[6] - v[24]) / max(v[6], v[24]) > 0.03:
        issues.append(f"6:24 = {v[6]/n:.2f}:{v[24]/n:.2f}")
    return issues


def check_run(h, ref):
    out = {}
    out["crvDigisPerChannel"] = check_channels(h["crvDigisPerChannel"][0], ref["crvDigisPerChannel"][0])
    out["PEr"] = check_per(h["PEr"][0], h["PEr"][1], ref["PEr"][0])
    out["h1_peakAdc"] = check_peakadc(h["h1_peakAdc"][0], h["h1_peakAdc"][1], ref["h1_peakAdc"][0])
    out["logchi2"] = check_logchi2(h["logchi2"][0], h["logchi2"][1], ref["logchi2"][0])
    out["errorBits"] = check_errorbits(h["errorBits"][0], h["activeFebCount"][0].sum())
    out["activeFebCount"] = check_febcount(h["activeFebCount"][0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.data)
    meta = json.load(open(d / "runs.json"))
    fname = lambda run: d / f"dqm.mu2e.cosmics_crv.run{run}.root"
    ref = load(fname(meta["reference"]["cosmics"]))
    vis = {(r["run"], r["histogram"]): r["ml"] for r in csv.DictReader(open(a.vision))}
    rows = []
    for run in sorted(meta["runs"]):
        for hist, issues in check_run(load(fname(run)), ref).items():
            rows.append({"run": run, "histogram": hist, "rule": "BAD" if issues else "GOOD",
                         "vision": vis.get((run, hist), "-"), "issues": "; ".join(issues)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rulecheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| Histogram | both BAD | both GOOD | vision BAD only | rule BAD only | agree |",
             "|---|---|---|---|---|---|"]
    for hist in PATHS:
        rs = [r for r in rows if r["histogram"] == hist and r["vision"] != "-"]
        n = lambda v, ru: sum(r["vision"] == v and r["rule"] == ru for r in rs)
        lines.append(f"| {hist} | {n('BAD','BAD')} | {n('GOOD','GOOD')} | {n('BAD','GOOD')} | "
                     f"{n('GOOD','BAD')} | {n('BAD','BAD')+n('GOOD','GOOD')}/{len(rs)} |")
    lines += ["", "Disagreements:", ""]
    lines += [f"- {r['run']} {r['histogram']}: vision {r['vision']}, rule {r['rule']} {r['issues']}"
              for r in rows if r["vision"] not in ("-", r["rule"])]
    (out / "rulecheck.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""
Figures for the real-data report: run vs reference overlays and a per-half-block summary

    python -m histcompare.report_figures --data real/006-001 --reference 124228 \
        --results results/real-006-001-ref124228-opus55 --out results/report_figures
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ERROR_BITS = ["FEBuBMismatch", "FEBBufferIssue", "FEBOverflow", "Group1Issue",
              "Group2Issue", "Group3Issue", "uBMatchError", "Truncation"]
REF_STYLE = dict(color="0.45", lw=1.6, label=None)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})


class Runs:
    def __init__(self, data: Path):
        self.data = data
        self._cache = {}

    def get(self, run: str, path: str):
        key = (run, path)
        if key not in self._cache:
            with uproot.open(self.data / f"dqm.mu2e.cosmics_crv.run{run}.root") as f:
                self._cache[key] = f[path].to_numpy()
        return self._cache[key]


def half_blocks(counts):
    """Median rate of the even and odd channels of each 64-channel block, relative to 0-1023."""
    base = np.median(counts[:1024][counts[:1024] > 0])
    return np.array([[np.median(counts[i * 64:(i + 1) * 64][p::2]) / base for p in (0, 1)]
                     for i in range(len(counts) // 64)])


def fig_heatmap(R: Runs, runs, ref, verdicts, out):
    ref_hb = half_blocks(R.get(ref, "crvDigiDQM/crvDigisPerChannel")[0])
    live = ref_hb > 0.05
    rows = []
    for r in runs:
        hb = half_blocks(R.get(r, "crvDigiDQM/crvDigisPerChannel")[0])
        ratio = np.where(live, hb / np.where(live, ref_hb, 1), np.nan)
        rows.append(ratio.reshape(-1))  # block0 even, block0 odd, block1 even, ...
    m = np.array(rows)
    m = np.clip(m, 1e-2, 1e2)
    fig, ax = plt.subplots(figsize=(13, 0.33 * len(runs) + 1.8))
    im = ax.imshow(m, aspect="auto", cmap="RdBu_r", norm=LogNorm(1e-2, 1e2), interpolation="nearest")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{r}  {verdicts.get(r, '')}" for r in runs], fontsize=8.5, family="monospace")
    ticks = np.arange(0, 72, 8)
    ax.set_xticks(ticks - 0.5)
    ax.set_xticklabels([str(t * 32) for t in ticks])
    ax.set_xlabel("Offline channel (start of 64-channel block; each block split into even | odd channels)")
    ax.set_title(f"Half-block digi rate relative to reference {ref} (each normalised to channels 0-1023)\n"
                 "grey: half-block not populated in the reference; label = model verdict on crvDigisPerChannel",
                 fontsize=10)
    for x in range(0, 72, 2):
        ax.axvline(x - 0.5, color="white", lw=0.4)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("rate / reference rate (log)")
    ax.set_facecolor("0.85")
    fig.tight_layout()
    fig.savefig(out / "fig1_channel_halfblocks.png")
    plt.close(fig)


def _norm(counts):
    s = counts.sum()
    return counts / s if s else counts


def fig_digis(R: Runs, ref, out):
    panels = [("124192", None, "124192: blocks missing (with FEB buffer/overflow errors)"),
              ("124529", None, "124529: only 1024-1535 and 2048-2303 read out"),
              ("125158", None, "125158: 1024-1535 and 2048-2303 ~40x too hot"),
              ("124336", (1536, 2048), "124336 (first reference), channels 1536-2047: dead half-blocks")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
    rc, edges = R.get(ref, "crvDigiDQM/crvDigisPerChannel")
    ch = np.arange(len(rc))
    for ax, (run, zoom, title) in zip(axs.flat, panels):
        c, _ = R.get(run, "crvDigiDQM/crvDigisPerChannel")
        rn, n = _norm(rc) * 1e3, _norm(c) * 1e3
        if zoom:
            sl = slice(*zoom)
            for par, mk in ((0, "o"), (1, "s")):
                sel = ch[sl][par::2]
                ax.plot(sel, rn[sel], mk, ms=2.2, color="0.55",
                        label=f"ref {ref}, {'even' if par == 0 else 'odd'} channels")
                ax.plot(sel, n[sel], mk, ms=2.2, color="tab:red" if par == 0 else "tab:orange",
                        label=f"run {run}, {'even' if par == 0 else 'odd'} channels")
            ax.legend(fontsize=7.5, ncol=2, loc="upper left")
            ax.set_xlim(zoom[0] - 8, zoom[1] + 8)
            ax.set_ylim(-0.02, 0.85)
        else:
            ax.stairs(rn, edges, color="0.45", lw=1.0, label=f"reference {ref}")
            ax.stairs(n, edges, color="tab:red", lw=0.7, label=f"run {run}")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("Offline channel (bar x 4 + SiPM)")
        ax.set_ylabel("Digis per 1000 (normalised)")
        for b in range(0, 2305, 64 if zoom else 512):
            ax.axvline(b - 0.5, color="0.9", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "fig2_channel_overlays.png")
    plt.close(fig)


def fig_pe(R: Runs, ref, out):
    runs = [("124203", "tab:blue"), ("125026", "tab:orange"), ("125152", "tab:red"),
            ("125707", "tab:green"), ("125846", "tab:purple")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    rc, e = R.get(ref, "crvRecoDQM/PEr")
    cen = 0.5 * (e[1:] + e[:-1])
    m = (cen >= 20) & (cen < 200)
    rn = rc / rc[m].sum()
    a1.stairs(rn, e, color="0.3", lw=2, label=f"reference {ref}")
    for run, col in runs:
        c, _ = R.get(run, "crvRecoDQM/PEr")
        n = c / c[m].sum()
        a1.stairs(n, e, color=col, lw=1.1, label=run)
        with np.errstate(divide="ignore", invalid="ignore"):
            a2.plot(cen[m], (n / rn)[m], "-", color=col, lw=1.3, label=run)
    a1.set_yscale("log")
    a1.set_xlim(0, 250)
    a1.set_ylim(1e-4, 2)
    a1.set_xlabel("PEs")
    a1.set_ylabel("Fraction of pulses with 20-200 PE (log)")
    a1.set_title("PEr, normalised to the 20-200 PE region")
    a1.legend(fontsize=8)
    a2.axhline(1, color="0.3", lw=1)
    a2.set_xlim(20, 200)
    a2.set_ylim(0.6, 1.6)
    a2.set_xlabel("PEs")
    a2.set_ylabel(f"Ratio to reference {ref}")
    a2.set_title("Shape relative to the reference: change from 125026 onwards")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_pe_spectrum.png")
    plt.close(fig)


def fig_chi2(R: Runs, ref, runs_all, out):
    show = [("124203", "tab:blue"), ("125026", "tab:orange"), ("125152", "tab:red"),
            ("125850", "tab:purple")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    rc, e = R.get(ref, "crvRecoDQM/logchi2")
    a1.stairs(_norm(rc), e, color="0.3", lw=2, label=f"reference {ref}")
    for run, col in show:
        a1.stairs(_norm(R.get(run, "crvRecoDQM/logchi2")[0]), e, color=col, lw=1.1, label=run)
    a1.axvline(3.5, color="0.5", ls="--", lw=1)
    a1.set_yscale("log")
    a1.set_ylim(1e-6, 0.1)
    a1.set_xlabel("log10(pulse fit chi2)")
    a1.set_ylabel("Fraction of pulses (log)")
    a1.set_title("logchi2: high-chi2 tail")
    a1.legend(fontsize=8)
    cen = 0.5 * (e[1:] + e[:-1])
    xs = sorted(runs_all + [ref])
    fr = [100 * R.get(r, "crvRecoDQM/logchi2")[0][cen > 3.5].sum() / R.get(r, "crvRecoDQM/logchi2")[0].sum()
          for r in xs]
    cols = ["0.3" if r == ref else ("tab:red" if f > 1 else "tab:blue") for r, f in zip(xs, fr)]
    a2.bar(range(len(xs)), fr, color=cols)
    a2.axhline(1.0, color="0.5", ls="--", lw=1)
    a2.text(-0.4, 1.03, "draft limit 1%", ha="left", va="bottom", fontsize=8, color="0.4")
    a2.set_xticks(range(len(xs)))
    a2.set_xticklabels(xs, rotation=90, fontsize=8)
    a2.set_ylabel("Pulses with log10(chi2) > 3.5 (%)")
    a2.set_title("Fraction of badly fitted pulses per run")
    fig.tight_layout()
    fig.savefig(out / "fig4_logchi2.png")
    plt.close(fig)


def fig_status(R: Runs, ref, runs_all, out):
    xs = sorted(runs_all + [ref])
    eb = []
    for r in xs:
        bits = R.get(r, "crvStatusDQM/errorBits")[0]
        headers = R.get(r, "crvStatusDQM/activeFebCount")[0].sum()
        eb.append(bits / headers)
    eb = np.array(eb)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    im = a1.imshow(np.clip(eb, 1e-6, 1), aspect="auto", cmap="magma_r", norm=LogNorm(1e-6, 1))
    a1.set_yticks(range(len(xs)))
    a1.set_yticklabels(xs, fontsize=8)
    a1.set_xticks(range(8))
    a1.set_xticklabels([f"{i} {n}" for i, n in enumerate(ERROR_BITS)], rotation=35, ha="right", fontsize=8)
    for i in range(len(xs)):
        for j in range(8):
            if eb[i, j] >= 0.01:
                a1.text(j, i, f"{eb[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if eb[i, j] > 0.1 else "black")
    a1.set_title("Firmware error bits: fraction of ROC headers with the bit set")
    fig.colorbar(im, ax=a1, pad=0.01).set_label("fraction (log)")
    rc, e = R.get(ref, "crvStatusDQM/activeFebCount")
    c, _ = R.get("125014", "crvStatusDQM/activeFebCount")
    a2.stairs(_norm(rc), e, color="0.3", lw=2, label=f"reference {ref}")
    a2.stairs(_norm(c), e, color="tab:red", lw=1.4, label="125014")
    a2.set_xlabel("N(active FEBs) per ROC header")
    a2.set_ylabel("Fraction of ROC headers")
    a2.set_title("activeFebCount: 125014 has 26% / 74% instead of 50% / 50%")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_status.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="real/006-001")
    ap.add_argument("--reference", default="124228")
    ap.add_argument("--results", default="results/real-006-001-ref124228-opus55")
    ap.add_argument("--out", default="results/report_figures")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Runs(Path(a.data))
    runs = sorted(r for r in json.loads((Path(a.data) / "runs.json").read_text())["runs"] if r != a.reference)
    verdicts = {r["run"]: r["ml"] for r in csv.DictReader(open(Path(a.results) / "results.csv"))
                if r["histogram"] == "crvDigisPerChannel"}
    fig_heatmap(R, runs, a.reference, verdicts, out)
    fig_digis(R, a.reference, out)
    fig_pe(R, a.reference, out)
    fig_chi2(R, a.reference, runs, out)
    fig_status(R, a.reference, runs, out)
    print(f"✓ figures in {out}")


if __name__ == "__main__":
    main()

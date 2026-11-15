"""
Synthetic CRV-like DQM histograms with known answers

Writes one ROOT file per run, in the layout dqm.hist.read expects, plus runs.json
(run -> run type) and truth.json (run -> histogram -> GOOD/BAD and why). Every run holds
the same three histograms:

  CrvDQM/crvPEs_CRVsector0             PEs per reconstructed pulse, sector 0 (Landau-like)
  CrvDQM/crvPEs_CRVsector1             the same for sector 1
  CrvDQM/crvDigisPerChannel_CRVsector0 digis per channel, 256 channels on 4 FEBs of 64

Channel efficiencies are a fixed "detector map" shared by every run, and channels 37 and
190 are masked (always empty), so the reference and the runs differ only by statistics
and by the problems injected below. Run 105000 is the golden cosmics reference.

    python -m histcompare.synthetic --out synthetic
"""

import argparse
import json
from pathlib import Path

import numpy as np
import uproot
from scipy.stats import moyal

DIR = "CrvDQM"
PES = ("crvPEs_CRVsector0", "crvPEs_CRVsector1")
DIGIS = "crvDigisPerChannel_CRVsector0"

N_CHANNELS, FEB_SIZE = 256, 64
MASKED = (37, 190)
PE_EDGES = np.linspace(0.0, 200.0, 101)
CH_EDGES = np.arange(N_CHANNELS + 1) - 0.5
PE_THRESHOLD = 5.0  # pulses below this are not reconstructed

REFERENCE_RUN = 105000

# run -> (run type, statistics relative to a normal run, injected problem, truth note)
RUNS = {
    105000: ("cosmics", 2.0, None, "golden reference"),
    105001: ("cosmics", 1.0, None, "normal run"),
    105002: ("cosmics", 3.0, None, "longer run: more entries, same shapes"),
    105003: ("cosmics", 0.05, None, "short run: low statistics, same shapes"),
    105004: ("cosmics", 1.0, "gain_drop", "sector 0 PE peak 15% low (SiPM gain drop)"),
    105005: ("cosmics", 1.0, "dead_feb", "FEB 2 (channels 128-191) dead"),
    105006: ("cosmics", 1.0, "hot_channel", "channel 77 hot (25x)"),
    105007: ("cosmics", 1.0, "one_dead_channel", "one extra isolated dead channel (101)"),
    105008: ("led", 1.0, None, "LED calibration run: shapes differ from cosmics by design"),
    105009: ("cosmics", 1.0, "low_pe_noise", "sector 1 excess of 5-12 PE pulses (noise)"),
}

# (problem, histogram) pairs a shifter must report; everything else is GOOD
BAD = {
    ("gain_drop", PES[0]),
    ("dead_feb", DIGIS),
    ("hot_channel", DIGIS),
    ("low_pe_noise", PES[1]),
}
# histograms a problem shows up in, including the tolerated one
AFFECTS = BAD | {("one_dead_channel", DIGIS)}

N_PULSES = 100_000  # per sector, normal run
N_DIGIS = 250_000


def detector_map(seed: int = 0) -> np.ndarray:
    """Relative digi rate of each channel: FEB gains, channel spread, masked channels."""
    rng = np.random.default_rng(seed)
    feb = np.repeat(rng.uniform(0.9, 1.1, N_CHANNELS // FEB_SIZE), FEB_SIZE)
    channel = rng.uniform(0.85, 1.15, N_CHANNELS)
    # cosmics hit the middle of a sector a little more often than its ends
    shape = 1.0 + 0.15 * np.sin(np.pi * (np.arange(N_CHANNELS) + 0.5) / N_CHANNELS)
    w = feb * channel * shape
    w[list(MASKED)] = 0.0
    return w / w.sum()


def pe_hist(rng, n: int, run_type: str, mpv: float = 40.0, noise: bool = False) -> np.ndarray:
    if run_type == "led":
        pes = rng.normal(60.0, 6.0, n)  # LED pulser: narrow peak, every channel lit
    else:
        pes = moyal.rvs(loc=mpv, scale=6.0, size=n, random_state=rng)
    if noise:
        pes = np.concatenate([pes, rng.uniform(5.0, 12.0, n // 8)])
    pes = pes[pes >= PE_THRESHOLD]
    return np.histogram(pes, PE_EDGES)[0].astype(float)


def digi_hist(rng, n: int, run_type: str, weights: np.ndarray, problem) -> np.ndarray:
    w = weights.copy()
    if run_type == "led":
        w = np.where(w > 0, 1.0, 0.0)
    if problem == "dead_feb":
        w[2 * FEB_SIZE:3 * FEB_SIZE] = 0.0
    elif problem == "hot_channel":
        w[77] *= 25.0
    elif problem == "one_dead_channel":
        w[101] = 0.0
    return rng.poisson(n * w / w.sum()).astype(float)


def generate(out: str, seed: int = 0) -> Path:
    """Write the synthetic runs to `out` and return it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weights = detector_map(seed)
    runs, truth = {}, {}
    for run, (run_type, scale, problem, note) in RUNS.items():
        rng = np.random.default_rng(seed + run)
        n_pulses, n_digis = int(N_PULSES * scale), int(N_DIGIS * scale)
        hists = {
            PES[0]: pe_hist(rng, n_pulses, run_type, mpv=34.0 if problem == "gain_drop" else 40.0),
            PES[1]: pe_hist(rng, n_pulses, run_type, noise=problem == "low_pe_noise"),
            DIGIS: digi_hist(rng, n_digis, run_type, weights, problem),
        }
        edges = {PES[0]: PE_EDGES, PES[1]: PE_EDGES, DIGIS: CH_EDGES}
        with uproot.recreate(out / f"dqm.mu2e.synthetic_crv.run{run}.root") as f:
            for name, counts in hists.items():
                f[f"{DIR}/{name}"] = (counts, edges[name])
        runs[str(run)] = run_type
        truth[str(run)] = {
            name: {"verdict": "BAD" if (problem, name) in BAD else "GOOD",
                   "note": note if problem is None or (problem, name) in AFFECTS else "normal"}
            for name in hists
        }
    (out / "runs.json").write_text(json.dumps(
        {"reference": {"cosmics": REFERENCE_RUN}, "runs": runs}, indent=2))
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"✓ {len(RUNS)} synthetic runs written to {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="synthetic", help="output directory (default: synthetic)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, args.seed)

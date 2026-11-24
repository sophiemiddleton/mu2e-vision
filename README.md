# mu2e-vision

Vision-LLM data quality monitoring for Mu2e: https://github.com/sophiemiddleton/mu2e-vision
(formerly Mu2e-VisionDQM). The Python package is `histcompare`.

```bash
git clone https://github.com/sophiemiddleton/mu2e-vision.git
```

Compares Mu2e DQM histograms with a golden reference run in two ways and scores both:

- **Standard:** `dqm.hist.compare` from Mu2e/DQM, the KS probability against the reference.
- **ML:** a vision LLM given the input and reference plots as images, the plot's shifter
  instructions, and the run type. This follows version 2 of the CMS DQM-vision benchmark
  (Kwok and Jarugula, HAAI hackathon, Aug 2026). See [PLAN.md](PLAN.md). A survey of
  related ML-for-DQM work in HEP is kept outside this repository, in `HistCompare/literature/`
  of the CrossExperimentalAIDQM working area.

This is a test setup on synthetic CRV-like histograms whose correct answers are known.

## Running it

```bash
source mu2e-vision/setup_env.sh      # ana env + Mu2e/DQM python (set DQM_DIR, or ../DQM)
cd mu2e-vision

python -m histcompare.synthetic --out synthetic                      # 10 runs, 3 histograms each
git show 4fd943f:instructions/crv.yaml > crv_synthetic.yaml           # its instructions (removed)
python -m histcompare.benchmark --data synthetic --instructions crv_synthetic.yaml --out results/dry-run
```

Without `--model`, the benchmark runs the standard comparison and writes each prompt the
model would get to `results/<name>/prompts/`, with its images in `images/`. Nothing is
sent anywhere.

To run the vision review through Fermilab's LiteLLM gateway (an OpenAI-compatible API to
Azure-hosted models; the `openai` package is already in the ana env):

1. On site or on the VPN, open https://litellm.fnal.gov, go to the Admin Panel (`/ui`), log
   in with SSO, and choose "Create New Key" (leave Team empty). Copy the key: it is shown
   only once.
2. Run:

```bash
export LITELLM_API_KEY=sk-...
python -m histcompare.benchmark --base-url https://litellm.fnal.gov/v1 --list-models --out x
python -m histcompare.benchmark --data synthetic --instructions crv_synthetic.yaml --out results/sonnet \
    --base-url https://litellm.fnal.gov/v1 --model azure/claude-sonnet-4-6
```

The same options work for a vLLM server (`--base-url https://<server>/v1`, key in
`VLLM_API_KEY` if it needs one). The model must accept images.

Or through the Anthropic API:

```bash
python -m pip install --user anthropic   # not in the ana env
export ANTHROPIC_API_KEY=...             # or an `ant auth login` profile
python -m histcompare.benchmark --data synthetic --instructions crv_synthetic.yaml \
    --out results/opus --model claude-opus-5-5
```

The output is `summary.md`, which shows the score of each method against the truth and one
row per comparison, linked to its overlay plot. There are also `results.csv` and the
model's answers in `responses/`.

## The synthetic set

`histcompare/synthetic.py` writes ROOT files in the layout `dqm.hist.read` expects. Each
run has the same three histograms: PEs per pulse for sectors 0 and 1, and digis per channel
for sector 0 (256 channels, 4 FEBs, channels 37 and 190 masked). Run 105000 is the golden
cosmics reference.

| Run | What it is | Correct answer |
|---|---|---|
| 105001 | Normal run | All GOOD |
| 105002 | Three times longer run | All GOOD |
| 105003 | Short run, 5% of the statistics | All GOOD |
| 105004 | Sector 0 PE peak 15% low (gain drop) | Sector 0 PEs BAD |
| 105005 | FEB 2 (channels 128–191) dead | Digis BAD |
| 105006 | Channel 77 hot (25x) | Digis BAD |
| 105007 | One extra isolated dead channel | All GOOD (tolerated by the instructions) |
| 105008 | LED calibration run | All GOOD (the cosmics reference doesn't apply) |
| 105009 | Low-PE noise in sector 1 | Sector 1 PEs BAD |

The synthetic set's instructions (`instructions/crv.yaml`) were removed on 2026-11-24, so
only the instructions for real data remain. To rerun the synthetic benchmark, restore them
from git history (`git show 4fd943f:instructions/crv.yaml > crv_synthetic.yaml`) and pass
`--instructions crv_synthetic.yaml`.

## Real data (first test, 2026-11-15)

`histcompare/realdata.py` reads the cosmics_crv DQM datasets on `/pnfs`, which hold one
file per subrun. It sums each run's subruns into one file, in the layout the benchmark
reads. There is no truth for real runs.

```bash
python -m histcompare.realdata survey --out real/survey.csv      # all runs: events, live channels, FEBs
python -m histcompare.realdata build --out real/006-001 --reference 124336 \
    --runs 124189 124192 ... --instructions instructions/crv_real_draft.yaml
python -m histcompare.benchmark --data real/006-001 \
    --instructions instructions/crv_real_draft.yaml --out results/real-006-001-dry
# then add --base-url https://litellm.fnal.gov/v1 --model azure/claude-opus-5-5
```

The current test uses `crvreco-006-001`. It has 20 runs from 124189–125850 plus the
reference, now **125026** (set in `real/006-001/runs.json`; earlier passes used 124336,
then 124228, kept in `runs_ref124228.json`), and 6 histograms: `PEr`, `crvDigisPerChannel`, `h1_peakAdc`, `logchi2`,
`errorBits` and `activeFebCount`.

- **[instructions/crv_real_draft.yaml](instructions/crv_real_draft.yaml) is a draft.** It
  was written by hand from the CRVDQM source and the bin contents of these runs, and the
  CRV group hasn't reviewed it. Its tolerances are guesses. Version 0.2 (2026-11-24)
  describes reference 125026. Every pass before that used version 0.1, which described
  124336 (`git show 4fd943f:instructions/crv_real_draft.yaml`). `histcompare.rulecheck`
  codes the same criteria as numeric tests, so keep the two in step.
- **The reference choice matters.** The detector configuration changes across the dataset:
  the live channels go 1720 → 1792 → 1536 → 1600 → 1920. Even within 124xxx–125xxx, the
  PEr shape changes twice. From about 125026, the highest bin moves from 8–12 to 4–8 PE and
  the muon shoulder becomes a flat plateau at about 28–48 PE. From about 125707, the peak is
  back at 8–12 PE, with a plateau at about 24–40 PE. Run 125846, the
  first candidate reference, turned out to have channels 1024–1535 and 2048–2303 dead.
  124336 has extra dead half-blocks (even channels 1664–1791, odd 1920–2047). 125026 has
  the same half-block pattern as 124228 and is in the period after the PEr/logchi2 change
  (from about 125026), so earlier runs are now judged against the later state.
- **Candidate problems seen by eye:** in 124192, 124324 and 124547, 64-channel blocks are
  missing and FEBBufferIssue and FEBOverflow are set on every ROC header. In 124529 only
  channels 1024–1535 and 2048–2303 read out. 125158 is similar, and in 125014 those
  ranges are low. 125850 has Group1, Group2 and uBMatch errors on about 35% of headers. In
  125014 the 6- and 24-FEB headers are unbalanced, at 26% and 74%. From 125707 the PE
  shoulder is higher.
- **KS is no use at these statistics:** 86 of 120 comparisons are BAD, and every run fails
  on all four shape histograms. With 10⁸–10⁹ entries, any systematic difference gives
  p ≈ 0.

## Choices to know about

- **KS thresholds.** `dqm.hist.compare` defaults to calling a comparison tight above
  KS > 0.999 and loose above 0.99. Those defaults suit comparing a sample with itself (a
  validation rerun). Two independent runs of the same detector give a KS probability that
  is roughly uniform between 0 and 1, so almost every run would fail. The benchmark
  instead uses `independent=True` with tight > 0.05 and loose > 0.01 (`--ks-tight`,
  `--ks-loose`).
- **One reference.** Every run is compared with the golden cosmics run, LED runs included.
  The LED instructions say that reference doesn't apply. That is the situation the talk's
  "circulating" run tested.
- **The model call** uses the Anthropic API with Claude Opus 5.5 by default, at high effort,
  with refusal fallbacks enabled. The talk reached its models through FNAL askSage; that
  route isn't implemented.

## Files

| File | Role |
|---|---|
| `histcompare/synthetic.py` | Generates the synthetic runs, `runs.json` and `truth.json` |
| `histcompare/context.py` | Instructions, run metadata, rendering through `dqm.plot`, and the prompt |
| `histcompare/vision.py` | The model call and verdict parsing |
| `histcompare/benchmark.py` | Runs both comparisons over all runs and scores them |
| `instructions/crv_real_draft.yaml` | Draft shifter instructions for the six crvreco-006-001 histograms |
| `setup_env.sh` | Environment for a Mu2e gpvm |

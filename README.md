# HistCompare

Compares Mu2e DQM histograms with a golden reference run in two ways and scores both:

- **Standard:** `dqm.hist.compare` from Mu2e/DQM, the KS probability against the reference.
- **ML:** a vision LLM given the input and reference plots as images, the plot's shifter
  instructions, and the run type. This follows version 2 of the CMS DQM-vision benchmark
  (Kwok and Jarugula, HAAI hackathon, Aug 2026). See [PLAN.md](PLAN.md), and
  [literature/](literature/README.md) for related ML-for-DQM work in HEP.

This is a test setup on synthetic CRV-like histograms whose correct answers are known.

## Running it

```bash
source Mu2e-VisionDQM/setup_env.sh      # ana env + Mu2e/DQM python (set DQM_DIR, or ../DQM)
cd Mu2e-VisionDQM

python -m histcompare.synthetic --out synthetic                      # 10 runs, 3 histograms each
python -m histcompare.benchmark --data synthetic --out results/dry-run
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
python -m histcompare.benchmark --data synthetic --out results/sonnet \
    --base-url https://litellm.fnal.gov/v1 --model azure/claude-sonnet-4-6
```

The same options work for a vLLM server (`--base-url https://<server>/v1`, key in
`VLLM_API_KEY` if it needs one). The model must accept images.

Or through the Anthropic API:

```bash
python -m pip install --user anthropic   # not in the ana env
export ANTHROPIC_API_KEY=...             # or an `ant auth login` profile
python -m histcompare.benchmark --data synthetic --out results/opus --model claude-opus-5-5
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

The instructions are in [instructions/crv.yaml](instructions/crv.yaml), in the talk's
format. They were written for this synthetic set, and should be replaced by the real CRV
shifter instructions.

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
| `instructions/crv.yaml` | Shifter instructions for the three histograms |
| `setup_env.sh` | Environment for a Mu2e gpvm |

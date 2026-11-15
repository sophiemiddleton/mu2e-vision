# HistCompare: ML-assisted histogram comparison (proposed plan)

HistCompare would compare Mu2e DQM histograms with a golden reference run in two ways and
show the results side by side:

- **Standard:** overlay the golden run and run the existing `dqm.hist.compare`: ROOT's KS
  probability (reproduced in numpy) plus the running-sum fraction, with tight/loose/fail.
- **ML (new):** the vision-LLM approach from the CMS "DQM vision" work (M. Kwok,
  S. Jarugula, HAAI hackathon, 18 Aug 2026). A vision model is given the input plot as an
  image, the golden-run image, the plot's shifter instructions (one YAML entry per plot)
  and the run metadata (e.g. cosmics or beam). It answers in four parts: the relevant
  instructions, what it sees, how that compares with the reference and instructions, and
  a GOOD/BAD verdict with the action to take. In their tests, the best model (Claude
  Opus 5) scored 4.8/5 overall. It ignored plotting artefacts, compared dead-pixel
  patterns, and recognised that a different colour scale came from a longer run. Each
  answer took about 10 s.

An agent would put this together for shifters: fetch the histograms, render them, run
both comparisons, and show the overlay with the model's answer. It would not raise alarms
itself.

To evaluate this, we would run both on the same set of runs and compare the GOOD/BAD
verdict with the KS status and with labels from a human expert. The talk found that its
LLM judge sometimes scored correct answers as wrong, so human labels are needed, at least
for the verdict.

## Questions to settle before starting

1. Can we reuse their setup (version 2: instructions from YAML, reference image passed in
   directly, no retrieval), or work with them directly?
2. Which histograms should we start with? The CRV ones, such as PE MPV and digis per
   channel for each sector? Who writes the instructions for each one? Do shifter
   instructions already exist, for example on a wiki?
3. Which run or runs are the golden reference? Who chooses them? Do we need one per run
   type (cosmics, beam) or per calibration epoch?
4. Which model, and how do we access it? The talk used FNAL askSage for privacy.
5. Should this go in the DQM web Histograms tab, which already overlays two files and runs
   the comparison, or be a separate tool?

## Tools

Most of the histogram handling already exists in Mu2e/DQM (`python/dqm/...`).

| Step | Tool | Needs |
|---|---|---|
| Find the files and the golden run | `dqm.catalog`: looks up DQM datasets in metacat and turns a file name into a path uproot can open | DQM's `catalog` extra |
| Read the histograms | `dqm.hist.read`: uproot, no ROOT; `common_histograms` pairs the same histogram across two files | Present |
| Standard comparison | `dqm.hist.compare`: KS probability and fraction, tight/loose/fail | Present |
| Render images for the model | `dqm.plot` (matplotlib PNG rendering), drawn the same way for input and reference | Present |
| Plot instructions | One YAML file per subsystem, one entry per histogram: what it shows, quality criteria for each run type, known issues, action if bad (the format in the talk) | To be written |
| Run metadata | The run type for each run (cosmics, beam, ...), from the DQM database or run records | Source to be decided |
| ML description | A vision LLM with the four-part prompt from the talk. The model is chosen by the access question above. | Model access |
| Display | The Histograms tab in the DQM web app (`dqm.web.histogram`), which would gain the model's answer next to the KS status | Present |
| Agent | Claude through the Claude Agent SDK, given a small MCP server that wraps the steps above, plus the existing DQM MCP server for metrics and alarms. It only reads and summarises. | Claude Agent SDK |

New code would be small: the rendering and context-building step (image, reference,
instruction entry, run type), the model call, and a script that runs both comparisons over
a set of runs for evaluation.

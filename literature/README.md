# Literature: ML for data quality monitoring in HEP

This is a survey of arXiv for ML tools used in HEP DQM, with notes on how each relates to
HistCompare. It was searched on 2026-11-06 through the arXiv API. The search covered "data
quality monitoring", "data certification", "detector monitoring", "online monitoring" and
"histogram" combined with ML, autoencoder, anomaly, LLM, VLM or agent terms, limited to
hep-ex, physics.ins-det and physics.data-an. Known items were looked up by name. BibTeX for
everything below is in [references.bib](references.bib).

**Main finding:** no paper on arXiv uses a vision-language model to judge DQM histograms
against a reference. The CMS DQM-vision work (Kwok and Jarugula, HAAI hackathon, Aug 2026)
seems to exist only as a talk. The deployed ML in DQM is autoencoders and statistical
tests. The closest thing to our image-based approach is JLab's Hydra, a trained CNN rather
than a prompted VLM.

## 1. Closest to HistCompare: a histogram vs a reference, a per-run verdict

| arXiv | Tool, experiment | Method | Status | Relevance |
|---|---|---|---|---|
| [2501.13789](https://arxiv.org/abs/2501.13789) | **AutoDQM**, CMS | Compares histograms with reference runs using a beta-binomial test, PCA, and an autoencoder on histogram images. Tested on all 2022 pp data. | Used in CMS. EPJ RI 10, 4 (2026). | The most direct counterpart. A stronger standard method than KS: beta-binomial handles low statistics and per-bin pulls. **Candidate second baseline.** |
| [2501.19237](https://arxiv.org/abs/2501.19237) | **DINAMO**, LHCb | Histogram templates that evolve over time, with uncertainties: EWMA (statistical) and a transformer-encoder version. | The EWMA version is being commissioned in LHCb. MLST 6 (2025). [Code](https://github.com/ArseniiGav/DINAMO). | Addresses the drift of a fixed golden run. Validated on synthetic data, like ours. |
| [2609.23329](https://arxiv.org/abs/2609.23329) | **ARGUS**, not tied to an experiment | Config-driven framework with four monitors that are compared, never merged: expert cuts, an ML second opinion, trend forecasting, and reference-histogram comparison with pluggable tests. | Under evaluation in two unnamed experiments. | Almost the same design as ours: synthetic detector, planted faults as truth, ML kept beside the standard method. A model for how to report results. |
| [2611.07819](https://arxiv.org/abs/2611.07819) | **ATLAS Open Data DQM benchmark** | 1289-bin histograms for each of 39,828 luminosity blocks, with injected defects and labels from the Good Run Lists. Gradient boosting on residuals, calibrated by conformal prediction. | Posted 2026-11-05. Public. | Key result: shapes vary far more between runs than within one (χ²/ndf 5.9 against 1.1), so **references built across runs fail**. This challenges the single-golden-run setup. It is also a public real-data test set with GRL truth. |
| [2301.08917](https://arxiv.org/abs/2301.08917), [2303.05413](https://arxiv.org/abs/2303.05413) | Kernel goodness-of-fit, CMS-style drift tubes (Padova) | A learned likelihood-ratio test of new data against a reference sample (NPLM / Falkon kernels). | Prototype. | An ML replacement for KS as the two-sample test, sensitive to any kind of anomaly. |

## 2. Deployed anomaly detection on detector maps (autoencoders, GNNs)

| arXiv | Experiment | Method | Status |
|---|---|---|---|
| [2309.10157](https://arxiv.org/abs/2309.10157) (also [2308.16659](https://arxiv.org/abs/2308.16659), [2407.20278](https://arxiv.org/abs/2407.20278)) | CMS ECAL | Autoencoder on occupancy maps, with corrections for spatial variation and the time evolution of anomalies. False discovery rate 10⁻²–10⁻⁴. | **Deployed online in Run 3.** CSBS 8, 11 (2024). |
| [2311.04190](https://arxiv.org/abs/2311.04190) | CMS HCAL (GraphSTAD) | CNN + GNN for channels on shared backend electronics + RNN over time. | Being integrated into CMS production. [Code](https://github.com/muleina/CMS_HCAL_ML_OnlineDQM). |
| [2603.02369](https://arxiv.org/abs/2603.02369) | CMS HCAL (DepthViT) | Masked-autoencoder ViT ensemble with continual learning against drift. | Research. |
| [1808.00911](https://arxiv.org/abs/1808.00911) | CMS muon (Pol et al.) | Supervised CNN + convolutional autoencoder on geographic images of the detector. | The seminal CMS ML4DQM paper. |
| [2512.05977](https://arxiv.org/abs/2512.05977) | ATLAS LAr | LSTM autoencoder on time series of moments of cluster properties. | Proceedings. |
| [2403.00689](https://arxiv.org/abs/2403.00689) | **Hydra**, JLab (GlueX, CLAS12, all halls) | Computer vision (trained CNN classifiers) on DQM plot images, with a web front end for shifters. | **Deployed since 2019.** |

Hydra is the nearest deployed analogue to HistCompare's vision route: it classifies the same
plot images shifters see. Unlike a VLM, it needs labelled training images per plot type.

For Mu2e, the HCAL GraphSTAD idea fits the CRV: channels share FEBs and ROCs, and that
grouping is where correlated faults show up.

## 3. Human-in-the-loop and workflow

- [1709.08607](https://arxiv.org/abs/1709.08607): Borisyak et al., CMS. Auto-classify the
  clear good and bad cases and send the "grey area" to humans. About 20% of the load was
  automated with no loss. This is a useful framing for how a vision verdict would be used.
- [1711.07051](https://arxiv.org/abs/1711.07051): Azzolini et al., CMS. Locate the channels
  responsible for an anomaly using only a global good/bad flag. This matters if run-level
  good-run bits become our only labels.
- [2405.15508](https://arxiv.org/abs/2405.15508): RL with a human in the loop for DQM. A
  multi-agent PPO prototype on synthetic data that asks the shifter only when needed.
- [2511.18172](https://arxiv.org/abs/2511.18172): MEDIC. A detector simulated with Delphes,
  with regions switched off as faults. Simulation-driven DQM development (JINST 2026).

## 4. LLMs, VLMs and agents near HEP operations (none do DQM verdicts)

- [2606.04755](https://arxiv.org/abs/2606.04755): **Archi**. A RAG + agent framework
  deployed for CMS Computing Operations since Feb 2026. Open-weight local models were
  competitive. A pattern for giving the reviewer shifter docs and the elog.
- [2511.17154](https://arxiv.org/abs/2511.17154): ALICE-FIT. A proposed LLM + RAG assistant
  for detector-control troubleshooting.
- [2605.17318](https://arxiv.org/abs/2605.17318): **RooAgent**. An LLM agent with PyROOT
  tools for histogram inspection and fitting, including an MCP server for Claude.
  Tool use rather than vision. It could give an agent the ability to query bin contents.
- [2609.00107](https://arxiv.org/abs/2609.00107): Agentic harness for HEP (Menzo et al.).
  Design principles for reusable, versioned agent tools.
- [2604.24589](https://arxiv.org/abs/2604.24589): **AstroVLBench**, a caution for us. Six
  frontier VLMs were tested on astronomy data. **Giving 1-D data as numerical tables instead
  of rendered plots improved accuracy by up to 23 points.** 26% of correct answers had
  invalid reasoning ("right answer, wrong reason"). This argues for (a) an ablation giving
  the model bin contents as text and (b) grading the Observations, not only the Verdict.
- [2508.19376](https://arxiv.org/abs/2508.19376): Fine-tuned LLaMA-3.2 VLM matching CNNs on
  NOvA/DUNE-style neutrino event images. Shows a VLM can be fine-tuned on detector images.

## 5. Review

- [2312.14190](https://arxiv.org/abs/2312.14190): Belis, Odagiu, Årrestad, *Machine Learning
  for Anomaly Detection in Particle Physics*, Reviews in Physics 12 (2024). Covers detector
  monitoring as well as BSM searches.

## What this suggests for HistCompare

1. **Add AutoDQM's beta-binomial test as a second standard method** beside KS. Reviewers
   familiar with CMS will expect it, and it is simple to implement.
2. **Treat the choice of reference as a variable to test.** The ATLAS benchmark and DINAMO
   both find that a fixed golden run degrades. Compare a fixed golden run, the previous good
   run, and a DINAMO-style EWMA template.
3. **Run a text-input ablation for the VLM** (bin contents and per-bin pulls as a table),
   following AstroVLBench, and spot-check the reasoning, not only the GOOD/BAD verdict.
4. **Real-data truth:** the ATLAS Open Data benchmark is a public set labelled with good-run
   lists that we could run on today, before Mu2e's good-run database has real entries.
5. **Novelty:** the vision-LLM DQM approach has no arXiv paper yet, so a Mu2e result would
   be early. Check with DJ and the CMS authors whether a write-up is in preparation.

This survey covers arXiv only. Worth getting directly: the CMS DQM-vision slides (Kwok and
Jarugula). A search of CHEP/ACAT proceedings and Indico (not done here) would likely turn up
more ML-for-DQM work from LHCb and ALICE, which is sparse on arXiv.

"""
Compare every histogram of every run with the golden run, two ways, and score both

  standard  dqm.hist.compare: the KS probability against the reference, GOOD unless the
            status is fail
  ML        a vision LLM given the two images, the plot instructions and the run type
            (only with --model; otherwise each prompt is written out and not sent)

Both are scored against truth.json when the data has one (the synthetic set does).

    python -m histcompare.benchmark --data synthetic --out results/dry-run
    python -m histcompare.benchmark --data synthetic --out results/opus --model claude-opus-5-5
"""

import argparse
import csv
import json
from pathlib import Path

from dqm.hist.compare import ComparePars, compare_histograms
from dqm.hist.read import open_file

from . import context
from .vision import parse_verdict

FIELDS = ["run", "run_type", "histogram", "truth", "truth_note", "ks", "ks_status",
          "standard", "ml"]


def run_file(data_dir: Path, run) -> Path:
    """The one file of a run: dqm.mu2e.<dataset>.run<run>.root (synthetic or real/realdata.py)."""
    found = sorted(data_dir.glob(f"dqm.mu2e.*.run{run}.root"))
    if len(found) != 1:
        raise SystemExit(f"expected one file for run {run} in {data_dir}, found {len(found)}")
    return found[0]


def score(rows, method: str) -> dict:
    """Counts with BAD as the positive class, over rows that have both answers."""
    pairs = [(r["truth"], r[method]) for r in rows if r["truth"] and r[method]]
    c = {"BAD caught": 0, "BAD missed": 0, "GOOD flagged": 0, "GOOD passed": 0}
    for truth, said in pairs:
        if truth == "BAD":
            c["BAD caught" if said == "BAD" else "BAD missed"] += 1
        else:
            c["GOOD flagged" if said == "BAD" else "GOOD passed"] += 1
    c["agreement"] = f"{sum(t == s for t, s in pairs)}/{len(pairs)}" if pairs else "-"
    return c


def write_summary(out: Path, rows, args) -> None:
    lines = ["# HistCompare benchmark", "",
             f"Data: `{args.data}`. Standard: KS against the golden run, "
             f"tight > {args.ks_tight}, loose > {args.ks_loose}, otherwise fail (BAD). "
             f"ML: {args.model or 'not run (prompts written to prompts/)'}.", "",
             "| | BAD caught | BAD missed | GOOD flagged | GOOD passed | agreement with truth |",
             "|---|---|---|---|---|---|"]
    for method in ("standard", "ml"):
        s = score(rows, method)
        lines.append(f"| {method} | {s['BAD caught']} | {s['BAD missed']} | "
                     f"{s['GOOD flagged']} | {s['GOOD passed']} | {s['agreement']} |")
    lines += ["", "| run | type | histogram | truth | KS | standard | ML | note |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = lambda v: f"**{v}**" if v and r["truth"] and v != r["truth"] else (v or "-")
        lines.append(f"| {r['run']} | {r['run_type']} | [{r['histogram']}]"
                     f"(images/{r['run']}_{r['histogram']}_overlay.png) | {r['truth'] or '-'} | "
                     f"{r['ks']:.3g} | {mark(r['standard'])} | {mark(r['ml'])} | "
                     f"{r['truth_note']} |")
    lines += ["", "Bold: disagrees with the truth."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="synthetic", help="directory from histcompare.synthetic")
    ap.add_argument("--instructions",
                    help="shifter instructions YAML whose plot names match the data's histograms, "
                         "e.g. instructions/crv_real_draft.yaml for crvreco-006-001")
    ap.add_argument("--out", required=True, help="output directory (a rerun into the same "
                    "directory reuses the saved model responses, so an interrupted run resumes)")
    ap.add_argument("--runs", nargs="*", help="only these runs (default: all but the references)")
    ap.add_argument("--model", help="run the vision review with this model, e.g. claude-opus-5-5")
    ap.add_argument("--base-url", help="send the review to this OpenAI-compatible server "
                    "(e.g. a vLLM server's .../v1) instead of the Anthropic API")
    ap.add_argument("--list-models", action="store_true",
                    help="with --base-url: list the server's models and exit")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    # dqm.hist.compare's own defaults (0.999 / 0.99) are for comparing a sample with itself,
    # e.g. a validation rerun; two independent runs need p-value thresholds
    ap.add_argument("--ks-tight", type=float, default=0.05)
    ap.add_argument("--ks-loose", type=float, default=0.01)
    args = ap.parse_args(argv)
    if not args.instructions and not args.list_models:
        ap.error("--instructions is required (e.g. instructions/crv_real_draft.yaml)")

    reviewer = None
    if args.base_url:
        from .vision import OpenAIReviewer
        reviewer = OpenAIReviewer(args.base_url, args.model)
        if args.list_models:
            print("\n".join(reviewer.models()))
            return
        if not args.model:
            ap.error("--base-url needs --model (see --list-models)")
    elif args.model:
        from .vision import Reviewer
        reviewer = Reviewer(args.model, args.effort)

    data, out = Path(args.data), Path(args.out)
    for sub in ("images", "prompts", "responses"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    plots = context.load_instructions(args.instructions)
    references, run_types = context.load_runs(data)
    meta = json.loads((data / "runs.json").read_text())
    ref_for = {r: int(v) for r, v in meta.get("reference_for", {}).items()}
    all_refs = set(references.values()) | set(meta.get("references", []))
    truth_path = data / "truth.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
    pars = ComparePars(independent=True, tight=args.ks_tight, loose=args.ks_loose)

    runs = args.runs or [r for r in run_types if int(r) not in all_refs]
    rows = []
    for run in runs:
        run_type = run_types[run]
        # one golden run, or one per configuration period when runs.json has reference_for;
        # the LED instructions say a cosmics reference doesn't apply
        ref_run = ref_for.get(run, references["cosmics"])
        with open_file(str(run_file(data, run))) as f, open_file(str(run_file(data, ref_run))) as fr:
            for name, instr in plots.items():
                path = f"{instr.get('folder', '')}/{name}".lstrip("/")
                h, ref = f.read(path), fr.read(path)
                cmp = compare_histograms(ref, h, pars, path=path)
                xlabel = instr.get("xlabel") or context.xlabel_for(name)

                img = out / "images" / f"{run}_{name}_input.png"
                ref_img = out / "images" / f"{ref_run}_{name}_reference.png"
                img.write_bytes(context.render(h, f"{instr['title']}, run {run}", xlabel, f"run {run}"))
                if not ref_img.exists():
                    ref_img.write_bytes(context.render(
                        ref, f"{instr['title']}, run {ref_run}", xlabel, f"run {ref_run}"))
                (out / "images" / f"{run}_{name}_overlay.png").write_bytes(context.render_overlay(
                    h, ref, f"{instr['title']}, run {run}", xlabel, f"run {run}", f"run {ref_run}"))

                content = context.user_content(name, run, run_type, ref_run, instr,
                                               img.read_bytes(), ref_img.read_bytes())
                (out / "prompts" / f"{run}_{name}.md").write_text(
                    f"## System\n\n{context.SYSTEM_PROMPT}\n\n## User\n\n"
                    + context.as_text(content, [f"../images/{ref_img.name}", f"../images/{img.name}"]))

                ml = None
                saved = out / "responses" / f"{run}_{name}.md"
                if reviewer and saved.exists() and parse_verdict(saved.read_text()):
                    # resume: a response from an earlier, interrupted run of the same --out
                    ml = parse_verdict(saved.read_text())
                    print(f"  {run} {name}: ML {ml} (saved)")
                elif reviewer:
                    text, ml = reviewer.review(content)
                    (out / "responses" / f"{run}_{name}.md").write_text(text + "\n")
                    print(f"  {run} {name}: ML {ml}")

                t = truth.get(run, {}).get(name, {})
                rows.append({"run": run, "run_type": run_type, "histogram": name,
                             "truth": t.get("verdict"), "truth_note": t.get("note", ""),
                             "ks": cmp.kolmogorov, "ks_status": cmp.status.name,
                             "standard": "GOOD" if cmp.ok else "BAD", "ml": ml})

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_summary(out, rows, args)
    for method in ("standard", "ml"):
        print(f"{method:8s}", score(rows, method))
    print(f"✓ {len(rows)} comparisons; see {out / 'summary.md'}")


if __name__ == "__main__":
    main()

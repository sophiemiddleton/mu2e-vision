"""
What a reviewer is given for one histogram

Follows version 2 of the CMS DQM-vision benchmark (Kwok, Jarugula, HAAI hackathon,
Aug 2026): the input plot and the reference plot as images, the plot's shifter
instructions from a YAML file, and the run type. Rendering goes through dqm.plot in its
ROOT style, so the images look like what a Mu2e shifter sees, stats box included.
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from dqm.plot.spec import PlotSpec
from dqm.plot.static import to_png_bytes

# The talk's prompt, with CMS replaced by Mu2e
SYSTEM_PROMPT = """\
You are an assistant to Mu2e detector operations shifters, helping evaluate DQM monitoring plots.
Assess whether the input plot indicates a detector problem requiring action.
Respond in 4 sections:
Instructions: Quote the shift instructions most relevant to this plot type.
Observations: Describe what you see in the input plot (shape, notable features, anomalies). \
If a reference plot is shown, describe it and note how the input differs.
Assessment: Compare the input plot against the quoted instructions. If a reference is \
provided, use it to calibrate what normal looks like.
Verdict: State GOOD or BAD. If BAD, specify the required action from the instructions."""


def load_instructions(path: str) -> Dict[str, dict]:
    """Plot name -> its instruction entry."""
    with open(path) as f:
        return yaml.safe_load(f)["plots"]


def load_runs(data_dir: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """(run type -> reference run, run -> run type) from runs.json."""
    meta = json.loads((Path(data_dir) / "runs.json").read_text())
    return meta["reference"], meta["runs"]


def stats_lines(h) -> List[str]:
    """Entries, mean and standard deviation, as in a ROOT stats box (binned estimate)."""
    counts = np.asarray(h.values(), dtype=float)
    centers = h.axes[0].centers
    n = counts.sum()
    mean = float((counts * centers).sum() / n) if n else 0.0
    std = float(np.sqrt((counts * (centers - mean) ** 2).sum() / n)) if n else 0.0
    return [f"Entries  {n:.0f}", f"Mean  {mean:.4g}", f"Std Dev  {std:.4g}"]


def render(h, title: str, xlabel: str, label: str) -> bytes:
    """One histogram on its own, as the model sees it."""
    # the static renderer draws annotations, not HistTrace.stats, as the stats box
    spec = PlotSpec(title=title, xlabel=xlabel, ylabel="Entries", legend=False,
                    annotations=stats_lines(h))
    spec.add_hist(h, label=label)
    return to_png_bytes(spec, figsize=(10.0, 6.0), style="root", dpi=100)


def render_overlay(h, ref, title: str, xlabel: str, label: str, ref_label: str) -> bytes:
    """The input with the reference scaled to the same number of entries, for people."""
    scale = h.sum() / ref.sum() if ref.sum() else 1.0
    spec = PlotSpec(title=title, xlabel=xlabel, ylabel="Entries", annotations=stats_lines(h))
    spec.add_hist(ref * scale, label=f"{ref_label} (scaled)", color="gray")
    spec.add_hist(h, label=label)
    return to_png_bytes(spec, figsize=(10.0, 6.0), style="root", dpi=100)


def xlabel_for(name: str) -> str:
    return "Channel" if "PerChannel" in name else "PEs"


def user_content(name: str, run: str, run_type: str, ref_run: int, instruction: dict,
                 image: bytes, ref_image: bytes) -> List[dict]:
    """The user turn: reference image, input image, instructions and run type."""
    def png(data: bytes) -> dict:
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.standard_b64encode(data).decode()}}
    return [
        {"type": "text", "text": f"Reference plot: golden cosmics run {ref_run}."},
        png(ref_image),
        {"type": "text", "text": f"Input plot: run {run}."},
        png(image),
        {"type": "text", "text": (
            f"Plot: {instruction.get('folder', '')}/{name} ({instruction.get('title', name)})\n"
            f"Run metadata: {json.dumps({run: run_type, str(ref_run): 'cosmics'})}\n\n"
            f"Plot instructions:\n{instruction['instruction']}\n"
            f"Is the input plot GOOD or BAD?")},
    ]


def as_text(content: List[dict], image_paths: List[str]) -> str:
    """The user turn written out for reading, with image blocks replaced by their files."""
    paths = iter(image_paths)
    parts = [f"![image]({next(paths)})" if b["type"] == "image" else b["text"] for b in content]
    return "\n\n".join(parts)

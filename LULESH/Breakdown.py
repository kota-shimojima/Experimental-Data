#!/usr/bin/env python3
"""
Generate stacked bar charts per rank for four execution environments:
- Native + Unenc
- Native + Enc
- SGX + Unenc
- SGX + Enc

Extracts the required metrics from the provided breakdown text files and
produces a single figure with four subplots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "Native + Unenc": BASE_DIR / "Native" / "Breakdown-PS60-Iter50-Unenc.csv",
    "Native + Enc": BASE_DIR / "Native" / "Breakdown-PS60-Iter50-Enc.csv",
    "SGX + Unenc": BASE_DIR / "SGX" / "Breakdown-PS60-Iter50-Unenc.csv",
    "SGX + Enc": BASE_DIR / "SGX" / "Breakdown-PS60-Iter50-Enc.csv",
}

UNENC_METRICS = ["isend", "irecv", "wait", "waitall", "allreduce"]
ENC_METRICS = ["enc-isend", "enc-irecv", "enc-wait", "enc-waitall", "enc-allreduce"]


@dataclass
class ParsedData:
    env: str
    df: pd.DataFrame


def parse_file(path: Path, metrics: List[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by rank with the requested metrics."""
    metric_to_regex = {m: m for m in metrics}
    # Handle the "wait all time" spelling used in Unenc files.
    if "waitall" in metric_to_regex:
        metric_to_regex["waitall"] = r"wait\s*all"

    pattern_map = {
        m: re.compile(
            rf"rank is (\d+).*?{pattern}\s*time is\s*([0-9.]+)s", re.IGNORECASE
        )
        for m, pattern in metric_to_regex.items()
    }
    data: Dict[int, Dict[str, float]] = {}

    for line in path.read_text().splitlines():
        for metric, pat in pattern_map.items():
            m = pat.search(line)
            if not m:
                continue
            rank = int(m.group(1))
            value = float(m.group(2))
            data.setdefault(rank, {})
            # Do not overwrite if already captured; keep the first occurrence.
            data[rank].setdefault(metric, value)

    # Ensure ranks are ordered and all requested metrics exist (fill missing with 0).
    rows = []
    for rank in sorted(data.keys()):
        row = {"rank": rank}
        for metric in metrics:
            row[metric] = data[rank].get(metric, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_env(ax, parsed: ParsedData, metrics: List[str], colors: List[str]) -> None:
    df = parsed.df.copy()
    if df.empty:
        ax.set_title(f"{parsed.env}\n(no data)")
        return

    df["rank_label"] = df["rank"] + 1  # Display ranks as 1-based.
    bottoms = pd.Series([0.0] * len(df))

    for metric, color in zip(metrics, colors):
        ax.bar(df["rank_label"], df[metric], bottom=bottoms, label=metric, color=color)
        bottoms += df[metric]

    ax.set_title(parsed.env)
    ax.set_xlabel("Rank (1-8)")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(df["rank_label"])
    ax.legend(fontsize="small", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def main() -> None:
    parsed_list: List[ParsedData] = []
    for env_name, file_path in FILES.items():
        if "Enc" in env_name:
            metrics = ENC_METRICS
        else:
            metrics = UNENC_METRICS
        df = parse_file(file_path, metrics)
        parsed_list.append(ParsedData(env=env_name, df=df))

    colors_unenc = ["#4C78A8", "#9C755F", "#72B7B2", "#E45756", "#59A14F"]
    colors_enc = ["#F28E2B", "#EDC948", "#B07AA1", "#FF9DA7", "#8CD17D"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, parsed in zip(axes.flat, parsed_list):
        if "Enc" in parsed.env:
            plot_env(ax, parsed, ENC_METRICS, colors_enc)
        else:
            plot_env(ax, parsed, UNENC_METRICS, colors_unenc)

    fig.suptitle("Communication Breakdown per Rank (Stacked)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = BASE_DIR / "breakdown.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()


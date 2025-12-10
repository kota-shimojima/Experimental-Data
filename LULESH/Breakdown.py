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


def plot_combined(ax, parsed_list: List[ParsedData]) -> None:
    """Plot all environments into a single stacked chart with shared legend/colors."""
    if not parsed_list:
        ax.set_title("No data")
        return

    color_map = {
        "isend": "#4C78A8",
        "irecv": "#9C755F",
        "wait": "#72B7B2",
        "waitall": "#E45756",
        "allreduce": "#59A14F",
    }

    bar_width = 0.8
    gap = 1.5  # gap between environments
    x_positions = []
    x_labels = []
    env_centers = []
    env_names = []

    max_rank = max((len(p.df) for p in parsed_list if not p.df.empty), default=0)
    current_offset = 0.0

    for parsed in parsed_list:
        df = parsed.df.copy()
        df["rank_label"] = df["rank"] + 1  # 1-based rank for display
        bottoms = pd.Series([0.0] * len(df))

        metrics = ENC_METRICS if "Enc" in parsed.env else UNENC_METRICS
        for metric in metrics:
            base_metric = metric.replace("enc-", "")
            color = color_map[base_metric]
            ax.bar(
                df["rank_label"] + current_offset,
                df[metric],
                bottom=bottoms,
                width=bar_width,
                color=color,
                label=base_metric,
            )
            bottoms += df[metric]

        # Tick labels: only rank names; environment is annotated separately.
        x_positions.extend((df["rank_label"] + current_offset).tolist())
        x_labels.extend([f"R{r}" for r in df["rank_label"]])

        # Track center position for environment label.
        if not df.empty:
            start = df["rank_label"].min() + current_offset
            end = df["rank_label"].max() + current_offset
            env_centers.append((start + end) / 2)
            env_names.append(parsed.env)

        current_offset += max_rank + gap

    # Unique legend entries preserving order.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)

    ax.set_title("Communication Breakdown per Rank (All Environments)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.legend(uniq_handles, uniq_labels, fontsize="small", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate environment names once per group above the bars.
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.1)
    for center, name in zip(env_centers, env_names):
        ax.text(
            center,
            y_max * 1.02,
            name,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main() -> None:
    parsed_list: List[ParsedData] = []
    for env_name, file_path in FILES.items():
        if "Enc" in env_name:
            metrics = ENC_METRICS
        else:
            metrics = UNENC_METRICS
        df = parse_file(file_path, metrics)
        parsed_list.append(ParsedData(env=env_name, df=df))

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_combined(ax, parsed_list)
    fig.tight_layout()

    output_path = BASE_DIR / "breakdown.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()


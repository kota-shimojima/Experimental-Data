#!/usr/bin/env python3
"""
Generate a combined stacked bar chart per rank for four environments:
- Native + Unenc
- Native + Enc
- SGX + Unenc
- SGX + Enc

Additionally, annotate how many times each communication component increased
from Unenc to Enc within Native and SGX using arrows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "Native + Unenc": BASE_DIR / "Native" / "PS=80" / "Breakdown-PS80-Iter50-Unenc.csv",
    "Native + Enc": BASE_DIR / "Native" / "PS=80" / "Breakdown-PS80-Iter50-Enc.csv",
    "SGX + Unenc": BASE_DIR / "SGX" / "PS=80" / "Breakdown-PS80-Iter50-Unenc.csv",
    "SGX + Enc": BASE_DIR / "SGX" / "PS=80" / "Breakdown-PS80-Iter50-Enc.csv",
}

UNENC_METRICS = ["isend", "irecv", "wait", "waitall", "allreduce"]
ENC_METRICS = ["enc-isend", "enc-irecv", "enc-wait", "enc-waitall", "enc-allreduce"]
METRIC_PAIRS: List[Tuple[str, str]] = [
    ("isend", "enc-isend"),
    ("irecv", "enc-irecv"),
    ("wait", "enc-wait"),
    ("waitall", "enc-waitall"),
    ("allreduce", "enc-allreduce"),
]


@dataclass
class ParsedData:
    env: str
    df: pd.DataFrame


def parse_file(path: Path, metrics: List[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by rank with the requested metrics."""
    # Check if this is an Enc file by checking if "enc" is in any metric name
    is_enc = any("enc-" in m for m in metrics)
    
    metric_to_regex = {}
    for m in metrics:
        if is_enc:
            # Enc file format: "isend enc time", "irecv dec time", etc.
            if m == "enc-isend":
                metric_to_regex[m] = r"isend\s+enc"
            elif m == "enc-irecv":
                metric_to_regex[m] = r"irecv\s+dec"
            elif m == "enc-wait":
                # wait doesn't have enc/dec, use regular wait time
                metric_to_regex[m] = r"wait\s+time"
            elif m == "enc-waitall":
                # waitall doesn't have enc/dec, use regular waitall time
                metric_to_regex[m] = r"waitall\s+time"
            elif m == "enc-allreduce":
                # allreduce has both enc and dec, we'll sum them
                metric_to_regex[m] = r"allreduce\s+(?:enc|dec)"
            else:
                metric_to_regex[m] = m
        else:
            # Unenc file format: "isend time", "wait all time", etc.
            if m == "waitall":
                metric_to_regex[m] = r"wait\s*all"
            else:
                metric_to_regex[m] = m

    pattern_map = {}
    for m, pattern in metric_to_regex.items():
        # For wait and waitall in enc files, pattern already includes "time"
        if is_enc and (m == "enc-wait" or m == "enc-waitall"):
            pattern_map[m] = re.compile(
                rf"rank is (\d+).*?{pattern}\s+is\s*([0-9.]+)s", re.IGNORECASE
            )
        else:
            pattern_map[m] = re.compile(
                rf"rank is (\d+).*?{pattern}\s*time is\s*([0-9.]+)s", re.IGNORECASE
            )
    data: Dict[int, Dict[str, float]] = {}

    for line in path.read_text().splitlines():
        for metric, pat in pattern_map.items():
            m = pat.search(line)
            if not m:
                continue
            rank = int(m.group(1))
            value = float(m.group(2))
            data.setdefault(rank, {})
            
            # For enc-allreduce, sum both enc and dec times
            if is_enc and metric == "enc-allreduce":
                data[rank].setdefault(metric, 0.0)
                data[rank][metric] += value
            else:
                data[rank].setdefault(metric, value)

    rows = []
    for rank in sorted(data.keys()):
        row = {"rank": rank}
        for metric in metrics:
            row[metric] = data[rank].get(metric, 0.0)
        rows.append(row)

    if not rows:
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=["rank"] + metrics)
    
    return pd.DataFrame(rows)


def plot_combined(ax, parsed_list: List[ParsedData]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Plot all environments into a single stacked chart with shared legend/colors."""
    if not parsed_list:
        ax.set_title("No data")
        return {}, {}

    color_map = {
        "isend": "#4C78A8",
        "irecv": "#9C755F",
        "wait": "#72B7B2",
        "waitall": "#E45756",
        "allreduce": "#59A14F",
    }

    bar_width = 0.8
    gap = 1.5  # gap between environments
    x_positions: List[float] = []
    x_labels: List[str] = []
    env_centers: Dict[str, float] = {}
    env_heights: Dict[str, float] = {}

    max_rank = max((len(p.df) for p in parsed_list if not p.df.empty), default=0)
    current_offset = 0.0

    for parsed in parsed_list:
        df = parsed.df.copy()
        if df.empty or "rank" not in df.columns:
            continue
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

        x_positions.extend((df["rank_label"] + current_offset).tolist())
        x_labels.extend([f"R{r}" for r in df["rank_label"]])

        if not df.empty:
            start = df["rank_label"].min() + current_offset
            end = df["rank_label"].max() + current_offset
            env_centers[parsed.env] = (start + end) / 2
            env_heights[parsed.env] = bottoms.max()

        current_offset += max_rank + gap

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

    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.2)
    for env, center in env_centers.items():
        ax.text(
            center,
            y_max * 1.05,
            env,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    return env_centers, env_heights


def compute_totals(parsed_list: List[ParsedData]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for parsed in parsed_list:
        env_totals: Dict[str, float] = {}
        for metric in (ENC_METRICS if "Enc" in parsed.env else UNENC_METRICS):
            env_totals[metric] = parsed.df[metric].sum() if not parsed.df.empty else 0.0
        totals[parsed.env] = env_totals
    return totals


def annotate_ratios(ax, centers, totals):
    """Annotate enc/unenc ratios for each metric within Native and SGX."""
    pairs = [
        ("Native + Unenc", "Native + Enc"),
        ("SGX + Unenc", "SGX + Enc"),
    ]
    for unenc_env, enc_env in pairs:
        if unenc_env not in centers or enc_env not in centers:
            continue
        center_unenc = centers[unenc_env]
        center_enc = centers[enc_env]
        height_unenc = sum(totals[unenc_env].values())
        height_enc = sum(totals[enc_env].values())
        base_y = max(height_unenc, height_enc) * 1.05 if max(height_unenc, height_enc) > 0 else 0.05
        step = base_y * 0.15 if base_y > 0 else 0.05

        for idx, (m_unenc, m_enc) in enumerate(METRIC_PAIRS):
            u_val = totals[unenc_env].get(m_unenc, 0.0)
            e_val = totals[enc_env].get(m_enc, 0.0)
            if u_val <= 0:
                continue
            ratio = e_val / u_val
            y = base_y + idx * step
            ax.annotate(
                f"{m_unenc}: x{ratio:.2f}",
                xy=(center_enc, y),
                xytext=(center_unenc, y),
                ha="center",
                va="bottom",
                arrowprops=dict(arrowstyle="<->", color="black", shrinkA=5, shrinkB=5, lw=1),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            )


def annotate_native_only(ax, centers, totals):
    """Annotate only Native Enc/Unenc ratios (explicit) on the combined plot."""
    unenc_env = "Native + Unenc"
    enc_env = "Native + Enc"
    if unenc_env not in centers or enc_env not in centers:
        return

    center_unenc = centers[unenc_env]
    center_enc = centers[enc_env]
    height_unenc = sum(totals[unenc_env].values())
    height_enc = sum(totals[enc_env].values())
    base_y = max(height_unenc, height_enc) * 1.05 if max(height_unenc, height_enc) > 0 else 0.05
    step = base_y * 0.15 if base_y > 0 else 0.05

    for idx, (m_unenc, m_enc) in enumerate(METRIC_PAIRS):
        u_val = totals[unenc_env].get(m_unenc, 0.0)
        e_val = totals[enc_env].get(m_enc, 0.0)
        if u_val <= 0:
            continue
        ratio = e_val / u_val
        y = base_y + idx * step
        ax.annotate(
            f"Native {m_unenc}: x{ratio:.2f}",
            xy=(center_enc, y),
            xytext=(center_unenc, y),
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="<->", color="black", shrinkA=5, shrinkB=5, lw=1),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )


def main() -> None:
    parsed_list: List[ParsedData] = []
    for env_name, file_path in FILES.items():
        metrics = ENC_METRICS if "Enc" in env_name else UNENC_METRICS
        df = parse_file(file_path, metrics)
        parsed_list.append(ParsedData(env=env_name, df=df))

    totals = compute_totals(parsed_list)

    fig, ax = plt.subplots(figsize=(14, 6))
    centers, _ = plot_combined(ax, parsed_list)
    annotate_ratios(ax, centers, totals)
    annotate_native_only(ax, centers, totals)
    fig.tight_layout()

    output_path = BASE_DIR / "breakdown-PS=80.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()


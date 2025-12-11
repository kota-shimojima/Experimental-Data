#!/usr/bin/env python3
"""
Plot FOM for four environments in order:
- Native + Unenc
- Native + Enc
- SGX + Unenc
- SGX + Enc
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FILES = {
    "Native + Unenc": BASE_DIR / "one-proccess" / "multi-threads" / "Native-Unenc-PS=80.csv",
    "Native + Enc": BASE_DIR / "one-proccess" / "multi-threads" / "Native-Enc-PS=80.csv",
    "SGX + Unenc": BASE_DIR / "one-proccess" / "multi-threads" / "SGX-Unenc-PS=80.csv",
    "SGX + Enc": BASE_DIR / "one-proccess" / "multi-threads" / "SGX-Enc-PS=80.csv",
}


def extract_fom(path: Path) -> float:
    text = path.read_text()
    m = re.search(r"FOM\s*=\s*([0-9.+-eE]+)", text)
    if not m:
        raise ValueError(f"FOM not found in {path}")
    return float(m.group(1))


def main() -> None:
    foms: Dict[str, float] = {}
    for name, file in FILES.items():
        foms[name] = extract_fom(file)

    labels = list(FILES.keys())
    values = [foms[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F28E2B", "#72B7B2", "#E45756"])

    ax.set_ylabel("FOM (z/s)")
    ax.set_title("FOM Comparison (PS80 Iter50, Multi Threads)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom")

    output_path = BASE_DIR / "fom_comparison-PS=80.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()


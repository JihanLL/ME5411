#!/usr/bin/env python3
"""Reproducible experiments for ME5411 Task 4: binary image conversion.

Run from the project root:
    /opt/miniconda3/envs/cvtask/bin/python task_4/task_4.py

The script converts the Task 3 sub-image to a binary image.  It compares a
global Otsu threshold with a local adaptive mean threshold and selects the
method that keeps the ten characters as separate, noise-free foreground
components.  All thresholds and connected-component analysis are implemented
from first principles in ``common.py``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from common import (  # noqa: E402
    adaptive_threshold_mean,
    binary_open,
    box_filter,
    connected_components,
    otsu_threshold,
)


SUBIMAGE_PATH = PROJECT_ROOT / "task_3" / "outputs" / "task3_subimage.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

SMOOTH_SIZE = 5  # averaging filter applied before thresholding
OPEN_SIZE = 3  # structuring element of the morphological opening
ADAPTIVE_WINDOW = 41  # local window of the adaptive mean threshold
ADAPTIVE_OFFSET = 25.0  # foreground is >25 grey levels above the local mean
MIN_CHARACTER_AREA = 500  # components above this area are character strokes


def component_summary(binary: np.ndarray) -> dict[str, float]:
    """Return component statistics used to judge binarization quality."""
    labels, count = connected_components(binary)
    areas = [int((labels == label).sum()) for label in range(1, count + 1)]
    large = [area for area in areas if area >= MIN_CHARACTER_AREA]
    noise = [area for area in areas if area < MIN_CHARACTER_AREA]
    return {
        "foreground_ratio_percent": 100.0 * float(binary.mean()),
        "n_components": float(count),
        "n_character_components": float(len(large)),
        "largest_noise_area": float(max(noise)) if noise else 0.0,
    }


def save_comparison(
    subimage: np.ndarray, methods: list[tuple[str, np.ndarray]]
) -> None:
    """Side-by-side comparison of the sub-image and the binary candidates."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 3.9), constrained_layout=True)
    axes[0, 0].imshow(subimage, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Task 3 sub-image")
    axes[0, 0].axis("off")
    for axis, (name, binary, _params) in zip(axes.ravel()[1:], methods):
        axis.imshow(binary, cmap="gray", vmin=0, vmax=1)
        axis.set_title(name)
        axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task4_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subimage = np.asarray(
        Image.open(SUBIMAGE_PATH).convert("L"), dtype=np.float64
    )
    smoothed = box_filter(subimage, SMOOTH_SIZE)

    otsu_threshold_value, binary_otsu = otsu_threshold(subimage)
    smoothed_threshold, binary_smoothed = otsu_threshold(smoothed)
    binary_cleaned = binary_open(binary_smoothed, OPEN_SIZE)
    binary_adaptive = adaptive_threshold_mean(
        smoothed, ADAPTIVE_WINDOW, c=-ADAPTIVE_OFFSET
    )

    methods = [
        (
            f"Global Otsu, T={otsu_threshold_value:.0f}",
            binary_otsu,
            {"threshold": otsu_threshold_value},
        ),
        (
            "Otsu after 5x5 smoothing + 3x3 opening",
            binary_cleaned,
            {"threshold": smoothed_threshold},
        ),
        (
            f"Adaptive mean 41x41, +{ADAPTIVE_OFFSET:.0f} (chosen)",
            binary_adaptive,
            {"threshold": "local"},
        ),
    ]

    rows = []
    for name, binary, params in methods:
        row = {"method": name}
        row.update(component_summary(binary))
        row["threshold_or_params"] = params["threshold"]
        rows.append(row)

    with (OUTPUT_DIR / "task4_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    Image.fromarray((binary_otsu * 255).astype(np.uint8)).save(
        OUTPUT_DIR / "task4_binary_otsu.png"
    )
    Image.fromarray((binary_cleaned * 255).astype(np.uint8)).save(
        OUTPUT_DIR / "task4_binary_otsu_cleaned.png"
    )
    Image.fromarray((binary_adaptive * 255).astype(np.uint8)).save(
        OUTPUT_DIR / "task4_binary.png"
    )
    save_comparison(subimage, methods)

    metadata = {
        "subimage": str(SUBIMAGE_PATH.relative_to(PROJECT_ROOT)),
        "smoothing_size": SMOOTH_SIZE,
        "otsu_threshold": otsu_threshold_value,
        "adaptive_window": ADAPTIVE_WINDOW,
        "adaptive_offset": ADAPTIVE_OFFSET,
        "chosen_method": f"adaptive mean threshold, window={ADAPTIVE_WINDOW}, offset=+{ADAPTIVE_OFFSET:.0f}",
        "output": "task4_binary.png",
    }
    with (OUTPUT_DIR / "task4_thresholds.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Input: {SUBIMAGE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Otsu threshold on raw sub-image: {otsu_threshold_value:.1f}")
    print(f"Otsu threshold on 5x5 smoothed sub-image: {smoothed_threshold:.1f}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    for row in rows:
        print(
            f"{row['method']}: foreground={row['foreground_ratio_percent']:.2f}%, "
            f"components={int(row['n_components'])}, "
            f"character components={int(row['n_character_components'])}, "
            f"largest noise area={int(row['largest_noise_area'])}"
        )


if __name__ == "__main__":
    main()

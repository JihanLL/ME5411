#!/usr/bin/env python3
"""Reproducible experiments for ME5411 Task 2: averaging-filter smoothing.

Run from the project root:
    /opt/miniconda3/envs/cvtask/bin/python task_2/task_2.py

The script applies a self-implemented separable box (averaging) filter of five
sizes to the Task 1 enhanced image, compares the smoothing quality visually
and quantitatively, and writes report-ready figures plus a CSV metrics table.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from common import box_filter  # noqa: E402


ENHANCED_PATH = (
    PROJECT_ROOT / "task_1" / "outputs" / "task1_percentile_stretch_p02_p98.png"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FILTER_SIZES = (3, 5, 7, 11, 15)

# Fixed, manually selected patches on bright character strokes and nearby
# character-free background.  The same pixels are used for every filter size.
FOREGROUND_PATCHES = (
    (370, 110, 384, 150),  # T vertical stroke
    (430, 105, 441, 150),  # 1 vertical stroke
    (535, 235, 548, 282),  # 7 vertical stroke
    (690, 235, 704, 286),  # D vertical stroke
    (835, 235, 847, 286),  # A left stroke
)
BACKGROUND_PATCHES = (
    (25, 25, 70, 70),
    (250, 25, 295, 70),
    (720, 25, 765, 70),
    (270, 165, 315, 205),
    (600, 105, 645, 145),
)

# Crop of the bottom text row showing the characters HD4, used to make the
# loss of stroke detail visible at the different filter sizes.
ZOOM_BOX = (52, 194, 330, 345)


def patch_pool(
    image: np.ndarray, patches: tuple[tuple[int, int, int, int], ...]
) -> np.ndarray:
    """Concatenate pixels from (x0, y0, x1, y1) patches."""
    return np.concatenate([image[y0:y1, x0:x1].ravel() for x0, y0, x1, y1 in patches])


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Return the magnitude of the central-difference gradient."""
    gy, gx = np.gradient(image)
    return np.hypot(gx, gy)


def background_std(image: np.ndarray) -> float:
    """Pooled standard deviation of the fixed background patches."""
    return float(patch_pool(image, BACKGROUND_PATCHES).std())


def foreground_gradient(image: np.ndarray) -> float:
    """Mean gradient magnitude over the fixed character-stroke patches."""
    magnitude = gradient_magnitude(image)
    return float(patch_pool(magnitude, FOREGROUND_PATCHES).mean())


def psnr(reference: np.ndarray, filtered: np.ndarray) -> float:
    """Peak signal-to-noise ratio of ``filtered`` relative to ``reference``."""
    squared_error = np.mean((reference - filtered) ** 2)
    if squared_error == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / squared_error))


def save_comparison(
    baseline: np.ndarray, images: dict[str, np.ndarray]
) -> None:
    """Side-by-side comparison of the baseline and all filter sizes."""
    fig, axes = plt.subplots(1, len(images) + 1, figsize=(18, 4.0), constrained_layout=True)
    axes[0].imshow(baseline, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Enhanced (no filtering)")
    axes[0].axis("off")
    for axis, (name, image) in zip(axes[1:], images.items()):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
        axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task2_comparison.png", dpi=220)
    plt.close(fig)


def save_zoom(baseline: np.ndarray, images: dict[str, np.ndarray]) -> None:
    """Zoomed view of the HD4 characters to expose stroke blur."""
    x0, y0, x1, y1 = ZOOM_BOX
    crops = [baseline[y0:y1, x0:x1]] + [image[y0:y1, x0:x1] for image in images.values()]
    names = ["Enhanced (no filtering)"] + list(images.keys())
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), constrained_layout=True)
    for axis, name, crop in zip(axes.ravel(), names, crops):
        axis.imshow(crop, cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
        axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task2_zoom.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = np.asarray(
        Image.open(ENHANCED_PATH).convert("L"), dtype=np.float64
    )

    images: dict[str, np.ndarray] = {}
    filenames: dict[str, str] = {}
    for size in FILTER_SIZES:
        name = f"{size}x{size} box"
        images[name] = box_filter(baseline, size)
        filenames[name] = f"task2_box_{size}x{size}.png"

    rows = [
        {
            "filter": "none (enhanced baseline)",
            "background_std": background_std(baseline),
            "foreground_gradient": foreground_gradient(baseline),
            "psnr_db": float("inf"),
        }
    ]
    for name, image in images.items():
        rows.append(
            {
                "filter": name,
                "background_std": background_std(image),
                "foreground_gradient": foreground_gradient(image),
                "psnr_db": psnr(baseline, image),
            }
        )
    with (OUTPUT_DIR / "task2_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for name, image in images.items():
        Image.fromarray(np.rint(np.clip(image, 0, 255)).astype(np.uint8)).save(
            OUTPUT_DIR / filenames[name]
        )

    save_comparison(baseline, images)
    save_zoom(baseline, images)

    print(f"Input: {ENHANCED_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    for row in rows:
        psnr_text = "inf" if np.isinf(row["psnr_db"]) else f"{row['psnr_db']:.2f}"
        print(
            f"{row['filter']}: background std={row['background_std']:.2f}, "
            f"foreground gradient={row['foreground_gradient']:.3f}, "
            f"PSNR={psnr_text} dB"
        )


if __name__ == "__main__":
    main()

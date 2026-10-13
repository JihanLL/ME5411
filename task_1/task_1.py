#!/usr/bin/env python3
"""Reproducible experiments for ME5411 Task 1.

Run from the project root:
    conda run -n cvtask python task_1/task_1.py

The script compares exactly three contrast-enhancement methods on the same
grayscale input and writes report-ready figures plus a CSV metrics table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "dataset_2026" / "Test_Image_2026.bmp"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Fixed, manually selected patches on bright character strokes and nearby
# character-free background.  The same pixels are used for every method.
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


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 data to luminance using the BT.601 coefficients."""
    rgb_float = rgb.astype(np.float64)
    return 0.299 * rgb_float[..., 0] + 0.587 * rgb_float[..., 1] + 0.114 * rgb_float[..., 2]


def percentile_stretch(gray: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Robust linear contrast stretch, clipping only the outer percentiles."""
    lower, upper = np.percentile(gray, (low, high))
    stretched = 255.0 * (gray - lower) / (upper - lower)
    return np.clip(stretched, 0, 255)


def gamma_correction(gray: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply a power-law transform after normalising intensities to [0, 1]."""
    return 255.0 * np.power(gray / 255.0, gamma)


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Implement 256-level global histogram equalisation with NumPy."""
    values = np.rint(np.clip(gray, 0, 255)).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256)
    cdf = histogram.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0 or cdf[-1] == nonzero[0]:
        return values.astype(np.float64)
    lookup = 255.0 * (cdf - nonzero[0]) / (cdf[-1] - nonzero[0])
    return np.clip(lookup[values], 0, 255)


def patch_values(image: np.ndarray, patches: tuple[tuple[int, int, int, int], ...]) -> np.ndarray:
    """Concatenate pixels from (x0, y0, x1, y1) patches."""
    return np.concatenate([image[y0:y1, x0:x1].ravel() for x0, y0, x1, y1 in patches])


def cnr(image: np.ndarray) -> float:
    """Character-to-background contrast-to-noise ratio over fixed patches."""
    foreground = patch_values(image, FOREGROUND_PATCHES)
    background = patch_values(image, BACKGROUND_PATCHES)
    denominator = np.sqrt(foreground.var() + background.var())
    return float(abs(foreground.mean() - background.mean()) / denominator) if denominator else 0.0


def otsu_separability(image: np.ndarray) -> float:
    """Return Otsu's normalised between-class variance, in [0, 1]."""
    values = np.rint(np.clip(image, 0, 255)).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    levels = np.arange(256, dtype=np.float64)
    cumulative_weight = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * levels)
    total_mean = cumulative_mean[-1]
    valid = (cumulative_weight > 0) & (cumulative_weight < 1)
    between = np.zeros(256, dtype=np.float64)
    between[valid] = (
        (total_mean * cumulative_weight[valid] - cumulative_mean[valid]) ** 2
        / (cumulative_weight[valid] * (1.0 - cumulative_weight[valid]))
    )
    total_variance = np.sum(probabilities * (levels - total_mean) ** 2)
    return float(between.max() / total_variance) if total_variance else 0.0


def saturation_ratio(image: np.ndarray) -> float:
    """Percentage of pixels clipped to either endpoint after enhancement."""
    return float(100.0 * np.mean((image <= 0.5) | (image >= 254.5)))


def save_comparison(rgb: np.ndarray, images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(images) + 1, figsize=(18, 4.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")
    for axis, (name, image) in zip(axes[1:], images.items()):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
        axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task1_comparison.png", dpi=220)
    plt.close(fig)


def save_histograms(images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    for axis, (name, image) in zip(axes.ravel(), images.items()):
        axis.hist(image.ravel(), bins=256, range=(0, 255), color="#2c7fb8")
        axis.set(title=name, xlabel="Grayscale intensity", ylabel="Pixel count", xlim=(0, 255))
    fig.savefig(OUTPUT_DIR / "task1_histograms.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(INPUT_PATH) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    gray = rgb_to_gray(rgb)

    images = {
        "Grayscale baseline": gray,
        "2--98% percentile stretch": percentile_stretch(gray),
        "Gamma correction (gamma=1.5)": gamma_correction(gray),
        "Global histogram equalization": histogram_equalization(gray),
    }
    filenames = {
        "Grayscale baseline": "task1_gray.png",
        "2--98% percentile stretch": "task1_percentile_stretch_p02_p98.png",
        "Gamma correction (gamma=1.5)": "task1_gamma_150.png",
        "Global histogram equalization": "task1_histogram_equalization.png",
    }

    Image.fromarray(rgb).save(OUTPUT_DIR / "task1_original_rgb.png")
    for name, image in images.items():
        Image.fromarray(np.rint(np.clip(image, 0, 255)).astype(np.uint8)).save(OUTPUT_DIR / filenames[name])

    rows = []
    for name, image in images.items():
        rows.append(
            {
                "method": name,
                "cnr": cnr(image),
                "otsu_separability": otsu_separability(image),
                "saturation_ratio_percent": saturation_ratio(image),
            }
        )
    with (OUTPUT_DIR / "task1_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    save_comparison(rgb, images)
    save_histograms(images)
    print(f"Input: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    for row in rows:
        print(
            f"{row['method']}: CNR={row['cnr']:.4f}, "
            f"Otsu separability={row['otsu_separability']:.4f}, "
            f"saturation={row['saturation_ratio_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproducible experiment for ME5411 Task 3: sub-image extraction.

Run from the project root:
    /opt/miniconda3/envs/cvtask/bin/python task_3/task_3.py

The script locates the text row that contains ``HD44780A00`` with projection
profiles computed on the Task 1 enhanced image, extracts a sub-image that
contains the complete row, and writes report-ready figures plus JSON metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from common import box_filter, otsu_threshold  # noqa: E402


INPUT_PATH = PROJECT_ROOT / "dataset_2026" / "Test_Image_2026.bmp"
ENHANCED_PATH = PROJECT_ROOT / "task_1" / "outputs" / "task1_percentile_stretch_p02_p98.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

MARGIN = 10  # pixels kept around the detected text row
MIN_ROW_COUNT = 25  # coarse stage: foreground pixels per row
MIN_COL_COUNT = 5  # coarse stage: foreground pixels per column
GAP_MERGE = 12  # coarse stage: merge column spans up to this gap
REFINE_PERCENTILE = 80  # refinement threshold as a percentile of the band
REFINE_MIN_ROW = 5  # refinement stage: foreground pixels per row
REFINE_MIN_COL = 3  # refinement stage: foreground pixels per column
REFINE_COL_GAP = 30  # refinement stage: merge across inter-character gaps


def contiguous_spans(
    counts: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """Return (start, stop) spans of indices whose count exceeds threshold."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(counts):
        if counts[index] > threshold:
            start = index
            while index < len(counts) and counts[index] > threshold:
                index += 1
            spans.append((start, index))
        else:
            index += 1
    return spans


def merge_spans(spans: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Merge spans separated by at most ``gap`` indices."""
    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def coarse_band(
    binary: np.ndarray, height: int, start_below: int
) -> tuple[int, int, int, int]:
    """Locate the character row below ``start_below`` with the coarse mask.

    Returns ``(x0, y0, x1, y1)`` using half-open coordinates.
    """
    # Row projection: candidate bands with enough foreground pixels.
    row_counts = binary.sum(axis=1)
    row_bands = merge_spans(
        contiguous_spans(row_counts, MIN_ROW_COUNT), gap=8
    )
    lower_bands = [band for band in row_bands if band[0] >= start_below]
    if not lower_bands:
        raise RuntimeError("no text band detected below the target row")
    # The required row is the last (lowest) band, i.e. HD44780A00.
    y0, y1 = max(lower_bands, key=lambda band: band[1] - band[0])

    # Column projection inside the band, keep the widest merged span.
    col_counts = binary[y0:y1].sum(axis=0)
    col_spans = merge_spans(
        contiguous_spans(col_counts, MIN_COL_COUNT), gap=GAP_MERGE
    )
    if not col_spans:
        raise RuntimeError("no column span detected inside the text band")
    x0, x1 = max(col_spans, key=lambda span: span[1] - span[0])
    return x0, y0, x1, y1


def refine_box(
    smoothed: np.ndarray, coarse: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Tighten the coarse box with a band-adaptive threshold.

    The coarse band contains both bright character strokes and textured
    background.  Its 80th percentile is above the texture but below the stroke
    cores, so thresholding the band at that level keeps only the characters.
    """
    cx0, cy0, cx1, cy1 = coarse
    band = smoothed[cy0:cy1, cx0:cx1]
    threshold = float(np.percentile(band, REFINE_PERCENTILE))
    binary = band > threshold

    # Row projection: keep contiguous bands of sufficient height so isolated
    # speckle rows above and below the characters are discarded.
    row_counts = binary.sum(axis=1)
    row_bands = merge_spans(
        contiguous_spans(row_counts, REFINE_MIN_ROW), gap=8
    )
    tall_bands = [band_ for band_ in row_bands if band_[1] - band_[0] > 40]
    if not tall_bands:
        raise RuntimeError("refinement stage found no tall row band")
    y0, y1 = max(tall_bands, key=lambda band_: band_[1] - band_[0])

    # Column projection inside the refined rows.  The merge gap must bridge
    # the spaces between characters, which reach about 20 pixels in this row.
    col_counts = binary[y0:y1].sum(axis=0)
    col_spans = merge_spans(
        contiguous_spans(col_counts, REFINE_MIN_COL), gap=REFINE_COL_GAP
    )
    if not col_spans:
        raise RuntimeError("refinement stage found no column span")
    x0, x1 = max(col_spans, key=lambda span: span[1] - span[0])
    return cx0 + x0, cy0 + y0, cx0 + x1, cy0 + y1


def apply_margin(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Expand a box by ``margin`` pixels and clamp it to the image bounds."""
    height, width = shape
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def save_extraction_region(enhanced: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """Draw the extraction rectangle over the enhanced image."""
    x0, y0, x1, y1 = box
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    rectangle = Rectangle(
        (x0 - 0.5, y0 - 0.5),
        x1 - x0,
        y1 - y0,
        linewidth=2,
        edgecolor="#d62728",
        facecolor="none",
    )
    axis.add_patch(rectangle)
    axis.set_title("Detected HD44780A00 row (red rectangle)")
    axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task3_extraction_region.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enhanced = np.asarray(Image.open(ENHANCED_PATH).convert("L"), dtype=np.uint8)
    original = np.asarray(Image.open(INPUT_PATH).convert("RGB"), dtype=np.uint8)

    # A light 5x5 box filter suppresses isolated speckles so the projection
    # profiles are dominated by the character strokes instead of noise.
    smoothed = box_filter(enhanced.astype(np.float64), 5)
    threshold, binary = otsu_threshold(smoothed)

    coarse = coarse_band(binary, enhanced.shape[0], start_below=185)
    detected = refine_box(smoothed, coarse)
    box = apply_margin(detected, MARGIN, enhanced.shape)
    x0, y0, x1, y1 = box

    # Extract and save the sub-images.
    subimage = enhanced[y0:y1, x0:x1]
    subimage_rgb = original[y0:y1, x0:x1]
    Image.fromarray(subimage).save(OUTPUT_DIR / "task3_subimage.png")
    Image.fromarray(subimage_rgb).save(OUTPUT_DIR / "task3_subimage_rgb.png")
    save_extraction_region(enhanced, box)

    metadata = {
        "input_image": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "enhanced_image": str(ENHANCED_PATH.relative_to(PROJECT_ROOT)),
        "image_size_px": list(enhanced.shape[::-1]),
        "otsu_threshold": threshold,
        "coarse_box": list(coarse),
        "detected_row_box": list(detected),
        "margin_px": MARGIN,
        "extraction_box": list(box),
        "subimage_size_px": list(subimage.shape[::-1]),
        "target_string": "HD44780A00",
    }
    with (OUTPUT_DIR / "task3_extraction.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Otsu threshold on smoothed image: {threshold:.1f}")
    print(f"Coarse box: {coarse}")
    print(f"Detected row box: {detected}")
    print(f"Extraction box (with {MARGIN} px margin): {box}")
    print(f"Sub-image size: {subimage.shape[1]} x {subimage.shape[0]} pixels")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

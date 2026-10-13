#!/usr/bin/env python3
"""Reproducible experiment for ME5411 Task 6: segmentation and labelling.

Run from the project root:
    /opt/miniconda3/envs/cvtask/bin/python task_6/task_6.py

The script labels the connected components of the Task 4 binary image with a
self-implemented two-pass union-find, removes the small noise components with
an area criterion, orders the ten surviving characters left to right, draws
the labels and bounding boxes, and saves one crop per character for Task 7.
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
from common import connected_components  # noqa: E402


BINARY_PATH = PROJECT_ROOT / "task_4" / "outputs" / "task4_binary.png"
OTSU_PATH = PROJECT_ROOT / "task_4" / "outputs" / "task4_binary_otsu.png"
GRAY_PATH = PROJECT_ROOT / "task_3" / "outputs" / "task3_subimage.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHARS_DIR = OUTPUT_DIR / "chars"

MIN_CHARACTER_AREA = 300  # noise components are at most 224 px
CROP_PADDING = 8  # context pixels kept around every character crop
GROUND_TRUTH = "HD44780A00"
COLORS = matplotlib.colormaps["tab10"]


def label_characters(
    binary: np.ndarray, min_area: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return a label image and one record per character, ordered left to
    right."""
    labels, count = connected_components(binary)
    records: list[dict[str, object]] = []
    for label in range(1, count + 1):
        mask = labels == label
        area = int(mask.sum())
        if area < min_area:
            labels[mask] = 0
            continue
        pixels = np.nonzero(mask)
        y0, y1 = int(pixels[0].min()), int(pixels[0].max())
        x0, x1 = int(pixels[1].min()), int(pixels[1].max())
        records.append(
            {
                "label": label,
                "bbox": [x0, y0, x1, y1],
                "area": area,
                "centroid": [
                    float(pixels[1].mean()),
                    float(pixels[0].mean()),
                ],
            }
        )
    records.sort(key=lambda record: record["bbox"][0])
    for order, record in enumerate(records, start=1):
        record["order"] = order
        record["expected"] = GROUND_TRUTH[order - 1]
    return labels, records


def plot_labelled(
    gray: np.ndarray,
    records: list[dict[str, object]],
    filename: str,
    show_boxes: bool,
) -> None:
    """Draw coloured labels (and optionally bounding boxes) on the image."""
    fig, axis = plt.subplots(figsize=(13, 3.4), constrained_layout=True)
    axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for record in records:
        x0, y0, x1, y1 = record["bbox"]
        colour = COLORS(record["order"] - 1)
        axis.text(
            (x0 + x1) / 2,
            y0 - 6,
            str(record["order"]),
            color=colour,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        if show_boxes:
            axis.add_patch(
                plt.Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0 + 1,
                    y1 - y0 + 1,
                    fill=False,
                    edgecolor=colour,
                    lw=1.4,
                )
            )
    axis.set_xlim(0, gray.shape[1])
    axis.set_ylim(gray.shape[0], 0)
    axis.axis("off")
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


def save_crops(
    gray: np.ndarray, binary: np.ndarray, records: list[dict[str, object]]
) -> None:
    """Save one padded grayscale and binary crop per character."""
    CHARS_DIR.mkdir(parents=True, exist_ok=True)
    height, width = binary.shape
    for record in records:
        x0, y0, x1, y1 = record["bbox"]
        cx0, cy0 = max(0, x0 - CROP_PADDING), max(0, y0 - CROP_PADDING)
        cx1, cy1 = min(width, x1 + 1 + CROP_PADDING), min(height, y1 + 1 + CROP_PADDING)
        name = f"char_{record['order']:02d}"
        record["crop_gray"] = f"chars/{name}_gray.png"
        record["crop_binary"] = f"chars/{name}_bin.png"
        Image.fromarray(gray[cy0:cy1, cx0:cx1]).save(CHARS_DIR / f"{name}_gray.png")
        Image.fromarray((binary[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)).save(
            CHARS_DIR / f"{name}_bin.png"
        )


def plot_crop_grid(
    gray: np.ndarray, records: list[dict[str, object]]
) -> None:
    """Arrange the ten character crops in a two-row grid."""
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6), constrained_layout=True)
    for axis, record in zip(axes.ravel(), records):
        x0, y0, x1, y1 = record["bbox"]
        cx0, cy0 = max(0, x0 - CROP_PADDING), max(0, y0 - CROP_PADDING)
        cx1, cy1 = min(gray.shape[1], x1 + 1 + CROP_PADDING), min(gray.shape[0], y1 + 1 + CROP_PADDING)
        axis.imshow(gray[cy0:cy1, cx0:cx1], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"Label {record['order']}")
        axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task6_crops.png", dpi=220)
    plt.close(fig)


def plot_segmentation_comparison(
    binary: np.ndarray,
    otsu_binary: np.ndarray,
    adaptive_records: list[dict[str, object]],
    otsu_records: list[dict[str, object]],
) -> None:
    """Compare character counts obtained on the two Task 4 binaries."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), constrained_layout=True)
    for axis, image, records, title in (
        (axes[0], otsu_binary, otsu_records, "Global Otsu binary"),
        (axes[1], binary, adaptive_records, "Adaptive binary (selected)"),
    ):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        for record in records:
            x0, y0, x1, y1 = record["bbox"]
            colour = COLORS(record["order"] - 1)
            axis.add_patch(
                plt.Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0 + 1,
                    y1 - y0 + 1,
                    fill=False,
                    edgecolor=colour,
                    lw=1.4,
                )
            )
        axis.set_title(f"{title}: {len(records)} groups")
        axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task6_segmentation_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(Image.open(BINARY_PATH).convert("L")) > 0
    otsu_binary = np.asarray(Image.open(OTSU_PATH).convert("L")) > 0
    gray = np.asarray(Image.open(GRAY_PATH).convert("L"), dtype=np.uint8)

    labels, records = label_characters(binary, MIN_CHARACTER_AREA)
    _, otsu_records = label_characters(otsu_binary, MIN_CHARACTER_AREA)
    if len(records) != 10:
        raise RuntimeError(f"expected 10 characters, found {len(records)}")

    cleaned = (labels > 0).astype(np.uint8) * 255
    Image.fromarray(cleaned).save(OUTPUT_DIR / "task6_cleaned_binary.png")

    plot_labelled(gray, records, "task6_labeled.png", show_boxes=False)
    plot_labelled(gray, records, "task6_boxes.png", show_boxes=True)
    save_crops(gray, binary, records)
    plot_crop_grid(gray, records)
    plot_segmentation_comparison(binary, otsu_binary, records, otsu_records)

    # Inter-character gap between adjacent bounding boxes.
    gaps: list[int] = []
    for previous, current in zip(records, records[1:]):
        gaps.append(int(current["bbox"][0] - previous["bbox"][2] - 1))

    rows = []
    for record in records:
        x0, y0, x1, y1 = record["bbox"]
        rows.append(
            {
                "order": record["order"],
                "expected": record["expected"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "width": x1 - x0 + 1,
                "height": y1 - y0 + 1,
                "area": record["area"],
                "centroid_x": round(record["centroid"][0], 1),
                "centroid_y": round(record["centroid"][1], 1),
                "crop_gray": record["crop_gray"],
                "crop_binary": record["crop_binary"],
            }
        )
    with (OUTPUT_DIR / "task6_components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / "task6_components.json").open("w") as handle:
        json.dump(
            {
                "ground_truth": GROUND_TRUTH,
                "min_character_area": MIN_CHARACTER_AREA,
                "n_characters": len(records),
                "inter_character_gaps_px": gaps,
                "otsu_binary_character_groups": len(otsu_records),
                "characters": rows,
            },
            handle,
            indent=2,
        )

    print(f"Input binary: {BINARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Characters segmented: {len(records)}, ground truth: {GROUND_TRUTH}")
    print(f"Inter-character gaps (px): {gaps}")
    print(f"Character groups on the Otsu binary: {len(otsu_records)}")
    for row in rows:
        print(
            f"label {row['order']} ({row['expected']}): "
            f"bbox=({row['x0']},{row['y0']})-({row['x1']},{row['y1']}), "
            f"area={row['area']}"
        )


if __name__ == "__main__":
    main()

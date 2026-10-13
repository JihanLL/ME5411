#!/usr/bin/env python3
"""Reproducible experiment for ME5411 Task 5: character outline detection.

Run from the project root:
    python task_5/task_5.py

The script loads the binary image selected in Task 4, keeps the ten
character-sized connected components, and traces the outer boundary of every
character together with the boundaries of its holes using a self-implemented
Moore-neighbor contour follower.  Report-ready figures and outline metadata
are written to ``task_5/outputs/``.
"""

from __future__ import annotations

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
    component_outlines,
    connected_components,
    freeman_chain_code,
)


BINARY_PATH = PROJECT_ROOT / "task_4" / "outputs" / "task4_binary.png"
GRAY_PATH = PROJECT_ROOT / "task_3" / "outputs" / "task3_subimage.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

MIN_CHARACTER_AREA = 300  # every Task 4 noise component is below 225 px
CHARACTER_COLORS = matplotlib.colormaps["tab10"]


def character_components(
    binary: np.ndarray, min_area: int
) -> list[tuple[int, np.ndarray]]:
    """Return ``(label, mask)`` for the character-sized components, ordered
    left to right by their minimum x coordinate."""
    labels, count = connected_components(binary)
    characters: list[tuple[int, np.ndarray]] = []
    for label in range(1, count + 1):
        mask = labels == label
        if int(mask.sum()) >= min_area:
            characters.append((label, mask))
    characters.sort(key=lambda item: int(np.nonzero(item[1])[1].min()))
    return characters


def validate_outline_coordinates(
    mask: np.ndarray,
    outer: list[tuple[int, int]],
    holes: list[list[tuple[int, int]]],
) -> None:
    """Check that traces use the input sub-image's coordinate system.

    Outer pixels must belong to this character, while hole traces must lie
    on background inside its bounding box.  These checks catch a missing
    crop-origin offset before an incorrect overlay or JSON file is written.
    """
    rows, cols = np.nonzero(mask)
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    for is_outer, trace in [(True, outer), *[(False, hole) for hole in holes]]:
        if not trace:
            raise RuntimeError("contour follower returned an empty trace")
        for row, col in trace:
            if not (y0 <= row <= y1 and x0 <= col <= x1):
                raise RuntimeError("outline coordinates are outside the character bbox")
            if bool(mask[row, col]) != is_outer:
                raise RuntimeError("outline coordinates do not match the character mask")
        # Validate every step, including the non-degenerate closed contour.
        chain = freeman_chain_code(trace)
        closes_explicitly = len(trace) > 1 and trace[-1] == trace[0]
        expected_steps = len(trace) - 1 if closes_explicitly else len(trace)
        if len(trace) > 1 and len(chain) != expected_steps:
            raise RuntimeError("contour does not close with an 8-connected step")


def plot_overlays(
    gray: np.ndarray,
    characters: list[tuple[int, np.ndarray]],
    outlines: list[dict[str, object]],
) -> None:
    """Overlay the traced outlines on the grayscale sub-image."""
    fig, axis = plt.subplots(figsize=(13, 3.2), constrained_layout=True)
    axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for index, (_, _mask) in enumerate(characters):
        colour = CHARACTER_COLORS(index)
        outer = outlines[index]["outer"]
        pixels = np.asarray(outer)
        axis.plot(pixels[:, 1], pixels[:, 0], color=colour, lw=1.2)
        for hole in outlines[index]["holes"]:
            hole_pixels = np.asarray(hole)
            axis.plot(
                hole_pixels[:, 1],
                hole_pixels[:, 0],
                color=colour,
                lw=1.0,
                ls=(0, (2, 2)),
            )
    axis.set_title("Traced character outlines (solid: outer, dashed: holes)")
    axis.set_xlim(0, gray.shape[1])
    axis.set_ylim(gray.shape[0], 0)
    axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task5_outlines.png", dpi=220)
    plt.close(fig)


def plot_outline_mask(shape: tuple[int, int], outlines: list[dict[str, object]]) -> None:
    """Draw only the traced boundary pixels as white on black."""
    mask = np.zeros(shape, dtype=np.uint8)
    for outline in outlines:
        for trace in [outline["outer"], *outline["holes"]]:
            for row, col in trace:
                mask[row, col] = 255
    Image.fromarray(mask).save(OUTPUT_DIR / "task5_outline_mask.png")


def plot_trace_example(
    gray: np.ndarray, mask: np.ndarray, outline: dict[str, object]
) -> None:
    """Enlarged view of one character with the trace start and direction."""
    pixels = np.nonzero(mask)
    y0, y1 = int(pixels[0].min()), int(pixels[0].max())
    x0, x1 = int(pixels[1].min()), int(pixels[1].max())
    crop = gray[y0 : y1 + 1, x0 : x1 + 1]

    outer = np.asarray(outline["outer"]) - [y0, x0]
    holes = [np.asarray(hole) - [y0, x0] for hole in outline["holes"]]

    fig, axis = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    axis.imshow(crop, cmap="gray", vmin=0, vmax=255)
    axis.plot(outer[:, 1], outer[:, 0], color="#d62728", lw=1.6)
    for hole in holes:
        axis.plot(hole[:, 1], hole[:, 0], color="#1f77b4", lw=1.4, ls="--")
    axis.plot(outer[0, 1], outer[0, 0], "go", ms=7, mfc="none")
    arrow_end = outer[min(12, len(outer) - 1)]
    axis.annotate(
        "",
        xy=(arrow_end[1], arrow_end[0]),
        xytext=(outer[0, 1], outer[0, 0]),
        arrowprops={"arrowstyle": "->", "color": "lime", "lw": 2},
    )
    axis.set_title("Moore trace (start: circle)")
    axis.axis("off")
    fig.savefig(OUTPUT_DIR / "task5_example_trace.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(Image.open(BINARY_PATH).convert("L")) > 0
    gray = np.asarray(Image.open(GRAY_PATH).convert("L"), dtype=np.uint8)

    characters = character_components(binary, MIN_CHARACTER_AREA)
    if len(characters) != 10:
        raise RuntimeError(
            f"expected 10 character components, found {len(characters)}"
        )

    records: list[dict[str, object]] = []
    outlines: list[dict[str, object]] = []
    for order, (label, mask) in enumerate(characters, start=1):
        pixels = np.nonzero(mask)
        y0, y1 = int(pixels[0].min()), int(pixels[0].max())
        x0, x1 = int(pixels[1].min()), int(pixels[1].max())
        pad = 1  # so the virtual western neighbour of the start pixel exists
        padded = np.pad(mask[y0 : y1 + 1, x0 : x1 + 1], pad)
        traced = component_outlines(padded)

        # The follower uses padded crop coordinates.  Restore both the pad
        # offset and the crop origin before drawing or saving the traces.
        outer = [(row - pad + y0, col - pad + x0) for row, col in traced["outer"]]
        holes = [
            [(row - pad + y0, col - pad + x0) for row, col in hole]
            for hole in traced["holes"]
        ]
        validate_outline_coordinates(mask, outer, holes)
        outlines.append({"outer": outer, "holes": holes})

        chain = freeman_chain_code(outer)
        hole_chains = [freeman_chain_code(hole) for hole in holes]
        records.append(
            {
                "order": order,
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "bbox_full_image": [int(x0 + 52), int(y0 + 194), int(x1 + 52), int(y1 + 194)],
                "area": int(mask.sum()),
                "outer_boundary_length": len(outer),
                "n_holes": int(traced["n_holes"]),
                "hole_boundary_lengths": [len(hole) for hole in holes],
                "coordinate_reference": "task3_subimage",
                "coordinate_order": ["row", "col"],
                "outer_trace": outer,
                "hole_traces": holes,
                "chain_code_start": outer[0],
                "chain_code": chain,
                "hole_chain_code_starts": [hole[0] for hole in holes],
                "hole_chain_codes": hole_chains,
                "chain_code_prefix": chain[:24],
                "chain_code_length": len(chain),
            }
        )

    plot_overlays(gray, characters, outlines)
    plot_outline_mask(gray.shape, outlines)
    # Enlarged example: the second '4' (order 4) has a diagonal stroke and a
    # clearly enclosed triangle, which makes the trace easy to follow.
    plot_trace_example(gray, characters[3][1], outlines[3])

    with (OUTPUT_DIR / "task5_outlines.json").open("w") as handle:
        json.dump(records, handle, indent=2)

    print(f"Input binary: {BINARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    total = 0
    for record in records:
        total += record["outer_boundary_length"]
        holes = ", ".join(
            str(length) for length in record["hole_boundary_lengths"]
        ) or "none"
        print(
            f"character {record['order']}: outer length="
            f"{record['outer_boundary_length']}, holes=[{holes}]"
        )
    print(f"Total outer boundary length of the row: {total} pixels")


if __name__ == "__main__":
    main()

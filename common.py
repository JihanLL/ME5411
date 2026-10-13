"""Shared image-processing helpers for the ME5411 group project.

The course brief strongly encourages self-implemented algorithms instead of
calling ready-made packages.  Every routine in this module is implemented from
first principles with NumPy only:

  * ``rgb_to_gray``                 -- BT.601 luminance conversion
  * ``percentile_stretch``          -- robust linear contrast stretch
  * ``box_filter``                  -- separable box (averaging) filter
  * ``otsu_threshold``              -- Otsu's global thresholding method
  * ``adaptive_threshold_mean``     -- local mean-based adaptive thresholding
  * ``binary_dilate`` / ``binary_erode`` / ``binary_open`` / ``binary_close``
  * ``connected_components``        -- two-pass labelling with union-find
  * ``trace_outer_boundary``        -- Moore-neighbor contour following
  * ``component_outlines``          -- outer and hole outlines of a component
  * ``freeman_chain_code``          -- 8-direction chain code of a trace
  * ``integral_image``              -- summed-area table used by the filters

Tasks import this module from the project root, e.g.

    from common import box_filter, otsu_threshold
"""

from __future__ import annotations

import numpy as np


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8/float array to BT.601 luminance."""
    rgb_float = rgb.astype(np.float64)
    return (
        0.299 * rgb_float[..., 0]
        + 0.587 * rgb_float[..., 1]
        + 0.114 * rgb_float[..., 2]
    )


def percentile_stretch(
    gray: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Map the low/high percentiles of ``gray`` to 0/255 and clip the rest."""
    lower, upper = np.percentile(gray, (low, high))
    stretched = 255.0 * (gray - lower) / (upper - lower)
    return np.clip(stretched, 0.0, 255.0)


def integral_image(image: np.ndarray) -> np.ndarray:
    """Return a summed-area table ``S`` with ``S[0, :] == S[:, 0] == 0``.

    The sum of ``image[i0:i1, j0:j1]`` is
    ``S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]``.
    """
    table = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return table


def box_filter(gray: np.ndarray, size: int) -> np.ndarray:
    """Apply a separable ``size x size`` box (averaging) filter.

    Boundary pixels are handled by replicating the edge values.  The kernel is
    separable, so the filter is computed in O(hw) time with a summed-area
    table instead of the O(hw * size^2) sliding-window implementation.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("filter size must be a positive odd integer")
    radius = size // 2
    padded = np.pad(gray, radius, mode="edge")
    table = integral_image(padded)
    sums = (
        table[size:, size:]
        - table[:-size, size:]
        - table[size:, :-size]
        + table[:-size, :-size]
    )
    return sums / float(size * size)


def otsu_threshold(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """Return Otsu's threshold and the resulting binary mask.

    The threshold maximises the between-class variance of the 256-bin
    intensity histogram.  Foreground pixels are ``True``.
    """
    values = np.rint(np.clip(gray, 0, 255)).astype(np.uint8)
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
    threshold = float(np.argmax(between))
    return threshold, gray > threshold


def local_mean(gray: np.ndarray, window: int) -> np.ndarray:
    """Compute the local mean over a ``window x window`` neighbourhood."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(gray, radius, mode="edge")
    table = integral_image(padded)
    sums = (
        table[window:, window:]
        - table[:-window, window:]
        - table[window:, :-window]
        + table[:-window, :-window]
    )
    return sums / float(window * window)


def adaptive_threshold_mean(
    gray: np.ndarray, window: int, c: float = 10.0
) -> np.ndarray:
    """Adaptive threshold ``T(x, y) = mean_{window}(x, y) - c``.

    A pixel is foreground when it is brighter than its local mean minus the
    offset ``c``.  Positive ``c`` is the conventional setting for bright
    strokes on a dark background; a negative ``c`` marks only pixels that are
    substantially brighter than the local mean.
    """
    mean = local_mean(gray, window)
    return gray > (mean - c)


def binary_dilate(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological dilation with a ``size x size`` square structuring
    element, implemented as the union of shifted copies."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    radius = size // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    result = np.zeros_like(mask)
    for dy in range(size):
        for dx in range(size):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def binary_erode(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological erosion with a ``size x size`` square structuring
    element, implemented as the intersection of shifted copies."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    radius = size // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    result = np.ones_like(mask)
    for dy in range(size):
        for dx in range(size):
            result &= padded[dy : dy + height, dx : dx + width]
    return result


def binary_open(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological opening (erosion followed by dilation)."""
    return binary_dilate(binary_erode(mask, size), size)


def binary_close(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological closing (dilation followed by erosion)."""
    return binary_erode(binary_dilate(mask, size), size)


def connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected foreground components with a two-pass union-find.

    Returns ``(labels, count)`` where labels of background pixels are 0 and
    the foreground components are numbered 1..count.
    """
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int64)
    parent = [0]

    def find(label: int) -> int:
        root = label
        while parent[root] != root:
            root = parent[root]
        while parent[label] != root:  # path compression
            parent[label], label = root, parent[label]
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    next_label = 1
    for y in range(height):
        row = binary[y]
        for x in range(width):
            if not row[x]:
                continue
            neighbours = []
            if x > 0 and labels[y, x - 1]:
                neighbours.append(labels[y, x - 1])
            if y > 0:
                if labels[y - 1, x]:
                    neighbours.append(labels[y - 1, x])
                if x > 0 and labels[y - 1, x - 1]:
                    neighbours.append(labels[y - 1, x - 1])
                if x + 1 < width and labels[y - 1, x + 1]:
                    neighbours.append(labels[y - 1, x + 1])
            if not neighbours:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                smallest = min(neighbours)
                labels[y, x] = smallest
                for neighbour in neighbours:
                    union(smallest, neighbour)

    remap = {}
    count = 0
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label:
                root = find(label)
                if root not in remap:
                    count += 1
                    remap[root] = count
                labels[y, x] = remap[root]
    return labels, count


# Clockwise 8-neighbour offsets starting from the west neighbour.
_NEIGHBOURS = (
    (-1, 0),   # 0: W
    (-1, -1),  # 1: NW
    (0, -1),   # 2: N
    (1, -1),   # 3: NE
    (1, 0),    # 4: E
    (1, 1),    # 5: SE
    (0, 1),    # 6: S
    (-1, 1),   # 7: SW
)

# Freeman chain-code direction for every pixel offset.
_DIRECTION = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, -1): 3,
    (-1, 0): 4,
    (-1, 1): 5,
    (0, 1): 6,
    (1, 1): 7,
}


def trace_outer_boundary(mask: np.ndarray) -> list[tuple[int, int]]:
    """Trace the outer boundary of one component with Moore-neighbor following.

    ``mask`` must contain exactly one 8-connected foreground component.  The
    algorithm starts at the top-leftmost foreground pixel and walks clockwise
    around the component, stopping with Jacob's criterion (the start pixel is
    re-entered from the same direction).  The returned list contains the
    ordered ``(row, col)`` coordinates of the boundary pixels.
    """
    foreground = np.nonzero(mask)
    if foreground[0].size == 0:
        return []
    rows, cols = foreground
    start_row = int(rows.min())
    start_col = int(cols[rows == start_row].min())
    start = (start_row, start_col)

    height, width = mask.shape
    boundary: list[tuple[int, int]] = []
    current = start
    back = (start_row, start_col - 1)  # virtual western neighbour
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while (current, back) not in seen_states:
        seen_states.add((current, back))
        boundary.append(current)

        # Locate the backtrace pixel in the clockwise neighbour table so the
        # search can resume just after it.
        offset = (back[1] - current[1], back[0] - current[0])
        try:
            back_index = _NEIGHBOURS.index(offset)
        except ValueError:
            back_index = 0

        found: tuple[int, int] | None = None
        for step in range(1, 9):
            dx, dy = _NEIGHBOURS[(back_index + step) % 8]
            neighbour = (current[0] + dy, current[1] + dx)
            if (
                0 <= neighbour[0] < height
                and 0 <= neighbour[1] < width
                and mask[neighbour]
            ):
                found = neighbour
                previous_dx, previous_dy = _NEIGHBOURS[(back_index + step - 1) % 8]
                back = (current[0] + previous_dy, current[1] + previous_dx)
                break

        if found is None:
            # The start pixel is isolated; the boundary is that single pixel.
            break
        current = found

    return boundary


def _hole_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Return one mask per hole (enclosed background region) of ``mask``."""
    height, width = mask.shape
    inverted = ~mask
    labels, count = connected_components(inverted)
    holes: list[np.ndarray] = []
    for label in range(1, count + 1):
        region = labels == label
        pixels = np.nonzero(region)
        touches_border = (
            pixels[0].min() == 0
            or pixels[0].max() == height - 1
            or pixels[1].min() == 0
            or pixels[1].max() == width - 1
        )
        if not touches_border:
            holes.append(region)
    return holes


def component_outlines(mask: np.ndarray) -> dict[str, object]:
    """Trace the outer boundary and the hole boundaries of one component.

    Returns a dictionary with keys ``outer``, ``holes`` (list of ordered
    pixel traces) and ``n_holes``.  Coordinates are relative to ``mask``.
    """
    outer = trace_outer_boundary(mask)
    holes = [trace_outer_boundary(hole) for hole in _hole_masks(mask)]
    return {"outer": outer, "holes": holes, "n_holes": len(holes)}


def freeman_chain_code(trace: list[tuple[int, int]]) -> list[int]:
    """Return the 8-direction Freeman chain code of an ordered trace."""
    codes: list[int] = []
    for (row_a, col_a), (row_b, col_b) in zip(trace, trace[1:]):
        offset = (col_b - col_a, row_b - row_a)
        if offset not in _DIRECTION:
            raise ValueError(f"trace step {offset} is not 8-connected")
        codes.append(_DIRECTION[offset])
    if trace and len(trace) > 1:
        # Close the loop back to the start pixel.
        offset = (trace[0][1] - trace[-1][1], trace[0][0] - trace[-1][0])
        if offset in _DIRECTION:
            codes.append(_DIRECTION[offset])
    return codes

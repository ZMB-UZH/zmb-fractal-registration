"""Analytic output bounding boxes via coordinate compression of tile edges."""

import numpy as np
from multiview_stitcher import msi_utils

from zmb_fractal_registration._stitch_register.sim_geometry import (
    _get_antipode_of_sim,
    _get_origin_of_sim,
)


def _get_tile_rects(
    msims_per_cycle: dict[str, list], cycles: list[str]
) -> tuple[dict[str, list[tuple[dict, dict]]], list[str]]:
    """Return per-cycle (origin, antipode) world rectangles and the spatial dims.

    Each registered tile is an axis-aligned world-space box delimited by its
    transformed origin (lower corner) and antipode (upper corner).
    """
    rects: dict[str, list[tuple[dict, dict]]] = {}
    dims: list[str] | None = None
    for cycle in cycles:
        cycle_rects = []
        for msim in msims_per_cycle[cycle]:
            sim = msi_utils.get_sim_from_msim(msim)
            origin = _get_origin_of_sim(sim, transform_key="affine_registered")
            antipode = _get_antipode_of_sim(sim, transform_key="affine_registered")
            if dims is None:
                dims = list(origin.keys())
            cycle_rects.append((origin, antipode))
        rects[cycle] = cycle_rects
    return rects, dims


def _coverage_cell_grid(
    rects: dict[str, list[tuple[dict, dict]]], cycles: list[str], dims: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Coordinate-compress tile edges into a cell grid flagged with full coverage.

    Every tile edge along each dim becomes a cell boundary, so each cell lies
    entirely inside or outside every tile (no rasterization error). Returns
    (breaks, covered): breaks[d] is the sorted array of cell boundaries along
    dim d, and covered is a boolean array with one entry per cell, True where
    every cycle has a tile.
    """
    breaks: dict[str, np.ndarray] = {}
    for d in dims:
        edges = set()
        for cycle in cycles:
            for origin, antipode in rects[cycle]:
                edges.add(origin[d])
                edges.add(antipode[d])
        breaks[d] = np.array(sorted(edges))

    n_cells = tuple(len(breaks[d]) - 1 for d in dims)
    covered = np.ones(n_cells, dtype=bool)
    for cycle in cycles:
        cyc = np.zeros(n_cells, dtype=bool)
        for origin, antipode in rects[cycle]:
            slc = tuple(
                slice(
                    int(np.searchsorted(breaks[d], origin[d])),
                    int(np.searchsorted(breaks[d], antipode[d])),
                )
                for d in dims
            )
            cyc[slc] = True
        covered &= cyc
    return breaks, covered


def _tight_covered_box(
    breaks: dict[str, np.ndarray], covered: np.ndarray, dims: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """World-space bounding box enclosing all cells covered by every cycle."""
    origin, antipode = {}, {}
    for ax, d in enumerate(dims):
        other_axes = tuple(i for i in range(covered.ndim) if i != ax)
        present = covered.any(axis=other_axes) if other_axes else covered
        idx = np.nonzero(present)[0]
        origin[d] = float(breaks[d][idx[0]])
        antipode[d] = float(breaks[d][idx[-1] + 1])
    return origin, antipode


def _max_run_1d(mask: np.ndarray, widths: np.ndarray) -> tuple[int, int, float]:
    """Longest (by physical width) contiguous True run; returns (lo, hi, width)."""
    best_lo, best_hi, best_w = 0, 0, 0.0
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            w = float(widths[i:j].sum())
            if w > best_w:
                best_lo, best_hi, best_w = i, j, w
            i = j
        else:
            i += 1
    return best_lo, best_hi, best_w


def _max_rect_2d(
    mask: np.ndarray, w_rows: np.ndarray, w_cols: np.ndarray
) -> tuple[int, int, int, int, float]:
    """Largest (by physical area) all-True axis-aligned rectangle in a 2D mask.

    Returns (row_lo, row_hi, col_lo, col_hi, area) with exclusive upper indices.
    """
    best = (0, 0, 0, 0, 0.0)
    n_rows = mask.shape[0]
    for r_lo in range(n_rows):
        acc = np.ones(mask.shape[1], dtype=bool)
        for r_hi in range(r_lo + 1, n_rows + 1):
            acc &= mask[r_hi - 1]
            if not acc.any():
                break
            c_lo, c_hi, run_w = _max_run_1d(acc, w_cols)
            area = float(w_rows[r_lo:r_hi].sum()) * run_w
            if area > best[4]:
                best = (r_lo, r_hi, c_lo, c_hi, area)
    return best


def _largest_covered_box(
    breaks: dict[str, np.ndarray], covered: np.ndarray, dims: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """World-space largest axis-aligned box that is fully covered by every cycle.

    Works in 2D and 3D. For 3D, the smallest axis is swept as outer ranges and a
    2D maximal rectangle is solved on the AND of the swept slabs.
    """
    widths = [np.diff(breaks[d]) for d in dims]
    ndim = covered.ndim
    if ndim == 1:
        lo, hi, _ = _max_run_1d(covered, widths[0])
        box = {dims[0]: (lo, hi)}
    elif ndim == 2:
        r_lo, r_hi, c_lo, c_hi, _ = _max_rect_2d(covered, widths[0], widths[1])
        box = {dims[0]: (r_lo, r_hi), dims[1]: (c_lo, c_hi)}
    else:
        ax = int(np.argmin(covered.shape))
        others = [i for i in range(ndim) if i != ax]
        cov = np.moveaxis(covered, ax, 0)
        w_rows, w_cols = widths[others[0]], widths[others[1]]
        ax_breaks = breaks[dims[ax]]
        best_vol, best = -1.0, None
        n_outer = cov.shape[0]
        for lo in range(n_outer):
            acc = np.ones(cov.shape[1:], dtype=bool)
            for hi in range(lo + 1, n_outer + 1):
                acc &= cov[hi - 1]
                if not acc.any():
                    break
                r_lo, r_hi, c_lo, c_hi, area = _max_rect_2d(acc, w_rows, w_cols)
                vol = float(ax_breaks[hi] - ax_breaks[lo]) * area
                if vol > best_vol:
                    best_vol, best = vol, (lo, hi, r_lo, r_hi, c_lo, c_hi)
        lo, hi, r_lo, r_hi, c_lo, c_hi = best
        box = {
            dims[ax]: (lo, hi),
            dims[others[0]]: (r_lo, r_hi),
            dims[others[1]]: (c_lo, c_hi),
        }
    origin = {d: float(breaks[d][box[d][0]]) for d in dims}
    antipode = {d: float(breaks[d][box[d][1]]) for d in dims}
    return origin, antipode


def _compute_global_bbox(
    msims_per_cycle: dict[str, list],
    cycles: list[str],
    spacing_ref: dict[str, float],
    fusion_region: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute the global output origin and shape across all cycles.

    Bounding boxes are derived analytically from the registered tile rectangles
    via coordinate compression (see _coverage_cell_grid), so no mask needs to be
    rasterized to find them:

    - 'union': spans every tile of any cycle.
    - 'intersection': tight box around the region covered by every cycle (the
      box can still contain holes, which are zeroed during fusion).
    - 'intersection_bbox': largest axis-aligned box fully covered by every cycle,
      so no pixels need to be zeroed.

    Raises:
        ValueError: If an intersection-based region is requested but the cycles
            share no common overlap region.
    """
    rects, dims = _get_tile_rects(msims_per_cycle, cycles)

    if fusion_region == "union":
        all_rects = [r for cycle in cycles for r in rects[cycle]]
        global_origin = {d: min(o[d] for o, _ in all_rects) for d in dims}
        global_antipode = {d: max(a[d] for _, a in all_rects) for d in dims}
    else:
        breaks, covered = _coverage_cell_grid(rects, cycles, dims)
        if not covered.any():
            raise ValueError(
                "Cycles share no common overlap region; cannot fuse with "
                f"fusion_region='{fusion_region}'."
            )
        if fusion_region == "intersection":
            global_origin, global_antipode = _tight_covered_box(breaks, covered, dims)
        else:  # intersection_bbox
            global_origin, global_antipode = _largest_covered_box(breaks, covered, dims)

    global_shape = {
        d: int(np.ceil((global_antipode[d] - global_origin[d]) / spacing_ref[d]))
        for d in dims
    }
    return global_origin, global_shape

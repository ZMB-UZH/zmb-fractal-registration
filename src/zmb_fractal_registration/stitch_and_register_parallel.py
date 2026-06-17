"""Fractal task to stitch and register multiple acquisitions."""

# TODO:
# - add option to get initial positions from grid alignment instead of original stage
#   positions in metadata
# - add option to input different ROI table
# - handle larger shifts between cycles by performing a pre-registration step
# - optimize dask parallelization

import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
from dask import compute, delayed
from multiview_stitcher import (
    fusion,
    msi_utils,
    mv_graph,
    param_utils,
    registration,
)
from multiview_stitcher import (
    spatial_image_utils as si_utils,
)
from multiview_stitcher.spatial_image_utils import (
    get_affine_from_sim,
    get_ndim_from_sim,
    get_origin_from_sim,
    get_shape_from_sim,
    get_spacing_from_sim,
    get_spatial_dims_from_sim,
)
from ngio import (
    ChannelSelectionModel,
    Roi,
    open_ome_zarr_container,
)
from ngio.ome_zarr_meta import Channel
from ngio.tables import RoiTable
from pydantic import BaseModel, validate_call

from zmb_fractal_registration.stitch_and_register_init import (
    TileCorrectionModel,
)

logger = logging.getLogger(__name__)


class InitArgsStitchAndRegisterParallel(BaseModel):
    """Init Args for stitch_and_register_parallel task.

    Args:
        zarr_urls_to_register: List of urls to the individual OME-Zarr images
            to be stitched/registered.
        cycle_names: Optional cycle names for acquisitions. Used to disambiguate
            channels across cycles.
        reference_acquisition_index: Index in zarr_urls_to_register that points
            to the reference acquisition.
        reference_channel: Channel selection used as reference during
            stitching/registration.
        pyramid_level: Pyramid level used for stitching/registration.
        z_project: If True, perform stitching/registration on a z-projection.
            If False, operate on the full image volume.
        keep_original_acquisitions: If True, keep the original acquisitions.
            If False, remove them after processing.
        tile_correction: Settings for correcting non-overlapping tiles and
            filtering outliers.
        fusion_region: Which region of the registered cycles to save.
            'union': save the full extent covered by any cycle (default).
            'intersection': save the tight bounding box of the region covered by
            every cycle; pixels inside it not covered by all cycles are set to 0.
            'intersection_bbox': save the largest box fully covered by every
            cycle; no pixels are set to 0.
        interpolation_order: Spline interpolation order for resampling tiles
            into the fused output. 0 (default) is nearest-neighbor (preserves
            original pixel values), 1 is linear.
    """

    zarr_urls_to_register: list[str]
    cycle_names: list[str]
    reference_acquisition_index: int
    reference_channel: ChannelSelectionModel
    pyramid_level: int = 0
    z_project: bool = True
    keep_original_acquisitions: bool = True
    tile_correction: TileCorrectionModel = TileCorrectionModel()
    fusion_region: Literal["union", "intersection", "intersection_bbox"] = "union"
    interpolation_order: int = 0


def _get_original_translation(roi: Roi, spatial_dims: list[str]) -> dict[str, float]:
    """Get original stage translation from ROI table entries."""
    translation = {}
    for dim in spatial_dims:
        try:
            translation[dim] = getattr(roi, f"{dim}_micrometer_original")
        except AttributeError:
            translation[dim] = getattr(roi, dim)
    return translation


def _get_msims(
    *,
    image,
    fov_roi_table: RoiTable,
    z_project: bool,
    channel_suffix: str = "",
) -> list:
    """Load all FOVs as multiscale spatial images."""
    msims = []
    for roi in fov_roi_table.rois():
        data_da = image.get_roi(roi, mode="dask")
        axes = list(image.axes)
        spatial_dims = [dim for dim in axes if dim in ["z", "y", "x"]]

        if z_project and "z" in axes:
            data_da = data_da.max(axis=axes.index("z"))
            axes.remove("z")
            spatial_dims.remove("z")

        sim = si_utils.get_sim_from_array(
            data_da,
            dims=axes,
            scale={dim: getattr(image.pixel_size, dim) for dim in spatial_dims},
            c_coords=[label + channel_suffix for label in image.channel_labels],
            translation=_get_original_translation(roi, spatial_dims=spatial_dims),
            transform_key="fractal_input",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
    return msims


def _xaffine_to_matrix(xaffine: xr.DataArray) -> np.ndarray:
    """Extract a (ndim+1, ndim+1) numpy matrix from an affine DataArray.

    multiview-stitcher affine DataArrays always carry a 't' dimension
    internally. This function selects the first coordinate along any
    non-spatial dimension to collapse it down to a plain 2-D matrix.
    """
    sel_dict = {
        dim: xaffine.coords[dim][0].values
        for dim in xaffine.dims
        if dim not in ["x_in", "x_out"]
    }
    return np.array(xaffine.sel(sel_dict))


def _get_origin_of_sim(sim, transform_key: str | None = None) -> dict[str, float]:
    """Get transformed origin for a spatial image."""
    ndim = get_ndim_from_sim(sim)
    origin = get_origin_from_sim(sim, asarray=False)
    origin = np.array([origin[dim] for dim in get_spatial_dims_from_sim(sim)])

    if transform_key is not None:
        affine = _xaffine_to_matrix(
            get_affine_from_sim(sim, transform_key=transform_key)
        )
        origin = np.concatenate([origin, np.ones(1)])
        origin = np.matmul(affine, origin)[:ndim]

    return dict(zip(get_spatial_dims_from_sim(sim), origin, strict=True))


def _get_antipode_of_sim(sim, transform_key: str | None = None) -> dict[str, float]:
    """Get transformed antipode for a spatial image."""
    ndim = get_ndim_from_sim(sim)
    spacing = get_spacing_from_sim(sim, asarray=False)
    origin = get_origin_from_sim(sim, asarray=False)
    shape = get_shape_from_sim(sim, asarray=False)

    antipode = np.array(
        [
            origin[dim] + spacing[dim] * shape[dim]
            for dim in get_spatial_dims_from_sim(sim)
        ]
    )

    if transform_key is not None:
        affine = _xaffine_to_matrix(
            get_affine_from_sim(sim, transform_key=transform_key)
        )
        antipode = np.concatenate([antipode, np.ones(1)])
        antipode = np.matmul(affine, antipode)[:ndim]

    return dict(zip(get_spatial_dims_from_sim(sim), antipode, strict=True))


def _resolve_registration_channel(image, selector: ChannelSelectionModel) -> str:
    """Resolve a ChannelSelectionModel to a channel label for registration."""
    if selector.mode == "index":
        return image.channel_labels[int(selector.identifier)]
    if selector.mode == "wavelength_id":
        idx = image.get_channel_idx(selector.identifier)
        return image.channel_labels[idx]
    return selector.identifier


def _fuse_masked(sims: list):
    """Fuse spatial images, mask non-tile regions with NaN, and alias the transform.

    All sims must have an "affine_registered" transform. The returned image
    has NaN outside the union of tile footprints and an additional
    "fractal_input" transform alias pointing to "affine_registered".
    """
    # TODO: optimize chunksize
    sim_fused = fusion.fuse(
        sims, transform_key="affine_registered", output_chunksize=1024
    )
    # Coverage is channel-independent: fuse a single-channel ones mask with
    # max_fusion (skips the blending-weight computation), then drop the channel
    # dim so it broadcasts across all channels of sim_fused.
    mask = fusion.fuse(
        [xr.ones_like(s.isel(c=[0])) for s in sims],
        transform_key="affine_registered",
        fusion_func=fusion.max_fusion,
        output_chunksize=1024,
    )
    mask = mask.isel(c=0, drop=True)
    sim_fused = xr.where(mask > 0, sim_fused, np.nan)
    sim_fused.transforms["fractal_input"] = sim_fused.transforms["affine_registered"]
    return sim_fused


def _stitch_and_fuse_reference(msims_ref: list, reg_channel: str):
    """Stitch reference tiles and fuse them into a masked reference image.

    Returns a spatial image (down-sampled, lazy) that covers the full stitched
    FOV and has NaN outside the tile coverage area.
    """
    registration.register(
        msims_ref,
        reg_channel=reg_channel,
        transform_key="fractal_input",
        new_transform_key="affine_registered",
        pre_registration_pruning_method="keep_axis_aligned",
    )
    return _fuse_masked([msi_utils.get_sim_from_msim(msim) for msim in msims_ref])


def _has_overlap_with_reference_tiles(
    msim, ref_msims: list, transform_key: str, ref_transform_key: str
) -> bool:
    """Return True if msim has spatial overlap with any tile in ref_msims.

    Overlap is checked using axis-aligned bounding boxes in world space.
    transform_key is used for msim; ref_transform_key is used for each
    reference tile (typically the stitched transform after Step 2).
    A return value of False means the tile does not spatially overlap with
    any reference tile and registration would produce an unreliable result.
    """
    sim = msi_utils.get_sim_from_msim(msim)
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim)
    if nsdims:
        sim = si_utils.sim_sel_coords(sim, {nd: sim.coords[nd][0] for nd in nsdims})
    tile_sp = si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key)

    for ref_msim in ref_msims:
        ref_sim = msi_utils.get_sim_from_msim(ref_msim)
        ref_nsdims = si_utils.get_nonspatial_dims_from_sim(ref_sim)
        if ref_nsdims:
            ref_sim = si_utils.sim_sel_coords(
                ref_sim, {nd: ref_sim.coords[nd][0] for nd in ref_nsdims}
            )
        ref_sp = si_utils.get_stack_properties_from_sim(
            ref_sim, transform_key=ref_transform_key
        )
        overlap_area, _ = mv_graph.get_overlap_between_pair_of_stack_props(
            tile_sp, ref_sp
        )
        if overlap_area > 0:
            return True
    return False


def _register_cycle_tiles(
    msims: list,
    sim_fused_ref,
    reg_channel: str,
    ref_msims: list,
) -> list[int]:
    """Register all tiles in one non-reference cycle against the fused reference.

    Tiles that have no spatial overlap with any reference tile are skipped and
    their indices are returned for re-registration in Step 4.
    Overlapping tiles are registered via dask-delayed tasks (computed here).

    Returns:
        no_overlap_indices: Indices of tiles that were skipped due to no overlap.
    """
    no_overlap_indices = []
    delayed_tasks = []

    for i, msim in enumerate(msims):
        if not _has_overlap_with_reference_tiles(
            msim,
            ref_msims,
            transform_key="fractal_input",
            ref_transform_key="affine_registered",
        ):
            no_overlap_indices.append(i)
            continue
        task = delayed(registration.register)(
            [msi_utils.get_msim_from_sim(sim_fused_ref), msim],
            reg_channel=reg_channel,
            transform_key="fractal_input",
            new_transform_key="affine_registered",
            pre_registration_pruning_method=None,
            groupwise_resolution_kwargs={"reference_view": 0},
            reg_res_level=0,
        )
        delayed_tasks.append(task)

    compute(*delayed_tasks)
    return no_overlap_indices


def _collect_shifts(msims: list, no_overlap_set: set) -> tuple[list[int], list]:
    """Collect per-tile (registered - input) shifts, skipping no-overlap tiles.

    Returns:
        reg_tile_indices: Index of each tile whose shift was collected.
        shifts: Corresponding shift vectors (ndarray per tile).
    """
    reg_tile_indices = []
    shifts = []
    for i, msim in enumerate(msims):
        if i in no_overlap_set:
            continue
        sim = msi_utils.get_sim_from_msim(msim)
        t_reg = param_utils.translation_from_affine(
            _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key="affine_registered")
            )
        )
        t_in = param_utils.translation_from_affine(
            _xaffine_to_matrix(get_affine_from_sim(sim, transform_key="fractal_input"))
        )
        reg_tile_indices.append(i)
        shifts.append(t_reg - t_in)
    return reg_tile_indices, shifts


def _detect_outlier_tiles(
    shifts: list,
    reg_tile_indices: list[int],
    tcm: "TileCorrectionModel",
    cycle: str,
) -> set[int]:
    """Detect tiles whose registration shift deviates too much from the mean.

    Outlier detection is performed iteratively: after each pass, the inlier
    mean (and std for zscore mode) are recomputed from the remaining inliers
    and another pass is run. Iteration stops when no new outliers are found or
    all tiles are flagged as outliers.

    Returns:
        outlier_tile_indices: Set of tile indices flagged as outliers.
    """
    if not shifts or tcm.outlier_filter_mode == "disabled":
        return set()

    shifts_arr = np.array(shifts)
    initial_mean = np.mean(shifts_arr, axis=0)
    initial_deviations = np.array(
        [float(np.linalg.norm(s - initial_mean)) for s in shifts_arr]
    )

    logger.info(
        f"Cycle '{cycle}': mean shift of all registered tiles: "
        f"mean = {np.round(initial_mean, 3)} um, "
        f"std = {np.round(np.std(initial_deviations, axis=0), 3)} um"
    )

    score_label = "z-score" if tcm.outlier_filter_mode == "zscore" else "deviation (um)"
    inlier_mask = np.ones(len(shifts), dtype=bool)
    n_iterations = 0
    scores = np.zeros(len(shifts))

    while np.any(inlier_mask):
        n_iterations += 1
        inlier_mean = np.mean(shifts_arr[inlier_mask], axis=0)
        deviations = np.array(
            [float(np.linalg.norm(s - inlier_mean)) for s in shifts_arr]
        )
        if tcm.outlier_filter_mode == "zscore":
            inlier_devs = deviations[inlier_mask]
            mean_dev = float(np.mean(inlier_devs))
            std_dev = float(np.std(inlier_devs))
            scores = (
                (deviations - mean_dev) / std_dev
                if std_dev > 0
                else np.zeros_like(deviations)
            )
            new_outliers = (scores > tcm.threshold) & inlier_mask
        else:
            scores = deviations
            new_outliers = (deviations > tcm.threshold) & inlier_mask

        if not np.any(new_outliers):
            break

        inlier_mask[new_outliers] = False

    n_inliers = int(np.sum(inlier_mask))
    n_outliers = int(np.sum(~inlier_mask))
    _final_mean = np.round(np.mean(shifts_arr[inlier_mask], axis=0), 3)
    _final_std = np.round(np.std(shifts_arr[inlier_mask], axis=0), 3)
    logger.info(
        f"Cycle '{cycle}': found {n_outliers} outlier(s) "
        f"after {n_iterations} iterations. "
        f"{n_inliers} inlier(s) left. "
        f"Final inlier shift: mean={_final_mean} um, std={_final_std} um."
    )

    outlier_tile_indices: set[int] = set()
    for list_idx, tile_idx in enumerate(reg_tile_indices):
        if not inlier_mask[list_idx]:
            logger.warning(
                f"Cycle '{cycle}', tile {tile_idx}: shift "
                f"{np.round(shifts[list_idx], 3)} um, "
                f"({score_label}={scores[list_idx]:.2f}) "
                f"flagged as outlier."
            )
            outlier_tile_indices.add(tile_idx)

    return outlier_tile_indices


def _apply_mean_shift_to_tiles(
    msims: list,
    tile_indices: list[int],
    mean_shift: np.ndarray,
    ndim: int,
    transform_key: str = "affine_registered",
) -> None:
    """Store stage_position + mean_shift as the named transform for each tile."""
    for tile_idx in tile_indices:
        sim = msi_utils.get_sim_from_msim(msims[tile_idx])
        t_in = param_utils.translation_from_affine(
            _xaffine_to_matrix(get_affine_from_sim(sim, transform_key="fractal_input"))
        )
        matrix = np.eye(ndim + 1)
        matrix[:ndim, ndim] = t_in + mean_shift
        msi_utils.set_affine_transform(
            msims[tile_idx],
            param_utils.affine_to_xaffine(matrix, t_coords=[0]),
            transform_key,
        )


def _register_leftover_tiles(
    msims: list,
    tiles_to_correct: set[int],
    reg_channel: str,
    cycle: str,
    correction_method: str = "reregister",
) -> None:
    """Correct outlier and no-overlap tiles using inlier tile information.

    Two correction methods are supported (controlled by correction_method):

    - ``"mean_shift"``: Apply the mean (registered - stage) translation of all
      inlier tiles directly to each leftover tile. Fast and deterministic, but
      ignores tile-specific image content.
    - ``"reregister"``: Fuse the inlier tiles into a reference image and
      re-register each leftover tile against it, seeded from stage position +
      mean inlier shift. Falls back to mean_shift if there is not enough overlap
      with the fused inlier.

    In both cases, falls back to the raw stage position when there are no
    inlier tiles.
    """
    if not tiles_to_correct:
        return

    ok_indices = [i for i in range(len(msims)) if i not in tiles_to_correct]

    if not ok_indices:
        logger.warning(
            f"Cycle '{cycle}': no inlier tiles available; "
            f"leftover tiles will keep their stage position."
        )
        for tile_idx in sorted(tiles_to_correct):
            msim = msims[tile_idx]
            sim = msi_utils.get_sim_from_msim(msim)
            matrix = _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key="fractal_input")
            )
            msi_utils.set_affine_transform(
                msim,
                param_utils.affine_to_xaffine(matrix, t_coords=[0]),
                "affine_registered",
            )
        return

    # Reuse _collect_shifts to compute mean (registered - stage) shift across inliers.
    _, inlier_shifts = _collect_shifts(msims, tiles_to_correct)
    mean_shift = np.mean(inlier_shifts, axis=0)
    ndim = len(mean_shift)
    sorted_tile_indices = sorted(tiles_to_correct)
    logger.info(
        f"Cycle '{cycle}': mean inlier shift for leftover tiles: "
        f"{np.round(mean_shift, 3)} um"
    )

    if correction_method == "mean_shift":
        logger.info(
            f"Cycle '{cycle}': applying mean inlier shift to "
            f"{len(sorted_tile_indices)} leftover tile(s) (no re-registration)."
        )
        _apply_mean_shift_to_tiles(msims, sorted_tile_indices, mean_shift, ndim)
        return

    # correction_method == "reregister"
    _INIT_KEY = "fractal_input_mean_shifted"
    logger.info(
        f"Cycle '{cycle}': fusing {len(ok_indices)} inlier tile(s) as "
        f"reference for re-registration of {len(sorted_tile_indices)} leftover tile(s)."
    )
    sim_fused_inliers = _fuse_masked(
        [msi_utils.get_sim_from_msim(msims[i]) for i in ok_indices]
    )

    # Seed each leftover tile from stage position + mean inlier shift.
    _apply_mean_shift_to_tiles(
        msims, sorted_tile_indices, mean_shift, ndim, transform_key=_INIT_KEY
    )

    # Alias the fused inlier's position under the shared init key.
    msim_fused_inliers = msi_utils.get_msim_from_sim(sim_fused_inliers)
    msi_utils.set_affine_transform(
        msim_fused_inliers,
        msi_utils.get_transform_from_msim(msim_fused_inliers, "fractal_input"),
        _INIT_KEY,
    )

    logger.info(
        f"Cycle '{cycle}': re-registering {len(sorted_tile_indices)} leftover "
        f"tile(s) against fused inlier image."
    )
    try:
        registration.register(
            [msim_fused_inliers] + [msims[i] for i in sorted_tile_indices],
            reg_channel=reg_channel,
            transform_key=_INIT_KEY,
            new_transform_key="affine_registered",
            pre_registration_pruning_method=None,
            groupwise_resolution_kwargs={"reference_view": 0},
            reg_res_level=0,
        )
    except mv_graph.NotEnoughOverlapError:
        logger.warning(
            f"Cycle '{cycle}': leftover tile registration failed (not enough overlap "
            f"with fused inlier); falling back to mean inlier shift."
        )
        _apply_mean_shift_to_tiles(msims, sorted_tile_indices, mean_shift, ndim)


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


@validate_call
def stitch_and_register_parallel(
    *,
    zarr_url: str,
    init_args: InitArgsStitchAndRegisterParallel,
) -> dict[str, Any]:
    """Stitch and register acquisitions, then fuse into a single output image.

    Args:
        zarr_url: Absolute path to the new OME-Zarr image.
        init_args: Initialization arguments from the init task.
    """
    logger.info(
        f"Starting stitch_and_register_parallel for zarr_url={zarr_url} with "
        f"{len(init_args.zarr_urls_to_register)} acquisitions "
        f"(reference index: {init_args.reference_acquisition_index}, "
        f"pyramid_level: {init_args.pyramid_level}, z_project: {init_args.z_project})"
    )

    if len(init_args.zarr_urls_to_register) < 2:
        raise ValueError("At least two acquisitions are required for registration.")
    if len(init_args.cycle_names) != len(init_args.zarr_urls_to_register):
        raise ValueError("cycle_names length must match zarr_urls_to_register length.")
    if not (0 <= init_args.reference_acquisition_index < len(init_args.cycle_names)):
        raise ValueError("reference_acquisition_index is out of range.")

    cycles = list(init_args.cycle_names)
    ref_cycle = cycles[init_args.reference_acquisition_index]
    z_project = init_args.z_project

    logger.info(f"Opening OME-Zarr containers for cycles: {cycles}")
    containers = {
        cycle: open_ome_zarr_container(url)
        for cycle, url in zip(cycles, init_args.zarr_urls_to_register, strict=True)
    }

    for cycle in cycles:
        if containers[cycle].is_time_series:
            raise ValueError(
                f"Acquisition '{cycle}' is a timeseries. "
                f"Timeseries data is not supported."
            )

    reg_image_ref = containers[ref_cycle].get_image(path=str(init_args.pyramid_level))
    reg_channel = _resolve_registration_channel(
        reg_image_ref, init_args.reference_channel
    )
    logger.info(
        f"Reference cycle: '{ref_cycle}', registration channel: '{reg_channel}'"
    )

    # ------------------------------------------------------------------
    # Step 1: Load all FOVs from each cycle at the registration pyramid
    # level, optionally projecting along z.
    # ------------------------------------------------------------------
    logger.info(
        f"[Step 1/7] Loading FOVs at pyramid level {init_args.pyramid_level}"
        f"{' (z-projected)' if z_project else ''}."
    )
    msims_reg = {}
    for cycle in cycles:
        reg_image = containers[cycle].get_image(path=str(init_args.pyramid_level))
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        msims_reg[cycle] = _get_msims(
            image=reg_image, fov_roi_table=fov_roi_table, z_project=z_project
        )
        logger.info(f"Cycle '{cycle}': loaded {len(msims_reg[cycle])} FOV(s).")

    # ------------------------------------------------------------------
    # Step 2: Stitch the reference cycle and fuse into a masked reference
    # image used as the fixed target for per-tile registration.
    # ------------------------------------------------------------------
    logger.info(
        f"[Step 2/7] Stitching reference cycle '{ref_cycle}' "
        f"({len(msims_reg[ref_cycle])} tile(s))."
    )
    sim_fused_ref_ds = _stitch_and_fuse_reference(msims_reg[ref_cycle], reg_channel)
    logger.info("Reference stitching and fusion complete.")

    # ------------------------------------------------------------------
    # Step 3: Register each non-reference cycle's tiles against the fused
    # reference. Tiles without spatial overlap are deferred to Step 4.
    # ------------------------------------------------------------------
    logger.info(
        f"[Step 3/7] Registering {len(cycles) - 1} non-reference cycle(s) "
        f"against fused reference."
    )
    no_overlap_indices: dict[str, list[int]] = {}
    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        logger.info(
            f"Cycle '{cycle}': registering {len(msims_reg[cycle])} tile(s) "
            f"against fused reference."
        )
        no_overlap = _register_cycle_tiles(
            msims_reg[cycle], sim_fused_ref_ds, reg_channel, msims_reg[ref_cycle]
        )
        no_overlap_indices[cycle] = no_overlap
        n_reg = len(msims_reg[cycle]) - len(no_overlap)
        logger.info(
            f"Cycle '{cycle}': {n_reg} tile(s) registered"
            + (
                f"; {len(no_overlap)} had no overlap with the reference "
                "(deferred to Step 4)."
                if no_overlap
                else "."
            )
        )
    logger.info("Tile registration complete.")

    # ------------------------------------------------------------------
    # Step 4: For each non-reference cycle, detect outlier tiles (shifts
    # that deviate too much from the cycle mean) and collect no-overlap
    # tiles. Re-register all such leftover tiles against a fused image of
    # the remaining inlier tiles, with the fused inlier held fixed.
    # ------------------------------------------------------------------
    tcm = init_args.tile_correction
    _outlier_desc = tcm.outlier_filter_mode + (
        f" (threshold={tcm.threshold})" if tcm.outlier_filter_mode != "disabled" else ""
    )
    logger.info(
        f"[Step 4/7] Correcting leftover tiles "
        f"(outlier detection: {_outlier_desc}, correction: {tcm.correction_method})."
    )

    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        no_overlap_set = set(no_overlap_indices.get(cycle, []))
        reg_tile_indices, shifts = _collect_shifts(msims_reg[cycle], no_overlap_set)
        outlier_indices = _detect_outlier_tiles(shifts, reg_tile_indices, tcm, cycle)
        tiles_to_correct = no_overlap_set | outlier_indices
        if tiles_to_correct:
            logger.info(
                f"Cycle '{cycle}': {len(tiles_to_correct)} leftover tile(s) to correct "
                f"({len(no_overlap_set)} no-overlap, {len(outlier_indices)} outliers)."
            )
        _register_leftover_tiles(
            msims_reg[cycle],
            tiles_to_correct,
            reg_channel,
            cycle,
            tcm.correction_method,
        )

    # ------------------------------------------------------------------
    # Step 5: Reload FOVs at full resolution and transfer the computed
    # transforms. Expand 2D affines to 3D when z_project was used.
    # ------------------------------------------------------------------
    logger.info(
        "[Step 5/7] Reloading FOVs at full resolution and transferring transforms."
    )
    msims_fusion = {}
    for cycle in cycles:
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        msims_fusion[cycle] = _get_msims(
            image=containers[cycle].get_image(),
            fov_roi_table=fov_roi_table,
            z_project=False,
            channel_suffix=f"_{cycle}",
        )
        for msim_reg, msim_fus in zip(
            msims_reg[cycle], msims_fusion[cycle], strict=True
        ):
            affine = msi_utils.get_transform_from_msim(msim_reg, "affine_registered")
            if z_project:
                affine_3d = param_utils.identity_transform(
                    ndim=3,
                    t_coords=affine.coords["t"] if "t" in affine.dims else None,
                )
                affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = (
                    affine
                )
                affine = affine_3d
            msi_utils.set_affine_transform(msim_fus, affine, "affine_registered")

    # ------------------------------------------------------------------
    # Step 6: Determine the global bounding box across all cycles and
    # fuse every cycle into a shared output canvas.
    # ------------------------------------------------------------------
    fusion_region = init_args.fusion_region
    interpolation_order = init_args.interpolation_order
    logger.info(
        f"[Step 6/7] Computing global bounding box ({fusion_region}) and "
        f"fusing all cycles (interpolation_order={interpolation_order})."
    )
    spacing_ref = get_spacing_from_sim(
        msi_utils.get_sim_from_msim(msims_fusion[ref_cycle][0]), asarray=False
    )
    global_origin, global_shape = _compute_global_bbox(
        msims_fusion, cycles, spacing_ref, fusion_region
    )
    _rounded_origin = {k: round(v, 3) for k, v in global_origin.items()}
    logger.info(f"Global output shape: {global_shape}, origin: {_rounded_origin}")

    sims_fused = {}
    masks_fused = {}
    for cycle in cycles:
        logger.info(f"Cycle '{cycle}': fusing {len(msims_fusion[cycle])} tile(s).")
        cycle_sims = [msi_utils.get_sim_from_msim(msim) for msim in msims_fusion[cycle]]
        sims_fused[cycle] = fusion.fuse(
            cycle_sims,
            transform_key="affine_registered",
            interpolation_order=interpolation_order,
            output_chunksize=1024,
            output_origin=global_origin,
            output_shape=global_shape,
        )
        if fusion_region == "intersection":
            # Coverage is channel-independent, so fuse a single channel only.
            masks_fused[cycle] = fusion.fuse(
                [xr.ones_like(sim.isel(c=[0])) for sim in cycle_sims],
                transform_key="affine_registered",
                fusion_func=fusion.max_fusion,
                interpolation_order=interpolation_order,
                output_chunksize=1024,
                output_origin=global_origin,
                output_shape=global_shape,
            )

    sim_fused_all = xr.concat([sims_fused[cycle] for cycle in cycles], dim="c")

    if fusion_region == "intersection":
        # Keep only pixels covered by every cycle; zero the rest (preserves dtype).
        dims_order = sim_fused_all.dims
        out_dtype = sim_fused_all.dtype
        coverage = None
        for cycle in cycles:
            cov = (masks_fused[cycle] > 0).any(dim="c")
            coverage = cov if coverage is None else (coverage & cov)
        sim_fused_all = (
            xr.where(coverage, sim_fused_all, 0)
            .astype(out_dtype)
            .transpose(*dims_order)
        )

    axes_in = containers[ref_cycle].get_image().axes
    sim_fused_all = sim_fused_all.squeeze(
        [dim for dim in sim_fused_all.dims if dim not in axes_in]
    )

    # ------------------------------------------------------------------
    # Step 7: Write the fused image to the output OME-Zarr store.
    # ------------------------------------------------------------------
    logger.info(
        f"[Step 7/7] Writing fused image to '{zarr_url}' "
        f"(shape: {sim_fused_all.shape}, dims: {sim_fused_all.dims})."
    )
    channels_meta_all = [
        Channel(
            label=f"{ch.label}_{cycle}",
            wavelength_id=ch.wavelength_id,
            channel_visualisation=ch.channel_visualisation,
        )
        for cycle in cycles
        for ch in containers[cycle].images_container.channels_meta.channels
    ]
    output_container = containers[ref_cycle].derive_image(
        store=zarr_url,
        shape=sim_fused_all.shape,
        channels_meta=channels_meta_all,
        chunks=tuple(c[0] for c in sim_fused_all.chunks),
        overwrite=True,
    )
    out_image = output_container.get_image()
    out_image.set_array(patch=sim_fused_all.data, axes_order=sim_fused_all.dims)
    out_image.consolidate()
    logger.info("Output image written and consolidated successfully.")

    image_list_updates = [
        {
            "zarr_url": zarr_url,
            "origin": init_args.zarr_urls_to_register[0],
            "attributes": {
                "acquisition": Path(zarr_url).as_posix().split("/")[-1],
            },
            # TODO: better passing of acquisition metadata (maybe pass from init task)
        }
    ]

    if init_args.keep_original_acquisitions:
        logger.info("Keeping original acquisitions. Task complete.")
        return {"image_list_updates": image_list_updates}

    logger.info(
        f"Removing {len(init_args.zarr_urls_to_register)} original acquisition(s)..."
    )
    for url in init_args.zarr_urls_to_register:
        logger.info(f"Deleting original acquisition at '{url}'.")
        shutil.rmtree(url)
    logger.info("Task complete.")
    return {
        "image_list_updates": image_list_updates,
        "image_list_removals": init_args.zarr_urls_to_register,
    }


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitch_and_register_parallel)

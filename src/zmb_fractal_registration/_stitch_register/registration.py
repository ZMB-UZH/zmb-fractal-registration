"""Tile registration, outlier detection, and leftover-tile correction."""

import logging

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
from multiview_stitcher.spatial_image_utils import get_affine_from_sim

from zmb_fractal_registration._stitch_register.sim_geometry import _xaffine_to_matrix
from zmb_fractal_registration.stitch_and_register_init import TileCorrectionModel

logger = logging.getLogger(__name__)


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
    tcm: TileCorrectionModel,
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

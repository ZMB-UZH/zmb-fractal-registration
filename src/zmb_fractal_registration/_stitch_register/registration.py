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

# Chunks for the *registration references* are sized relative to one input
# tile. Fusing a chunk holds every tile overlapping it (resampled, plus
# blending weights) in memory at once, so a chunk larger than a tile pulls in
# ~(chunk/tile)**2 tiles and the memory grows quadratically. A chunk of one
# tile keeps that count at ~4 per dimension at *any* pyramid level, which an
# absolute chunk size cannot do since a tile is a different number of pixels at
# each level. Measured on 36 tiles / 8 workers: factor 2 needed 0.67 GB, factor
# 1 0.53 GB, factor 0.5 0.50 GB but 27% more time, and factor 0.25 was worse on
# both (0.82 GB, 2.6x the time) as per-chunk overhead took over - which is why
# the references are not simply chunked like their input.
_CHUNKSIZE_TILE_FACTOR = 1.0

# Chunks below this many pixels along a dimension cost more in per-chunk
# overhead than they save in memory (a tile can be a handful of pixels at the
# coarsest pyramid levels, and a chunk that small holds almost nothing).
_MIN_CHUNKSIZE = 64

# Upper bound on the voxels in one chunk, so that a chunk stays small in
# absolute terms even where a tile does not: at full resolution a tile can be
# millions of voxels, and tracking it exactly would undo the point of chunking.
# 2**24 is what the previous fixed 1024x1024 chunks held in 2D.
_MAX_CHUNK_VOXELS = 2**24


def _cap_chunk_voxels(chunks: dict[str, int]) -> dict[str, int]:
    """Shrink chunks until a single chunk fits the voxel budget.

    Every dimension shrinks by the same factor, rounding down so the result
    cannot come back over the budget. No minimum is applied: staying inside the
    budget matters more than avoiding small chunks, since exceeding it is what
    runs a job out of memory. Dimensions already at 1 cannot shrink further, so
    the pass repeats with the remaining ones - a store chunked one plane at a
    time (z=1) would otherwise stay over budget however far y and x shrink.
    """
    chunks = dict(chunks)
    for _ in range(len(chunks)):
        if int(np.prod(list(chunks.values()))) <= _MAX_CHUNK_VOXELS:
            break
        shrinkable = {dim: size for dim, size in chunks.items() if size > 1}
        if not shrinkable:
            break
        n_shrinkable = int(np.prod(list(shrinkable.values())))
        fixed = int(np.prod([s for d, s in chunks.items() if d not in shrinkable]))
        scale = (_MAX_CHUNK_VOXELS / (fixed * n_shrinkable)) ** (1 / len(shrinkable))
        chunks = {
            dim: max(1, int(size * scale)) if dim in shrinkable else size
            for dim, size in chunks.items()
        }
    return chunks


def _registration_chunksize(sims: list) -> dict[str, int]:
    """Chunks for fusing a registration reference, scaled to the input tile.

    Deliberately independent of how the input happens to be chunked: these
    images are fused repeatedly (once per tile registered against them), so
    chunks smaller than a tile pay their per-chunk overhead many times over.
    """
    tile_shape = si_utils.get_shape_from_sim(sims[0], asarray=False)
    return _cap_chunk_voxels(
        {
            dim: max(_MIN_CHUNKSIZE, round(_CHUNKSIZE_TILE_FACTOR * size))
            for dim, size in tile_shape.items()
        }
    )


def _output_chunksize(sims: list) -> dict[str, int]:
    """Chunks for the fused output image: the input tiles' own chunking.

    This size is also written as the chunking of the output OME-Zarr, so
    inheriting it keeps the output laid out like the images it was built from.
    Tiles are read as ROIs from their store, so their chunks never exceed a
    tile - which is exactly the bound the fusion needs anyway. Chunks are only
    shrunk if the input carries pathologically large ones.
    """
    spatial_dims = si_utils.get_spatial_dims_from_sim(sims[0])
    data = sims[0].data
    if hasattr(data, "chunksize"):
        chunks = dict(
            zip(spatial_dims, data.chunksize[-len(spatial_dims) :], strict=True)
        )
    else:
        # Not a chunked array; fall back to the tile itself.
        chunks = si_utils.get_shape_from_sim(sims[0], asarray=False)
    return _cap_chunk_voxels(chunks)


def _fuse_masked(
    sims: list,
    transform_key: str = "affine_registered",
    alias_key: str = "fractal_input",
    interpolation_order: int = 0,
    output_origin: dict[str, float] | None = None,
    output_shape: dict[str, int] | None = None,
    output_spacing: dict[str, float] | None = None,
):
    """Fuse spatial images, mask non-tile regions with NaN, and alias the transform.

    All sims must have a transform under `transform_key`. The returned image
    has NaN outside the union of tile footprints and an additional `alias_key`
    transform alias pointing to `transform_key`, so that the fused image can be
    registered together with tiles that use `alias_key` as their input position.
    Masking makes the result floating point (float32 for the usual integer
    inputs, since NaN cannot be represented in an integer dtype).

    The output canvas defaults to the union of the input tiles; passing
    `output_origin`/`output_shape`/`output_spacing` fuses onto an explicitly
    given canvas instead, which is how several cycles are put onto a shared
    grid.

    Resampling defaults to nearest-neighbour: these fused images only serve as
    registration references, so keeping the original pixel values is preferable
    to smoothing them.
    """
    chunksize = _registration_chunksize(sims)
    canvas_kwargs = {
        "output_origin": output_origin,
        "output_shape": output_shape,
        "output_spacing": output_spacing,
    }
    sim_fused = fusion.fuse(
        sims,
        transform_key=transform_key,
        interpolation_order=interpolation_order,
        output_chunksize=chunksize,
        **canvas_kwargs,
    )
    # Coverage is channel-independent: fuse a single-channel ones mask with
    # max_fusion (skips the blending-weight computation), then drop the channel
    # dim so it broadcasts across all channels of sim_fused.
    mask = fusion.fuse(
        [xr.ones_like(s.isel(c=[0])) for s in sims],
        transform_key=transform_key,
        fusion_func=fusion.max_fusion,
        interpolation_order=interpolation_order,
        output_chunksize=chunksize,
        **canvas_kwargs,
    )
    mask = mask.isel(c=0, drop=True)
    # np.float32 rather than a bare np.nan (a python float, i.e. float64): NaN
    # forces a float dtype anyway, and multiview-stitcher casts to float32
    # before registering, so float64 would only double the memory of every
    # intermediate fused image for precision that is discarded downstream.
    sim_fused = xr.where(mask > 0, sim_fused, np.float32(np.nan))
    sim_fused.transforms[alias_key] = sim_fused.transforms[transform_key]
    return sim_fused


def _stitch_and_fuse_reference(
    msims_ref: list, reg_channel: str, init_transform_key: str = "fractal_input"
):
    """Stitch reference tiles and fuse them into a masked reference image.

    `init_transform_key` is the transform the tiles start from, and the key the
    fused image is aliased under so that other cycles can be registered against
    it.

    Returns a spatial image (down-sampled, lazy) that covers the full stitched
    FOV and has NaN outside the tile coverage area. It stays lazy on purpose:
    registering a tile against it only computes the chunks its overlap region
    touches, so the whole image is never materialized (see _output_chunksize).
    """
    registration.register(
        msims_ref,
        reg_channel=reg_channel,
        transform_key=init_transform_key,
        new_transform_key="affine_registered",
        pre_registration_pruning_method="keep_axis_aligned",
    )
    return _fuse_masked(
        [msi_utils.get_sim_from_msim(msim) for msim in msims_ref],
        alias_key=init_transform_key,
    )


def _stack_props(msim, transform_key: str):
    """Stack properties of a tile in world space, without non-spatial dims."""
    sim = msi_utils.get_sim_from_msim(msim)
    nsdims = si_utils.get_nonspatial_dims_from_sim(sim)
    if nsdims:
        sim = si_utils.sim_sel_coords(sim, {nd: sim.coords[nd][0] for nd in nsdims})
    return si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key)


def _has_overlap_with_reference_tiles(
    msim, ref_msims: list, transform_key: str, ref_transform_key: str
) -> bool:
    """Return True if msim has spatial overlap with any tile in ref_msims.

    Overlap is checked using axis-aligned bounding boxes in world space.
    transform_key is used for msim; ref_transform_key is used for each
    reference tile (typically the stitched transform after Step 3).
    A return value of False means the tile does not spatially overlap with
    any reference tile and registration would produce an unreliable result.
    """
    tile_sp = _stack_props(msim, transform_key)
    for ref_msim in ref_msims:
        ref_sp = _stack_props(ref_msim, ref_transform_key)
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
    init_transform_key: str = "fractal_input",
) -> list[int]:
    """Register all tiles in one non-reference cycle against the fused reference.

    `sim_fused_ref` stays lazy: multiview-stitcher crops both images to their
    overlap region before computing anything, and fusion only materializes the
    tiles overlapping each output chunk, so a tile's registration computes just
    the few chunks its own footprint touches. That only holds while the chunks
    are no larger than a tile, which is what _output_chunksize guarantees.

    Tiles that have no spatial overlap with any reference tile are skipped and
    their indices are returned for re-registration in Step 5.
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
            transform_key=init_transform_key,
            ref_transform_key="affine_registered",
        ):
            no_overlap_indices.append(i)
            continue
        # A fresh msim per tile: register() writes its result onto every msim it
        # is given, so sharing one across the delayed tasks would be a data race.
        task = delayed(registration.register)(
            [msi_utils.get_msim_from_sim(sim_fused_ref), msim],
            reg_channel=reg_channel,
            transform_key=init_transform_key,
            new_transform_key="affine_registered",
            pre_registration_pruning_method=None,
            groupwise_resolution_kwargs={"reference_view": 0},
            reg_res_level=0,
        )
        delayed_tasks.append(task)

    compute(*delayed_tasks)
    return no_overlap_indices


def _collect_shifts(
    msims: list, no_overlap_set: set, init_transform_key: str = "fractal_input"
) -> tuple[list[int], list]:
    """Collect per-tile (registered - input) shifts, skipping no-overlap tiles.

    The shifts are relative to `init_transform_key`, i.e. relative to whatever
    position the tiles were registered from.

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
            _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key=init_transform_key)
            )
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
    init_transform_key: str = "fractal_input",
) -> None:
    """Store input_position + mean_shift as the named transform for each tile.

    The input position is taken from `init_transform_key`, i.e. the same
    position the collected shifts are relative to.
    """
    for tile_idx in tile_indices:
        sim = msi_utils.get_sim_from_msim(msims[tile_idx])
        t_in = param_utils.translation_from_affine(
            _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key=init_transform_key)
            )
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
    init_transform_key: str = "fractal_input",
) -> None:
    """Correct outlier and no-overlap tiles using inlier tile information.

    Two correction methods are supported (controlled by correction_method):

    - ``"mean_shift"``: Apply the mean (registered - input) translation of all
      inlier tiles directly to each leftover tile. Fast and deterministic, but
      ignores tile-specific image content.
    - ``"reregister"``: Fuse the inlier tiles into a reference image and
      re-register each leftover tile against it, seeded from its input position
      + mean inlier shift. Falls back to mean_shift if there is not enough
      overlap with the fused inlier.

    In both cases, falls back to the input position (`init_transform_key`) when
    there are no inlier tiles.
    """
    if not tiles_to_correct:
        return

    ok_indices = [i for i in range(len(msims)) if i not in tiles_to_correct]

    if not ok_indices:
        logger.warning(
            f"Cycle '{cycle}': no inlier tiles available; "
            f"leftover tiles will keep their input position."
        )
        for tile_idx in sorted(tiles_to_correct):
            msim = msims[tile_idx]
            sim = msi_utils.get_sim_from_msim(msim)
            matrix = _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key=init_transform_key)
            )
            msi_utils.set_affine_transform(
                msim,
                param_utils.affine_to_xaffine(matrix, t_coords=[0]),
                "affine_registered",
            )
        return

    # Reuse _collect_shifts to compute mean (registered - input) shift across inliers.
    _, inlier_shifts = _collect_shifts(msims, tiles_to_correct, init_transform_key)
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
        _apply_mean_shift_to_tiles(
            msims,
            sorted_tile_indices,
            mean_shift,
            ndim,
            init_transform_key=init_transform_key,
        )
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

    # Seed each leftover tile from its input position + mean inlier shift.
    _apply_mean_shift_to_tiles(
        msims,
        sorted_tile_indices,
        mean_shift,
        ndim,
        transform_key=_INIT_KEY,
        init_transform_key=init_transform_key,
    )

    # Alias the fused inlier's position under the shared init key.
    msim_fused_inliers = msi_utils.get_msim_from_sim(sim_fused_inliers)
    msi_utils.set_affine_transform(
        msim_fused_inliers,
        msi_utils.get_transform_from_msim(msim_fused_inliers, "fractal_input"),
        _INIT_KEY,
    )

    # Only re-register tiles that land on an actual inlier tile. The fused
    # inlier image is NaN wherever no inlier tile is, but its *bounding box*
    # covers those gaps too, so a tile seeded into such a gap passes
    # multiview-stitcher's overlap check and then hands phase correlation an
    # all-NaN reference, which it cannot handle.
    inlier_msims = [msims[i] for i in ok_indices]
    reregister_indices, isolated_indices = [], []
    for tile_idx in sorted_tile_indices:
        if _has_overlap_with_reference_tiles(
            msims[tile_idx],
            inlier_msims,
            transform_key=_INIT_KEY,
            ref_transform_key="affine_registered",
        ):
            reregister_indices.append(tile_idx)
        else:
            isolated_indices.append(tile_idx)

    if isolated_indices:
        logger.warning(
            f"Cycle '{cycle}': {len(isolated_indices)} leftover tile(s) do not "
            f"overlap any inlier tile even after the mean shift; keeping the mean "
            f"inlier shift for them without re-registration."
        )
        _apply_mean_shift_to_tiles(
            msims,
            isolated_indices,
            mean_shift,
            ndim,
            init_transform_key=init_transform_key,
        )

    if not reregister_indices:
        logger.warning(
            f"Cycle '{cycle}': no leftover tile overlaps an inlier tile; "
            f"skipping re-registration."
        )
        return

    logger.info(
        f"Cycle '{cycle}': re-registering {len(reregister_indices)} leftover "
        f"tile(s) against fused inlier image."
    )
    try:
        registration.register(
            [msim_fused_inliers] + [msims[i] for i in reregister_indices],
            reg_channel=reg_channel,
            transform_key=_INIT_KEY,
            new_transform_key="affine_registered",
            pre_registration_pruning_method=None,
            groupwise_resolution_kwargs={"reference_view": 0},
            reg_res_level=0,
        )
    # ValueError covers phase correlation choking on a degenerate overlap region
    # (e.g. an all-NaN reference), which must not take the whole well down.
    except (mv_graph.NotEnoughOverlapError, ValueError) as exc:
        logger.warning(
            f"Cycle '{cycle}': leftover tile registration failed ({exc}); "
            f"falling back to mean inlier shift."
        )
        _apply_mean_shift_to_tiles(
            msims,
            reregister_indices,
            mean_shift,
            ndim,
            init_transform_key=init_transform_key,
        )

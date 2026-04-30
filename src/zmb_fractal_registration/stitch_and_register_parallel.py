"""Fractal task to stitch and register multiple acquisitions."""

# TODO:
# - add option to get initial positions from grid alignment instead of original stage
#   positions in metadata
# - add option to input different ROI table
# - handle larger shifts between cycles by performing a pre-registration step
# - optimize dask parallelization

import logging
import shutil
from typing import Any

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
    OutlierFilterModel,
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
        outlier_filter: Settings for outlier registration-shift correction.
    """

    zarr_urls_to_register: list[str]
    cycle_names: list[str]
    reference_acquisition_index: int
    reference_channel: ChannelSelectionModel
    pyramid_level: int = 0
    z_project: bool = True
    keep_original_acquisitions: bool = True
    outlier_filter: OutlierFilterModel = OutlierFilterModel()


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
        affine = _xaffine_to_matrix(get_affine_from_sim(sim, transform_key=transform_key))
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
        affine = _xaffine_to_matrix(get_affine_from_sim(sim, transform_key=transform_key))
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
    sim_fused = fusion.fuse(sims, transform_key="affine_registered", output_chunksize=1024)
    mask = fusion.fuse(
        [xr.ones_like(s) for s in sims],
        transform_key="affine_registered",
        output_chunksize=1024,
    )
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
        sim = si_utils.sim_sel_coords(
            sim, {nd: sim.coords[nd][0] for nd in nsdims}
        )
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
            _xaffine_to_matrix(get_affine_from_sim(sim, transform_key="affine_registered"))
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
    osc: "OutlierFilterModel",
    cycle: str,
) -> set[int]:
    """Detect tiles whose registration shift deviates too much from the mean.

    Returns:
        outlier_tile_indices: Set of tile indices flagged as outliers.
    """
    if not shifts or osc.mode == "disabled":
        return set()

    initial_mean = np.mean(shifts, axis=0)
    deviations = np.array(
        [float(np.linalg.norm(s - initial_mean)) for s in shifts]
    )

    if osc.mode == "zscore":
        mean_dev = float(np.mean(deviations))
        std_dev = float(np.std(deviations))
        zscores = (
            (deviations - mean_dev) / std_dev
            if std_dev > 0
            else np.zeros_like(deviations)
        )
        is_outlier = zscores > osc.threshold
        score_label, scores = "z-score", zscores
    else:
        is_outlier = deviations > osc.threshold
        score_label, scores = "deviation (µm)", deviations

    outlier_tile_indices: set[int] = set()
    for list_idx, tile_idx in enumerate(reg_tile_indices):
        if is_outlier[list_idx]:
            logger.warning(
                f"  Cycle '{cycle}', tile {tile_idx}: shift "
                f"{np.round(shifts[list_idx], 3)} "
                f"({score_label}={scores[list_idx]:.2f}) "
                f"flagged as outlier."
            )
            outlier_tile_indices.add(tile_idx)

    return outlier_tile_indices


def _register_leftover_tiles(
    msims: list,
    tiles_to_correct: set[int],
    reg_channel: str,
    cycle: str,
) -> None:
    """Re-register outlier and no-overlap tiles against the fused inlier image.

    Inlier tiles (all tiles not in tiles_to_correct) are fused into a single
    reference image. Each leftover tile is then registered against this fused
    inlier, starting from its stage position (fractal_input), with the fused
    inlier held fixed.

    For both no-overlap tiles (affine_registered never set) and outlier tiles
    (affine_registered unreliable), fractal_input is used as the starting
    transform so the registration is seeded from the stage position.

    Falls back to the stage position if there are no inlier tiles to fuse.
    """
    if not tiles_to_correct:
        return

    ok_indices = [i for i in range(len(msims)) if i not in tiles_to_correct]

    if not ok_indices:
        logger.warning(
            f"  Cycle '{cycle}': no inlier tiles to fuse as reference. "
            f"Leftover tiles will keep their stage position."
        )
        for tile_idx in sorted(tiles_to_correct):
            msim = msims[tile_idx]
            sim = msi_utils.get_sim_from_msim(msim)
            matrix = _xaffine_to_matrix(get_affine_from_sim(sim, transform_key="fractal_input"))
            xaffine = param_utils.affine_to_xaffine(matrix, t_coords=[0])
            msi_utils.set_affine_transform(msim, xaffine, "affine_registered")
        return

    logger.info(
        f"  Cycle '{cycle}': fusing {len(ok_indices)} inlier tile(s) as "
        f"reference for re-registration of {len(tiles_to_correct)} leftover tile(s)."
    )
    sim_fused_inliers = _fuse_masked(
        [msi_utils.get_sim_from_msim(msims[i]) for i in ok_indices]
    )

    sorted_tile_indices = sorted(tiles_to_correct)
    logger.info(
        f"  Cycle '{cycle}': re-registering {len(sorted_tile_indices)} leftover tile(s) "
        f"against fused inlier image."
    )
    try:
        registration.register(
            [msi_utils.get_msim_from_sim(sim_fused_inliers)]
            + [msims[i] for i in sorted_tile_indices],
            reg_channel=reg_channel,
            transform_key="fractal_input",
            new_transform_key="affine_registered",
            pre_registration_pruning_method=None,
            groupwise_resolution_kwargs={"reference_view": 0},
            reg_res_level=0,
        )
    except mv_graph.NotEnoughOverlapError:
        logger.warning(
            f"  Cycle '{cycle}': leftover tile registration failed (not enough overlap "
            f"with fused inlier); all leftover tiles will keep their stage position."
        )
        for tile_idx in sorted_tile_indices:
            msim = msims[tile_idx]
            sim = msi_utils.get_sim_from_msim(msim)
            matrix = _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key="fractal_input")
            )
            xaffine = param_utils.affine_to_xaffine(matrix, t_coords=[0])
            msi_utils.set_affine_transform(msim, xaffine, "affine_registered")


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
    reg_channel = _resolve_registration_channel(reg_image_ref, init_args.reference_channel)
    logger.info(f"Reference cycle: '{ref_cycle}', registration channel: '{reg_channel}'")

    # ------------------------------------------------------------------
    # Step 1: Load all FOVs from each cycle at the registration pyramid
    # level, optionally projecting along z.
    # ------------------------------------------------------------------
    logger.info(
        f"Loading FOVs at pyramid level {init_args.pyramid_level}"
        f"{' (z-projected)' if z_project else ''}"
    )
    msims_reg = {}
    for cycle in cycles:
        reg_image = containers[cycle].get_image(path=str(init_args.pyramid_level))
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        msims_reg[cycle] = _get_msims(
            image=reg_image, fov_roi_table=fov_roi_table, z_project=z_project
        )
        logger.info(f"  Cycle '{cycle}': loaded {len(msims_reg[cycle])} FOV(s)")

    # ------------------------------------------------------------------
    # Step 2: Stitch the reference cycle and fuse into a masked reference
    # image used as the fixed target for per-tile registration.
    # ------------------------------------------------------------------
    logger.info(
        f"Stitching reference cycle '{ref_cycle}' ({len(msims_reg[ref_cycle])} tiles)"
    )
    sim_fused_ref_ds = _stitch_and_fuse_reference(msims_reg[ref_cycle], reg_channel)
    logger.info("  Reference stitching and fusion complete.")

    # ------------------------------------------------------------------
    # Step 3: Register each non-reference cycle's tiles against the fused
    # reference. Tiles without spatial overlap are deferred to Step 4.
    # ------------------------------------------------------------------
    logger.info(
        f"Registering {len(cycles) - 1} non-reference cycle(s) against fused reference."
    )
    no_overlap_indices: dict[str, list[int]] = {}
    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        logger.debug(
            f"  Queuing {len(msims_reg[cycle])} tile registration task(s) "
            f"for cycle '{cycle}'"
        )
        no_overlap = _register_cycle_tiles(
            msims_reg[cycle], sim_fused_ref_ds, reg_channel, msims_reg[ref_cycle]
        )
        no_overlap_indices[cycle] = no_overlap
        if no_overlap:
            logger.warning(
                f"  Cycle '{cycle}': {len(no_overlap)} tile(s) had no spatial "
                f"overlap with the reference and will be re-registered in Step 4."
            )
    logger.info("  Parallel tile registration complete.")

    # ------------------------------------------------------------------
    # Step 4: For each non-reference cycle, detect outlier tiles (shifts
    # that deviate too much from the cycle mean) and collect no-overlap
    # tiles. Re-register all such leftover tiles against a fused image of
    # the remaining inlier tiles, with the fused inlier held fixed.
    # ------------------------------------------------------------------
    osc = init_args.outlier_filter
    if osc.mode == "zscore":
        logger.info(f"Outlier detection enabled (z-score threshold: {osc.threshold}).")
    elif osc.mode != "disabled":
        logger.info(f"Outlier detection enabled (absolute threshold: {osc.threshold} µm).")

    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        no_overlap_set = set(no_overlap_indices.get(cycle, []))
        reg_tile_indices, shifts = _collect_shifts(msims_reg[cycle], no_overlap_set)
        outlier_indices = _detect_outlier_tiles(shifts, reg_tile_indices, osc, cycle)
        tiles_to_correct = no_overlap_set | outlier_indices
        _register_leftover_tiles(msims_reg[cycle], tiles_to_correct, reg_channel, cycle)

    # ------------------------------------------------------------------
    # Step 5: Reload FOVs at full resolution and transfer the computed
    # transforms. Expand 2D affines to 3D when z_project was used.
    # ------------------------------------------------------------------
    logger.info("Reloading FOVs at full resolution and transferring transforms.")
    msims_fusion = {}
    for cycle in cycles:
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        msims_fusion[cycle] = _get_msims(
            image=containers[cycle].get_image(),
            fov_roi_table=fov_roi_table,
            z_project=False,
            channel_suffix=f"_{cycle}",
        )
        for msim_reg, msim_fus in zip(msims_reg[cycle], msims_fusion[cycle], strict=True):
            affine = msi_utils.get_transform_from_msim(msim_reg, "affine_registered")
            if z_project:
                affine_3d = param_utils.identity_transform(
                    ndim=3,
                    t_coords=affine.coords["t"] if "t" in affine.dims else None,
                )
                affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = affine
                affine = affine_3d
            msi_utils.set_affine_transform(msim_fus, affine, "affine_registered")

    # ------------------------------------------------------------------
    # Step 6: Determine the global bounding box across all cycles and
    # fuse every cycle into a shared output canvas.
    # ------------------------------------------------------------------
    logger.info("Computing global bounding box and fusing all cycles.")
    origins_all, antipodes_all = [], []
    for cycle in cycles:
        for msim in msims_fusion[cycle]:
            sim = msi_utils.get_sim_from_msim(msim)
            origins_all.append(_get_origin_of_sim(sim, transform_key="affine_registered"))
            antipodes_all.append(_get_antipode_of_sim(sim, transform_key="affine_registered"))

    spacing_ref = get_spacing_from_sim(
        msi_utils.get_sim_from_msim(msims_fusion[ref_cycle][0]), asarray=False
    )
    global_origin = {
        dim: min(o[dim] for o in origins_all) for dim in origins_all[0]
    }
    global_shape = {
        dim: int(
            np.ceil(
                (max(a[dim] for a in antipodes_all) - global_origin[dim])
                / spacing_ref[dim]
            )
        )
        for dim in global_origin
    }
    logger.info(
        f"  Global output shape: {global_shape}, "
        f"origin: {{k: round(v, 3) for k, v in global_origin.items()}}"
    )

    sims_fused = {}
    for cycle in cycles:
        logger.info(f"  Fusing cycle '{cycle}'...")
        sims_fused[cycle] = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim) for msim in msims_fusion[cycle]],
            transform_key="affine_registered",
            output_chunksize=1024,
            output_origin=global_origin,
            output_shape=global_shape,
        )

    sim_fused_all = xr.concat([sims_fused[cycle] for cycle in cycles], dim="c")
    axes_in = containers[ref_cycle].get_image().axes
    sim_fused_all = sim_fused_all.squeeze(
        [dim for dim in sim_fused_all.dims if dim not in axes_in]
    )

    # ------------------------------------------------------------------
    # Step 7: Write the fused image to the output OME-Zarr store.
    # ------------------------------------------------------------------
    logger.info(
        f"Writing fused image to '{zarr_url}' "
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
        overwrite=True,
    )
    out_image = output_container.get_image()
    out_image.set_array(patch=sim_fused_all.data, axes_order=sim_fused_all.dims)
    out_image.consolidate()
    logger.info("  Output image written and consolidated successfully.")

    image_list_updates = [
        {
            "zarr_url": zarr_url,
            "origin": init_args.zarr_urls_to_register[0],
            # TODO: ask Joel how to handle passing the acquisition attribute here
        }
    ]

    if init_args.keep_original_acquisitions:
        logger.info("Keeping original acquisitions. Task complete.")
        return {"image_list_updates": image_list_updates}

    logger.info(
        f"Removing {len(init_args.zarr_urls_to_register)} original acquisition(s)..."
    )
    for url in init_args.zarr_urls_to_register:
        logger.info(f"  Deleting original acquisition at '{url}'")
        shutil.rmtree(url)
    logger.info("Task complete.")
    return {
        "image_list_updates": image_list_updates,
        "image_list_removals": init_args.zarr_urls_to_register,
    }


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitch_and_register_parallel)

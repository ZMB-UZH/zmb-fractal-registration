"""Fractal task to stitch and register multiple acquisitions."""

# TODO:
# - add option to get initial positions from grid alignment instead of original stage
#   positions in metadata
# - add option to input different ROI table
# - handle larger shifts between cycles by performing a pre-registration step
# - optimize dask parallelization
# - clean up function structure and break into smaller functions where possible

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


def _get_origin_of_sim(sim, transform_key: str | None = None) -> dict[str, float]:
    """Get transformed origin for a spatial image."""
    ndim = get_ndim_from_sim(sim)
    origin = get_origin_from_sim(sim, asarray=False)
    origin = np.array([origin[dim] for dim in get_spatial_dims_from_sim(sim)])

    if transform_key is not None:
        affine = get_affine_from_sim(sim, transform_key=transform_key)
        sel_dict = {
            dim: affine.coords[dim][0].values
            for dim in affine.dims
            if dim not in ["x_in", "x_out"]
        }
        affine = np.array(affine.sel(sel_dict))
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
        affine = get_affine_from_sim(sim, transform_key=transform_key)
        sel_dict = {
            dim: affine.coords[dim][0].values
            for dim in affine.dims
            if dim not in ["x_in", "x_out"]
        }
        affine = np.array(affine.sel(sel_dict))
        antipode = np.concatenate([antipode, np.ones(1)])
        antipode = np.matmul(affine, antipode)[:ndim]

    return dict(zip(get_spatial_dims_from_sim(sim), antipode, strict=True))


def _xaffine_to_matrix(xaffine: xr.DataArray) -> np.ndarray:
    """Extract a (ndim+1, ndim+1) numpy matrix from an affine DataArray.

    Selects the first coordinate along any non-spatial dimensions (e.g. 't').
    """
    sel_dict = {
        dim: xaffine.coords[dim][0].values
        for dim in xaffine.dims
        if dim not in ["x_in", "x_out"]
    }
    return np.array(xaffine.sel(sel_dict))


def _resolve_registration_channel(image, selector: ChannelSelectionModel) -> str:
    """Resolve a ChannelSelectionModel to a channel label for registration."""
    if selector.mode == "index":
        return image.channel_labels[int(selector.identifier)]
    if selector.mode == "wavelength_id":
        idx = image.get_channel_idx(selector.identifier)
        return image.channel_labels[idx]
    return selector.identifier


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

    logger.info(f"Opening OME-Zarr containers for cycles: {init_args.cycle_names}")
    containers = {
        cycle: open_ome_zarr_container(url)
        for cycle, url in zip(
            init_args.cycle_names, init_args.zarr_urls_to_register, strict=True
        )
    }
    cycles = list(init_args.cycle_names)
    ref_cycle = cycles[init_args.reference_acquisition_index]
    logger.info(f"Reference cycle: {ref_cycle} (registration channel: resolved below)")

    reg_image_ref = containers[ref_cycle].get_image(path=str(init_args.pyramid_level))
    reg_channel = _resolve_registration_channel(
        reg_image_ref, init_args.reference_channel
    )
    logger.info(f"Resolved registration channel: '{reg_channel}'")
    z_project = init_args.z_project

    # ------------------------------------------------------------------
    # Load all FOVs from each cycle at the registration pyramid
    # level, optionally projecting along z.
    # ------------------------------------------------------------------
    logger.info(
        f"Loading FOVs at pyramid level {init_args.pyramid_level}"
        f"{' (z-projected)' if z_project else ''}"
    )
    msims_reg = {}
    for cycle in cycles:
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        reg_image = containers[cycle].get_image(path=str(init_args.pyramid_level))
        msims_reg[cycle] = _get_msims(
            image=reg_image,
            fov_roi_table=fov_roi_table,
            z_project=z_project,
        )
        logger.info(f"  Cycle '{cycle}': loaded {len(msims_reg[cycle])} FOV(s)")

    # ------------------------------------------------------------------
    # Stitch the reference cycle - align its tiles relative to
    # each other using the stage positions as the starting point.
    # ------------------------------------------------------------------
    logger.info(
        f"Stitching reference cycle '{ref_cycle}' ({len(msims_reg[ref_cycle])} tiles)"
    )
    registration.register(
        msims_reg[ref_cycle],
        reg_channel=reg_channel,
        transform_key="fractal_input",
        new_transform_key="affine_registered",
        pre_registration_pruning_method="keep_axis_aligned",
    )

    logger.info(
        "  Reference cycle stitching complete. Fusing reference tiles (lazy)..."
    )
    # Fuse the stitched reference tiles into a single reference image (lazily and at the
    # sub-resolution). This is then used as the fixed target for registering the tiles
    # of the other cycles.
    sim_fused_ref_ds = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims_reg[ref_cycle]],
        transform_key="affine_registered",
        output_chunksize=1024,  # TODO: optimize chunksize
    )
    sim_fused_ref_ds.transforms["fractal_input"] = sim_fused_ref_ds.transforms[
        "affine_registered"
    ]
    # Mask out regions not covered by any tile (outside the stitched FOV).
    sim_fused_ref_ds_mask = fusion.fuse(
        [
            xr.ones_like(msi_utils.get_sim_from_msim(msim))
            for msim in msims_reg[ref_cycle]
        ],
        transform_key="affine_registered",
        output_chunksize=1024,
    )
    sim_fused_ref_ds = xr.where(sim_fused_ref_ds_mask > 0, sim_fused_ref_ds, np.nan)

    # ------------------------------------------------------------------
    # Register each non-reference cycle tile individually against
    # the fused reference image. Tiles across all cycles are registered
    # in parallel using dask.
    # ------------------------------------------------------------------
    logger.info(
        f"Registering {len(cycles) - 1} non-reference cycle(s) against fused reference."
    )
    # Tracks per-cycle indices of tiles with no spatial overlap with the
    # reference; their transforms are set later together with outlier correction.
    no_overlap_indices: dict[str, list[int]] = {}
    delayed_tasks = []
    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        no_overlap_indices[cycle] = []
        logger.debug(
            f"  Queuing {len(msims_reg[cycle])} tile registration task(s) "
            f"for cycle '{cycle}'"
        )
        for i, msim in enumerate(msims_reg[cycle]):
            # Check overlap before registering. Defer transform assignment
            # to the outlier correction step so these tiles receive the inlier
            # mean shift rather than a fallback stage position.
            if not _has_overlap_with_reference_tiles(
                msim,
                msims_reg[ref_cycle],
                transform_key="fractal_input",
                ref_transform_key="affine_registered",
            ):
                logger.warning(
                    f"  Cycle '{cycle}', tile {i}: no spatial overlap with any "
                    f"reference tile. Deferring transform to outlier correction."
                )
                no_overlap_indices[cycle].append(i)
                continue
            task = delayed(registration.register)(
                [msi_utils.get_msim_from_sim(sim_fused_ref_ds), msim],
                reg_channel=reg_channel,
                transform_key="fractal_input",
                new_transform_key="affine_registered",
                pre_registration_pruning_method=None,
                groupwise_resolution_kwargs={"reference_view": 0},
                reg_res_level=0,
            )
            delayed_tasks.append(task)
    logger.info(
        f"  Running {len(delayed_tasks)} tile registration task(s) in parallel..."
    )
    compute(*delayed_tasks)
    logger.info("  Parallel tile registration complete.")

    # ------------------------------------------------------------------
    # Shift correction - for each non-reference cycle:
    #   1. If outlier correction is enabled, detect tiles whose shift
    #      deviates too much and add them to the set of tiles to correct.
    #   2. No-overlap tiles (which were skipped during registration) are
    #      always added to that set.
    #   3. Compute the mean shift from the remaining inlier registered tiles.
    #   4. Apply the mean shift to all tiles in the set.
    # ------------------------------------------------------------------
    # TODO: review this part
    # TODO: correct not with mean shift of all inliers, but interpolate from nearby
    # inliers
    osc = init_args.outlier_filter
    if osc.mode == "zscore":
        logger.info(f"Outlier shift correction (z-score threshold: {osc.threshold}).")
    elif osc.mode != "disabled":
        logger.info(
            f"Outlier shift correction (absolute threshold: {osc.threshold} µm)."
        )

    for cycle in cycles:
        if cycle == ref_cycle:
            continue

        no_overlap_set = set(no_overlap_indices.get(cycle, []))

        # Collect shifts for all registered (overlapping) tiles.
        reg_tile_indices = []
        shifts = []
        for i, msim in enumerate(msims_reg[cycle]):
            if i in no_overlap_set:
                continue
            reg_tile_indices.append(i)
            sim = msi_utils.get_sim_from_msim(msim)
            t_reg = param_utils.translation_from_affine(
                _xaffine_to_matrix(get_affine_from_sim(sim, transform_key="affine_registered"))
            )
            t_in = param_utils.translation_from_affine(
                _xaffine_to_matrix(get_affine_from_sim(sim, transform_key="fractal_input"))
            )
            shifts.append(t_reg - t_in)

        # Build the set of tiles to correct (always includes no-overlap tiles).
        tiles_to_correct = set(no_overlap_set)

        if not shifts:
            logger.warning(
                f"  Cycle '{cycle}': no registered tiles; cannot compute "
                f"inlier mean shift. All tiles will keep their stage position."
            )
            mean_shift = None
        else:
            inlier_mask = [True] * len(shifts)

            if osc.mode != "disabled":
                # Euclidean distance of each tile's shift from the mean.
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

                for list_idx, tile_idx in enumerate(reg_tile_indices):
                    if is_outlier[list_idx]:
                        logger.warning(
                            f"  Cycle '{cycle}', tile {tile_idx}: shift "
                            f"{np.round(shifts[list_idx], 3)} "
                            f"({score_label}={scores[list_idx]:.2f}) "
                            f"flagged as outlier."
                        )
                        inlier_mask[list_idx] = False
                        tiles_to_correct.add(tile_idx)

            inlier_shifts = [
                s for s, inlier in zip(shifts, inlier_mask, strict=True) if inlier
            ]

            if inlier_shifts:
                mean_shift = np.mean(inlier_shifts, axis=0)
            else:
                logger.warning(
                    f"  Cycle '{cycle}': all registered tiles flagged as outliers; "
                    f"falling back to full-tile mean."
                )
                mean_shift = np.mean(shifts, axis=0)

            logger.info(
                f"  Cycle '{cycle}': inlier mean shift (µm) = "
                f"{np.round(mean_shift, 3)} "
                f"({len(inlier_shifts)}/{len(shifts)} inlier registered tiles)"
            )

        # Apply mean shift to all tiles that need correction.
        for tile_idx in sorted(tiles_to_correct):
            msim = msims_reg[cycle][tile_idx]
            sim = msi_utils.get_sim_from_msim(msim)
            affine_in_xr = get_affine_from_sim(sim, transform_key="fractal_input")
            t_coords = affine_in_xr.coords["t"].values if "t" in affine_in_xr.dims else None
            a = _xaffine_to_matrix(affine_in_xr)
            if mean_shift is not None:
                if tile_idx in no_overlap_set:
                    logger.warning(
                        f"  Cycle '{cycle}', tile {tile_idx}: no overlap with "
                        f"reference tiles. Applying inlier mean shift "
                        f"{np.round(mean_shift, 3)}."
                    )
                else:
                    logger.warning(
                        f"  Cycle '{cycle}', tile {tile_idx}: replacing outlier "
                        f"shift with inlier mean {np.round(mean_shift, 3)}."
                    )
                a = a.copy()
                a[: len(mean_shift), -1] = param_utils.translation_from_affine(a) + mean_shift
            else:
                logger.warning(
                    f"  Cycle '{cycle}', tile {tile_idx}: no overlap with "
                    f"reference tiles and no inlier mean available. "
                    f"Keeping stage position."
                )
            msi_utils.set_affine_transform(
                msim,
                param_utils.affine_to_xaffine(a, t_coords=t_coords),
                "affine_registered",
            )

    # ------------------------------------------------------------------
    # Reload FOVs at full resolution and transfer the computed
    # transforms. When z_project was used, expand the 2D affine back to
    # 3D (identity in z) before applying it to the full volume.
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
    # Determine the bounding box that contains all transformed
    # tiles across all cycles, then fuse each cycle into that canvas.
    # ------------------------------------------------------------------
    logger.info("Computing global bounding box and fusing all cycles.")
    origins_all = []
    antipodes_all = []
    for cycle in cycles:
        for msim in msims_fusion[cycle]:
            sim = msi_utils.get_sim_from_msim(msim)
            origins_all.append(
                _get_origin_of_sim(sim, transform_key="affine_registered")
            )
            antipodes_all.append(
                _get_antipode_of_sim(sim, transform_key="affine_registered")
            )

    global_origin = {}
    global_shape = {}
    spacing_ref = get_spacing_from_sim(
        msi_utils.get_sim_from_msim(msims_fusion[ref_cycle][0]), asarray=False
    )
    for dim in origins_all[0]:
        global_origin[dim] = min(origin[dim] for origin in origins_all)
        global_antipode = max(antipode[dim] for antipode in antipodes_all)
        global_shape[dim] = int(
            np.ceil((global_antipode - global_origin[dim]) / spacing_ref[dim])
        )

    rounded_origin = {k: round(v, 3) for k, v in global_origin.items()}
    logger.info(f"  Global output shape: {global_shape}, origin: {rounded_origin}")
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

    # Concatenate all cycles along the channel axis and drop any singleton
    # dimensions that are not present in the reference image axes.
    sim_fused_all = xr.concat([sims_fused[cycle] for cycle in cycles], dim="c")
    axes_in = containers[ref_cycle].get_image().axes
    sim_fused_all = sim_fused_all.squeeze(
        [dim for dim in sim_fused_all.dims if dim not in axes_in]
    )

    # ------------------------------------------------------------------
    # Write the fused image to the output OME-Zarr store.
    # ------------------------------------------------------------------
    logger.info(
        f"Writing fused image to '{zarr_url}' "
        f"(shape: {sim_fused_all.shape}, dims: {sim_fused_all.dims})."
    )
    # Build per-channel metadata, appending _{cycle} to each label so
    # channels from different cycles can be distinguished.
    channels_meta_all = []
    for cycle in cycles:
        channels = containers[cycle].images_container.channels_meta.channels
        for channel in channels:
            channels_meta_all.append(
                Channel(
                    label=f"{channel.label}_{cycle}",
                    wavelength_id=channel.wavelength_id,
                    channel_visualisation=channel.channel_visualisation,
                )
            )

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

    # Remove the original per-cycle acquisitions from disk.
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

"""Fractal task to stitch and register multiple acquisitions."""

# TODO:
# - add option to get initial positions from grid alignment instead of original stage
#   positions in metadata
# - add option to input different ROI table
# - optimize dask parallelization
# - consider materializing intermediate fused images to a temporary zarr instead
#   of holding them in memory. The fused reference in Steps 3-4 stays lazy and
#   is only ever computed chunk-wise, but the pre-registration in Step 2
#   computes one shared canvas into memory, and each cycle's chunks are
#   recomputed once per tile registered against them (dask cannot share results
#   across the delayed tasks, since register() computes internally). Writing
#   those to scratch with da.store and reading them back lazily would remove the
#   recomputation and bound the memory at any pyramid level, at the cost of disk
#   I/O and temp-file handling.
# - interpolation_order=0 leaves a 1-pixel zero frame around the outer edge of a
#   fused cycle whenever the output canvas is not aligned to that cycle's pixel
#   grid (i.e. whenever the registration shift is sub-pixel). The frame follows
#   the canvas edge, not the tile edge, so shrinking the output box does not
#   remove it - meaning fusion_region='intersection_bbox' cannot fully guarantee
#   "no pixels set to 0" at order 0. order=1 is unaffected. Fixing it likely
#   means snapping global_origin onto the reference cycle's pixel grid (which
#   would also stop order-0 nearest-neighbour from introducing up to half a
#   pixel of jitter per tile), though non-reference cycles cannot be aligned at
#   the same time. Possibly an off-by-one in multiview-stitcher's order-0
#   resampling path - worth checking upstream first.

import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import xarray as xr
from multiview_stitcher import fusion, msi_utils, param_utils
from multiview_stitcher.spatial_image_utils import get_spacing_from_sim
from ngio import ChannelSelectionModel, open_ome_zarr_container
from ngio.ome_zarr_meta import Channel
from pydantic import BaseModel, validate_call

from zmb_fractal_registration._stitch_register.loading import (
    _get_msims,
    _resolve_registration_channel,
    _validate_registration_channel,
)
from zmb_fractal_registration._stitch_register.output_bbox import _compute_global_bbox
from zmb_fractal_registration._stitch_register.pre_registration import (
    PREREG_TRANSFORM_KEY,
    _pre_register_cycles,
)
from zmb_fractal_registration._stitch_register.registration import (
    _collect_shifts,
    _detect_outlier_tiles,
    _output_chunksize,
    _register_cycle_tiles,
    _register_leftover_tiles,
    _stitch_and_fuse_reference,
)
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
        pre_registration: If True, roughly align whole cycles against the
            reference cycle before the accurate stitching/registration. Each
            cycle is fused from its stage coordinates at the coarsest pyramid
            level and registered as a whole, assuming all cycles cover roughly
            the same area.
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
    pre_registration: bool = False
    keep_original_acquisitions: bool = True
    tile_correction: TileCorrectionModel = TileCorrectionModel()
    fusion_region: Literal["union", "intersection", "intersection_bbox"] = "union"
    interpolation_order: int = 0


def _load_registration_msims(
    containers: dict, cycles: list[str], pyramid_level: int, z_project: bool
) -> dict[str, list]:
    """Step 1: load all FOVs per cycle at the registration pyramid level."""
    logger.info(
        f"[Step 1/8] Loading FOVs at pyramid level {pyramid_level}"
        f"{' (z-projected)' if z_project else ''}."
    )
    msims_reg = {}
    for cycle in cycles:
        reg_image = containers[cycle].get_image(path=str(pyramid_level))
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        msims_reg[cycle] = _get_msims(
            image=reg_image, fov_roi_table=fov_roi_table, z_project=z_project
        )
        logger.info(f"Cycle '{cycle}': loaded {len(msims_reg[cycle])} FOV(s).")
    return msims_reg


def _register_cycles_to_reference(
    msims_reg: dict[str, list],
    cycles: list[str],
    ref_cycle: str,
    sim_fused_ref,
    reg_channel: str,
    init_transform_key: str,
) -> dict[str, list[int]]:
    """Step 4: register each non-reference cycle's tiles against the reference.

    Returns the indices of tiles per cycle that had no overlap with the
    reference and are deferred to Step 5.
    """
    logger.info(
        f"[Step 4/8] Registering {len(cycles) - 1} non-reference cycle(s) "
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
            msims_reg[cycle],
            sim_fused_ref,
            reg_channel,
            msims_reg[ref_cycle],
            init_transform_key,
        )
        no_overlap_indices[cycle] = no_overlap
        n_reg = len(msims_reg[cycle]) - len(no_overlap)
        logger.info(
            f"Cycle '{cycle}': {n_reg} tile(s) registered"
            + (
                f"; {len(no_overlap)} had no overlap with the reference "
                "(deferred to Step 5)."
                if no_overlap
                else "."
            )
        )
    logger.info("Tile registration complete.")
    return no_overlap_indices


def _correct_leftover_tiles(
    msims_reg: dict[str, list],
    cycles: list[str],
    ref_cycle: str,
    no_overlap_indices: dict[str, list[int]],
    tile_correction: TileCorrectionModel,
    reg_channel: str,
    init_transform_key: str,
) -> None:
    """Step 5: detect outlier tiles and re-register leftover tiles per cycle."""
    tcm = tile_correction
    _outlier_desc = tcm.outlier_filter_mode + (
        f" (threshold={tcm.threshold})" if tcm.outlier_filter_mode != "disabled" else ""
    )
    logger.info(
        f"[Step 5/8] Correcting leftover tiles "
        f"(outlier detection: {_outlier_desc}, correction: {tcm.correction_method})."
    )

    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        no_overlap_set = set(no_overlap_indices.get(cycle, []))
        reg_tile_indices, shifts = _collect_shifts(
            msims_reg[cycle], no_overlap_set, init_transform_key
        )
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
            init_transform_key,
        )


def _transfer_transforms_to_full_res(
    containers: dict, msims_reg: dict[str, list], cycles: list[str], z_project: bool
) -> dict[str, list]:
    """Step 6: reload FOVs at full resolution and transfer the transforms.

    Expands 2D affines to 3D when z_project was used during registration.
    """
    logger.info(
        "[Step 6/8] Reloading FOVs at full resolution and transferring transforms."
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
    return msims_fusion


def _fuse_cycles(
    containers: dict,
    msims_fusion: dict[str, list],
    cycles: list[str],
    ref_cycle: str,
    fusion_region: str,
    interpolation_order: int,
):
    """Step 7: compute the global bounding box and fuse all cycles into one image.

    For 'intersection', pixels not covered by every cycle inside the box are set
    to 0.
    """
    logger.info(
        f"[Step 7/8] Computing global bounding box ({fusion_region}) and "
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
        # Inherited from the input tiles, so the output is chunked like the
        # images it was built from (this is also the on-disk chunking).
        chunksize = _output_chunksize(cycle_sims)
        logger.info(f"Cycle '{cycle}': output chunks {chunksize}.")
        sims_fused[cycle] = fusion.fuse(
            cycle_sims,
            transform_key="affine_registered",
            interpolation_order=interpolation_order,
            output_chunksize=chunksize,
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
                output_chunksize=chunksize,
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
    return sim_fused_all


def _write_fused_image(
    containers: dict,
    ref_cycle: str,
    cycles: list[str],
    zarr_url: str,
    sim_fused_all,
) -> None:
    """Step 8: write the fused image to the output OME-Zarr store."""
    logger.info(
        f"[Step 8/8] Writing fused image to '{zarr_url}' "
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
        f"pyramid_level: {init_args.pyramid_level}, z_project: {init_args.z_project}, "
        f"pre_registration: {init_args.pre_registration})"
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
    _validate_registration_channel(
        containers, cycles, init_args.pyramid_level, reg_channel
    )

    # Step 1: load FOVs at the registration pyramid level.
    msims_reg = _load_registration_msims(
        containers, cycles, init_args.pyramid_level, z_project
    )

    # Step 2: roughly pre-register whole cycles against the reference cycle.
    # The rough result is stored under its own transform key, which the
    # following steps then start from instead of the raw stage coordinates.
    if init_args.pre_registration:
        logger.info(
            f"[Step 2/8] Rough pre-registration of {len(cycles) - 1} non-reference "
            f"cycle(s) at the coarsest pyramid level."
        )
        _pre_register_cycles(
            containers, msims_reg, cycles, ref_cycle, reg_channel, z_project
        )
        init_transform_key = PREREG_TRANSFORM_KEY
        logger.info("Rough pre-registration complete.")
    else:
        init_transform_key = "fractal_input"
        logger.info("[Step 2/8] Rough pre-registration disabled; skipping.")

    # Step 3: stitch the reference cycle into a masked reference image.
    logger.info(
        f"[Step 3/8] Stitching reference cycle '{ref_cycle}' "
        f"({len(msims_reg[ref_cycle])} tile(s))."
    )
    sim_fused_ref = _stitch_and_fuse_reference(
        msims_reg[ref_cycle], reg_channel, init_transform_key
    )
    logger.info("Reference stitching and fusion complete.")

    # Step 4: register each non-reference cycle against the fused reference.
    no_overlap_indices = _register_cycles_to_reference(
        msims_reg, cycles, ref_cycle, sim_fused_ref, reg_channel, init_transform_key
    )

    # Step 5: correct outlier and no-overlap (leftover) tiles.
    _correct_leftover_tiles(
        msims_reg,
        cycles,
        ref_cycle,
        no_overlap_indices,
        init_args.tile_correction,
        reg_channel,
        init_transform_key,
    )

    # Step 6: reload at full resolution and transfer the computed transforms.
    msims_fusion = _transfer_transforms_to_full_res(
        containers, msims_reg, cycles, z_project
    )

    # Step 7: fuse every cycle into a shared output canvas.
    sim_fused_all = _fuse_cycles(
        containers,
        msims_fusion,
        cycles,
        ref_cycle,
        init_args.fusion_region,
        init_args.interpolation_order,
    )

    # Step 8: write the fused image to the output OME-Zarr store.
    _write_fused_image(containers, ref_cycle, cycles, zarr_url, sim_fused_all)

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

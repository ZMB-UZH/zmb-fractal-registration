"""Fractal task to stitch and register multiple acquisitions."""

# TODO:
# - add option to get initial positions from grid alignment instead of original stage
#   positions in metadata
# - add option to input different ROI table
# - handle larger shifts between cycles by performing a pre-registration step

import logging
import shutil
from typing import Any

import numpy as np
import xarray as xr
from multiview_stitcher import (
    fusion,
    msi_utils,
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
    """

    zarr_urls_to_register: list[str]
    cycle_names: list[str]
    reference_acquisition_index: int
    reference_channel: ChannelSelectionModel
    pyramid_level: int = 0
    z_project: bool = True
    keep_original_acquisitions: bool = True


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


def _resolve_registration_channel(image, selector: ChannelSelectionModel) -> str:
    """Resolve a ChannelSelectionModel to a channel label for registration."""
    if selector.mode == "index":
        return image.channel_labels[int(selector.identifier)]
    if selector.mode == "wavelength_id":
        idx = image.get_channel_idx(selector.identifier)
        return image.channel_labels[idx]
    return selector.identifier


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
    if len(init_args.zarr_urls_to_register) < 2:
        raise ValueError("At least two acquisitions are required for registration.")

    if len(init_args.cycle_names) != len(init_args.zarr_urls_to_register):
        raise ValueError("cycle_names length must match zarr_urls_to_register length.")

    if not (0 <= init_args.reference_acquisition_index < len(init_args.cycle_names)):
        raise ValueError("reference_acquisition_index is out of range.")

    containers = {
        cycle: open_ome_zarr_container(url)
        for cycle, url in zip(
            init_args.cycle_names, init_args.zarr_urls_to_register, strict=True
        )
    }
    cycles = list(init_args.cycle_names)
    ref_cycle = cycles[init_args.reference_acquisition_index]

    reg_image_ref = containers[ref_cycle].get_image(path=str(init_args.pyramid_level))
    reg_channel = _resolve_registration_channel(
        reg_image_ref, init_args.reference_channel
    )
    z_project = init_args.z_project

    msims_reg = {}
    for cycle in cycles:
        fov_roi_table = containers[cycle].get_table("FOV_ROI_table")
        reg_image = containers[cycle].get_image(path=str(init_args.pyramid_level))
        msims_reg[cycle] = _get_msims(
            image=reg_image,
            fov_roi_table=fov_roi_table,
            z_project=z_project,
        )

    registration.register(
        msims_reg[ref_cycle],
        reg_channel=reg_channel,
        transform_key="fractal_input",
        new_transform_key="affine_registered",
        pre_registration_pruning_method="keep_axis_aligned",
    )

    sim_fused_ref_ds = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims_reg[ref_cycle]],
        transform_key="affine_registered",
        output_chunksize=1024,
    )
    sim_fused_ref_ds.transforms["fractal_input"] = sim_fused_ref_ds.transforms[
        "affine_registered"
    ]
    sim_fused_ref_ds_mask = fusion.fuse(
        [
            xr.ones_like(msi_utils.get_sim_from_msim(msim))
            for msim in msims_reg[ref_cycle]
        ],
        transform_key="affine_registered",
        output_chunksize=1024,
    )
    sim_fused_ref_ds = xr.where(sim_fused_ref_ds_mask > 0, sim_fused_ref_ds, np.nan)

    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        for msim in msims_reg[cycle]:
            registration.register(
                [msi_utils.get_msim_from_sim(sim_fused_ref_ds), msim],
                reg_channel=reg_channel,
                transform_key="fractal_input",
                new_transform_key="affine_registered",
                pre_registration_pruning_method=None,
                groupwise_resolution_kwargs={"reference_view": 0},
                reg_res_level=0,
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

    sims_fused = {}
    for cycle in cycles:
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

    image_list_updates = [
        {
            "zarr_url": zarr_url,
            "origin": init_args.zarr_urls_to_register[0],
        }
    ]

    if init_args.keep_original_acquisitions:
        logging.info("Keeping original acquisitions.")
        return {"image_list_updates": image_list_updates}

    for url in init_args.zarr_urls_to_register:
        logging.info(f"Deleting original acquisition at {url}")
        shutil.rmtree(url)

    return {
        "image_list_updates": image_list_updates,
        "image_list_removals": init_args.zarr_urls_to_register,
    }


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitch_and_register_parallel)

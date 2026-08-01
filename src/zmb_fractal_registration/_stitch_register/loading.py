"""Loading FOVs as multiscale spatial images and resolving channels."""

import logging

from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from ngio import ChannelSelectionModel, Roi
from ngio.tables import RoiTable

logger = logging.getLogger(__name__)


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


def _resolve_registration_channel(image, selector: ChannelSelectionModel) -> str:
    """Resolve a ChannelSelectionModel to a channel label for registration."""
    if selector.mode == "index":
        return image.channel_labels[int(selector.identifier)]
    if selector.mode == "wavelength_id":
        idx = image.get_channel_idx(selector.identifier)
        return image.channel_labels[idx]
    return selector.identifier


def _validate_registration_channel(
    containers: dict,
    cycles: list[str],
    pyramid_level: int,
    reg_channel: str,
) -> None:
    """Log each cycle's channel labels and check the registration channel exists.

    The registration channel is resolved on the reference acquisition only, but is
    then selected by label on every cycle. Acquisitions with differing channel sets
    would otherwise fail with an opaque KeyError deep inside the registration of the
    first offending cycle, after all preceding cycles have already been registered.
    """
    missing = []
    for cycle in cycles:
        labels = containers[cycle].get_image(path=str(pyramid_level)).channel_labels
        logger.info(f"Cycle '{cycle}': channels {labels}")
        if reg_channel not in labels:
            missing.append((cycle, labels))

    if missing:
        details = "; ".join(f"'{cycle}' has {labels}" for cycle, labels in missing)
        raise ValueError(
            f"Registration channel '{reg_channel}' is missing from "
            f"{len(missing)} of {len(cycles)} acquisition(s): {details}. "
            "Choose a channel that is present in every acquisition. Note that "
            "'index' mode resolves the channel on the reference acquisition only, "
            "so the same index may refer to a different channel in the others."
        )

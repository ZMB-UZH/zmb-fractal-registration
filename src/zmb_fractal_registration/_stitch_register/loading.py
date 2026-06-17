"""Loading FOVs as multiscale spatial images and resolving channels."""

from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from ngio import ChannelSelectionModel, Roi
from ngio.tables import RoiTable


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

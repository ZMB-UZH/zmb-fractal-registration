"""Low-level spatial-image and affine geometry helpers."""

import numpy as np
import xarray as xr
from multiview_stitcher.spatial_image_utils import (
    get_affine_from_sim,
    get_ndim_from_sim,
    get_origin_from_sim,
    get_shape_from_sim,
    get_spacing_from_sim,
    get_spatial_dims_from_sim,
)


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

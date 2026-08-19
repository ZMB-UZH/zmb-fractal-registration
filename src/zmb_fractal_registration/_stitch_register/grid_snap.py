"""Snapping registered tiles onto the output pixel grid for order-0 fusion.

Nearest-neighbour fusion (interpolation_order=0) quantizes every output pixel
to the nearest tile pixel, so a tile at a sub-pixel offset to the output grid
gains nothing from that offset: the resampling adds up to half a pixel of
jitter within the tile, and multiview-stitcher's order-0 path leaves one-pixel
zero seams at the canvas edge and at chunk borders (the per-chunk tile crop
has ``interpolation_order`` = 0 pixels of margin, and scipy maps coordinates
even slightly outside the crop to the fill value).

Rounding each tile's *position* to the output grid instead moves the tile
rigidly by at most half a pixel - the same error budget order 0 has anyway -
after which fusion is an exact integer-shift mosaic: original pixel values,
no jitter, no seams.

The tiles are also rebased into a local frame anchored at the lower corner of
their bounding box. multiview-stitcher rounds its resampling offsets to 10
decimals (in pixels), so the snapped positions must be integer multiples of
the spacing to within ~5e-11 px. At stage coordinates of ~1e5 um (a well far
out on a plate) double precision cannot hold that, and even perfectly
grid-aligned tiles come out with zero seams; at local coordinates spanning
only the imaged area it is exact. The world position was never written to the
fused output, so nothing downstream observes the rebase.
"""

import logging

import numpy as np
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.spatial_image_utils import get_affine_from_sim

from zmb_fractal_registration._stitch_register.sim_geometry import _xaffine_to_matrix

logger = logging.getLogger(__name__)


def _snap_msims_to_output_grid(
    msims_per_cycle: dict[str, list],
    cycles: list[str],
    spacing_ref: dict[str, float],
) -> dict[str, list]:
    """Return tiles rebased to a local frame and snapped onto one pixel lattice.

    The returned tiles carry their (snapped, local) position in their
    coordinates with an identity "affine_registered" transform, which is the
    same convention the rest of the pipeline uses for fixed tiles.

    Snapping requires every tile to be a pure translation with the reference
    pixel size; if any tile is not, the input is returned unchanged (with a
    warning) and fusion falls back to plain order-0 resampling.
    """
    # Validate all tiles and collect their world positions first: rebasing
    # only some tiles would place them in different frames.
    world_origins: dict[str, list[dict[str, float]]] = {}
    sdims: list[str] | None = None
    for cycle in cycles:
        world_origins[cycle] = []
        for msim in msims_per_cycle[cycle]:
            sim = msi_utils.get_sim_from_msim(msim)
            if sdims is None:
                sdims = si_utils.get_spatial_dims_from_sim(sim)
            ndim = len(sdims)
            matrix = _xaffine_to_matrix(
                get_affine_from_sim(sim, transform_key="affine_registered")
            )
            spacing = si_utils.get_spacing_from_sim(sim, asarray=False)
            if not np.allclose(matrix[:ndim, :ndim], np.eye(ndim), atol=1e-12):
                logger.warning(
                    f"Cycle '{cycle}': a tile's registered transform is not a "
                    "pure translation; skipping grid snapping for order-0 "
                    "fusion (sub-pixel resampling seams are possible)."
                )
                return msims_per_cycle
            # The spacing is derived from the coordinate arrays, so at large
            # world coordinates it carries float error (~1e-11 relative) and
            # cannot be compared exactly. 1e-6 still cleanly separates that
            # from a genuine pixel-size mismatch between acquisitions. The
            # snapped tiles are rebuilt with spacing_ref exactly, which
            # removes the derived error rather than propagating it.
            if any(
                not np.isclose(spacing[dim], spacing_ref[dim], rtol=1e-6, atol=0.0)
                for dim in sdims
            ):
                logger.warning(
                    f"Cycle '{cycle}': a tile's pixel size {spacing} differs "
                    f"from the reference {spacing_ref}; skipping grid snapping "
                    "for order-0 fusion."
                )
                return msims_per_cycle
            origin = si_utils.get_origin_from_sim(sim, asarray=False)
            translation = matrix[:ndim, ndim]
            world_origins[cycle].append(
                {dim: origin[dim] + translation[i] for i, dim in enumerate(sdims)}
            )

    anchor = {
        dim: min(w[dim] for cycle in cycles for w in world_origins[cycle])
        for dim in sdims
    }

    snapped: dict[str, list] = {}
    max_shift = 0.0
    for cycle in cycles:
        snapped[cycle] = []
        for msim, world in zip(
            msims_per_cycle[cycle], world_origins[cycle], strict=True
        ):
            sim = msi_utils.get_sim_from_msim(msim)
            local = {}
            for dim in sdims:
                k = round((world[dim] - anchor[dim]) / spacing_ref[dim])
                local[dim] = k * spacing_ref[dim]
                max_shift = max(max_shift, abs(local[dim] - (world[dim] - anchor[dim])))
            new_sim = si_utils.get_sim_from_array(
                sim.data,
                dims=list(sim.dims),
                scale={dim: spacing_ref[dim] for dim in sdims},
                translation=local,
                c_coords=list(sim.coords["c"].values) if "c" in sim.dims else None,
                transform_key="affine_registered",
            )
            snapped[cycle].append(
                msi_utils.get_msim_from_sim(new_sim, scale_factors=[])
            )

    logger.info(
        "Snapped all tiles onto the output pixel grid for order-0 fusion "
        f"(local frame anchor: { {d: round(v, 3) for d, v in anchor.items()} } um, "
        f"max tile shift: {max_shift:.4f} um)."
    )
    return snapped

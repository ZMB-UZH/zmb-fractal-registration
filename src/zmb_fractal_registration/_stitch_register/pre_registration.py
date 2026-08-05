"""Rough pre-registration of whole cycles against the reference cycle.

Every cycle is fused from its raw stage coordinates at the coarsest pyramid
level and registered as a whole against the fused reference cycle. The
resulting per-cycle correction is composed onto each tile's "fractal_input"
transform and stored under PREREG_TRANSFORM_KEY, which the accurate stitching
and registration then use as their starting position instead of the raw stage
coordinates. This is what makes shifts between cycles that are large compared
to the tile overlap recoverable.

The stage coordinates cannot be used to seed this registration - they are the
thing that is wrong. Instead every cycle is anchored on the origin of its own
tile bounding box (i.e. the cycles are assumed to cover roughly the same area)
and fused onto one shared canvas that is the union of all cycle extents. Both
images handed to the registration therefore have identical geometry, so the
region multiview-stitcher registers on is the full canvas and no data is
cropped away, no matter how far apart the stage coordinates place the cycles.
The canvas is built at origin 0 and the world placement is kept in the
transforms, which keeps the large stage offsets out of the pixel coordinates.

"fractal_input" itself is never modified, so the raw stage position of every
tile stays available (and keeps meaning the same thing as on the full
resolution tiles rebuilt for fusion).
"""

import logging

import numpy as np
from multiview_stitcher import msi_utils, mv_graph, param_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.spatial_image_utils import get_affine_from_sim

from zmb_fractal_registration._stitch_register.loading import _get_msims
from zmb_fractal_registration._stitch_register.registration import _fuse_masked
from zmb_fractal_registration._stitch_register.sim_geometry import (
    _get_antipode_of_sim,
    _get_origin_of_sim,
    _xaffine_to_matrix,
)

logger = logging.getLogger(__name__)

# Transform key holding each tile's roughly pre-registered position, i.e.
# "fractal_input" with the whole-cycle correction applied. Written for every
# cycle (identity correction for the reference and for cycles whose
# pre-registration failed), so downstream steps can rely on it existing.
PREREG_TRANSFORM_KEY = "fractal_input_preregistered"

# Transform key placing a cycle's tiles relative to the origin of its own
# bounding box, which is the frame the cycles are registered in.
_ANCHOR_KEY = "fractal_input_anchored"

# Transform key of the registration result on the fused cycle images.
_FUSED_PREREG_KEY = "affine_preregistered"

# Minimum size (in pixels, along y and x) a pyramid level must have to be used
# for the pre-registration. The coarsest level of a small image can be a few
# pixels wide, which makes phase correlation meaningless.
_MIN_LEVEL_EXTENT_PX = 64

# Warn when the shared canvas is this much larger than the reference extent,
# which means some cycle covers a far larger area than the reference - usually
# a tile with a broken stage coordinate.
_CANVAS_WARN_FACTOR = 3.0


def _coarsest_usable_level(container) -> str:
    """Coarsest pyramid level that is still large enough to register on.

    Falls back to the finest level if every level is below the minimum extent.
    """
    for level_path in reversed(container.level_paths):
        image = container.get_image(path=level_path)
        shape = dict(zip(image.axes, image.shape, strict=True))
        if all(shape.get(dim, 0) >= _MIN_LEVEL_EXTENT_PX for dim in ["y", "x"]):
            return level_path
    return container.level_paths[0]


def _load_cycle_tiles(container, z_project: bool) -> tuple[list, str]:
    """Load one cycle's FOVs at the coarsest usable pyramid level."""
    level_path = _coarsest_usable_level(container)
    msims = _get_msims(
        image=container.get_image(path=level_path),
        fov_roi_table=container.get_table("FOV_ROI_table"),
        z_project=z_project,
    )
    return msims, level_path


def _cycle_box(msims: list) -> tuple[dict[str, float], dict[str, float]]:
    """World-space origin and extent of the union of a cycle's tiles."""
    origins, antipodes = [], []
    for msim in msims:
        sim = msi_utils.get_sim_from_msim(msim)
        origins.append(_get_origin_of_sim(sim, transform_key="fractal_input"))
        antipodes.append(_get_antipode_of_sim(sim, transform_key="fractal_input"))
    dims = list(origins[0].keys())
    origin = {d: min(o[d] for o in origins) for d in dims}
    extent = {d: max(a[d] for a in antipodes) - origin[d] for d in dims}
    return origin, extent


def _translation_matrix(shift: dict[str, float], dims: list[str]) -> np.ndarray:
    """Homogeneous translation matrix from a per-dimension shift."""
    return param_utils.affine_from_translation([shift[d] for d in dims])


def _set_anchor_transform(
    msims: list, origin: dict[str, float], dims: list[str]
) -> None:
    """Store each tile's position relative to the cycle's bounding box origin."""
    to_anchor = _translation_matrix({d: -origin[d] for d in dims}, dims)
    for msim in msims:
        sim = msi_utils.get_sim_from_msim(msim)
        matrix = to_anchor @ _xaffine_to_matrix(
            get_affine_from_sim(sim, transform_key="fractal_input")
        )
        msi_utils.set_affine_transform(
            msim,
            param_utils.affine_to_xaffine(matrix, t_coords=[0]),
            _ANCHOR_KEY,
        )


def _shared_canvas_shape(
    boxes: dict[str, tuple[dict, dict]], cycles: list[str], spacing: dict[str, float]
) -> dict[str, int]:
    """Canvas covering every cycle's extent, each anchored at its box origin."""
    dims = list(spacing.keys())
    return {
        d: int(np.ceil(max(boxes[c][1][d] for c in cycles) / spacing[d])) for d in dims
    }


def _fuse_on_canvas(msims: list, shape: dict[str, int], spacing: dict[str, float]):
    """Fuse one cycle's tiles onto the shared canvas, anchored at its box origin."""
    return _fuse_masked(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key=_ANCHOR_KEY,
        alias_key=_ANCHOR_KEY,
        output_origin=dict.fromkeys(shape, 0.0),
        output_shape=shape,
        output_spacing=spacing,
    )


def _apply_correction_to_tiles(msims: list, correction: np.ndarray | None) -> None:
    """Store "fractal_input" composed with the correction under the pre-reg key.

    A correction of None stores "fractal_input" unchanged, which is what the
    reference cycle and any cycle whose pre-registration failed get.
    """
    for msim in msims:
        sim = msi_utils.get_sim_from_msim(msim)
        matrix = _xaffine_to_matrix(
            get_affine_from_sim(sim, transform_key="fractal_input")
        )
        if correction is not None:
            matrix = correction @ matrix
        msi_utils.set_affine_transform(
            msim,
            param_utils.affine_to_xaffine(matrix, t_coords=[0]),
            PREREG_TRANSFORM_KEY,
        )


def _pre_register_cycles(
    containers: dict,
    msims_reg: dict[str, list],
    cycles: list[str],
    ref_cycle: str,
    reg_channel: str,
    z_project: bool,
) -> None:
    """Roughly align every cycle against the reference cycle.

    Every tile in ``msims_reg`` gets a PREREG_TRANSFORM_KEY transform holding
    its roughly aligned position, which downstream steps use as their starting
    point. Cycles whose pre-registration fails (e.g. no overlap with the
    reference) fall back to their raw stage coordinates.
    """
    for cycle in cycles:
        _apply_correction_to_tiles(msims_reg[cycle], None)

    # Load every cycle at the coarsest usable level and measure its box.
    tiles, boxes = {}, {}
    for cycle in cycles:
        msims, level_path = _load_cycle_tiles(containers[cycle], z_project)
        origin, extent = _cycle_box(msims)
        tiles[cycle] = msims
        boxes[cycle] = (origin, extent)
        logger.info(
            f"Cycle '{cycle}': {len(msims)} FOV(s) at pyramid level '{level_path}', "
            f"extent { ({d: round(v, 1) for d, v in extent.items()}) } um."
        )

    ref_origin, ref_extent = boxes[ref_cycle]
    dims = list(ref_extent.keys())
    spacing = si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(tiles[ref_cycle][0]), asarray=False
    )

    # One canvas for all cycles: the union of every cycle's extent, with each
    # cycle anchored on the origin of its own box. Sized once so the fused
    # reference can be reused for every pair.
    shape = _shared_canvas_shape(boxes, cycles, spacing)
    logger.info(f"Shared registration canvas: {shape} px (spacing {spacing} um).")
    for d in dims:
        if shape[d] * spacing[d] > _CANVAS_WARN_FACTOR * ref_extent[d]:
            logger.warning(
                f"Registration canvas along '{d}' is "
                f"{shape[d] * spacing[d] / ref_extent[d]:.1f}x the reference extent; "
                f"a cycle likely contains a tile with a broken stage coordinate."
            )

    for cycle in cycles:
        _set_anchor_transform(tiles[cycle], boxes[cycle][0], dims)

    # Fuse the reference once and hold it in memory for all pairs.
    sim_ref = _fuse_on_canvas(tiles[ref_cycle], shape, spacing).compute()
    msim_ref = msi_utils.get_msim_from_sim(sim_ref)

    for cycle in cycles:
        if cycle == ref_cycle:
            continue
        msim_cycle = msi_utils.get_msim_from_sim(
            _fuse_on_canvas(tiles[cycle], shape, spacing)
        )
        try:
            registration.register(
                [msim_ref, msim_cycle],
                reg_channel=reg_channel,
                transform_key=_ANCHOR_KEY,
                new_transform_key=_FUSED_PREREG_KEY,
                pre_registration_pruning_method=None,
                groupwise_resolution_kwargs={"reference_view": 0},
                # Both cycles sit on the same canvas, so the region
                # multiview-stitcher intersects is already the full union.
                overlap_tolerance=dict.fromkeys(dims, 0.0),
                reg_res_level=0,
            )
        # Only registration failures are tolerated here (no overlap between the
        # cycles, degenerate input to the phase correlation); anything else is a
        # bug and must not be hidden behind a warning.
        except (mv_graph.NotEnoughOverlapError, ValueError, RuntimeError) as exc:
            logger.warning(
                f"Cycle '{cycle}': rough pre-registration failed ({exc}); "
                f"continuing with the raw stage coordinates."
            )
            continue

        # The registration ran in the anchored frame, so undo the anchoring of
        # this cycle and re-enter the reference's world frame.
        displacement = _xaffine_to_matrix(
            msi_utils.get_transform_from_msim(msim_cycle, _FUSED_PREREG_KEY)
        )
        correction = (
            _translation_matrix(ref_origin, dims)
            @ displacement
            @ _translation_matrix({d: -boxes[cycle][0][d] for d in dims}, dims)
        )
        logger.info(
            f"Cycle '{cycle}': rough pre-registration shift "
            f"{np.round(param_utils.translation_from_affine(correction), 3)} um."
        )
        _apply_correction_to_tiles(msims_reg[cycle], correction)

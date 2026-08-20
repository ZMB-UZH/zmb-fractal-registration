from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.spatial_image_utils import get_affine_from_sim
from ngio import (
    ChannelSelectionModel,
    ImageInWellPath,
    Roi,
    create_empty_plate,
    create_synthetic_ome_zarr,
    open_ome_zarr_container,
    open_ome_zarr_plate,
)
from ngio.ome_zarr_meta import Channel
from ngio.tables import RoiTable

from zmb_fractal_registration._stitch_register.grid_snap import (
    _snap_msims_to_output_grid,
)
from zmb_fractal_registration._stitch_register.loading import (
    _get_msims,
    _resolve_registration_channel,
)
from zmb_fractal_registration._stitch_register.output_bbox import (
    _compute_global_bbox,
    _coverage_cell_grid,
    _largest_covered_box,
    _tight_covered_box,
)
from zmb_fractal_registration._stitch_register.pre_registration import (
    PREREG_TRANSFORM_KEY,
    _apply_correction_to_tiles,
    _coarsest_usable_level,
    _cycle_box,
    _fuse_on_canvas,
    _load_cycle_tiles,
    _pre_register_cycles,
    _set_anchor_transform,
    _shared_canvas_shape,
)
from zmb_fractal_registration._stitch_register.registration import (
    _MAX_CHUNK_VOXELS,
    _detect_outlier_tiles,
    _fuse_masked,
    _output_chunksize,
    _register_leftover_tiles,
    _registration_chunksize,
)
from zmb_fractal_registration._stitch_register.sim_geometry import _xaffine_to_matrix
from zmb_fractal_registration.stitch_and_register_init import (
    TileCorrectionModel,
    stitch_and_register_init,
)
from zmb_fractal_registration.stitch_and_register_parallel import (
    _fuse_cycles,
    stitch_and_register_parallel,
)

_PIXEL_SIZE = 0.325  # um/px (default for create_synthetic_ome_zarr Cardiomyocyte)
_FOV_PX = 64  # pixels per FOV side
_OVERLAP_PX = 32  # 50% overlap -> enough for multiview_stitcher adjacency detection


def _std_rois(
    fov_px: int = _FOV_PX, overlap_px: int = _OVERLAP_PX, x_offset_um: float = 0.0
) -> list[Roi]:
    """Two side-by-side FOVs with 50% overlap in x.

    FOV 1: world x=[0, fov_size_um] -> pixels x=[0:fov_px].
    FOV 2: world x=[fov_size_um-overlap_um, 2*fov_size_um-overlap_um]
      -> pixels x=[overlap_px:2*fov_px-overlap_px].
    `x_offset_um` shifts only the `*_micrometer_original` stage coordinates
    (the positions the task initializes the tiles from), not the pixel slices.
    """
    fov_size_um = fov_px * _PIXEL_SIZE
    overlap_um = overlap_px * _PIXEL_SIZE
    return [
        Roi.from_values(
            slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
            name="FOV_1",
            y_micrometer_original=0.0,
            x_micrometer_original=x_offset_um,
        ),
        Roi.from_values(
            slices={
                "y": (0.0, fov_size_um),
                "x": (fov_size_um - overlap_um, 2 * fov_size_um - overlap_um),
            },
            name="FOV_2",
            y_micrometer_original=0.0,
            x_micrometer_original=fov_size_um - overlap_um + x_offset_um,
        ),
    ]


def _displaced_roi(name: str, x_um: float, fov_px: int = _FOV_PX) -> Roi:
    """A tile reusing FOV_1's pixel region, but placed at `x_um` in world space.

    The slice coordinates reuse the first FOV's pixel region so that
    image.get_roi() succeeds; only the world-space stage coordinate is moved.
    """
    fov_size_um = fov_px * _PIXEL_SIZE
    return Roi.from_values(
        slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
        name=name,
        y_micrometer_original=0.0,
        x_micrometer_original=x_um,
    )


def _build_plate(
    plate_path: Path,
    rois_per_acquisition: list[list[Roi]],
    channels_per_acquisition: list[list] | None = None,
    fov_px: int = _FOV_PX,
    overlap_px: int = _OVERLAP_PX,
    levels: int | None = None,
) -> list[str]:
    """Create a plate with one image per acquisition and the given ROI tables.

    Each image is two FOVs wide (fov_px high, 2*fov_px-overlap_px wide) and
    lives in well A/1 under paths "0", "1", ... with matching acquisition IDs.
    Returns the zarr URL of each image.
    """
    n_acq = len(rois_per_acquisition)
    if channels_per_acquisition is None:
        channels_per_acquisition = [["DAPI"] for _ in range(n_acq)]
    plate = create_empty_plate(
        store=plate_path,
        name="test_plate",
        images=[
            ImageInWellPath(row="A", column=1, path=str(i), acquisition_id=i)
            for i in range(n_acq)
        ],
        overwrite=True,
    )
    zarr_urls = []
    for img_rel_path, rois, channels in zip(
        plate.images_paths(),
        rois_per_acquisition,
        channels_per_acquisition,
        strict=True,
    ):
        img_path = plate_path / img_rel_path
        levels_kwargs = {} if levels is None else {"levels": levels}
        container = create_synthetic_ome_zarr(
            store=img_path,
            shape=(len(channels), fov_px, 2 * fov_px - overlap_px),
            axes_names="cyx",
            channels_meta=channels,
            overwrite=True,
            **levels_kwargs,
        )
        container.add_table("FOV_ROI_table", RoiTable(rois=rois))
        zarr_urls.append(str(img_path))
    return zarr_urls


def _create_test_plate(plate_path: Path) -> list[str]:
    """A minimal plate: 2 acquisitions, each with 2 overlapping FOVs."""
    return _build_plate(plate_path, [_std_rois(), _std_rois()])


def _open_fused_image(plate_path: Path):
    """The single fused image in the `_fused` plate derived from plate_path.

    The task writes its output to a new plate next to the original (inside
    zarr_dir, which the tests pass as the original's parent directory).
    """
    fused_plate_path = plate_path.parent / f"{plate_path.stem}_fused.zarr"
    fused_plate = open_ome_zarr_plate(fused_plate_path)
    fused_images = list(fused_plate.get_images(acquisition=0).values())
    assert len(fused_images) == 1
    return fused_images[0].get_image()


def test_stitch_and_register(tmp_path: Path):
    """Smoke test for the stitch and register task."""
    plate_path = tmp_path / "test.zarr"
    zarr_urls = _create_test_plate(plate_path)

    ref_channel = ChannelSelectionModel(mode="label", identifier="DAPI")

    # Run init task
    result = stitch_and_register_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        reference_channel=ref_channel,
    )
    parallelization_list = result["parallelization_list"]
    assert len(parallelization_list) == 1  # one well in the plate

    # Run parallel task for each well
    for item in parallelization_list:
        stitch_and_register_parallel(
            zarr_url=item["zarr_url"],
            init_args=item["init_args"],
        )

    # Check that the fused plate was created, with channels from both cycles
    # (1 channel x 2 cycles = 2).
    fused_image = _open_fused_image(plate_path)
    assert len(fused_image.channel_labels) == 2
    assert any("DAPI" in label for label in fused_image.channel_labels)

    # The original plate is left untouched.
    plate = open_ome_zarr_plate(plate_path)
    assert set(plate.acquisition_ids) == {0, 1}
    assert len(plate.images_paths()) == 2

    # The fused image carries a well ROI table spanning its full extent.
    fused_plate = open_ome_zarr_plate(plate_path.parent / "test_fused.zarr")
    fused_container = next(iter(fused_plate.get_images(acquisition=0).values()))
    rois = fused_container.get_table("well_ROI_table").rois()
    assert len(rois) == 1
    x_extent_px = rois[0].get("x").length / fused_image.pixel_size.x
    assert round(x_extent_px) == fused_image.shape[-1]


def _create_plate_with_far_tiles(
    plate_path: Path,
    all_nonref_tiles_far: bool = False,
) -> list[str]:
    """A plate whose non-reference cycle has tiles far from the reference tiles.

    When ``all_nonref_tiles_far`` is False (default), the non-ref cycle has two
    normal overlapping tiles plus one extra tile positioned far in world space
    (no spatial overlap with the reference).

    When ``all_nonref_tiles_far`` is True, *all* non-ref tiles are placed far away,
    exercising the fallback path where no inlier tiles exist.
    """
    fov_size_um = _FOV_PX * _PIXEL_SIZE
    far_x = 10.0 * fov_size_um
    if all_nonref_tiles_far:
        nonref_rois = [
            _displaced_roi("FOV_1_far", far_x),
            _displaced_roi("FOV_2_far", far_x + fov_size_um),
        ]
    else:
        nonref_rois = [*_std_rois(), _displaced_roi("FOV_3_far", far_x)]
    return _build_plate(plate_path, [_std_rois(), nonref_rois])


def _run_stitch_and_register(
    zarr_urls: list[str],
    zarr_dir: str,
    fusion_region: str = "union",
    pre_registration: bool = False,
) -> None:
    """Run init + parallel stitch-and-register tasks for a list of zarr URLs."""
    ref_channel = ChannelSelectionModel(mode="label", identifier="DAPI")
    result = stitch_and_register_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_channel=ref_channel,
        fusion_region=fusion_region,
        pre_registration=pre_registration,
    )
    for item in result["parallelization_list"]:
        stitch_and_register_parallel(
            zarr_url=item["zarr_url"],
            init_args=item["init_args"],
        )


# ---------------------------------------------------------------------------
# Unit tests for _detect_outlier_tiles
# ---------------------------------------------------------------------------


def test_detect_outlier_tiles_zscore():
    """Zscore mode flags a tile whose shift is a clear statistical outlier."""
    # Four tiles with small, consistent shifts and one with a very large shift.
    # With 4+1 layout the outlier's z-score is exactly 2.0; use threshold < 2.0.
    shifts = [np.array([1.0, 0.0])] * 4 + [np.array([100.0, 0.0])]
    reg_tile_indices = list(range(5))
    osc = TileCorrectionModel(outlier_filter_mode="zscore", threshold=1.5)
    outliers = _detect_outlier_tiles(shifts, reg_tile_indices, osc, "cycle1")
    assert outliers == {4}


def test_detect_outlier_tiles_absolute():
    """Absolute mode flags a tile whose shift exceeds the threshold in um."""
    # Deviations are computed from the mean, not from zero.
    # With 4 inliers at [1,0] and 1 outlier at [100,0], mean=[20.8,0],
    # inlier deviation~19.8 um and outlier deviation~79.2 um.
    # Use threshold=50 to flag only the outlier.
    shifts = [np.array([1.0, 0.0])] * 4 + [np.array([100.0, 0.0])]
    reg_tile_indices = list(range(5))
    osc = TileCorrectionModel(outlier_filter_mode="absolute", threshold=50.0)
    outliers = _detect_outlier_tiles(shifts, reg_tile_indices, osc, "cycle1")
    assert outliers == {4}


def test_detect_outlier_tiles_disabled():
    """Disabled mode never flags any tile as an outlier."""
    shifts = [np.array([1.0, 0.0])] * 4 + [np.array([100.0, 0.0])]
    reg_tile_indices = list(range(5))
    osc = TileCorrectionModel(outlier_filter_mode="disabled")
    outliers = _detect_outlier_tiles(shifts, reg_tile_indices, osc, "cycle1")
    assert outliers == set()


def test_detect_outlier_tiles_empty_shifts():
    """Empty shift list always returns an empty outlier set."""
    osc = TileCorrectionModel(outlier_filter_mode="zscore", threshold=2.0)
    assert _detect_outlier_tiles([], [], osc, "cycle1") == set()


# ---------------------------------------------------------------------------
# Unit tests for the coordinate-compression bounding-box helpers
# ---------------------------------------------------------------------------


def _rect(y0, y1, x0, x1, z0=None, z1=None):
    """Build a (origin, antipode) tile rectangle dict pair."""
    origin = {"y": float(y0), "x": float(x0)}
    antipode = {"y": float(y1), "x": float(x1)}
    if z0 is not None:
        origin["z"] = float(z0)
        antipode["z"] = float(z1)
    return origin, antipode


def test_coverage_grid_full_overlap():
    """Two identical full grids -> every cell covered; tight box == bbox."""
    grid = [_rect(0, 1, 0, 1), _rect(0, 1, 1, 2), _rect(1, 2, 0, 1), _rect(1, 2, 1, 2)]
    rects = {"A": grid, "B": grid}
    dims = ["y", "x"]
    breaks, covered = _coverage_cell_grid(rects, ["A", "B"], dims)
    assert covered.all()
    tight_o, tight_a = _tight_covered_box(breaks, covered, dims)
    bbox_o, bbox_a = _largest_covered_box(breaks, covered, dims)
    assert (tight_o, tight_a) == ({"y": 0.0, "x": 0.0}, {"y": 2.0, "x": 2.0})
    assert (bbox_o, bbox_a) == (tight_o, tight_a)


def test_coverage_grid_missing_tile_differs():
    """A hole makes intersection (tight box) larger than intersection_bbox."""
    full = [_rect(0, 1, 0, 1), _rect(0, 1, 1, 2), _rect(1, 2, 0, 1), _rect(1, 2, 1, 2)]
    hole = [
        _rect(0, 1, 0, 1),
        _rect(0, 1, 1, 2),
        _rect(1, 2, 0, 1),
    ]  # missing (1-2,1-2)
    rects = {"A": full, "B": hole}
    dims = ["y", "x"]
    breaks, covered = _coverage_cell_grid(rects, ["A", "B"], dims)
    # Only the (1,1) cell is uncovered.
    assert covered.sum() == 3
    assert not covered[1, 1]

    tight_o, tight_a = _tight_covered_box(breaks, covered, dims)
    bbox_o, bbox_a = _largest_covered_box(breaks, covered, dims)

    # Tight box still spans the whole 2x2 (it encloses the hole).
    assert (tight_o, tight_a) == ({"y": 0.0, "x": 0.0}, {"y": 2.0, "x": 2.0})
    tight_vol = (tight_a["y"] - tight_o["y"]) * (tight_a["x"] - tight_o["x"])
    bbox_vol = (bbox_a["y"] - bbox_o["y"]) * (bbox_a["x"] - bbox_o["x"])
    assert bbox_vol == 2.0  # a 1x2 strip avoiding the hole
    assert bbox_vol < tight_vol


def test_largest_covered_box_3d():
    """3D path: hole in xy, full z range -> bbox keeps full z, avoids the hole."""
    full = [
        _rect(0, 1, 0, 1, 0, 1),
        _rect(0, 1, 1, 2, 0, 1),
        _rect(1, 2, 0, 1, 0, 1),
        _rect(1, 2, 1, 2, 0, 1),
    ]
    hole = full[:3]  # missing (y1-2, x1-2)
    rects = {"A": full, "B": hole}
    dims = ["z", "y", "x"]
    breaks, covered = _coverage_cell_grid(rects, ["A", "B"], dims)
    assert covered.shape == (1, 2, 2)
    bbox_o, bbox_a = _largest_covered_box(breaks, covered, dims)
    # Full z, and a 1x2 xy strip (volume 2).
    assert bbox_a["z"] - bbox_o["z"] == 1.0
    vol = (
        (bbox_a["z"] - bbox_o["z"])
        * (bbox_a["y"] - bbox_o["y"])
        * (bbox_a["x"] - bbox_o["x"])
    )
    assert vol == 2.0


# ---------------------------------------------------------------------------
# Unit tests for rounding the covered box to a whole pixel count
# ---------------------------------------------------------------------------

_ROUND_SPACING = 0.65  # um/px
_ROUND_TILE_PX = 100
# A registration shift that is deliberately not a whole number of pixels, so the
# covered box width is fractional in pixels and ceil/floor actually differ.
_ROUND_SUBPIXEL = 0.31 * _ROUND_SPACING


def _single_tile_msims(offset: float) -> list:
    """One cycle holding a single tile, translated by `offset` um in y and x."""
    sim = si_utils.get_sim_from_array(
        np.ones((1, _ROUND_TILE_PX, _ROUND_TILE_PX), dtype=np.uint16),
        dims=["c", "y", "x"],
        scale={"y": _ROUND_SPACING, "x": _ROUND_SPACING},
        translation={"y": offset, "x": offset},
        c_coords=["DAPI"],
        transform_key="affine_registered",
    )
    return [msi_utils.get_msim_from_sim(sim, scale_factors=[])]


def test_intersection_bbox_shape_rounds_down():
    """The intersection_bbox canvas must fit inside the covered box.

    output_origin is the centre of the first pixel, so the last pixel centre
    lies at origin + (shape - 1) * spacing. Rounding the pixel count up would
    push that centre past the covered box and yield a row/column with no data,
    breaking the mode's "no pixels set to 0" promise.
    """
    msims = {"A": _single_tile_msims(0.0), "B": _single_tile_msims(_ROUND_SUBPIXEL)}
    spacing_ref = {"y": _ROUND_SPACING, "x": _ROUND_SPACING}

    _, shape = _compute_global_bbox(msims, ["A", "B"], spacing_ref, "intersection_bbox")

    # Covered region is the overlap of the two tiles.
    box_width = _ROUND_TILE_PX * _ROUND_SPACING - _ROUND_SUBPIXEL
    exact_px = box_width / _ROUND_SPACING
    assert exact_px % 1 != 0, "test is only meaningful for a fractional pixel count"

    for dim in ("y", "x"):
        assert shape[dim] == int(np.floor(exact_px))
        # The canvas stays inside the covered box...
        assert shape[dim] * _ROUND_SPACING <= box_width
        # ...whereas rounding up would have overshot it.
        assert np.ceil(exact_px) * _ROUND_SPACING > box_width


def test_union_shape_still_rounds_up():
    """union keeps rounding up, so its canvas contains the whole extent."""
    msims = {"A": _single_tile_msims(0.0), "B": _single_tile_msims(_ROUND_SUBPIXEL)}
    spacing_ref = {"y": _ROUND_SPACING, "x": _ROUND_SPACING}

    _, shape = _compute_global_bbox(msims, ["A", "B"], spacing_ref, "union")

    box_width = _ROUND_TILE_PX * _ROUND_SPACING + _ROUND_SUBPIXEL
    for dim in ("y", "x"):
        assert shape[dim] == int(np.ceil(box_width / _ROUND_SPACING))
        assert shape[dim] * _ROUND_SPACING >= box_width


def test_intersection_bbox_shape_exact_multiple_keeps_all_pixels():
    """A grid-aligned box must not lose a pixel to floating-point error."""
    aligned_shift = 10 * _ROUND_SPACING  # exactly 10 px, so the box is 90 px wide
    msims = {"A": _single_tile_msims(0.0), "B": _single_tile_msims(aligned_shift)}
    spacing_ref = {"y": _ROUND_SPACING, "x": _ROUND_SPACING}

    _, shape = _compute_global_bbox(msims, ["A", "B"], spacing_ref, "intersection_bbox")

    for dim in ("y", "x"):
        assert shape[dim] == _ROUND_TILE_PX - 10


# ---------------------------------------------------------------------------
# Unit tests for leftover-tile correction
# ---------------------------------------------------------------------------


def _tile_msim(y: float, x: float, registered_shift: float | None = None):
    """A 50x50 um tile at (y, x), optionally with an affine_registered transform.

    The content is deliberately non-constant: multiview-stitcher short-circuits
    pairwise registration of constant images, which would hide the code path
    under test.
    """
    rng = np.random.default_rng(abs(int(y * 7 + x * 13)) + 1)
    sim = si_utils.get_sim_from_array(
        rng.integers(0, 4096, size=(1, 50, 50)).astype(np.uint16),
        dims=["c", "y", "x"],
        scale={"y": 1.0, "x": 1.0},
        translation={"y": y, "x": x},
        c_coords=["DAPI"],
        transform_key="fractal_input",
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    if registered_shift is not None:
        msi_utils.set_affine_transform(
            msim,
            param_utils.affine_to_xaffine(
                param_utils.affine_from_translation(
                    [registered_shift, registered_shift]
                ),
                t_coords=[0],
            ),
            "affine_registered",
        )
    return msim


def test_leftover_tile_in_hole_of_inlier_coverage(tmp_path: Path):
    """A leftover tile inside the inliers' bounding box but not on any inlier.

    The inliers sit diagonally, so the fused inlier image spans a box with a
    hole in it. A leftover tile in that hole overlaps the fused image's bounding
    box - so it is not caught as "not enough overlap" - while the pixels there
    are all NaN. Phase correlation used to crash on that empty reference with
    "zero-size array to reduction operation minimum".
    """
    # inliers on the diagonal, leftover in the empty off-diagonal corner
    msims = [
        _tile_msim(0.0, 0.0, registered_shift=1.0),
        _tile_msim(0.0, 100.0),  # leftover, in the hole
        _tile_msim(100.0, 100.0, registered_shift=1.0),
    ]

    _register_leftover_tiles(msims, {1}, "DAPI", "cycle1")

    # It falls back to the mean inlier shift rather than raising. The tile's world
    # position lives in its coordinates, so the transform holds the shift alone.
    assert np.allclose(_translations([msims[1]], "affine_registered")[0], [1.0, 1.0])


def _chunk_test_sim(z_planes: int | None, offset: float = 0.0):
    """A single tile, 3D when z_planes is given, 2D otherwise."""
    dims = ["c", "y", "x"] if z_planes is None else ["c", "z", "y", "x"]
    shape = (1, 300, 300) if z_planes is None else (1, z_planes, 300, 300)
    scale = {"y": 0.325, "x": 0.325}
    translation = {"y": 0.0, "x": offset}
    if z_planes is not None:
        scale["z"] = 1.0
        translation["z"] = 0.0
    return si_utils.get_sim_from_array(
        np.ones(shape, dtype=np.uint16),
        dims=dims,
        scale=scale,
        translation=translation,
        c_coords=["DAPI"],
        transform_key="affine_registered",
    )


def test_registration_chunksize_tracks_the_tile_size():
    """A tile that fits the voxel budget is used as the chunk verbatim."""
    assert _registration_chunksize([_chunk_test_sim(None)]) == {"y": 300, "x": 300}


def test_registration_chunksize_has_a_floor():
    """Tiny tiles (coarse pyramid levels) do not produce tiny chunks."""
    tiny = si_utils.get_sim_from_array(
        np.ones((1, 8, 8), dtype=np.uint16),
        dims=["c", "y", "x"],
        scale={"y": 1.0, "x": 1.0},
        c_coords=["DAPI"],
        transform_key="affine_registered",
    )
    assert _registration_chunksize([tiny]) == {"y": 64, "x": 64}


@pytest.mark.parametrize("tile_shape", [(1, 50, 2048, 2048), (1, 1, 8000, 8000)])
def test_registration_chunksize_caps_large_tiles(tile_shape):
    """A full-resolution tile is too big to use as a chunk verbatim.

    Without the cap a 2048x2048x50 tile would ask for a 200M-voxel chunk, which
    holds several GB once the overlapping tiles are resampled into it.
    """
    assert np.prod(tile_shape) > _MAX_CHUNK_VOXELS, "tile must exceed the budget"
    chunks = _registration_chunksize([_dask_tile(tile_shape, shape=tile_shape)])

    assert np.prod(list(chunks.values())) <= _MAX_CHUNK_VOXELS
    # still no larger than the tile in any dimension, so a chunk cannot span
    # more than one tile
    for dim, size in zip(["z", "y", "x"], tile_shape[1:], strict=True):
        assert chunks[dim] <= max(size, 64)


def _dask_tile(chunks: tuple[int, ...], shape=(1, 40, 300, 300)):
    """A 3D tile backed by a dask array with an explicit chunking."""
    return si_utils.get_sim_from_array(
        da.ones(shape, dtype=np.uint16, chunks=chunks),
        dims=["c", "z", "y", "x"],
        scale={"z": 1.0, "y": 0.325, "x": 0.325},
        c_coords=["DAPI"],
        transform_key="affine_registered",
    )


def test_output_chunksize_inherits_the_input_chunking():
    """The fused output is chunked like the tiles it was built from.

    This size is also written as the on-disk chunking of the output OME-Zarr,
    so it should follow the input rather than a number derived from a memory
    budget.
    """
    assert _output_chunksize([_dask_tile((1, 10, 128, 128))]) == {
        "z": 10,
        "y": 128,
        "x": 128,
    }
    # a store written one z-plane at a time, which is a common OME-Zarr layout
    assert _output_chunksize([_dask_tile((1, 1, 256, 256))]) == {
        "z": 1,
        "y": 256,
        "x": 256,
    }


def test_output_chunksize_caps_pathological_input_chunking():
    """A store with huge chunks must not dictate a huge fusion chunk.

    A whole-plane chunk would hold gigabytes once every overlapping tile is
    resampled into it, so the inherited size is still capped.
    """
    chunks = _output_chunksize(
        [_dask_tile((1, 1, 5000, 5000), shape=(1, 1, 5000, 5000))]
    )

    # z is already 1 and cannot shrink, so y and x must absorb the whole cap
    assert np.prod(list(chunks.values())) <= _MAX_CHUNK_VOXELS
    assert chunks["z"] == 1


def test_output_chunksize_falls_back_for_numpy_tiles():
    """A tile that is not a chunked array falls back to the tile shape."""
    assert _output_chunksize([_chunk_test_sim(None)]) == {"y": 300, "x": 300}


def test_fused_chunks_never_exceed_the_tile_size():
    """No fused chunk may span more than one tile, in any dimension.

    A chunk larger than a tile pulls ~(chunk/tile)**2 tiles into memory at once
    when it is fused. An absolute chunk size cannot guarantee this, because a
    tile is a different number of pixels at every pyramid level - which is how
    a whole fused reference once ended up inside a single chunk.
    """
    for z_planes in (None, 40):
        sims = [_chunk_test_sim(z_planes), _chunk_test_sim(z_planes, offset=50.0)]
        tile_shape = si_utils.get_shape_from_sim(sims[0], asarray=False)

        fused = _fuse_masked(sims)  # lazy, nothing is computed here

        for dim, tile_size in tile_shape.items():
            assert max(fused.chunksizes[dim]) <= max(tile_size, 64)


# ---------------------------------------------------------------------------
# Order-0 fusion: grid snapping
# ---------------------------------------------------------------------------

_SNAP_S = 0.325
_SNAP_BASE = 98765.43  # plate-scale stage coordinate in um


def _uniform_tile_msim(value: int, y: float, x: float, n: int = 64):
    """A constant-valued chunked tile at (y, x) with the position in its coords."""
    sim = si_utils.get_sim_from_array(
        da.full((1, n, n), value, dtype=np.uint16, chunks=(1, 16, 16)),
        dims=["c", "y", "x"],
        scale={"y": _SNAP_S, "x": _SNAP_S},
        translation={"y": y, "x": x},
        c_coords=["ch"],
        transform_key="affine_registered",
    )
    return msi_utils.get_msim_from_sim(sim, scale_factors=[])


def test_snap_msims_to_output_grid_properties():
    """Snapping rebases to a local frame and moves tiles by at most half a pixel."""
    sub = 0.31 * _SNAP_S
    msims = {
        "A": [_uniform_tile_msim(100, _SNAP_BASE, _SNAP_BASE)],
        "B": [
            _uniform_tile_msim(200, _SNAP_BASE + sub, _SNAP_BASE + 2 * _SNAP_S + sub)
        ],
    }
    spacing_ref = {"y": _SNAP_S, "x": _SNAP_S}

    snapped = _snap_msims_to_output_grid(msims, ["A", "B"], spacing_ref)

    # Anchored at the lowest tile corner; the 0.31 px offsets round down, the
    # 2.31 px offset rounds to 2 px.
    for cycle, k_exp in [("A", {"y": 0, "x": 0}), ("B", {"y": 0, "x": 2})]:
        sim = msi_utils.get_sim_from_msim(snapped[cycle][0])
        origin = si_utils.get_origin_from_sim(sim, asarray=False)
        for dim in ("y", "x"):
            assert origin[dim] == k_exp[dim] * _SNAP_S


def test_order0_fusion_is_exact_after_grid_snapping():
    """intersection_bbox at order 0 contains only original pixel values.

    Tiles at plate-scale world coordinates with sub-pixel registration shifts
    used to come out of the order-0 fusion with one-pixel zero seams at chunk
    borders and the canvas edge (and off-by-one values elsewhere); with grid
    snapping the fusion is an exact integer-shift mosaic.
    """
    from types import SimpleNamespace

    sub = 0.31 * _SNAP_S
    msims_fusion = {
        "A": [_uniform_tile_msim(100, _SNAP_BASE, _SNAP_BASE)],
        "B": [_uniform_tile_msim(200, _SNAP_BASE + sub, _SNAP_BASE + sub)],
    }
    # _fuse_cycles only uses the reference container for the axes lookup.
    containers = {
        "A": SimpleNamespace(get_image=lambda: SimpleNamespace(axes=["c", "y", "x"]))
    }

    fused = _fuse_cycles(
        containers, msims_fusion, ["A", "B"], "A", "intersection_bbox", 0
    )

    arr = np.asarray(fused.data)
    assert sorted(np.unique(arr).tolist()) == [100, 200]


def test_fused_output_chunks_are_uniform_across_cycles():
    """Cycles with different input chunkings fuse onto one regular chunk grid.

    Per-cycle chunk sizes made xr.concat unify the dask chunks into an
    irregular grid that no longer matched the on-disk chunking derived from
    it, sending zarr through its unaligned read-modify-write path with several
    tasks updating the same chunk file concurrently - silently dropping
    updates and failing on network storage.
    """
    from types import SimpleNamespace

    def tile_with_chunks(value, chunks):
        sim = si_utils.get_sim_from_array(
            da.full((1, 96, 96), value, dtype=np.uint16, chunks=(1, *chunks)),
            dims=["c", "y", "x"],
            scale={"y": _SNAP_S, "x": _SNAP_S},
            translation={"y": 0.0, "x": 0.0},
            c_coords=["ch"],
            transform_key="affine_registered",
        )
        return msi_utils.get_msim_from_sim(sim, scale_factors=[])

    msims_fusion = {
        "A": [tile_with_chunks(100, (40, 40))],
        "B": [tile_with_chunks(200, (32, 32))],
    }
    containers = {
        "A": SimpleNamespace(get_image=lambda: SimpleNamespace(axes=["c", "y", "x"]))
    }

    fused = _fuse_cycles(containers, msims_fusion, ["A", "B"], "A", "union", 0)

    # A regular grid: every chunk except the last must equal the first, in
    # every spatial dimension and identically for all channels.
    for dim in ("y", "x"):
        sizes = fused.chunksizes[dim]
        assert all(s == sizes[0] for s in sizes[:-1])
        assert sizes[0] == 32  # the per-dim minimum across cycles


def test_register_cycle_tiles_survives_registration_failure(monkeypatch):
    """A tile whose registration crashes is deferred to leftover correction.

    A tile can overlap a reference tile's bounding box while the overlap
    region of the fused reference is all NaN (coverage hole or seam), which
    used to crash the whole well with "zero-size array to reduction operation
    minimum" inside phase correlation.
    """
    from zmb_fractal_registration._stitch_register import registration as regmod

    ref_msims = [_tile_msim(0.0, 0.0, registered_shift=0.0)]
    sim_fused_ref = _fuse_masked([msi_utils.get_sim_from_msim(m) for m in ref_msims])
    moving = [_tile_msim(10.0, 10.0)]  # overlaps the reference tile's bbox

    def failing_register(*args, **kwargs):
        raise ValueError("zero-size array to reduction operation minimum")

    monkeypatch.setattr(regmod.registration, "register", failing_register)

    leftovers = regmod._register_cycle_tiles(
        moving, sim_fused_ref, "DAPI", ref_msims, "fractal_input"
    )

    assert leftovers == [0]


# ---------------------------------------------------------------------------
# Integration tests: non-overlapping and fallback tile handling
# ---------------------------------------------------------------------------


def test_non_overlapping_tile(tmp_path: Path):
    """Task completes when one non-ref tile has no spatial overlap with reference."""
    plate_path = tmp_path / "test.zarr"
    zarr_urls = _create_plate_with_far_tiles(plate_path, all_nonref_tiles_far=False)
    _run_stitch_and_register(zarr_urls, str(tmp_path))

    fused_image = _open_fused_image(plate_path)
    assert len(fused_image.channel_labels) == 2
    assert any("DAPI" in label for label in fused_image.channel_labels)


def _fused_shape(plate_path: Path) -> tuple[int, ...]:
    """Return the shape of the single fused image for a processed plate."""
    fused_image = _open_fused_image(plate_path)
    assert len(fused_image.channel_labels) == 2
    return fused_image.shape


def test_fusion_regions(tmp_path: Path):
    """The three fusion regions produce consistently sized outputs.

    The far-away non-ref tile means the non-ref cycle covers a much wider
    extent than the reference, so both intersection canvases must be strictly
    smaller than the union canvas along x. For a rectangular tile layout the
    per-pixel intersection already fills its bounding box, so 'intersection'
    and 'intersection_bbox' share the same shape.
    """
    shapes = {}
    for region in ("union", "intersection", "intersection_bbox"):
        plate_path = tmp_path / f"{region}.zarr"
        zarr_urls = _create_plate_with_far_tiles(plate_path)
        _run_stitch_and_register(zarr_urls, str(tmp_path), fusion_region=region)
        shapes[region] = _fused_shape(plate_path)

    # Intersection drops the far tile's exclusive region -> smaller along x.
    assert shapes["intersection"][-1] < shapes["union"][-1]
    assert shapes["intersection_bbox"] == shapes["intersection"]


# ---------------------------------------------------------------------------
# Rough pre-registration of whole cycles
# ---------------------------------------------------------------------------

_PREREG_FOV_PX = 128
_PREREG_OVERLAP_PX = 64
# With 2 levels the coarsest level of the plate below is 64 x 96 px, which is
# just large enough to be picked by _coarsest_usable_level.
_PREREG_LEVELS = 2


def _create_offset_plate(
    plate_path: Path, offset_um: float, stray_tile_um: float | None = None
) -> list[str]:
    """Two acquisitions with identical content, the second with wrong stage coords.

    The ROI slices (the pixel region each FOV is read from) are the same in both
    acquisitions, but the `*_micrometer_original` values of the second one - the
    stage coordinates the task initializes the tile positions from - are offset
    by `offset_um` along x. The second acquisition therefore believes it sits
    `offset_um` further along x than it really does.

    When `stray_tile_um` is given, the second acquisition gets an extra tile
    whose stage coordinate places it that far along x, which drags the origin
    of that cycle's bounding box away from its actual content.
    """
    nonref_rois = _std_rois(_PREREG_FOV_PX, _PREREG_OVERLAP_PX, x_offset_um=offset_um)
    if stray_tile_um is not None:
        nonref_rois.append(
            _displaced_roi("FOV_stray", stray_tile_um, fov_px=_PREREG_FOV_PX)
        )
    return _build_plate(
        plate_path,
        [_std_rois(_PREREG_FOV_PX, _PREREG_OVERLAP_PX), nonref_rois],
        fov_px=_PREREG_FOV_PX,
        overlap_px=_PREREG_OVERLAP_PX,
        levels=_PREREG_LEVELS,
    )


def _translations(msims: list, transform_key: str) -> list[np.ndarray]:
    """Translation of the named transform of each tile."""
    return [
        param_utils.translation_from_affine(
            _xaffine_to_matrix(
                get_affine_from_sim(
                    msi_utils.get_sim_from_msim(msim), transform_key=transform_key
                )
            )
        )
        for msim in msims
    ]


def test_coarsest_usable_level_skips_tiny_levels(tmp_path: Path):
    """Pyramid levels below the minimum extent are not used for pre-registration."""
    # 5 levels of a 64 x 96 image: only level 0 reaches 64 px in y.
    small_urls = _create_test_plate(tmp_path / "small.zarr")
    assert _coarsest_usable_level(open_ome_zarr_container(small_urls[0])) == "0"

    # 2 levels of a 128 x 192 image: level 1 (64 x 96) is still large enough.
    large_urls = _create_offset_plate(tmp_path / "large.zarr", offset_um=0.0)
    assert _coarsest_usable_level(open_ome_zarr_container(large_urls[0])) == "1"


def test_apply_correction_to_tiles_writes_separate_transform():
    """The correction goes to the pre-reg key and leaves "fractal_input" alone."""
    msims = _single_tile_msims(0.0)
    msi_utils.set_affine_transform(
        msims[0],
        param_utils.affine_to_xaffine(
            param_utils.affine_from_translation([1.0, 2.0]), t_coords=[0]
        ),
        "fractal_input",
    )
    correction = param_utils.affine_from_translation([10.0, 20.0])

    _apply_correction_to_tiles(msims, correction)

    assert np.allclose(_translations(msims, PREREG_TRANSFORM_KEY)[0], [11.0, 22.0])
    assert np.allclose(_translations(msims, "fractal_input")[0], [1.0, 2.0])


def test_apply_correction_to_tiles_without_correction_copies_input():
    """A correction of None just copies "fractal_input" to the pre-reg key."""
    msims = _single_tile_msims(0.0)
    msi_utils.set_affine_transform(
        msims[0],
        param_utils.affine_to_xaffine(
            param_utils.affine_from_translation([1.0, 2.0]), t_coords=[0]
        ),
        "fractal_input",
    )

    _apply_correction_to_tiles(msims, None)

    assert np.allclose(_translations(msims, PREREG_TRANSFORM_KEY)[0], [1.0, 2.0])


def test_pre_registration_recovers_offset(tmp_path: Path):
    """A cycle with wrong stage coordinates is shifted back onto the reference."""
    offset_um = 16 * _PIXEL_SIZE  # 16 px at full resolution, 8 px at level 1
    zarr_urls = _create_offset_plate(tmp_path / "prereg.zarr", offset_um)

    cycles = ["cycle0", "cycle1"]
    containers = {
        cycle: open_ome_zarr_container(url)
        for cycle, url in zip(cycles, zarr_urls, strict=True)
    }
    msims_reg = {
        cycle: _get_msims(
            image=containers[cycle].get_image(path="0"),
            fov_roi_table=containers[cycle].get_table("FOV_ROI_table"),
            z_project=True,
        )
        for cycle in cycles
    }
    stage = _translations(msims_reg["cycle1"], "fractal_input")

    _pre_register_cycles(containers, msims_reg, cycles, "cycle0", "DAPI", True)

    prereg = _translations(msims_reg["cycle1"], PREREG_TRANSFORM_KEY)
    # The offset was applied along x only, so the correction must undo it there
    # and leave y untouched. Tolerance is 2 px of the pre-registration level.
    atol = 2 * 2 * _PIXEL_SIZE
    for t_stage, t_prereg in zip(stage, prereg, strict=True):
        assert np.allclose(t_prereg - t_stage, [0.0, -offset_um], atol=atol)
    # The raw stage transform is left untouched...
    assert np.allclose(_translations(msims_reg["cycle1"], "fractal_input"), stage)
    # ...and the reference cycle gets the key too, with no correction applied.
    assert np.allclose(_translations(msims_reg["cycle0"], PREREG_TRANSFORM_KEY), 0.0)


def test_shared_canvas_shape_is_union_of_extents():
    """The canvas covers the largest extent of any cycle, in whole pixels."""
    boxes = {
        "a": ({"y": 0.0, "x": 0.0}, {"y": 10.0, "x": 10.0}),
        "b": ({"y": 5.0, "x": 900.0}, {"y": 10.0, "x": 25.5}),
    }
    spacing = {"y": 1.0, "x": 1.0}
    assert _shared_canvas_shape(boxes, ["a", "b"], spacing) == {"y": 10, "x": 26}


def test_cycles_fuse_onto_identical_grid(tmp_path: Path):
    """Two cycles fused on the shared canvas end up with the same geometry.

    This is the invariant the whole approach rests on: identical geometry means
    the region multiview-stitcher registers on is the full canvas.
    """
    cycles = ["cycle0", "cycle1"]
    zarr_urls = _create_offset_plate(
        tmp_path / "grid.zarr", offset_um=500.0, stray_tile_um=-300.0
    )
    containers = {
        c: open_ome_zarr_container(u) for c, u in zip(cycles, zarr_urls, strict=True)
    }
    tiles, boxes = {}, {}
    for c in cycles:
        tiles[c], _ = _load_cycle_tiles(containers[c], z_project=True)
        boxes[c] = _cycle_box(tiles[c])
    spacing = si_utils.get_spacing_from_sim(
        msi_utils.get_sim_from_msim(tiles["cycle0"][0]), asarray=False
    )
    shape = _shared_canvas_shape(boxes, cycles, spacing)

    fused = {}
    for c in cycles:
        _set_anchor_transform(tiles[c], boxes[c][0], list(spacing.keys()))
        fused[c] = _fuse_on_canvas(tiles[c], shape, spacing)

    for c in cycles:
        assert si_utils.get_shape_from_sim(fused[c], asarray=False) == shape
        assert np.allclose(si_utils.get_origin_from_sim(fused[c], asarray=True), 0.0)
        assert np.allclose(
            si_utils.get_spacing_from_sim(fused[c], asarray=True),
            si_utils.get_spacing_from_sim(fused["cycle0"], asarray=True),
        )
    # The stray tile widened the canvas well beyond a single cycle's extent.
    assert shape["x"] * spacing["x"] > boxes["cycle0"][1]["x"]


def test_pre_registration_recovers_offset_larger_than_extent(tmp_path: Path):
    """A cycle displaced by many times the imaged extent is still recovered.

    This is the case the stage coordinates cannot seed at all: the two cycles'
    bounding boxes do not overlap, so a registration started from them has
    nothing to work with.
    """
    offset_um = 500.0  # imaged extent is ~62 um, so ~8x
    zarr_urls = _create_offset_plate(tmp_path / "far.zarr", offset_um)

    cycles = ["cycle0", "cycle1"]
    containers = {
        c: open_ome_zarr_container(u) for c, u in zip(cycles, zarr_urls, strict=True)
    }
    msims_reg = {
        c: _get_msims(
            image=containers[c].get_image(path="0"),
            fov_roi_table=containers[c].get_table("FOV_ROI_table"),
            z_project=True,
        )
        for c in cycles
    }
    stage = _translations(msims_reg["cycle1"], "fractal_input")

    _pre_register_cycles(containers, msims_reg, cycles, "cycle0", "DAPI", True)

    prereg = _translations(msims_reg["cycle1"], PREREG_TRANSFORM_KEY)
    atol = 2 * 2 * _PIXEL_SIZE
    for t_stage, t_prereg in zip(stage, prereg, strict=True):
        assert np.allclose(t_prereg - t_stage, [0.0, -offset_um], atol=atol)


def test_pre_registration_survives_stray_tile_anchor(tmp_path: Path):
    """A tile with a broken stage coordinate drags the anchor, not the result.

    The stray tile becomes the origin of the cycle's bounding box, so the cycle
    is anchored ~300 um away from its actual content. The shared canvas grows to
    cover both, so no data is cropped and the registration still finds the real
    displacement.
    """
    offset_um = 500.0
    zarr_urls = _create_offset_plate(
        tmp_path / "stray.zarr", offset_um, stray_tile_um=-300.0
    )

    cycles = ["cycle0", "cycle1"]
    containers = {
        c: open_ome_zarr_container(u) for c, u in zip(cycles, zarr_urls, strict=True)
    }
    msims_reg = {
        c: _get_msims(
            image=containers[c].get_image(path="0"),
            fov_roi_table=containers[c].get_table("FOV_ROI_table"),
            z_project=True,
        )
        for c in cycles
    }
    stage = _translations(msims_reg["cycle1"], "fractal_input")

    _pre_register_cycles(containers, msims_reg, cycles, "cycle0", "DAPI", True)

    prereg = _translations(msims_reg["cycle1"], PREREG_TRANSFORM_KEY)
    atol = 2 * 2 * _PIXEL_SIZE
    for t_stage, t_prereg in zip(stage, prereg, strict=True):
        assert np.allclose(t_prereg - t_stage, [0.0, -offset_um], atol=atol)


def test_pre_registration_task(tmp_path: Path):
    """The full task runs with pre_registration enabled."""
    plate_path = tmp_path / "prereg_task.zarr"
    zarr_urls = _create_offset_plate(plate_path, offset_um=16 * _PIXEL_SIZE)
    _run_stitch_and_register(zarr_urls, str(tmp_path), pre_registration=True)

    fused_image = _open_fused_image(plate_path)
    assert len(fused_image.channel_labels) == 2
    # The recovered offset puts both cycles on top of each other, so the canvas
    # is a single cycle's extent (plus a few px of registration residue). A
    # coordinate mismatch between the fused cycles used to double it silently.
    assert abs(fused_image.shape[-1] - (2 * _PREREG_FOV_PX - _PREREG_OVERLAP_PX)) <= 16


# Labels and wavelength IDs are deliberately disjoint: a wavelength ID must not
# also be a valid label, or looking one up as the other would appear to work.
_WAVELENGTH_CHANNELS = [
    ("Hoechst", "465"),
    ("CD45", "647"),
]


def _create_plate_with_wavelength_ids(plate_path: Path) -> list[str]:
    """A plate whose channels carry wavelength IDs distinct from their labels."""
    channels_per_acquisition = [
        [
            Channel.default_init(label=label, wavelength_id=wavelength_id)
            for label, wavelength_id in _WAVELENGTH_CHANNELS
        ]
        for _ in range(2)
    ]
    return _build_plate(
        plate_path,
        [_std_rois(), _std_rois()],
        channels_per_acquisition=channels_per_acquisition,
    )


def test_resolve_registration_channel_by_wavelength_id(tmp_path: Path):
    """A wavelength_id selector resolves via the wavelength, not the label.

    ngio's get_channel_idx takes (channel_label, wavelength_id), so passing the
    identifier positionally silently looks a wavelength ID up as a label and
    fails with "Channel with label 465 not found".
    """
    zarr_urls = _create_plate_with_wavelength_ids(tmp_path / "wavelength.zarr")
    image = open_ome_zarr_container(zarr_urls[0]).get_image()

    for label, wavelength_id in _WAVELENGTH_CHANNELS:
        assert wavelength_id not in image.channel_labels  # the test would be vacuous
        selector = ChannelSelectionModel(mode="wavelength_id", identifier=wavelength_id)
        assert _resolve_registration_channel(image, selector) == label


def test_resolve_registration_channel_by_label_and_index(tmp_path: Path):
    """The other two selector modes keep resolving to the same labels."""
    zarr_urls = _create_plate_with_wavelength_ids(tmp_path / "wavelength.zarr")
    image = open_ome_zarr_container(zarr_urls[0]).get_image()

    for idx, (label, _) in enumerate(_WAVELENGTH_CHANNELS):
        by_index = ChannelSelectionModel(mode="index", identifier=str(idx))
        by_label = ChannelSelectionModel(mode="label", identifier=label)
        assert _resolve_registration_channel(image, by_index) == label
        assert _resolve_registration_channel(image, by_label) == label


def test_stitch_and_register_with_wavelength_id_channel(tmp_path: Path):
    """The whole task runs when the registration channel is given as a wavelength."""
    plate_path = tmp_path / "wavelength_task.zarr"
    zarr_urls = _create_plate_with_wavelength_ids(plate_path)

    result = stitch_and_register_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        reference_channel=ChannelSelectionModel(mode="wavelength_id", identifier="465"),
    )
    for item in result["parallelization_list"]:
        stitch_and_register_parallel(
            zarr_url=item["zarr_url"], init_args=item["init_args"]
        )

    fused = _open_fused_image(plate_path)
    # 2 channels x 2 cycles
    assert len(fused.channel_labels) == 4


def _create_plate_with_mismatched_channels(plate_path: Path) -> list[str]:
    """A plate where the second acquisition is missing a channel.

    Acquisition 0 has ["DAPI", "GFP"], acquisition 1 only has ["DAPI"], mimicking
    an imaging round that was acquired with fewer channels than the reference.
    """
    return _build_plate(
        plate_path,
        [_std_rois(), _std_rois()],
        channels_per_acquisition=[["DAPI", "GFP"], ["DAPI"]],
    )


def test_registration_channel_missing_in_other_acquisition(tmp_path: Path):
    """Task fails early and clearly if a cycle lacks the registration channel.

    The registration channel is resolved on the reference acquisition only, so a
    channel absent from another acquisition must be reported up front instead of
    surfacing as a KeyError once registration of that cycle starts.
    """
    plate_path = tmp_path / "test.zarr"
    zarr_urls = _create_plate_with_mismatched_channels(plate_path)

    # Index 1 resolves to "GFP" on the reference, which cycle1 does not have.
    ref_channel = ChannelSelectionModel(mode="index", identifier="1")
    result = stitch_and_register_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        reference_channel=ref_channel,
    )

    for item in result["parallelization_list"]:
        with pytest.raises(ValueError, match="GFP"):
            stitch_and_register_parallel(
                zarr_url=item["zarr_url"],
                init_args=item["init_args"],
            )


def test_all_tiles_non_overlapping_fallback(tmp_path: Path):
    """Task completes when *all* non-ref tiles are outside the reference region.

    This exercises the fallback path in _register_leftover_tiles where no
    inlier tiles exist and stage positions are copied as the final transform.
    """
    plate_path = tmp_path / "test.zarr"
    zarr_urls = _create_plate_with_far_tiles(plate_path, all_nonref_tiles_far=True)
    _run_stitch_and_register(zarr_urls, str(tmp_path))

    fused_image = _open_fused_image(plate_path)
    assert len(fused_image.channel_labels) == 2
    assert any("DAPI" in label for label in fused_image.channel_labels)
    # Union of the reference ([0, 2 fov]) and the far tiles ([10 fov, 12 fov]),
    # so 12 FOV widths. A coordinate mismatch between the fused cycles used to
    # double the canvas silently.
    assert abs(fused_image.shape[-1] - 12 * _FOV_PX) <= 8

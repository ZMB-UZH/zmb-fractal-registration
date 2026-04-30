from pathlib import Path

import numpy as np
from ngio import (
    ChannelSelectionModel,
    ImageInWellPath,
    Roi,
    create_empty_plate,
    create_synthetic_ome_zarr,
    open_ome_zarr_plate,
)
from ngio.tables import RoiTable

from zmb_fractal_registration.stitch_and_register_init import (
    OutlierFilterModel,
    stitch_and_register_init,
)
from zmb_fractal_registration.stitch_and_register_parallel import (
    _detect_outlier_tiles,
    stitch_and_register_parallel,
)

_PIXEL_SIZE = 0.325  # um/px (default for create_synthetic_ome_zarr Cardiomyocyte)
_FOV_PX = 64  # pixels per FOV side
_OVERLAP_PX = 32  # 50% overlap -> enough for multiview_stitcher adjacency detection


def _create_test_plate(plate_path: Path) -> list[str]:
    """Create a minimal test plate with 2 acquisitions, each with 2 overlapping FOVs.

    Each acquisition has one image containing 2 side-by-side tiles with 50% overlap.
    The FOV_ROI_table describes the world-space position of each tile.
    """
    img_shape = (1, _FOV_PX, 2 * _FOV_PX - _OVERLAP_PX)
    fov_size_um = _FOV_PX * _PIXEL_SIZE
    overlap_um = _OVERLAP_PX * _PIXEL_SIZE

    plate = create_empty_plate(
        store=plate_path,
        name="test_plate",
        images=[
            ImageInWellPath(row="A", column=1, path="0", acquisition_id=0),
            ImageInWellPath(row="A", column=1, path="1", acquisition_id=1),
        ],
        overwrite=True,
    )

    zarr_urls = []
    for img_rel_path in plate.images_paths():
        img_path = plate_path / img_rel_path
        container = create_synthetic_ome_zarr(
            store=img_path,
            shape=img_shape,
            axes_names="cyx",
            channels_meta=["DAPI"],
            overwrite=True,
        )

        # Two FOVs side-by-side with 50% overlap in x.
        # FOV 1: world x=[0, fov_size_um]
        #   -> pixels x=[0:FOV_PX]
        # FOV 2: world x=[fov_size_um-overlap_um, 2*fov_size_um-overlap_um]
        #   -> pixels x=[OVERLAP_PX:2*FOV_PX-OVERLAP_PX]
        rois = [
            Roi.from_values(
                slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
                name="FOV_1",
                y_micrometer_original=0.0,
                x_micrometer_original=0.0,
            ),
            Roi.from_values(
                slices={
                    "y": (0.0, fov_size_um),
                    "x": (fov_size_um - overlap_um, 2 * fov_size_um - overlap_um),
                },
                name="FOV_2",
                y_micrometer_original=0.0,
                x_micrometer_original=fov_size_um - overlap_um,
            ),
        ]
        container.add_table("FOV_ROI_table", RoiTable(rois=rois))
        zarr_urls.append(str(img_path))

    return zarr_urls


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

    # Check that the fused acquisition was created
    plate = open_ome_zarr_plate(plate_path)
    fused_acq_id = max(plate.acquisition_ids)
    fused_images = list(plate.get_images(acquisition=fused_acq_id).values())
    assert len(fused_images) == 1

    # Check that channels from both cycles are present (1 channel x 2 cycles = 2)
    fused_image = fused_images[0].get_image()
    assert len(fused_image.channel_labels) == 2
    assert any("DAPI" in label for label in fused_image.channel_labels)


def _create_plate_with_far_tiles(
    plate_path: Path,
    all_nonref_tiles_far: bool = False,
) -> list[str]:
    """Create a test plate where the non-reference cycle contains tiles placed far
    from the reference tiles (no spatial overlap).

    When ``all_nonref_tiles_far`` is False (default), the non-ref cycle has two
    normal overlapping tiles plus one extra tile positioned far in world space
    (reusing the same pixel data as FOV_1 so pixel extraction remains valid).

    When ``all_nonref_tiles_far`` is True, *all* non-ref tiles are placed far away,
    exercising the fallback path where no inlier tiles exist.
    """
    img_shape = (1, _FOV_PX, 2 * _FOV_PX - _OVERLAP_PX)
    fov_size_um = _FOV_PX * _PIXEL_SIZE
    overlap_um = _OVERLAP_PX * _PIXEL_SIZE

    plate = create_empty_plate(
        store=plate_path,
        name="test_plate",
        images=[
            ImageInWellPath(row="A", column=1, path="0", acquisition_id=0),
            ImageInWellPath(row="A", column=1, path="1", acquisition_id=1),
        ],
        overwrite=True,
    )

    zarr_urls = []
    for idx, img_rel_path in enumerate(plate.images_paths()):
        img_path = plate_path / img_rel_path
        container = create_synthetic_ome_zarr(
            store=img_path,
            shape=img_shape,
            axes_names="cyx",
            channels_meta=["DAPI"],
            overwrite=True,
        )

        if idx == 0 or not all_nonref_tiles_far:
            # Standard two overlapping FOVs (also used for ref cycle, idx==0).
            rois = [
                Roi.from_values(
                    slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
                    name="FOV_1",
                    y_micrometer_original=0.0,
                    x_micrometer_original=0.0,
                ),
                Roi.from_values(
                    slices={
                        "y": (0.0, fov_size_um),
                        "x": (fov_size_um - overlap_um, 2 * fov_size_um - overlap_um),
                    },
                    name="FOV_2",
                    y_micrometer_original=0.0,
                    x_micrometer_original=fov_size_um - overlap_um,
                ),
            ]
        else:
            rois = []

        if idx == 1:
            # Place one or two tiles far outside the reference region.
            # The slice coordinates reuse the FOV_1 pixel region so that
            # image.get_roi() succeeds; only the world-space origin is far.
            far_x = 10.0 * fov_size_um
            if all_nonref_tiles_far:
                rois = [
                    Roi.from_values(
                        slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
                        name="FOV_1_far",
                        y_micrometer_original=0.0,
                        x_micrometer_original=far_x,
                    ),
                    Roi.from_values(
                        slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
                        name="FOV_2_far",
                        y_micrometer_original=0.0,
                        x_micrometer_original=far_x + fov_size_um,
                    ),
                ]
            else:
                rois.append(
                    Roi.from_values(
                        slices={"y": (0.0, fov_size_um), "x": (0.0, fov_size_um)},
                        name="FOV_3_far",
                        y_micrometer_original=0.0,
                        x_micrometer_original=far_x,
                    )
                )

        container.add_table("FOV_ROI_table", RoiTable(rois=rois))
        zarr_urls.append(str(img_path))

    return zarr_urls


def _run_stitch_and_register(zarr_urls: list[str], zarr_dir: str) -> None:
    """Run init + parallel stitch-and-register tasks for a list of zarr URLs."""
    ref_channel = ChannelSelectionModel(mode="label", identifier="DAPI")
    result = stitch_and_register_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_channel=ref_channel,
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
    osc = OutlierFilterModel(mode="zscore", threshold=1.5)
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
    osc = OutlierFilterModel(mode="absolute", threshold=50.0)
    outliers = _detect_outlier_tiles(shifts, reg_tile_indices, osc, "cycle1")
    assert outliers == {4}


def test_detect_outlier_tiles_disabled():
    """Disabled mode never flags any tile as an outlier."""
    shifts = [np.array([1.0, 0.0])] * 4 + [np.array([100.0, 0.0])]
    reg_tile_indices = list(range(5))
    osc = OutlierFilterModel(mode="disabled")
    outliers = _detect_outlier_tiles(shifts, reg_tile_indices, osc, "cycle1")
    assert outliers == set()


def test_detect_outlier_tiles_empty_shifts():
    """Empty shift list always returns an empty outlier set."""
    osc = OutlierFilterModel(mode="zscore", threshold=2.0)
    assert _detect_outlier_tiles([], [], osc, "cycle1") == set()


# ---------------------------------------------------------------------------
# Integration tests: non-overlapping and fallback tile handling
# ---------------------------------------------------------------------------


def test_non_overlapping_tile(tmp_path: Path):
    """Task completes when one non-ref tile has no spatial overlap with reference."""
    plate_path = tmp_path / "test.zarr"
    zarr_urls = _create_plate_with_far_tiles(plate_path, all_nonref_tiles_far=False)
    _run_stitch_and_register(zarr_urls, str(tmp_path))

    plate = open_ome_zarr_plate(plate_path)
    fused_acq_id = max(plate.acquisition_ids)
    fused_images = list(plate.get_images(acquisition=fused_acq_id).values())
    assert len(fused_images) == 1
    fused_image = fused_images[0].get_image()
    assert len(fused_image.channel_labels) == 2
    assert any("DAPI" in label for label in fused_image.channel_labels)


def test_all_tiles_non_overlapping_fallback(tmp_path: Path):
    """Task completes when *all* non-ref tiles are outside the reference region.

    This exercises the fallback path in _register_leftover_tiles where no
    inlier tiles exist and stage positions are copied as the final transform.
    """
    plate_path = tmp_path / "test.zarr"
    zarr_urls = _create_plate_with_far_tiles(plate_path, all_nonref_tiles_far=True)
    _run_stitch_and_register(zarr_urls, str(tmp_path))

    plate = open_ome_zarr_plate(plate_path)
    fused_acq_id = max(plate.acquisition_ids)
    fused_images = list(plate.get_images(acquisition=fused_acq_id).values())
    assert len(fused_images) == 1
    fused_image = fused_images[0].get_image()
    assert len(fused_image.channel_labels) == 2
    assert any("DAPI" in label for label in fused_image.channel_labels)

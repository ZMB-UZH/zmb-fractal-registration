from pathlib import Path

from ngio import (
    ChannelSelectionModel,
    ImageInWellPath,
    Roi,
    create_empty_plate,
    create_synthetic_ome_zarr,
    open_ome_zarr_plate,
)
from ngio.tables import RoiTable

from zmb_fractal_registration.stitch_and_register_init import stitch_and_register_init
from zmb_fractal_registration.stitch_and_register_parallel import (
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

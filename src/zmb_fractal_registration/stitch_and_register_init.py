"""Fractal init task to stitch and register multiple acquisitions of a plate."""

import logging
from pathlib import Path
from typing import Literal, Optional

from ngio import (
    ChannelSelectionModel,
    ImageInWellPath,
    create_empty_plate,
    open_ome_zarr_plate,
)
from pydantic import BaseModel, ConfigDict, model_validator, validate_call


class TileCorrectionModel(BaseModel):
    """Settings for correcting non-overlapping tiles and filtering outliers."""

    model_config = ConfigDict(extra="forbid")

    correction_method: Literal["reregister", "mean_shift"] = "reregister"
    """How to correct leftover tiles (outliers and non-overlapping tiles).
    'reregister': re-stitch leftover tiles against fixed inlier tiles.
    'mean_shift': apply the mean (registered - stage) shift of the inlier tiles
    directly, without re-registration."""
    outlier_filter_mode: Literal["disabled", "absolute", "zscore"] = "disabled"
    """Outlier detection method. 'absolute': threshold in um; 'zscore': z-score
    threshold (2-3 is typical)."""
    threshold: float | None = None
    """Threshold value (um for 'absolute', z-score for 'zscore'). Required
    unless mode is 'disabled'."""

    @model_validator(mode="after")
    def _check_threshold(self) -> "TileCorrectionModel":
        if self.outlier_filter_mode != "disabled" and self.threshold is None:
            raise ValueError(
                f"`threshold` must be set when mode is '{self.outlier_filter_mode}'."
            )
        return self


class AcquisitionInputModel(BaseModel):
    """Input model for acquisitions."""

    model_config = ConfigDict(extra="forbid")

    acquisition_ID: int
    """Acquisition ID in plate."""
    optional_cycle_name: Optional[str] = None
    """Optional cycle name. Will be appended to original channel labels.
        If None, defaults to `cycle{acquisition_ID}`."""

    @property
    def cycle_name(self) -> str:
        """The cycle name, defaulting to `cycle{acquisition_ID}`."""
        return self.optional_cycle_name or f"cycle{self.acquisition_ID}"


class AcquisitionsSelectionModel(BaseModel):
    """Model to select which acquisitions to process."""

    model_config = ConfigDict(extra="forbid")

    use_all_acquisitions: bool = True
    """If True, all acquisitions in the plate are used and `acquisitions` must
        be empty."""
    acquisitions: list[AcquisitionInputModel] = []
    """List of acquisitions to include. Only used when `use_all_acquisitions`
        is False."""

    @model_validator(mode="after")
    def check_acquisitions_empty_when_use_all(self) -> "AcquisitionsSelectionModel":
        """Validate acquisitions list is empty when use_all_acquisitions is True."""
        if self.use_all_acquisitions and self.acquisitions:
            raise ValueError(
                "`acquisitions` must be empty when `use_all_acquisitions` is True."
            )
        return self


@validate_call
def stitch_and_register_init(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    acquisitions_to_include: AcquisitionsSelectionModel = AcquisitionsSelectionModel(),
    reference_acquisition: AcquisitionInputModel = AcquisitionInputModel(
        acquisition_ID=0
    ),
    reference_channel: ChannelSelectionModel = ChannelSelectionModel(
        mode="index", identifier="0"
    ),
    new_plate_suffix: str = "fused",
    pyramid_level: int = 0,
    z_project: bool = True,
    pre_registration: bool = True,
    tile_correction: TileCorrectionModel = TileCorrectionModel(),
    fusion_region: Literal["union", "intersection", "intersection_bbox"] = "union",
    interpolation_order: int = 0,
):
    """Stitch and register multiple acquisitions of a plate.

    Task to stitch and register multiple acquisitions of a plate using the
    `multiview_stitcher` package. In a first step, the tiles of the reference
    acquisition will stitched together to create a reference image. In a second
    step, the tiles of the other acquisitions will be registered individually
    to the reference image. This workflow is similar to what the Ashlar package
    does https://github.com/labsyspharm/ashlar, but also works in 3D.

    The fused output is written to a new plate (named after the original with
    `new_plate_suffix` appended); the original plate is not modified.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr images to
            be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: Directory in which the new plate holding the fused output is
            created.
            (Standard argument for Fractal tasks, managed by Fractal server).
        acquisitions_to_include: Selection of acquisitions to process. If
            `use_all_acquisitions` is True (default), all acquisitions in the
            plate are used. Otherwise, only the acquisitions listed in
            `acquisitions` are processed.
        reference_acquisition: Acquisition to use as reference for
            registration.
        reference_channel: Channel to use as reference for stitching and
            registration.
        new_plate_suffix: Suffix for the new plate holding the fused output:
            the fused images of `plate.zarr` are written to
            `plate_{new_plate_suffix}.zarr` inside `zarr_dir`. An existing
            plate at that path is overwritten.
        pyramid_level: Pyramid level to use for stitching and registration.
        z_project: If True, calculate stitching/registration on a z-projection
            and apply the calculated transformations to the full 3D image.
            If False, operate on the full image volume. Only used in case of
            3D images.
        pre_registration: If True, perform a rough pre-registration of the
            acquisitions before the accurate stitching and registration. Use
            this if there are significant global shifts between acquisitions.
        tile_correction: Settings for correcting non-overlapping tiles and
            filtering outliers.
        fusion_region: Which region of the registered cycles to save.
            'union': save the full extent covered by any cycle.
            'intersection': tight box of the region covered by every cycle;
            uncovered pixels inside it are set to 0.
            'intersection_bbox': largest box fully covered by every cycle
            (no pixels set to 0).
        interpolation_order: Spline interpolation order for resampling tiles
            into the fused output. 0 is nearest-neighbor (preserves
            original pixel values), 1 is linear.
    """
    # TODO: Currently, we ignore the zarr_urls, and process all acquisitions found in
    # the plate. -> think about how to filter the acquisitions based on the zarr_urls

    if not new_plate_suffix:
        raise ValueError("`new_plate_suffix` must not be empty.")

    zarr_paths = [Path(url) for url in zarr_urls]
    # extract all plate roots
    plate_roots = {p.parent.parent.parent for p in zarr_paths}
    parallelization_list = []
    for plate_root in plate_roots:
        ome_zarr_plate = open_ome_zarr_plate(plate_root)
        # filter acquisitions based on acquisitions_to_include
        acquisition_ids = ome_zarr_plate.acquisition_ids
        if not acquisitions_to_include.use_all_acquisitions:
            acquisition_ids_filtered = []
            cycle_names = []
            for acq in acquisitions_to_include.acquisitions:
                if acq.acquisition_ID in acquisition_ids:
                    acquisition_ids_filtered.append(acq.acquisition_ID)
                    cycle_names.append(acq.cycle_name)
                else:
                    logging.warning(
                        f"Acquisition ID {acq.acquisition_ID} not found in plate at "
                        f"{plate_root}. Skipping this acquisition."
                    )
        else:
            acquisition_ids_filtered = acquisition_ids
            cycle_names = [f"cycle{acq_id}" for acq_id in acquisition_ids]

        if len(acquisition_ids_filtered) < 2:
            logging.info(
                f"Plate at {plate_root} has less than two acquisitions. Skipping."
            )
            continue
        if reference_acquisition.acquisition_ID not in acquisition_ids:
            raise ValueError(
                f"Reference acquisition ID {reference_acquisition.acquisition_ID} not "
                f"found in plate at {plate_root}."
            )
        elif reference_acquisition.acquisition_ID not in acquisition_ids_filtered:
            logging.warning(
                f"Reference acquisition ID {reference_acquisition.acquisition_ID} not "
                f"in acquisitions_to_include for plate at {plate_root}. Adding it to "
                "the list of acquisitions to process."
            )
            acquisition_ids_filtered.append(reference_acquisition.acquisition_ID)
            cycle_names.append(reference_acquisition.cycle_name)

        # Collect and validate the input image per acquisition for every well.
        wells = []
        for well_path in ome_zarr_plate.wells_paths():
            row, column = well_path.split("/")
            acquisition_paths = []
            for acquisition_id in acquisition_ids_filtered:
                images = ome_zarr_plate.well_images_paths(
                    row=row, column=int(column), acquisition=acquisition_id
                )
                if len(images) == 0:
                    raise ValueError(
                        f"No images found for acquisition {acquisition_id} in well "
                        f"{row}_{column} of plate at {plate_root}."
                    )
                elif len(images) > 1:
                    raise ValueError(
                        f"Multiple images found for acquisition {acquisition_id} in "
                        f"well {row}_{column} of plate at {plate_root}. This task only "
                        "supports one image per acquisition per well."
                    )
                else:
                    acquisition_paths.append(images[0])
            wells.append((row, column, acquisition_paths))

        # Create a new plate holding one fused image per well; the input plate
        # is never modified, so a failed run leaves it fully intact.
        new_plate_root = Path(zarr_dir) / f"{plate_root.stem}_{new_plate_suffix}.zarr"
        if any(new_plate_root.resolve() == p.resolve() for p in plate_roots):
            raise ValueError(
                f"Output plate path {new_plate_root} collides with an input plate. "
                "Choose a different `new_plate_suffix`."
            )
        logging.info(f"Writing fused output to new plate at {new_plate_root}.")
        new_plate = create_empty_plate(
            store=new_plate_root,
            name=new_plate_root.stem,
            images=[
                ImageInWellPath(
                    row=row,
                    column=column,
                    path="0",
                    acquisition_id=0,
                    acquisition_name="fused",
                )
                for row, column, _ in wells
            ],
            overwrite=True,
        )

        for row, column, acquisition_paths in wells:
            zarr_url_new = (
                new_plate_root / new_plate.well_images_paths(row=row, column=column)[0]
            ).as_posix()
            init_args = {
                "zarr_urls_to_register": [
                    (plate_root / p).as_posix() for p in acquisition_paths
                ],
                "cycle_names": cycle_names,
                "reference_acquisition_index": acquisition_ids_filtered.index(
                    reference_acquisition.acquisition_ID
                ),
                "reference_channel": reference_channel.model_dump(),
                "pyramid_level": pyramid_level,
                "z_project": z_project,
                "pre_registration": pre_registration,
                "tile_correction": tile_correction.model_dump(),
                "fusion_region": fusion_region,
                "interpolation_order": interpolation_order,
            }
            parallelization_list.append(
                {
                    "zarr_url": zarr_url_new,
                    "init_args": init_args,
                }
            )

    logging.info("Returning parallelization list for combine_acquisitions_parallel.")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitch_and_register_init)

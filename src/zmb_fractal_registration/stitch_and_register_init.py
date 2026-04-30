"""Fractal init task to stitch and register multiple acquisitions of a plate."""

import logging
from pathlib import Path
from typing import Literal, Optional

from ngio import ChannelSelectionModel, open_ome_zarr_plate
from pydantic import BaseModel, model_validator, validate_call


class OutlierFilterModel(BaseModel):
    """Settings for outlier filtering."""

    mode: Literal["disabled", "absolute", "zscore"] = "disabled"
    """Outlier detection method. 'absolute': threshold in um; 'zscore': z-score
    threshold (2-3 is typical)."""
    threshold: float | None = None
    """Threshold value (um for 'absolute', z-score for 'zscore'). Required unless
    mode is 'disabled'."""

    @model_validator(mode="after")
    def _check_threshold(self) -> "OutlierFilterModel":
        if self.mode != "disabled" and self.threshold is None:
            raise ValueError(f"`threshold` must be set when mode is '{self.mode}'.")
        return self


class AcquisitionInputModel(BaseModel):
    """Input model for acquisitions."""

    acquisition_ID: int
    """Acquisition ID in plate."""
    optional_cycle_name: Optional[str] = None
    """Optional cycle name. Will be appended to original channel labels.
        If None, defaults to `cycle{acquisition_ID}`."""


class AcquisitionsSelectionModel(BaseModel):
    """Model to select which acquisitions to process."""

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
    pyramid_level: int = 0,
    z_project: bool = True,
    outlier_filter: OutlierFilterModel = OutlierFilterModel(),
    keep_original_acquisitions: bool = True,
):
    """Stitch and register multiple acquisitions of a plate.

    Task to stitch and register multiple acquisitions of a plate using the
    `multiview_stitcher` package. In a first step, the tiles of the reference
    acquisition will stitched together to create a reference image. In a second
    step, the tiles of the other acquisitions will be registered individually
    to the reference image. This workflow is similar to what the Ashlar package
    does https://github.com/labsyspharm/ashlar, but also works in 3D.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr images to
            be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: Not used for this task.
            (Standard argument for Fractal tasks, managed by Fractal server).
        acquisitions_to_include: Selection of acquisitions to process. If
            `use_all_acquisitions` is True (default), all acquisitions in the
            plate are used. Otherwise, only the acquisitions listed in
            `acquisitions` are processed.
        reference_acquisition: Acquisition to use as reference for
            registration.
        reference_channel: Channel to use as reference for stitching and
            registration.
        pyramid_level: Pyramid level to use for stitching and registration.
        z_project: If True, calculate stitching/registration on a z-projection
            and apply the calculated transformations to the full 3D image.
            If False, operate on the full image volume. Only used in case of
            3D images.
        outlier_filter: Settings for filtering out large shifts during
            registration.
        keep_original_acquisitions: If True, keep original acquisitions after
            registration. If False, remove them.
    """
    # TODO: Currently, we ignore the zarr_urls, and process all acquisitions found in
    # the plate. -> think about how to filter the acquisitions based on the zarr_urls

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
                    if acq.optional_cycle_name:
                        cycle_names.append(acq.optional_cycle_name)
                    else:
                        cycle_names.append(f"cycle{acq.acquisition_ID}")
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
            if reference_acquisition.optional_cycle_name:
                cycle_names.append(reference_acquisition.optional_cycle_name)
            else:
                cycle_names.append(f"cycle{reference_acquisition.acquisition_ID}")

        # Create a single merged output acquisition.
        new_acquisition_id = max(acquisition_ids) + 1
        ome_zarr_plate.add_acquisition(new_acquisition_id, "fused")
        logging.info(
            f"New combined acquisition will have ID {new_acquisition_id} and name "
            "'fused'."
        )

        for well_path in ome_zarr_plate.wells_paths():
            row = well_path.split("/")[0]
            column = int(well_path.split("/")[1])
            acquisition_paths = []
            for acquisition_id in acquisition_ids_filtered:
                images = ome_zarr_plate.well_images_paths(
                    row=row, column=column, acquisition=acquisition_id
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

            # TODO: think about changing new path simply to str(new_acquisition_id)
            zarr_url_new = (
                plate_root
                / ome_zarr_plate.add_image(
                    row=row,
                    column=column,
                    image_path="fused",
                    acquisition_id=new_acquisition_id,
                )
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
                "keep_original_acquisitions": keep_original_acquisitions,
                "outlier_filter": outlier_filter.model_dump(),
            }
            parallelization_list.append(
                {
                    "zarr_url": zarr_url_new,
                    "init_args": init_args,
                }
            )
            if not keep_original_acquisitions:
                # remove individual acquisitions from plate metadata
                for acquisition_path in acquisition_paths:
                    ome_zarr_plate.remove_image(
                        row=row,
                        column=column,
                        image_path=str(acquisition_path.split("/")[-1]),
                    )

    logging.info("Returning parallelization list for combine_acquisitions_parallel.")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitch_and_register_init)

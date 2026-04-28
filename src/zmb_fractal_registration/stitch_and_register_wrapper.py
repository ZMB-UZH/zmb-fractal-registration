"""Wrapper to run stitch_and_register_parallel locally without a Fractal server."""

import logging
from typing import Optional

from ngio import ChannelSelectionModel

from zmb_fractal_registration.stitch_and_register_init import (
    OutlierFilterModel,
)
from zmb_fractal_registration.stitch_and_register_parallel import (
    InitArgsStitchAndRegisterParallel,
    stitch_and_register_parallel,
)

logger = logging.getLogger(__name__)


def stitch_and_register(
    *,
    input_zarr_urls: list[str],
    output_zarr_url: str,
    reference_cycle_index: int = 0,
    reference_channel: ChannelSelectionModel = ChannelSelectionModel(
        mode="index", identifier="0"
    ),
    cycle_names: Optional[list[str]] = None,
    z_project: bool = True,
    pyramid_level: int = 0,
    show_logs: bool = False,
    log_level: int = logging.INFO,
    outlier_filter: OutlierFilterModel = OutlierFilterModel(),
) -> dict:
    """Stitch and register multiple acquisitions locally.

    Convenience wrapper around `stitch_and_register_parallel` for local use
    without a Fractal server.

    Args:
        input_zarr_urls: List of absolute paths to the OME-Zarr images to
            stitch and register. Each URL corresponds to one acquisition/cycle.
        output_zarr_url: Absolute path where the fused output OME-Zarr image
            will be written.
        reference_cycle_index: Index into `input_zarr_urls` pointing to the
            acquisition used as the stitching/registration reference.
            Defaults to 0.
        reference_channel: Channel used as reference for stitching and
            registration.
        cycle_names: Optional names for each acquisition. Used to disambiguate
            channels in the output (e.g. `DAPI_cycle0`). If None, defaults to
            `cycle0`, `cycle1`, etc.
        z_project: If True, compute stitching/registration on a maximum-
            intensity Z-projection and apply the transforms to the full 3D
            volume. If False, operate on the full volume directly.
        pyramid_level: Pyramid level used for stitching/registration.
        show_logs: If True, configure root logging so task logs are printed.
        log_level: Logging level used when show_logs is True.
        outlier_filter: Settings for outlier registration-shift correction.
            See `OutlierShiftCorrectionModel` for details.
    """
    if show_logs:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        else:
            logging.getLogger().setLevel(log_level)

        logger.info("Enabled console logging for stitch_and_register wrapper.")

    if cycle_names is None:
        cycle_names = [f"cycle{i}" for i in range(len(input_zarr_urls))]

    if len(cycle_names) != len(input_zarr_urls):
        raise ValueError("`cycle_names` length must match `input_zarr_urls` length.")

    init_args = InitArgsStitchAndRegisterParallel(
        zarr_urls_to_register=input_zarr_urls,
        cycle_names=cycle_names,
        reference_acquisition_index=reference_cycle_index,
        reference_channel=reference_channel,
        pyramid_level=pyramid_level,
        z_project=z_project,
        keep_original_acquisitions=True,
        outlier_filter=outlier_filter,
    )

    return stitch_and_register_parallel(
        zarr_url=output_zarr_url,
        init_args=init_args,
    )

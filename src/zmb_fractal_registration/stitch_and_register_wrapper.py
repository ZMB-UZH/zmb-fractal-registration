"""Wrapper to run stitch_and_register_parallel locally without a Fractal server."""

import logging
from typing import Literal, Optional

from ngio import ChannelSelectionModel

from zmb_fractal_registration.stitch_and_register_init import (
    TileCorrectionModel,
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
    pyramid_level: int = 0,
    z_project: bool = True,
    pre_registration: bool = True,
    tile_correction: TileCorrectionModel = TileCorrectionModel(),
    fusion_region: Literal["union", "intersection", "intersection_bbox"] = "union",
    interpolation_order: int = 0,
    show_logs: bool = False,
    log_level: int = logging.INFO,
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
        pyramid_level: Pyramid level used for stitching/registration.
        z_project: If True, compute stitching/registration on a maximum-
            intensity Z-projection and apply the transforms to the full 3D
            volume. If False, operate on the full volume directly.
        pre_registration: If True, perform a rough pre-registration of the
            acquisitions before the accurate stitching and registration. Use
            this if there are significant global shifts between acquisitions.
        tile_correction: Settings for correcting non-overlapping tiles and
            filtering outliers. See `TileCorrectionModel` for details.
        fusion_region: Which region of the registered cycles to save.
            'union': save the full extent covered by any cycle.
            'intersection': tight box of the region covered by every cycle;
            uncovered pixels inside it are set to 0.
            'intersection_bbox': largest box fully covered by every cycle
            (no pixels set to 0).
        interpolation_order: Spline interpolation order for resampling tiles
            into the fused output. 0 is nearest-neighbor (preserves
            original pixel values), 1 is linear.
        show_logs: If True, configure root logging so task logs are printed.
        log_level: Logging level used when show_logs is True.
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
        pre_registration=pre_registration,
        tile_correction=tile_correction,
        fusion_region=fusion_region,
        interpolation_order=interpolation_order,
    )

    return stitch_and_register_parallel(
        zarr_url=output_zarr_url,
        init_args=init_args,
    )

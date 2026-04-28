"""Wrapper to run stitch_and_register_parallel locally without a Fractal server."""

from typing import Optional

from ngio import ChannelSelectionModel

from zmb_fractal_registration.stitch_and_register_parallel import (
    InitArgsStitchAndRegisterParallel,
    stitch_and_register_parallel,
)


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
    """
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
    )

    return stitch_and_register_parallel(
        zarr_url=output_zarr_url,
        init_args=init_args,
    )

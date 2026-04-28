### Purpose

Stitches together tiles from multiple acquisitions of a plate and registers them into a single fused image per well. This task is designed for multi-cycle imaging experiments where the same well is imaged across multiple acquisitions (e.g., sequential rounds of staining).

The workflow proceeds in two steps:

1. **Stitching** — tiles within the reference acquisition are stitched together using phase cross correlation.
2. **Registration** — tiles from all other acquisitions are independently registered to the stitched reference image, correcting for shifts between cycles.

Key features:
- Uses the [`multiview_stitcher`](https://multiview-stitcher.readthedocs.io) package for stitching and registration.
- Conceptually similar to [Ashlar](https://github.com/labsyspharm/ashlar), but also supports **3D volumes** in addition to 2D.
- When `z_project` is enabled (default), stitching and registration are computed on a maximum-intensity Z-projection, and the resulting transformations are applied to the full 3D volume — reducing computation time.
- All acquisitions to process can be explicitly selected via `acquisitions_to_include`; if omitted, all acquisitions in the plate are used.
- Each acquisition can be assigned an optional **cycle name** to disambiguate channels from different rounds. If not specified, cycle names default to `cycle0`, `cycle1`, etc.

### Outputs

Creates a new OME-Zarr acquisition named **`fused`** within the same plate. For each well, the fused image contains **all channels from all registered acquisitions**, concatenated along the channel axis. Each channel is renamed with a `_{cycle_name}` suffix (e.g., `DAPI_cycle0`, `GFP_cycle1`) to distinguish channels across cycles.

If `keep_original_acquisitions` is `False`, the individual input acquisitions are removed from the plate after fusion.

### Limitations

- Each acquisition must contain a **`FOV_ROI_table`** with original stage coordinates to initialize the stitching.
- Supports only **one image per acquisition per well** — plates with multiple fields of view stored as separate images per acquisition are not supported.
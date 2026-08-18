### Purpose

Stitches together tiles from multiple acquisitions of a plate and registers them into a single fused image per well. This task is designed for multi-cycle imaging experiments where the same well is imaged across multiple acquisitions (e.g., sequential rounds of staining).

The workflow proceeds in three steps:

1. **Stitching** - tiles within the reference acquisition are stitched together using phase cross correlation.
2. **Registration** - tiles from all other acquisitions are independently registered to the stitched reference image, correcting for shifts between cycles.
3. **Fusion** - all cycles are resampled onto a shared output canvas and written as a single image per well.

Optionally, a rough **pre-registration** runs before step 1 (see `pre_registration` below).

Key features:
- Uses the [`multiview_stitcher`](https://multiview-stitcher.readthedocs.io) package for stitching and registration.
- Conceptually similar to [Ashlar](https://github.com/labsyspharm/ashlar), but also supports **3D volumes** in addition to 2D.
- When `z_project` is enabled (default), stitching and registration are computed on a maximum-intensity Z-projection, and the resulting transformations are applied to the full 3D volume - reducing computation time.
- All acquisitions to process can be explicitly selected via `acquisitions_to_include`; if omitted, all acquisitions in the plate are used.
- Each acquisition can be assigned an optional **cycle name** to disambiguate channels from different rounds. If not specified, cycle names default to `cycle0`, `cycle1`, etc.
- `pre_registration` (disabled by default) adds a rough alignment step in front of the accurate stitching and registration: every acquisition is fused from its original stage coordinates at the **coarsest pyramid level**, and those fused acquisitions are registered against the fused reference acquisition as a whole. The resulting per-acquisition shift is used as the starting point of the accurate registration. Enable it when the shift between acquisitions is large compared to the tile overlap - in that case the individual tiles of a cycle would otherwise not overlap the reference tile they belong to, and the per-tile registration cannot recover the shift. The step does **not** assume the stage coordinates are approximately right, since that is what it exists to correct: each acquisition is anchored on the origin of its own tile bounding box (i.e. the acquisitions are assumed to cover roughly the same area) and all of them are fused onto one shared canvas covering every acquisition's extent. Shifts of any size are therefore recoverable, including ones that place the acquisitions' stage coordinates completely apart. Acquisitions whose pre-registration fails (e.g. no image overlap at all) log a warning and keep their raw stage coordinates.
- The extent of the fused output is controlled by `fusion_region` (see **Outputs**), so cycles that do not cover exactly the same area can be cropped down to their common region.
- `interpolation_order` sets the spline order used when resampling tiles into the fused output. The default of `0` (nearest neighbour) preserves the original pixel values exactly.

### Outputs

Creates a **new OME-Zarr plate**, named after the original with `new_plate_suffix` appended (default: `plate.zarr` → `plate_fused.zarr`), holding one fused image per well under a single acquisition named `fused`. For each well, the fused image contains **all channels from all registered acquisitions**, concatenated along the channel axis. Each channel is renamed with a `_{cycle_name}` suffix (e.g., `DAPI_cycle0`, `GFP_cycle1`) to distinguish channels across cycles.

After registration the cycles rarely cover exactly the same area, so `fusion_region` selects which part of that area is written:

- **`union`** (default) - the full extent covered by *any* cycle. Nothing is cropped, but pixels near the border are present in only some cycles; where a cycle has no data it stays at the fill value.
- **`intersection`** - the tight bounding box around the region covered by *every* cycle. Pixels inside that box that are not covered by all cycles are set to `0`. Use this when you want the smallest output that still contains the whole common region.
- **`intersection_bbox`** - the largest rectangular box that is *entirely* covered by every cycle. No pixels are set to `0`, at the cost of discarding parts of the common region that do not fit into a single box.

`intersection` and `intersection_bbox` differ only when the covered region is not itself rectangular (e.g. a missing or badly shifted tile punches a hole in it). If the cycles share no common region at all, both raise an error.

### Limitations

- Each acquisition must contain a **`FOV_ROI_table`** with original stage coordinates to initialize the stitching.
- Supports only **one image per acquisition per well** - plates with multiple fields of view stored as separate images per acquisition are not supported.
- Large shifts between cycles that are not represented in the original stage coordinates are only handled when `pre_registration` is enabled, which assumes the acquisitions cover roughly the same area. A single tile with a broken stage coordinate widens the shared registration canvas (this is logged as a warning) but does not affect the result.
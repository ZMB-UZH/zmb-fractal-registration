"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    CompoundTask,
)

AUTHORS = "Flurin Sturzenegger"


DOCS_LINK = None


TASK_LIST = [
    CompoundTask(
        name="Stitch and register acquisitions",
        executable_init="stitch_and_register_init.py",
        executable="stitch_and_register_parallel.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 4, "mem": 8000},
        category="Registration",
        tags=["Registration", "Stitching", "Fusion", "multiview-stitcher", "2D", "3D"],
        docs_info="file:docs_info/stitch_and_register.md",
        modality="HCS",
        output_types={"stitched": True, "registered": True},
    ),
]

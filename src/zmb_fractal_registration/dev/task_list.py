"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Flurin Sturzenegger"


DOCS_LINK = None


TASK_LIST = [
    
    ParallelTask(
        name="Threshold Segmentation",
        executable="threshold_segmentation_task.py",
        # Modify the meta according to your task requirements
        # If the task requires a GPU, add "needs_gpu": True
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation"],
        docs_info="file:docs_info/threshold_segmentation_task.md",
    ),
    
    
    
]

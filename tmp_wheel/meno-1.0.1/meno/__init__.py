"""Meno - Topic Modeling Toolkit.

A toolkit for topic modeling with both traditional (LDA) and
modern embedding-based approaches, visualization, and reporting.
"""

__version__ = "1.0.0"

# Import key components for easy access
from .meno import MenoTopicModeler
from .workflow import (
    MenoWorkflow, create_workflow, 
    load_workflow_config, save_workflow_config
)
from .utils.config import WorkflowMenoConfig

# Re-export key functions
from .preprocessing import correct_spelling, expand_acronyms, normalize_text

__all__ = [
    "MenoTopicModeler",
    "MenoWorkflow",
    "create_workflow",
    "load_workflow_config",
    "save_workflow_config",
    "WorkflowMenoConfig",
    "correct_spelling",
    "expand_acronyms",
    "normalize_text",
]
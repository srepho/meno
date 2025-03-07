"""Meno - Topic Modeling Toolkit.

A toolkit for topic modeling with both traditional (LDA) and
modern embedding-based approaches, visualization, and reporting.
"""

__version__ = "1.0.3"

# Import key components for easy access
from .meno import MenoTopicModeler
from .workflow import (
    MenoWorkflow, create_workflow, 
    load_workflow_config, save_workflow_config
)
from .utils.config import WorkflowMenoConfig

# Re-export key functions
from .preprocessing import correct_spelling, expand_acronyms, normalize_text

# Download English spaCy model on import if not already available
import logging
import importlib
import subprocess
import sys
import os
import warnings

logger = logging.getLogger(__name__)

def _ensure_spacy_model():
    """Ensure the required spaCy model is installed."""
    try:
        # Try to import en_core_web_sm
        en_core_web_sm = importlib.import_module("en_core_web_sm")
        logger.debug("spaCy model en_core_web_sm already installed")
    except ImportError:
        try:
            # Try to load the model
            import spacy
            try:
                spacy.load("en_core_web_sm")
                logger.debug("spaCy model successfully loaded")
            except OSError:
                logger.info("Downloading spaCy language model (en_core_web_sm)...")
                warnings.warn(
                    "Downloading spaCy language model (en_core_web_sm). "
                    "This is a one-time operation."
                )
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL if not os.environ.get("MENO_DEBUG") else None
                )
                logger.info("spaCy language model downloaded successfully")
        except Exception as e:
            warnings.warn(
                f"Failed to download spaCy model: {e}. "
                "Text lemmatization may be limited. "
                "You can manually install the model with: python -m spacy download en_core_web_sm"
            )
            logger.warning(f"Failed to download spaCy model: {e}")

# Try to download the model when importing the package
try:
    _ensure_spacy_model()
except Exception as e:
    logger.debug(f"Non-critical error while ensuring spaCy model: {e}")

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
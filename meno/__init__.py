"""Meno - Topic Modeling Toolkit.

A toolkit for topic modeling with both traditional (LDA) and
modern embedding-based approaches, visualization, and reporting.
"""

__version__ = "1.1.1"

# Import key components for easy access
from .meno import MenoTopicModeler
from .workflow import (
    MenoWorkflow, create_workflow, 
    load_workflow_config, save_workflow_config
)
from .utils.config import WorkflowMenoConfig

# Re-export key functions
from .preprocessing import correct_spelling, expand_acronyms, normalize_text

# Re-export lightweight models for easy access
try:
    from .modeling.simple_models.lightweight_models import (
        SimpleTopicModel,
        TFIDFTopicModel,
        NMFTopicModel,
        LSATopicModel
    )
except ImportError:
    # Graceful fallback if scikit-learn is not installed
    pass

# Re-export feedback system
try:
    from .active_learning.simple_feedback import SimpleFeedback, TopicFeedbackManager
except ImportError:
    # Graceful fallback if dependencies are missing
    pass

# Re-export feedback visualization components
try:
    from .visualization.enhanced_viz.feedback_viz import (
        plot_feedback_impact,
        create_feedback_comparison_dashboard,
        plot_topic_feedback_distribution
    )
except ImportError:
    # Graceful fallback if visualization dependencies are missing
    pass

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
    # Core classes
    "MenoTopicModeler",
    "MenoWorkflow",
    "create_workflow",
    "load_workflow_config",
    "save_workflow_config",
    "WorkflowMenoConfig",
    
    # Preprocessing functions
    "correct_spelling",
    "expand_acronyms",
    "normalize_text",
    
    # Lightweight models
    "SimpleTopicModel",
    "TFIDFTopicModel",
    "NMFTopicModel",
    "LSATopicModel",
    
    # Feedback system
    "SimpleFeedback",
    "TopicFeedbackManager",
    
    # Feedback visualization
    "plot_feedback_impact",
    "create_feedback_comparison_dashboard",
    "plot_topic_feedback_distribution",
]
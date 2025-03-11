"""Topic modeling module for supervised and unsupervised topic discovery."""

from typing import List, Dict, Optional, Union

# Re-export key classes and functions
from .unsupervised import LDAModel, EmbeddingClusterModel
from .supervised import TopicMatcher
from .embeddings import DocumentEmbedding
from .simple_models import SimpleTopicModel, TFIDFTopicModel, NMFTopicModel, LSATopicModel

# Import LLM topic labeling if available
try:
    from .llm_topic_labeling import LLMTopicLabeler
    __all__ = [
        "LDAModel",
        "EmbeddingClusterModel",
        "TopicMatcher",
        "DocumentEmbedding",
        "SimpleTopicModel",
        "TFIDFTopicModel",
        "NMFTopicModel",
        "LSATopicModel",
        "LLMTopicLabeler",
    ]
except ImportError:
    __all__ = [
        "LDAModel",
        "EmbeddingClusterModel",
        "TopicMatcher",
        "DocumentEmbedding",
        "SimpleTopicModel",
        "TFIDFTopicModel",
        "NMFTopicModel",
        "LSATopicModel",
    ]
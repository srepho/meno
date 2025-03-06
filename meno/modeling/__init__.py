"""Topic modeling module for supervised and unsupervised topic discovery."""

from typing import List, Dict, Optional, Union

# Re-export key classes and functions
from .unsupervised import LDAModel, EmbeddingClusterModel
from .supervised import TopicMatcher
from .embeddings import DocumentEmbedding

__all__ = [
    "LDAModel",
    "EmbeddingClusterModel",
    "TopicMatcher",
    "DocumentEmbedding",
]
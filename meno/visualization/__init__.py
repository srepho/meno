"""Visualization module for topic model results."""

from typing import List, Dict, Optional, Union

# Re-export key functions and classes
from .static_plots import plot_topic_distribution, plot_word_cloud
from .interactive_plots import plot_embeddings, plot_topic_clusters
from .umap_viz import create_umap_projection

__all__ = [
    "plot_topic_distribution",
    "plot_word_cloud",
    "plot_embeddings",
    "plot_topic_clusters",
    "create_umap_projection",
]
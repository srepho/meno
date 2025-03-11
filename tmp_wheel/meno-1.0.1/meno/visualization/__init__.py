"""Visualization module for topic model results."""

from typing import List, Dict, Optional, Union

# Re-export key functions and classes
from .static_plots import plot_topic_distribution, plot_word_cloud
from .interactive_plots import plot_embeddings, plot_topic_clusters
from .umap_viz import create_umap_projection

# Import time series visualization functions
from .time_series import (
    create_topic_trend_plot,
    create_topic_heatmap,
    create_topic_stacked_area,
    create_topic_ridge_plot,
    create_topic_calendar_heatmap,
)

# Import geospatial visualization functions
from .geospatial import (
    create_topic_map,
    create_region_choropleth,
    create_topic_density_map,
    create_postcode_map,
)

# Import time-space visualization functions
from .time_space import (
    create_animated_map,
    create_space_time_heatmap,
    create_category_time_plot,
)

__all__ = [
    # Basic visualizations
    "plot_topic_distribution",
    "plot_word_cloud",
    "plot_embeddings",
    "plot_topic_clusters",
    "create_umap_projection",
    
    # Time series visualizations
    "create_topic_trend_plot",
    "create_topic_heatmap",
    "create_topic_stacked_area",
    "create_topic_ridge_plot",
    "create_topic_calendar_heatmap",
    
    # Geospatial visualizations
    "create_topic_map",
    "create_region_choropleth",
    "create_topic_density_map",
    "create_postcode_map",
    
    # Time-space visualizations
    "create_animated_map",
    "create_space_time_heatmap",
    "create_category_time_plot",
]
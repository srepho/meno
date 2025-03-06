"""Static plotting utilities for topic modeling results."""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter


def plot_topic_distribution(
    topics: Union[List[str], pd.Series],
    width: int = 900,
    height: int = 600,
    title: str = "Topic Distribution",
    colorscale: str = "Viridis",
    sort_by_count: bool = True,
) -> go.Figure:
    """Create a bar chart of topic distribution.
    
    Parameters
    ----------
    topics : Union[List[str], pd.Series]
        List or Series of topic assignments
    width : int, optional
        Plot width, by default 900
    height : int, optional
        Plot height, by default 600
    title : str, optional
        Plot title, by default "Topic Distribution"
    colorscale : str, optional
        Colorscale for bars, by default "Viridis"
    sort_by_count : bool, optional
        Whether to sort topics by frequency, by default True
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Convert to pandas Series if needed
    if not isinstance(topics, pd.Series):
        topics = pd.Series(topics)
    
    # Count topics
    topic_counts = topics.value_counts()
    
    # Sort by count if requested
    if sort_by_count:
        topic_counts = topic_counts.sort_values(ascending=False)
    
    # Create colormap
    colors = px.colors.sample_colorscale(
        colorscale,
        np.linspace(0, 1, len(topic_counts))
    )
    
    # Create figure
    fig = go.Figure(
        go.Bar(
            x=topic_counts.index,
            y=topic_counts.values,
            marker_color=colors,
            text=topic_counts.values,
            textposition="auto",
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Topic",
        yaxis_title="Document Count",
        width=width,
        height=height,
    )
    
    return fig


def plot_word_cloud(
    words: Dict[str, Union[float, int]],
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    max_words: int = 100,
    colormap: str = "viridis",
) -> plt.Figure:
    """Create a word cloud from word frequencies or importance scores.
    
    Parameters
    ----------
    words : Dict[str, Union[float, int]]
        Dictionary mapping words to frequencies or importance scores
    width : int, optional
        Plot width, by default 800
    height : int, optional
        Plot height, by default 400
    background_color : str, optional
        Background color, by default "white"
    max_words : int, optional
        Maximum number of words to include, by default 100
    colormap : str, optional
        Matplotlib colormap name, by default "viridis"
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        colormap=colormap,
        random_state=42,
    ).generate_from_frequencies(words)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    
    return fig


def plot_topic_word_clouds(
    topic_words: Dict[str, Dict[str, float]],
    ncols: int = 3,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    max_words: int = 50,
    colormap: str = "viridis",
) -> plt.Figure:
    """Create word clouds for multiple topics.
    
    Parameters
    ----------
    topic_words : Dict[str, Dict[str, float]]
        Dictionary mapping topic names to word frequency dictionaries
    ncols : int, optional
        Number of columns in the grid, by default 3
    width : int, optional
        Width of each word cloud, by default 800
    height : int, optional
        Height of each word cloud, by default 400
    background_color : str, optional
        Background color, by default "white"
    max_words : int, optional
        Maximum number of words per cloud, by default 50
    colormap : str, optional
        Matplotlib colormap name, by default "viridis"
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Calculate grid dimensions
    n_topics = len(topic_words)
    nrows = (n_topics + ncols - 1) // ncols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * width / 100, nrows * height / 100),
    )
    
    # Flatten axes if needed
    if nrows > 1 or ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create word cloud for each topic
    for i, (topic, words) in enumerate(topic_words.items()):
        if i < len(axes):
            # Create word cloud
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color=background_color,
                max_words=max_words,
                colormap=colormap,
                random_state=42,
            ).generate_from_frequencies(words)
            
            # Plot on axis
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(topic)
            axes[i].axis("off")
    
    # Hide empty subplots
    for i in range(n_topics, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    return fig


def plot_topic_proportions(
    topic_proportions: pd.DataFrame,
    width: int = 900,
    height: int = 600,
    title: str = "Topic Proportions",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Plot topic proportions from LDA or other probabilistic models.
    
    Parameters
    ----------
    topic_proportions : pd.DataFrame
        DataFrame with topic proportion columns (excluding topic/probability columns)
    width : int, optional
        Plot width, by default 900
    height : int, optional
        Plot height, by default 600
    title : str, optional
        Plot title, by default "Topic Proportions"
    colorscale : str, optional
        Colorscale for bars, by default "Viridis"
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Get topic columns (exclude non-topic columns)
    exclude_cols = ["topic", "primary_topic", "topic_probability", "all_topics", "all_scores"]
    topic_cols = [col for col in topic_proportions.columns if col not in exclude_cols]
    
    # Calculate average proportion for each topic
    avg_proportions = topic_proportions[topic_cols].mean().sort_values(ascending=False)
    
    # Create colormap
    colors = px.colors.sample_colorscale(
        colorscale,
        np.linspace(0, 1, len(avg_proportions))
    )
    
    # Create figure
    fig = go.Figure(
        go.Bar(
            x=avg_proportions.index,
            y=avg_proportions.values,
            marker_color=colors,
            text=[f"{p:.2f}" for p in avg_proportions.values],
            textposition="auto",
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Topic",
        yaxis_title="Average Proportion",
        width=width,
        height=height,
    )
    
    return fig
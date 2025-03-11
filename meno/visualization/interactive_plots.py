"""Interactive plotting utilities for topic modeling results."""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_embeddings(
    projection: np.ndarray,
    topics: Union[List[str], pd.Series],
    document_texts: Optional[Union[List[str], pd.Series]] = None,
    width: int = 900,
    height: int = 600,
    title: str = "Document Embeddings",
    colorscale: str = "Viridis",
    marker_size: int = 5,
    opacity: float = 0.7,
) -> go.Figure:
    """Create an interactive scatter plot of document embeddings colored by topic.
    
    Parameters
    ----------
    projection : np.ndarray
        UMAP or other projection with shape (n_documents, 2) or (n_documents, 3)
    topics : Union[List[str], pd.Series]
        Topic assignments for each document
    document_texts : Optional[Union[List[str], pd.Series]], optional
        Original document texts for hover information, by default None
    width : int, optional
        Plot width, by default 900
    height : int, optional
        Plot height, by default 600
    title : str, optional
        Plot title, by default "Document Embeddings"
    colorscale : str, optional
        Colorscale for points, by default "Viridis"
    marker_size : int, optional
        Size of markers, by default 5
    opacity : float, optional
        Opacity of markers, by default 0.7
    
    Returns
    -------
    go.Figure
        Plotly figure object with interactive scatter plot
    
    Raises
    ------
    ValueError
        If projection dimensions are not 2 or 3
    """
    # Check projection dimensions
    if projection.shape[1] not in [2, 3]:
        raise ValueError(
            f"Projection must have 2 or 3 dimensions, got {projection.shape[1]}"
        )
    
    # Convert to pandas Series if needed
    if not isinstance(topics, pd.Series):
        topics = pd.Series(topics)
    
    # Create data frame for plotting
    plot_df = pd.DataFrame()
    
    # Add projection coordinates
    if projection.shape[1] == 2:
        plot_df["x"] = projection[:, 0]
        plot_df["y"] = projection[:, 1]
    else:  # 3D
        plot_df["x"] = projection[:, 0]
        plot_df["y"] = projection[:, 1]
        plot_df["z"] = projection[:, 2]
    
    # Add topics and texts
    plot_df["topic"] = topics.values
    
    if document_texts is not None:
        if not isinstance(document_texts, pd.Series):
            document_texts = pd.Series(document_texts)
        plot_df["text"] = document_texts.values
    
    # Create scatter plot
    if projection.shape[1] == 2:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="topic",
            hover_name="text" if "text" in plot_df.columns else None,
            color_discrete_sequence=px.colors.qualitative.Bold,
            width=width,
            height=height,
            title=title,
        )
    else:  # 3D
        fig = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="z",
            color="topic",
            hover_name="text" if "text" in plot_df.columns else None,
            color_discrete_sequence=px.colors.qualitative.Bold,
            width=width,
            height=height,
            title=title,
        )
    
    # Update marker properties
    fig.update_traces(
        marker={
            "size": marker_size,
            "opacity": opacity,
        },
    )
    
    # Update layout
    fig.update_layout(
        legend_title_text="Topic",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
    )
    
    if projection.shape[1] == 3:
        fig.update_layout(
            scene={
                "xaxis_title": "Dimension 1",
                "yaxis_title": "Dimension 2",
                "zaxis_title": "Dimension 3",
            }
        )
    
    return fig


def plot_topic_clusters(
    projection: np.ndarray,
    topics: Union[List[str], pd.Series],
    cluster_centers: Optional[np.ndarray] = None,
    document_texts: Optional[Union[List[str], pd.Series]] = None,
    width: int = 900,
    height: int = 600,
    title: str = "Topic Clusters",
    colorscale: str = "Viridis",
    marker_size: int = 5,
    center_size: int = 15,
    opacity: float = 0.7,
) -> go.Figure:
    """Create an interactive scatter plot of document clusters with centers.
    
    Parameters
    ----------
    projection : np.ndarray
        UMAP or other projection with shape (n_documents, 2)
    topics : Union[List[str], pd.Series]
        Topic assignments for each document
    cluster_centers : Optional[np.ndarray], optional
        Cluster centers with shape (n_clusters, 2), by default None
    document_texts : Optional[Union[List[str], pd.Series]], optional
        Original document texts for hover information, by default None
    width : int, optional
        Plot width, by default 900
    height : int, optional
        Plot height, by default 600
    title : str, optional
        Plot title, by default "Topic Clusters"
    colorscale : str, optional
        Colorscale for points, by default "Viridis"
    marker_size : int, optional
        Size of markers, by default 5
    center_size : int, optional
        Size of cluster center markers, by default 15
    opacity : float, optional
        Opacity of markers, by default 0.7
    
    Returns
    -------
    go.Figure
        Plotly figure object with interactive scatter plot
    
    Raises
    ------
    ValueError
        If projection dimensions are not 2
    """
    # Check projection dimensions
    if projection.shape[1] != 2:
        raise ValueError(
            f"Projection must have 2 dimensions for cluster visualization, got {projection.shape[1]}"
        )
    
    # Convert to pandas Series if needed
    if not isinstance(topics, pd.Series):
        topics = pd.Series(topics)
    
    # Create data frame for plotting
    plot_df = pd.DataFrame()
    
    # Add projection coordinates
    plot_df["x"] = projection[:, 0]
    plot_df["y"] = projection[:, 1]
    
    # Add topics and texts
    plot_df["topic"] = topics.values
    
    if document_texts is not None:
        if not isinstance(document_texts, pd.Series):
            document_texts = pd.Series(document_texts)
        plot_df["text"] = document_texts.values
    
    # Create scatter plot for documents
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="topic",
        hover_name="text" if "text" in plot_df.columns else None,
        color_discrete_sequence=px.colors.qualitative.Bold,
        width=width,
        height=height,
        title=title,
    )
    
    # Update marker properties
    fig.update_traces(
        marker={
            "size": marker_size,
            "opacity": opacity,
        },
    )
    
    # Add cluster centers if provided
    if cluster_centers is not None:
        # Get unique topics and their colors
        unique_topics = plot_df["topic"].unique()
        color_map = {}
        
        for i, trace in enumerate(fig.data):
            if i < len(unique_topics):
                color_map[unique_topics[i]] = trace.marker.color
        
        # Map numeric topic labels to string labels
        if cluster_centers.shape[0] <= len(unique_topics):
            # Create center markers
            for i in range(cluster_centers.shape[0]):
                topic_name = f"Topic_{i}"
                if topic_name in unique_topics:
                    color = color_map.get(topic_name, "black")
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[cluster_centers[i, 0]],
                            y=[cluster_centers[i, 1]],
                            mode="markers",
                            marker={
                                "symbol": "x",
                                "size": center_size,
                                "color": color,
                                "line": {
                                    "width": 2,
                                    "color": "black",
                                },
                            },
                            name=f"{topic_name} Center",
                            showlegend=False,
                        )
                    )
    
    # Update layout
    fig.update_layout(
        legend_title_text="Topic",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
    )
    
    return fig


def plot_topic_similarity_heatmap(
    similarity_matrix: np.ndarray,
    topic_names: List[str],
    width: int = 700,
    height: int = 700,
    title: str = "Topic Similarity",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create a heatmap of topic similarities.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        Topic similarity matrix with shape (n_topics, n_topics)
    topic_names : List[str]
        Names of the topics
    width : int, optional
        Plot width, by default 700
    height : int, optional
        Plot height, by default 700
    title : str, optional
        Plot title, by default "Topic Similarity"
    colorscale : str, optional
        Colorscale for heatmap, by default "Viridis"
    
    Returns
    -------
    go.Figure
        Plotly figure object with interactive heatmap
    
    Raises
    ------
    ValueError
        If similarity_matrix shape doesn't match topic_names length
    """
    # Check input dimensions
    if similarity_matrix.shape != (len(topic_names), len(topic_names)):
        raise ValueError(
            f"Similarity matrix shape {similarity_matrix.shape} does not match "
            f"topic_names length {len(topic_names)}"
        )
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix,
            x=topic_names,
            y=topic_names,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )
    
    return fig


def plot_interactive_wordcloud(
    topic_words: Dict[str, Dict[str, float]],
    width: int = 900,
    height: int = 700,
    title: str = "Topic Word Clouds",
) -> go.Figure:
    """Create an interactive word cloud visualization with topic selector.
    
    Parameters
    ----------
    topic_words : Dict[str, Dict[str, float]]
        Dictionary mapping topic names to word frequency dictionaries
    width : int, optional
        Plot width, by default 900
    height : int, optional
        Plot height, by default 700
    title : str, optional
        Plot title, by default "Topic Word Clouds"
    
    Returns
    -------
    go.Figure
        Plotly figure object with interactive word cloud
    """
    import random
    
    # Get list of topics
    topics = list(topic_words.keys())
    
    # Create figure with dropdown
    fig = go.Figure()
    
    # Add word cloud for each topic as a separate trace
    for topic in topics:
        words = topic_words[topic]
        
        # Sort words by frequency and get top words
        sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:50] if len(sorted_words) > 50 else sorted_words
        
        # Normalize frequencies for size calculations
        max_freq = max([freq for _, freq in top_words]) if top_words else 1
        min_freq = min([freq for _, freq in top_words]) if top_words else 0
        
        # Generate positions for each word to create a proper word cloud
        # Use a spiral layout to distribute words
        x_positions = []
        y_positions = []
        word_sizes = []
        word_texts = []
        
        # Set boundaries for the word cloud
        x_range = 0.8  # Use 80% of the horizontal space
        y_range = 0.8  # Use 80% of the vertical space
        
        # Use a spiral algorithm to place words
        for i, (word, freq) in enumerate(top_words):
            # Normalize frequency to range [0.3, 1.0]
            norm_freq = 0.3 + 0.7 * (freq - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0.5
            
            # Calculate font size based on normalized frequency
            font_size = 10 + 30 * norm_freq
            
            # Generate position using outward spiral
            # More important words (higher freq) closer to center
            t = i / 10 * 2 * np.pi  # Angle parameter
            radial_pos = 0.05 + 0.4 * (1 - norm_freq)  # Radial position (important words closer to center)
            
            # Add some jitter for natural appearance
            jitter_x = random.uniform(-0.05, 0.05)
            jitter_y = random.uniform(-0.05, 0.05)
            
            x_pos = 0.5 + radial_pos * np.cos(t) * x_range + jitter_x
            y_pos = 0.5 + radial_pos * np.sin(t) * y_range + jitter_y
            
            x_positions.append(x_pos)
            y_positions.append(y_pos)
            word_sizes.append(font_size)
            word_texts.append(word)
        
        # Create a scatter trace with text for this topic
        for i, (word, x_pos, y_pos, size) in enumerate(zip(word_texts, x_positions, y_positions, word_sizes)):
            fig.add_trace(
                go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode="text",
                    text=[word],
                    textfont={
                        "size": size,
                        "color": px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    },
                    hoverinfo="text",
                    name=f"{topic}_{word}",
                    visible=(topic == topics[0]),  # Only first topic visible by default
                    showlegend=False
                )
            )
    
    # Create buttons for dropdown
    buttons = []
    for i, topic in enumerate(topics):
        # Calculate visible status for each trace
        # Each topic has multiple traces (one per word)
        topic_start_idx = 0
        for j in range(i):
            word_count = len(topic_words[topics[j]])
            word_count = min(word_count, 50)  # Max 50 words per topic
            topic_start_idx += word_count
            
        topic_word_count = len(topic_words[topic])
        topic_word_count = min(topic_word_count, 50)  # Max 50 words per topic
        
        # Set visibility for all traces
        visible_array = [False] * len(fig.data)
        for j in range(topic_start_idx, topic_start_idx + topic_word_count):
            if j < len(visible_array):
                visible_array[j] = True
        
        button = {
            "method": "update",
            "label": topic,
            "args": [
                {"visible": visible_array},
                {"title": f"{title}: {topic}"},
            ],
        }
        buttons.append(button)
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.15,
            }
        ]
    )
    
    # Update layout for proper word cloud display
    fig.update_layout(
        title=f"{title}: {topics[0]}",
        width=width,
        height=height,
        xaxis={
            "visible": False,
            "range": [0, 1],
            "fixedrange": True,
        },
        yaxis={
            "visible": False,
            "range": [0, 1],
            "fixedrange": True,
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig
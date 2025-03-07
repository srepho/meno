"""
BERTopic visualization utilities for Meno.

This module provides visualization utilities for BERTopic models, including:
- Topic similarity visualizations
- Topic hierarchy visualizations
- Topic distribution visualizations
- Topics over time visualizations
- Custom visualizations for domain-specific applications
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json

# Optional imports that will gracefully fail if dependencies are missing
try:
    from bertopic import BERTopic
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from umap import UMAP
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_bertopic_topic_similarity(
    model: 'BERTopic',
    width: int = 800,
    height: int = 600,
    title: Optional[str] = "Topic Similarity Network",
    colorscale: str = "Viridis",
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a topic similarity visualization using BERTopic's capabilities.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    width : int, optional
        Width of the figure, by default 800
    height : int, optional
        Height of the figure, by default 600
    title : Optional[str], optional
        Title of the figure, by default "Topic Similarity Network"
    colorscale : str, optional
        Colorscale for the visualization, by default "Viridis"
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Use BERTopic's built-in visualization
        fig = model.visualize_topics(width=width, height=height)
        
        # Update figure layout with custom settings
        fig.update_layout(title=title)
        
        # Update colorscale if different from default
        if colorscale != "Viridis":
            for trace in fig.data:
                if hasattr(trace, 'marker') and hasattr(trace.marker, 'colorscale'):
                    trace.marker.colorscale = colorscale
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating topic similarity visualization: {str(e)}")
        return None


def create_bertopic_hierarchy(
    model: 'BERTopic',
    width: int = 1000,
    height: int = 600,
    title: Optional[str] = "Topic Hierarchy",
    orientation: str = "left",
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a topic hierarchy visualization using BERTopic's capabilities.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    width : int, optional
        Width of the figure, by default 1000
    height : int, optional
        Height of the figure, by default 600
    title : Optional[str], optional
        Title of the figure, by default "Topic Hierarchy"
    orientation : str, optional
        Orientation of the tree, by default "left"
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Use BERTopic's built-in visualization
        fig = model.visualize_hierarchy(width=width, height=height, orientation=orientation)
        
        # Update figure layout with custom settings
        fig.update_layout(title=title)
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating topic hierarchy visualization: {str(e)}")
        return None


def create_bertopic_barchart(
    model: 'BERTopic',
    top_n_topics: int = 10,
    n_words: int = 5,
    width: int = 800,
    height: int = 500,
    title: Optional[str] = "Topic Word Scores",
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a topic barchart visualization using BERTopic's capabilities.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    top_n_topics : int, optional
        Number of topics to include, by default 10
    n_words : int, optional
        Number of words per topic, by default 5
    width : int, optional
        Width of the figure, by default 800
    height : int, optional
        Height of the figure, by default 500
    title : Optional[str], optional
        Title of the figure, by default "Topic Word Scores"
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Use BERTopic's built-in visualization
        fig = model.visualize_barchart(
            top_n_topics=top_n_topics,
            n_words=n_words,
            width=width,
            height=height
        )
        
        # Update figure layout with custom settings
        fig.update_layout(title=title)
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating topic barchart visualization: {str(e)}")
        return None


def create_bertopic_topic_distribution(
    model: 'BERTopic',
    topics: List[int],
    width: int = 800,
    height: int = 500,
    title: Optional[str] = "Topic Distribution",
    min_topic_size: int = 5,
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a topic distribution visualization using topic assignments.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    topics : List[int]
        List of topic assignments
    width : int, optional
        Width of the figure, by default 800
    height : int, optional
        Height of the figure, by default 500
    title : Optional[str], optional
        Title of the figure, by default "Topic Distribution"
    min_topic_size : int, optional
        Minimum topic size to include, by default 5
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Get topic info
        topic_info = model.get_topic_info()
        
        # Filter out very small topics and outliers
        topic_info = topic_info[
            (topic_info["Count"] >= min_topic_size) & 
            (topic_info["Topic"] != -1)
        ]
        
        # Get topic frequencies
        topic_counts = pd.Series(topics).value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        
        # Merge with topic info to get names
        merged_data = topic_counts.merge(topic_info[["Topic", "Name"]], on="Topic", how="left")
        
        # Create the figure
        fig = px.bar(
            merged_data,
            x="Name",
            y="Count",
            title=title,
            width=width,
            height=height,
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Number of Documents",
            xaxis={'categoryorder': 'total descending'}
        )
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating topic distribution visualization: {str(e)}")
        return None


def create_bertopic_over_time(
    model: 'BERTopic',
    docs_df: pd.DataFrame,
    timestamp_col: str,
    topic_col: str,
    width: int = 900,
    height: int = 600,
    title: Optional[str] = "Topics Over Time",
    frequency: str = "W",  # Weekly by default
    normalize: bool = False,
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a topics over time visualization.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    docs_df : pd.DataFrame
        DataFrame with documents, topics, and timestamps
    timestamp_col : str
        Name of the timestamp column
    topic_col : str
        Name of the topic column
    width : int, optional
        Width of the figure, by default 900
    height : int, optional
        Height of the figure, by default 600
    title : Optional[str], optional
        Title of the figure, by default "Topics Over Time"
    frequency : str, optional
        Frequency for resampling, by default "W"
    normalize : bool, optional
        Whether to normalize counts, by default False
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Ensure timestamp column is datetime
        docs_df = docs_df.copy()
        docs_df[timestamp_col] = pd.to_datetime(docs_df[timestamp_col])
        
        # Prepare the data for topics over time
        topics_over_time = model.topics_over_time(
            docs=None,  # We'll use our own data
            timestamps=docs_df[timestamp_col].tolist(),
            topics=docs_df[topic_col].tolist(),
            global_tuning=True,
            evolution_tuning=True,
            nr_bins=10
        )
        
        # Use BERTopic's built-in visualization
        fig = model.visualize_topics_over_time(
            topics_over_time=topics_over_time,
            width=width,
            height=height,
            topics=None,  # All topics
            normalize_frequency=normalize
        )
        
        # Update figure layout with custom settings
        fig.update_layout(title=title)
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating topics over time visualization: {str(e)}")
        return None


def create_bertopic_categories_comparison(
    model: 'BERTopic',
    docs_df: pd.DataFrame,
    topic_col: str,
    category_col: str,
    width: int = 900,
    height: int = 600,
    title: Optional[str] = "Topic Distribution by Category",
    top_n_topics: int = 8,
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a visualization comparing topic distributions across categories.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    docs_df : pd.DataFrame
        DataFrame with documents, topics, and categories
    topic_col : str
        Name of the topic column
    category_col : str
        Name of the category column
    width : int, optional
        Width of the figure, by default 900
    height : int, optional
        Height of the figure, by default 600
    title : Optional[str], optional
        Title of the figure, by default "Topic Distribution by Category"
    top_n_topics : int, optional
        Number of top topics to include, by default 8
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Get topic info
        topic_info = model.get_topic_info()
        
        # Filter topics
        top_topics = topic_info[topic_info["Topic"] != -1].head(top_n_topics)["Topic"].tolist()
        
        # Get category distribution for each topic
        categories = docs_df[category_col].unique()
        
        # Create DataFrame for visualization
        data = []
        for topic in top_topics:
            topic_name = topic_info[topic_info["Topic"] == topic]["Name"].iloc[0]
            topic_docs = docs_df[docs_df[topic_col] == topic]
            
            for category in categories:
                count = len(topic_docs[topic_docs[category_col] == category])
                data.append({
                    "Topic": topic_name,
                    "Category": category,
                    "Count": count
                })
        
        # Create DataFrame
        dist_df = pd.DataFrame(data)
        
        # Create the figure
        fig = px.bar(
            dist_df, 
            x="Topic", 
            y="Count", 
            color="Category",
            title=title,
            width=width,
            height=height,
            barmode="group"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Number of Documents",
            legend_title="Category"
        )
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating category comparison visualization: {str(e)}")
        return None


def create_bertopic_document_similarity(
    model: 'BERTopic',
    docs: List[str],
    topic_assignments: List[int],
    embeddings: np.ndarray,
    width: int = 900,
    height: int = 600,
    title: Optional[str] = "Document Similarity",
    use_3d: bool = False,
    color_by_topic: bool = True,
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a document similarity visualization using UMAP on embeddings.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    docs : List[str]
        List of document texts
    topic_assignments : List[int]
        List of topic assignments for each document
    embeddings : np.ndarray
        Document embeddings
    width : int, optional
        Width of the figure, by default 900
    height : int, optional
        Height of the figure, by default 600
    title : Optional[str], optional
        Title of the figure, by default "Document Similarity"
    use_3d : bool, optional
        Whether to create a 3D visualization, by default False
    color_by_topic : bool, optional
        Whether to color points by topic, by default True
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Get topic info for labels
        topic_info = model.get_topic_info()
        
        # Create topic labels
        topic_labels = []
        for topic in topic_assignments:
            if topic == -1:
                topic_labels.append("Outlier")
            else:
                topic_name = topic_info[topic_info["Topic"] == topic]["Name"].iloc[0]
                topic_labels.append(f"Topic {topic}: {topic_name}")
        
        # Apply UMAP for dimensionality reduction
        n_components = 3 if use_3d else 2
        umap_model = UMAP(n_components=n_components, random_state=42)
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "UMAP1": umap_embeddings[:, 0],
            "UMAP2": umap_embeddings[:, 1],
            "Topic": topic_assignments,
            "TopicLabel": topic_labels,
            "Text": [doc[:100] + "..." if len(doc) > 100 else doc for doc in docs]
        })
        
        if use_3d:
            plot_df["UMAP3"] = umap_embeddings[:, 2]
            
            if color_by_topic:
                fig = px.scatter_3d(
                    plot_df,
                    x="UMAP1",
                    y="UMAP2",
                    z="UMAP3",
                    color="Topic",
                    hover_data=["TopicLabel", "Text"],
                    title=title,
                    width=width,
                    height=height
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x="UMAP1",
                    y="UMAP2",
                    z="UMAP3",
                    hover_data=["TopicLabel", "Text"],
                    title=title,
                    width=width,
                    height=height
                )
        else:
            if color_by_topic:
                fig = px.scatter(
                    plot_df,
                    x="UMAP1",
                    y="UMAP2",
                    color="Topic",
                    hover_data=["TopicLabel", "Text"],
                    title=title,
                    width=width,
                    height=height
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x="UMAP1",
                    y="UMAP2",
                    hover_data=["TopicLabel", "Text"],
                    title=title,
                    width=width,
                    height=height
                )
        
        # Update marker size
        fig.update_traces(marker=dict(size=6))
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating document similarity visualization: {str(e)}")
        return None


def create_bertopic_word_score_heatmap(
    model: 'BERTopic',
    topics: Optional[List[int]] = None,
    top_n_topics: int = 10,
    n_words: int = 10,
    width: int = 900,
    height: int = 600,
    title: Optional[str] = "Topic-Word Score Heatmap",
    return_fig: bool = True
) -> Union[go.Figure, None]:
    """
    Create a heatmap of word scores for topics.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    topics : Optional[List[int]], optional
        List of specific topics to include, by default None
    top_n_topics : int, optional
        Number of top topics to include if topics is None, by default 10
    n_words : int, optional
        Number of words per topic, by default 10
    width : int, optional
        Width of the figure, by default 900
    height : int, optional
        Height of the figure, by default 600
    title : Optional[str], optional
        Title of the figure, by default "Topic-Word Score Heatmap"
    return_fig : bool, optional
        Whether to return the figure or show it, by default True
    
    Returns
    -------
    Union[go.Figure, None]
        Plotly figure if return_fig is True, otherwise None
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available. Cannot create visualization.")
        return None
    
    try:
        # Get topic info
        topic_info = model.get_topic_info()
        
        # Determine which topics to visualize
        if topics is None:
            # Get top N topics by count
            topics = topic_info[topic_info["Topic"] != -1].head(top_n_topics)["Topic"].tolist()
        
        # Get words and scores for each topic
        words_by_topic = {}
        for topic in topics:
            words_scores = model.get_topic(topic)[:n_words]
            words = [word for word, _ in words_scores]
            scores = [score for _, score in words_scores]
            words_by_topic[f"Topic {topic}"] = (words, scores)
        
        # Create a DataFrame for the heatmap
        all_words = list({word for topic_words, _ in words_by_topic.values() for word in topic_words})
        heatmap_data = {word: [] for word in all_words}
        
        for topic, (topic_words, topic_scores) in words_by_topic.items():
            for word in all_words:
                if word in topic_words:
                    idx = topic_words.index(word)
                    heatmap_data[word].append(topic_scores[idx])
                else:
                    heatmap_data[word].append(0)
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data, index=list(words_by_topic.keys()))
        
        # Create the heatmap
        fig = px.imshow(
            heatmap_df,
            x=all_words,
            y=list(words_by_topic.keys()),
            color_continuous_scale="Viridis",
            title=title,
            width=width,
            height=height
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Word",
            yaxis_title="Topic",
            coloraxis_colorbar=dict(title="Score")
        )
        
        if return_fig:
            return fig
        else:
            fig.show()
            return None
    except Exception as e:
        logger.error(f"Error creating word score heatmap: {str(e)}")
        return None
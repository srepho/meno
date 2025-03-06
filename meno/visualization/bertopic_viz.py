"""BERTopic-specific visualizations for Meno."""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import networkx as nx
import logging

try:
    from bertopic import BERTopic
    from bertopic.plotting import visualize_topics, visualize_hierarchy
    from bertopic.plotting import visualize_barchart, visualize_heatmap
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


def create_enhanced_topic_visualization(
    bertopic_model: Any,
    width: int = 800,
    height: int = 600,
    title: str = "Topic Similarity Map",
    color_by: str = "size",
    theme: str = "plotly",
    background_color: str = None,
) -> go.Figure:
    """Create an enhanced visualization of topics in 2D space.
    
    This visualization improves on the standard BERTopic visualization with
    better readability, interactivity, and customization options.
    
    Parameters
    ----------
    bertopic_model : Any
        Trained BERTopic model
    width : int, optional
        Width of the plot, by default 800
    height : int, optional
        Height of the plot, by default 600
    title : str, optional
        Plot title, by default "Topic Similarity Map"
    color_by : str, optional
        How to color the topics, by default "size"
        Options: "size", "id", "custom"
    theme : str, optional
        Plot theme, by default "plotly"
        Options: "plotly", "plotly_white", "plotly_dark", "ggplot2"
    background_color : str, optional
        Background color, by default None
        
    Returns
    -------
    go.Figure
        Interactive plot of topics
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic is required for this visualization. "
            "Install with 'pip install bertopic>=0.15.0'"
        )
    
    # Get topic info
    topic_info = bertopic_model.get_topic_info()
    
    # Create base visualization with BERTopic
    base_fig = visualize_topics(
        bertopic_model,
        topics=list(topic_info[topic_info['Topic'] != -1]['Topic']),
        width=width,
        height=height
    )
    
    # Extract data from base visualization
    x_data = []
    y_data = []
    topic_ids = []
    topic_names = []
    topic_sizes = []
    
    for trace in base_fig.data:
        if hasattr(trace, 'x') and hasattr(trace, 'y') and trace.x is not None and trace.y is not None:
            x_data.extend(trace.x)
            y_data.extend(trace.y)
            
            # Extract topic info from hovertext or name
            if hasattr(trace, 'hovertext') and trace.hovertext is not None:
                for text in trace.hovertext:
                    # Extract topic ID and add to list
                    if 'Topic' in text:
                        try:
                            topic_id = int(text.split('Topic ')[1].split(':')[0])
                            topic_ids.append(topic_id)
                            
                            # Get topic name and size
                            row = topic_info[topic_info['Topic'] == topic_id].iloc[0]
                            topic_names.append(f"Topic {topic_id}")
                            topic_sizes.append(row['Count'])
                        except (IndexError, ValueError):
                            topic_ids.append(-1)
                            topic_names.append("Unknown")
                            topic_sizes.append(0)
                    else:
                        topic_ids.append(-1)
                        topic_names.append("Unknown")
                        topic_sizes.append(0)
    
    # Create a new dataframe with all the data
    df = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        'topic_id': topic_ids,
        'topic_name': topic_names,
        'size': topic_sizes
    })
    
    # Get top words for each topic
    topics_with_words = {}
    for topic_id in df['topic_id'].unique():
        if topic_id != -1:
            words = [word for word, _ in bertopic_model.get_topic(topic_id)][:7]
            topics_with_words[topic_id] = words
    
    # Add topic words to dataframe
    df['topic_words'] = df['topic_id'].map(
        lambda x: ', '.join(topics_with_words.get(x, [])) if x in topics_with_words else ''
    )
    
    # Create marker sizes based on topic sizes
    size_min, size_max = 20, 50
    if len(topic_sizes) > 0:
        size_scale = (df['size'] - min(df['size'])) / (max(df['size']) - min(df['size']) + 0.1)
        df['marker_size'] = size_min + size_scale * (size_max - size_min)
    else:
        df['marker_size'] = size_min
    
    # Normalize topics for colors
    if color_by == "size":
        # Color by topic size
        df['color'] = df['size']
        color_title = "Topic Size"
    elif color_by == "id":
        # Color by topic ID
        df['color'] = df['topic_id']
        color_title = "Topic ID"
    else:
        # Default color by topic ID
        df['color'] = df['topic_id']
        color_title = "Topic ID"
    
    # Create hover text
    df['hover_text'] = df.apply(
        lambda row: (
            f"<b>Topic {row['topic_id']}</b><br>"
            f"Size: {row['size']} documents<br>"
            f"Words: {row['topic_words']}"
        ), axis=1
    )
    
    # Create enhanced visualization
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='color',
        size='marker_size',
        hover_name='topic_name',
        hover_data={
            'x': False,
            'y': False,
            'color': False,
            'marker_size': False,
            'topic_name': False,
            'topic_id': True,
            'size': True,
            'topic_words': True
        },
        color_continuous_scale='viridis',
        template=theme,
        title=title,
        labels={
            'color': color_title,
            'topic_id': 'Topic ID',
            'size': 'Documents',
            'topic_words': 'Top Words'
        }
    )
    
    # Update layout
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='DarkSlateGrey')
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        width=width,
        height=height,
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=True,
            zeroline=False,
            title=None
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=True,
            zeroline=False,
            title=None
        )
    )
    
    # Add topic labels for largest topics
    top_topics = df.sort_values('size', ascending=False).head(8)
    annotations = []
    
    for _, row in top_topics.iterrows():
        annotations.append(dict(
            x=row['x'],
            y=row['y'],
            xref='x',
            yref='y',
            text=f"Topic {row['topic_id']}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30
        ))
    
    fig.update_layout(annotations=annotations)
    
    # Set background color if specified
    if background_color:
        fig.update_layout(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color
        )
    
    return fig


def create_topic_timeline(
    bertopic_model: Any,
    timestamps: List[Any],
    topics: List[int] = None,
    width: int = 900,
    height: int = 600,
    title: str = "Topic Trends Over Time",
    time_format: str = None,
) -> go.Figure:
    """Create an interactive timeline of topic trends.
    
    Parameters
    ----------
    bertopic_model : Any
        Trained BERTopic model
    timestamps : List[Any]
        List of timestamps for each document
    topics : List[int], optional
        List of topics to include, by default None (all topics)
    width : int, optional
        Width of the plot, by default 900
    height : int, optional
        Height of the plot, by default 600
    title : str, optional
        Plot title, by default "Topic Trends Over Time"
    time_format : str, optional
        Format for timestamp display, by default None
        
    Returns
    -------
    go.Figure
        Interactive timeline plot
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic is required for this visualization. "
            "Install with 'pip install bertopic>=0.15.0'"
        )
    
    # Convert timestamps to pandas datetime
    timestamps = pd.to_datetime(timestamps)
    
    # Get document-topic mapping
    doc_topics, _ = bertopic_model.topics_, bertopic_model.probabilities_
    
    # Get topic info
    topic_info = bertopic_model.get_topic_info()
    
    # Filter out outlier topic (-1)
    valid_topic_info = topic_info[topic_info['Topic'] != -1]
    
    # Select topics to visualize
    if topics is None:
        # Use top 10 largest topics
        topics = valid_topic_info.sort_values('Count', ascending=False).head(10)['Topic'].tolist()
    
    # Create dataframe with timestamp and topic
    df = pd.DataFrame({
        'timestamp': timestamps,
        'topic': doc_topics
    })
    
    # Filter for selected topics only
    df = df[df['topic'].isin(topics)]
    
    # Group by date and topic, count documents
    if time_format == 'monthly':
        df['date'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()
    elif time_format == 'weekly':
        df['date'] = df['timestamp'].dt.to_period('W').dt.to_timestamp()
    elif time_format == 'daily':
        df['date'] = df['timestamp'].dt.date
    else:
        df['date'] = df['timestamp'].dt.date
    
    topic_counts = df.groupby(['date', 'topic']).size().reset_index(name='count')
    
    # Get topic words for each topic
    topic_words = {}
    for topic_id in topics:
        words = [word for word, _ in bertopic_model.get_topic(topic_id)][:5]
        topic_words[topic_id] = f"Topic {topic_id}: {', '.join(words)}"
    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each topic
    for topic_id in topics:
        topic_data = topic_counts[topic_counts['topic'] == topic_id]
        if not topic_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=topic_data['date'],
                    y=topic_data['count'],
                    mode='lines+markers',
                    name=topic_words.get(topic_id, f"Topic {topic_id}"),
                    hovertemplate=(
                        "<b>%{y} documents</b><br>"
                        "Date: %{x}<br>"
                        f"Topic {topic_id}<extra></extra>"
                    )
                )
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="Date",
        yaxis_title="Document Count",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title=None
        ),
        margin=dict(l=50, r=50, t=80, b=100),
        hovermode="closest"
    )
    
    return fig


def create_topic_comparison(
    bertopic_model: Any,
    topic_ids: List[int],
    width: int = 800,
    height: int = 500,
    title: str = "Topic Comparison",
) -> go.Figure:
    """Create a comparison visualization of selected topics.
    
    Parameters
    ----------
    bertopic_model : Any
        Trained BERTopic model
    topic_ids : List[int]
        List of topic IDs to compare
    width : int, optional
        Width of the plot, by default 800
    height : int, optional
        Height of the plot, by default 500
    title : str, optional
        Plot title, by default "Topic Comparison"
        
    Returns
    -------
    go.Figure
        Interactive comparison plot
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic is required for this visualization. "
            "Install with 'pip install bertopic>=0.15.0'"
        )
    
    # Get topic info
    topic_info = bertopic_model.get_topic_info()
    
    # Create figure with subplots
    n_topics = len(topic_ids)
    fig = make_subplots(
        rows=1, 
        cols=n_topics,
        subplot_titles=[f"Topic {topic_id}" for topic_id in topic_ids]
    )
    
    # Add barchart for each topic
    for i, topic_id in enumerate(topic_ids):
        if topic_id == -1:
            # Skip outlier topic
            continue
            
        # Get topic words and weights
        topic_words = bertopic_model.get_topic(topic_id)
        if not topic_words:
            continue
            
        words = [word for word, _ in topic_words[:10]]
        weights = [weight for _, weight in topic_words[:10]]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=weights[::-1],
                y=words[::-1],
                orientation='h',
                marker=dict(
                    color=weights[::-1],
                    colorscale='Viridis',
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Weight: %{x:.3f}<extra></extra>"
                ),
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Add topic size information
        topic_size = topic_info[topic_info['Topic'] == topic_id]['Count'].values[0]
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"{topic_size} documents",
            xref=f"x{i+1}", yref=f"paper",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="closest"
    )
    
    # Update x and y axes
    for i in range(1, n_topics+1):
        fig.update_xaxes(title_text="Weight", row=1, col=i)
        fig.update_yaxes(title_text="", row=1, col=i)
    
    return fig


def create_topic_network(
    bertopic_model: Any,
    min_similarity: float = 0.3,
    width: int = 800,
    height: int = 800,
    title: str = "Topic Network",
) -> go.Figure:
    """Create a network visualization of topic relationships.
    
    Parameters
    ----------
    bertopic_model : Any
        Trained BERTopic model
    min_similarity : float, optional
        Minimum similarity threshold for connecting topics, by default 0.3
    width : int, optional
        Width of the plot, by default 800
    height : int, optional
        Height of the plot, by default 800
    title : str, optional
        Plot title, by default "Topic Network"
        
    Returns
    -------
    go.Figure
        Interactive network plot
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic is required for this visualization. "
            "Install with 'pip install bertopic>=0.15.0'"
        )
    
    # Get topic info
    topic_info = bertopic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    # Get topic embeddings
    if not hasattr(bertopic_model, 'topic_embeddings_') or bertopic_model.topic_embeddings_ is None:
        # Get topic representations
        topic_list = []
        for topic in valid_topics:
            words = " ".join([word for word, _ in bertopic_model.get_topic(topic)][:10])
            topic_list.append(words)
        
        # Create embeddings
        from sentence_transformers import SentenceTransformer
        embedding_model = bertopic_model.embedding_model
        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        topic_embeddings = embedding_model.encode(topic_list)
    else:
        topic_embeddings = bertopic_model.topic_embeddings_
    
    # Calculate similarities
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(topic_embeddings)
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for i, topic_id in enumerate(valid_topics):
        topic_size = topic_info[topic_info['Topic'] == topic_id]['Count'].values[0]
        words = [word for word, _ in bertopic_model.get_topic(topic_id)][:5]
        G.add_node(
            topic_id,
            size=topic_size,
            words=", ".join(words)
        )
    
    # Add edges
    for i, topic_i in enumerate(valid_topics):
        for j, topic_j in enumerate(valid_topics):
            if i < j:  # Avoid duplicate edges and self-loops
                similarity = similarities[i, j]
                if similarity >= min_similarity:
                    G.add_edge(topic_i, topic_j, weight=similarity)
    
    # Create positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Scale node sizes based on topic sizes
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    if node_sizes:
        size_min, size_max = 20, 50
        size_scale = (np.array(node_sizes) - min(node_sizes)) / (max(node_sizes) - min(node_sizes) + 0.1)
        node_sizes = size_min + size_scale * (size_max - size_min)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[node for node in G.nodes()],
            colorscale='Viridis',
            size=node_sizes,
            line=dict(width=1, color='DarkSlateGrey')
        )
    )
    
    # Add node hover information
    node_text = []
    for node in G.nodes():
        node_text.append(
            f"Topic {node}<br>"
            f"Size: {G.nodes[node]['size']} documents<br>"
            f"Words: {G.nodes[node]['words']}"
        )
    node_trace.text = node_text
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        # Scale line width and opacity based on weight
        width = 1 + 3 * weight
        opacity = 0.2 + 0.8 * weight
        
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=width, color=f'rgba(120, 120, 120, {opacity})'),
            hoverinfo='text',
            text=f"Similarity: {weight:.3f}",
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height
        )
    )
    
    # Add node labels for larger topics
    top_topic_indices = np.argsort([G.nodes[node]['size'] for node in G.nodes()])[-8:]
    top_topics = [list(G.nodes())[i] for i in top_topic_indices]
    
    annotations = []
    for topic in top_topics:
        x, y = pos[topic]
        annotations.append(dict(
            x=x, y=y,
            xref='x', yref='y',
            text=f"Topic {topic}",
            showarrow=True,
            arrowhead=1,
            ax=0, ay=-30
        ))
    
    fig.update_layout(annotations=annotations)
    
    return fig
"""
Entity flow visualization for analyzing topic transitions.

This module provides visualizations to track how individual entities (customers, claims, etc.)
transition between topics over time, enabling analysis of sequential patterns in topic assignments.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import colorsys
import math
from datetime import datetime, timedelta

def visualize_entity_transitions(
    df: pd.DataFrame,
    entity_id_column: str,
    topic_column: str,
    time_column: str,
    min_transition_count: int = 1,
    node_color_map: Optional[Dict[str, str]] = None,
    title: str = "Entity Topic Transitions",
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """
    Create a Sankey diagram showing how entities transition between topics over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entity data with topic assignments over time
    entity_id_column : str
        Name of column containing entity IDs (customer ID, claim number, etc.)
    topic_column : str
        Name of column containing topic assignments
    time_column : str
        Name of column containing timestamp or datetime information
    min_transition_count : int, default 1
        Minimum number of transitions to include in the visualization
    node_color_map : Optional[Dict[str, str]], default None
        Mapping of topic names to colors
    title : str, default "Entity Topic Transitions"
        Title for the visualization
    width : int, default 900
        Width of the figure in pixels
    height : int, default 700
        Height of the figure in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure containing the Sankey diagram
    
    Notes
    -----
    This visualization is particularly useful for understanding how customers or claims
    move between topics over time, revealing common paths and potential intervention points.
    """
    # Ensure data is sorted by entity and time
    plot_df = df.copy().sort_values([entity_id_column, time_column])
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Get earliest topic assignment for each entity
    first_topics = plot_df.groupby(entity_id_column).first()[topic_column]
    
    # Calculate transitions between topics
    transitions = {}
    entity_paths = {}
    
    for entity in plot_df[entity_id_column].unique():
        entity_data = plot_df[plot_df[entity_id_column] == entity]
        
        if len(entity_data) <= 1:
            continue  # Skip entities with only one record
            
        # Extract sequence of topics for this entity
        topic_sequence = entity_data[topic_column].tolist()
        entity_paths[entity] = topic_sequence
        
        # Record transitions between topics
        for i in range(len(topic_sequence) - 1):
            from_topic = topic_sequence[i]
            to_topic = topic_sequence[i + 1]
            
            if from_topic == to_topic:
                continue  # Skip self-transitions
                
            transition_key = (from_topic, to_topic)
            if transition_key not in transitions:
                transitions[transition_key] = {
                    'count': 0,
                    'entities': set()
                }
                
            transitions[transition_key]['count'] += 1
            transitions[transition_key]['entities'].add(entity)
    
    # Filter transitions by minimum count
    filtered_transitions = {k: v for k, v in transitions.items() if v['count'] >= min_transition_count}
    
    if not filtered_transitions:
        # Create an empty figure with a message if no transitions meet the criteria
        fig = go.Figure()
        fig.add_annotation(
            text="No transitions found matching the criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(width=width, height=height, title=title)
        return fig
    
    # Create node list for Sankey diagram
    all_topics = set()
    for source, target in filtered_transitions.keys():
        all_topics.add(source)
        all_topics.add(target)
    
    node_list = sorted(list(all_topics))
    node_indices = {node: i for i, node in enumerate(node_list)}
    
    # Create link data
    sources = []
    targets = []
    values = []
    labels = []
    
    for (source, target), data in filtered_transitions.items():
        sources.append(node_indices[source])
        targets.append(node_indices[target])
        values.append(data['count'])
        labels.append(f"{source} → {target}: {data['count']} transitions")
    
    # Create node colors
    if node_color_map is None:
        # Generate colors automatically
        node_colors = []
        for i, node in enumerate(node_list):
            # Generate colors in HSV space for better distribution
            hue = i / len(node_list)
            # Convert to RGB for Plotly
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            node_colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)")
    else:
        # Use provided color map
        node_colors = [node_color_map.get(node, "gray") for node in node_list]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_list,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels,
            hovertemplate='%{label}<extra></extra>'
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        font=dict(size=12),
        width=width,
        height=height
    )
    
    return fig

def analyze_entity_metrics(
    df: pd.DataFrame,
    entity_id_column: str,
    topic_column: str,
    time_column: str,
) -> Dict[str, Any]:
    """
    Analyze key metrics for entity transitions between topics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entity data with topic assignments over time
    entity_id_column : str
        Name of column containing entity IDs (customer ID, claim number, etc.)
    topic_column : str
        Name of column containing topic assignments
    time_column : str
        Name of column containing timestamp or datetime information
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing various entity flow metrics:
        - average_time_in_topic: Average time spent in each topic
        - common_entry_topics: Most common entry topics for entities
        - common_exit_topics: Most common exit topics for entities
        - topic_retention: How often entities stay in the same topic
        - frequent_paths: Most common multi-step paths between topics
    """
    # Ensure data is sorted by entity and time
    plot_df = df.copy().sort_values([entity_id_column, time_column])
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Initialize metrics
    time_in_topic = {}  # {topic: [durations]}
    entry_topics = {}  # {topic: count}
    exit_topics = {}  # {topic: count}
    topic_retention = {}  # {topic: percentage}
    common_paths = {}  # {(topic1, topic2, topic3): count}
    
    # Process each entity
    for entity in plot_df[entity_id_column].unique():
        entity_data = plot_df[plot_df[entity_id_column] == entity].sort_values(time_column)
        
        if len(entity_data) <= 1:
            continue  # Skip entities with only one record
            
        # Extract sequence of topics and times for this entity
        topic_sequence = entity_data[topic_column].tolist()
        time_sequence = entity_data[time_column].tolist()
        
        # Record entry and exit topics
        entry_topic = topic_sequence[0]
        exit_topic = topic_sequence[-1]
        
        entry_topics[entry_topic] = entry_topics.get(entry_topic, 0) + 1
        exit_topics[exit_topic] = exit_topics.get(exit_topic, 0) + 1
        
        # Calculate time spent in each topic
        for i in range(len(topic_sequence) - 1):
            topic = topic_sequence[i]
            duration = time_sequence[i+1] - time_sequence[i]
            
            if topic not in time_in_topic:
                time_in_topic[topic] = []
            
            time_in_topic[topic].append(duration.total_seconds() / (60 * 60 * 24))  # Convert to days
            
        # Record paths (sequences of 2-4 topics)
        for path_length in range(2, min(5, len(topic_sequence) + 1)):
            for i in range(len(topic_sequence) - path_length + 1):
                path = tuple(topic_sequence[i:i+path_length])
                common_paths[path] = common_paths.get(path, 0) + 1
        
        # Calculate topic retention (how often entities stay in the same topic)
        for topic in set(topic_sequence):
            same_topic_count = 0
            transition_count = 0
            
            for i in range(len(topic_sequence) - 1):
                if topic_sequence[i] == topic:
                    transition_count += 1
                    if topic_sequence[i+1] == topic:
                        same_topic_count += 1
            
            if transition_count > 0:
                retention = same_topic_count / transition_count
                if topic not in topic_retention:
                    topic_retention[topic] = []
                topic_retention[topic].append(retention)
    
    # Process the collected metrics
    avg_time_in_topic = {topic: np.mean(durations) for topic, durations in time_in_topic.items()}
    
    # Sort entry and exit topics by frequency
    sorted_entry_topics = sorted(entry_topics.items(), key=lambda x: x[1], reverse=True)
    sorted_exit_topics = sorted(exit_topics.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate average retention by topic
    avg_topic_retention = {topic: np.mean(rates) for topic, rates in topic_retention.items()}
    
    # Sort common paths by frequency
    sorted_paths = sorted(common_paths.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare return value
    return {
        "average_time_in_topic": avg_time_in_topic,
        "common_entry_topics": sorted_entry_topics,
        "common_exit_topics": sorted_exit_topics,
        "topic_retention": avg_topic_retention,
        "frequent_paths": sorted_paths[:20]  # Return top 20 paths
    }

def visualize_entity_metrics(
    metrics: Dict[str, Any],
    title_prefix: str = "Entity Flow Analysis",
    width: int = 800,
    height: int = 600,
) -> Dict[str, go.Figure]:
    """
    Create visualizations for entity flow metrics.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of metrics from analyze_entity_metrics
    title_prefix : str, default "Entity Flow Analysis"
        Prefix to use for all chart titles
    width : int, default 800
        Width for all charts
    height : int, default 600
        Height for all charts
        
    Returns
    -------
    Dict[str, go.Figure]
        Dictionary of Plotly figures for each metric visualization
    """
    figures = {}
    
    # 1. Average time in topic (bar chart)
    if metrics["average_time_in_topic"]:
        topics = list(metrics["average_time_in_topic"].keys())
        avg_times = list(metrics["average_time_in_topic"].values())
        
        # Sort by time spent
        sorted_indices = np.argsort(avg_times)[::-1]
        sorted_topics = [topics[i] for i in sorted_indices]
        sorted_times = [avg_times[i] for i in sorted_indices]
        
        fig_time = go.Figure(data=[
            go.Bar(
                x=sorted_topics,
                y=sorted_times,
                text=[f"{t:.1f} days" for t in sorted_times],
                textposition='auto',
                marker_color=px.colors.qualitative.Plotly
            )
        ])
        
        fig_time.update_layout(
            title=f"{title_prefix}: Average Time Spent in Each Topic (Days)",
            xaxis_title="Topic",
            yaxis_title="Average Days",
            width=width,
            height=height
        )
        
        figures["time_in_topic"] = fig_time
    
    # 2. Entry and Exit topics (side-by-side bars)
    if metrics["common_entry_topics"] and metrics["common_exit_topics"]:
        # Create dictionary lookup for both
        entry_dict = dict(metrics["common_entry_topics"])
        exit_dict = dict(metrics["common_exit_topics"])
        
        # Get union of all topics
        all_topics = list(set(entry_dict.keys()) | set(exit_dict.keys()))
        
        # Sort by total frequency (entry + exit)
        total_freq = {t: entry_dict.get(t, 0) + exit_dict.get(t, 0) for t in all_topics}
        sorted_topics = sorted(all_topics, key=lambda t: total_freq[t], reverse=True)
        
        # Limit to top 10 for readability
        top_topics = sorted_topics[:10]
        
        fig_entry_exit = go.Figure(data=[
            go.Bar(
                name="Entry Topic",
                x=top_topics,
                y=[entry_dict.get(t, 0) for t in top_topics],
                marker_color='lightblue'
            ),
            go.Bar(
                name="Exit Topic",
                x=top_topics,
                y=[exit_dict.get(t, 0) for t in top_topics],
                marker_color='salmon'
            )
        ])
        
        fig_entry_exit.update_layout(
            title=f"{title_prefix}: Common Entry and Exit Topics",
            xaxis_title="Topic",
            yaxis_title="Frequency",
            barmode='group',
            width=width,
            height=height
        )
        
        figures["entry_exit_topics"] = fig_entry_exit
    
    # 3. Topic retention (horizontal bar chart)
    if metrics["topic_retention"]:
        topics = list(metrics["topic_retention"].keys())
        retention_rates = list(metrics["topic_retention"].values())
        
        # Sort by retention rate
        sorted_indices = np.argsort(retention_rates)[::-1]
        sorted_topics = [topics[i] for i in sorted_indices]
        sorted_rates = [retention_rates[i] for i in sorted_indices]
        
        fig_retention = go.Figure(data=[
            go.Bar(
                y=sorted_topics,
                x=sorted_rates,
                text=[f"{r:.1%}" for r in sorted_rates],
                textposition='auto',
                marker_color='lightgreen',
                orientation='h'
            )
        ])
        
        fig_retention.update_layout(
            title=f"{title_prefix}: Topic Retention Rates",
            xaxis_title="Retention Rate",
            yaxis_title="Topic",
            xaxis=dict(tickformat=".0%"),
            width=width,
            height=height
        )
        
        figures["topic_retention"] = fig_retention
    
    # 4. Common paths (network diagram)
    if metrics["frequent_paths"]:
        top_paths = metrics["frequent_paths"][:10]  # Use only top 10 for clarity
        
        # Create nodes (unique topics in paths)
        unique_topics = set()
        for path, _ in top_paths:
            for topic in path:
                unique_topics.add(topic)
                
        node_list = list(unique_topics)
        node_indices = {node: i for i, node in enumerate(node_list)}
        
        # Create edges
        sources = []
        targets = []
        values = []
        labels = []
        
        for path, count in top_paths:
            # For each path, connect consecutive topics
            for i in range(len(path) - 1):
                from_topic = path[i]
                to_topic = path[i + 1]
                
                sources.append(node_indices[from_topic])
                targets.append(node_indices[to_topic])
                values.append(count)
                labels.append(f"{from_topic} → {to_topic}: {count}")
        
        # Generate colors for nodes
        node_colors = []
        for i, node in enumerate(node_list):
            hue = i / len(node_list)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            node_colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)")
            
        # Create network diagram
        fig_paths = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_list,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=labels,
                hovertemplate='%{label}<extra></extra>'
            )
        )])
        
        fig_paths.update_layout(
            title=f"{title_prefix}: Common Topic Paths",
            font=dict(size=12),
            width=width,
            height=height
        )
        
        figures["common_paths"] = fig_paths
        
    return figures

def filter_entity_transitions(
    df: pd.DataFrame,
    entity_id_column: str,
    topic_column: str,
    time_column: str,
    min_time: Optional[datetime] = None,
    max_time: Optional[datetime] = None,
    categories: Optional[List[str]] = None,
    category_column: Optional[str] = None,
    min_value: Optional[float] = None,
    value_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter entity transition data based on various criteria.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entity data with topic assignments over time
    entity_id_column : str
        Name of column containing entity IDs
    topic_column : str
        Name of column containing topic assignments
    time_column : str
        Name of column containing timestamp information
    min_time : Optional[datetime], default None
        Filter transitions after this time
    max_time : Optional[datetime], default None
        Filter transitions before this time
    categories : Optional[List[str]], default None
        List of categories to include
    category_column : Optional[str], default None
        Name of column containing category information
    min_value : Optional[float], default None
        Minimum value to include
    value_column : Optional[str], default None
        Name of column containing values to filter by
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    # Make a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(filtered_df[time_column]):
        filtered_df[time_column] = pd.to_datetime(filtered_df[time_column])
    
    # Apply time filters
    if min_time is not None:
        filtered_df = filtered_df[filtered_df[time_column] >= min_time]
    
    if max_time is not None:
        filtered_df = filtered_df[filtered_df[time_column] <= max_time]
    
    # Apply category filter
    if categories is not None and category_column is not None and category_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[category_column].isin(categories)]
    
    # Apply value filter
    if min_value is not None and value_column is not None and value_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[value_column] >= min_value]
    
    # Filter out entities with only a single record (no transitions)
    entity_counts = filtered_df[entity_id_column].value_counts()
    entities_with_transitions = entity_counts[entity_counts > 1].index
    filtered_df = filtered_df[filtered_df[entity_id_column].isin(entities_with_transitions)]
    
    return filtered_df

def create_entity_flow_sankey(
    df: pd.DataFrame,
    entity_id_column: str,
    topic_column: str,
    time_column: str,
    value_column: Optional[str] = None,
    include_self_transitions: bool = False,
    time_segments: Optional[int] = None,
    title: str = "Entity Flow Between Topics Over Time",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """
    Create an enhanced Sankey diagram showing entity flow between topics across time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entity data with topic assignments over time
    entity_id_column : str
        Name of column containing entity IDs (customer ID, claim number, etc.)
    topic_column : str
        Name of column containing topic assignments
    time_column : str
        Name of column containing timestamp or datetime information
    value_column : Optional[str], default None
        Name of column containing values to aggregate for flow weight
        (if None, uses counts of entities)
    include_self_transitions : bool, default False
        Whether to include transitions where the topic stays the same
    time_segments : Optional[int], default None
        Number of time segments to divide the data into (if None, uses natural time divisions)
    title : str, default "Entity Flow Between Topics Over Time"
        Title for the visualization
    width : int, default 1000
        Width of the figure in pixels
    height : int, default 800
        Height of the figure in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure containing the enhanced Sankey diagram
        
    Notes
    -----
    This visualization shows how entities flow between topics across time periods,
    with nodes representing topic-time combinations and links showing transitions.
    """
    # Ensure data is sorted by entity and time
    plot_df = df.copy().sort_values([entity_id_column, time_column])
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Create time segments if requested
    if time_segments is not None:
        # Divide time range into equal segments
        min_time = plot_df[time_column].min()
        max_time = plot_df[time_column].max()
        time_range = max_time - min_time
        segment_duration = time_range / time_segments
        
        # Assign time segment to each row
        plot_df['time_segment'] = ((plot_df[time_column] - min_time) / segment_duration).astype(int)
        plot_df['time_segment'] = plot_df['time_segment'].clip(upper=time_segments-1)
        
        # Create readable time segment labels
        time_labels = {}
        for i in range(time_segments):
            segment_start = min_time + i * segment_duration
            segment_end = min_time + (i + 1) * segment_duration
            time_labels[i] = f"{segment_start.strftime('%Y-%m-%d')} to {segment_end.strftime('%Y-%m-%d')}"
            
        plot_df['time_label'] = plot_df['time_segment'].map(time_labels)
    else:
        # Use existing time values, aggregated by month
        plot_df['time_segment'] = plot_df[time_column].dt.to_period('M').astype(str)
        plot_df['time_label'] = plot_df['time_segment']
    
    # Create combined topic-time nodes
    plot_df['node'] = plot_df[topic_column].astype(str) + " (" + plot_df['time_label'].astype(str) + ")"
    
    # Calculate transitions between nodes
    transitions = {}
    
    for entity in plot_df[entity_id_column].unique():
        entity_data = plot_df[plot_df[entity_id_column] == entity].sort_values(time_column)
        
        if len(entity_data) <= 1:
            continue  # Skip entities with only one record
            
        # Extract sequence of nodes for this entity
        node_sequence = entity_data['node'].tolist()
        time_sequence = entity_data['time_segment'].tolist()
        
        # Record transitions between nodes
        for i in range(len(node_sequence) - 1):
            from_node = node_sequence[i]
            to_node = node_sequence[i + 1]
            from_time = time_sequence[i]
            to_time = time_sequence[i + 1]
            
            # Skip self-transitions if requested
            if from_node == to_node and not include_self_transitions:
                continue
                
            # Get flow value
            if value_column:
                flow_value = entity_data.iloc[i][value_column]
            else:
                flow_value = 1
                
            transition_key = (from_node, to_node)
            if transition_key not in transitions:
                transitions[transition_key] = {
                    'value': 0,
                    'entities': set(),
                    'from_time': from_time,
                    'to_time': to_time
                }
                
            transitions[transition_key]['value'] += flow_value
            transitions[transition_key]['entities'].add(entity)
    
    if not transitions:
        # Create an empty figure with a message if no transitions found
        fig = go.Figure()
        fig.add_annotation(
            text="No transitions found matching the criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(width=width, height=height, title=title)
        return fig
    
    # Create node list for Sankey diagram
    all_nodes = set()
    for source, target in transitions.keys():
        all_nodes.add(source)
        all_nodes.add(target)
    
    # Sort nodes by time period then topic
    def node_sort_key(node):
        # Extract time and topic from node string
        parts = node.split('(')
        topic = parts[0].strip()
        time_part = parts[1].replace(')', '').strip() if len(parts) > 1 else ''
        return (time_part, topic)
    
    node_list = sorted(list(all_nodes), key=node_sort_key)
    node_indices = {node: i for i, node in enumerate(node_list)}
    
    # Create link data
    sources = []
    targets = []
    values = []
    labels = []
    link_colors = []
    
    for (source, target), data in transitions.items():
        sources.append(node_indices[source])
        targets.append(node_indices[target])
        values.append(data['value'])
        entity_count = len(data['entities'])
        labels.append(f"{source} → {target}: {entity_count} entities, value: {data['value']}")
        
        # Extract topics for color
        source_topic = source.split('(')[0].strip()
        target_topic = target.split('(')[0].strip()
        
        # Color based on whether topic changed
        if source_topic == target_topic:
            link_colors.append("rgba(100, 100, 100, 0.3)")  # Gray for same topic
        else:
            link_colors.append("rgba(255, 165, 0, 0.6)")  # Orange for topic change
    
    # Create node colors by topic
    unique_topics = set(node.split('(')[0].strip() for node in node_list)
    topic_colors = {}
    
    # Generate colors for topics
    for i, topic in enumerate(sorted(unique_topics)):
        hue = i / len(unique_topics)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        topic_colors[topic] = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
    
    # Assign colors to nodes based on topic
    node_colors = []
    for node in node_list:
        topic = node.split('(')[0].strip()
        node_colors.append(topic_colors[topic])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_list,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels,
            color=link_colors,
            hovertemplate='%{label}<extra></extra>'
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        font=dict(size=12),
        width=width,
        height=height
    )
    
    return fig
"""
Time-Space Visualization Module for Meno Topic Modeling.

This module provides visualizations for analyzing topics across both time and 
geographic dimensions simultaneously, enabling discovery of spatiotemporal
patterns in topic distribution.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

def create_animated_map(
    df: pd.DataFrame,
    topic_column: str,
    time_column: str,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    value_column: Optional[str] = None,
    time_interval: str = "M",
    zoom: int = 3,
    center: Optional[Dict[str, float]] = None,
    marker_size_range: Tuple[int, int] = (5, 15),
    title: str = "Topic Distribution Over Time and Space",
    height: int = 600,
    width: int = 800,
    mapbox_style: str = "open-street-map",
) -> go.Figure:
    """
    Create an animated map showing topic distribution over time and space.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    time_column : str
        Name of column containing datetime values
    lat_column : str, default "latitude"
        Name of column containing latitude values
    lon_column : str, default "longitude"
        Name of column containing longitude values
    value_column : str, optional
        Name of column containing values for sizing markers (if None, all markers are equal size)
    time_interval : str, default "M"
        Time interval for animation frames: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    zoom : int, default 3
        Initial zoom level for the map
    center : Dict[str, float], optional
        Center coordinates for the map, e.g. {"lat": -33.8688, "lon": 151.2093}
        If None, automatically centers based on data
    marker_size_range : Tuple[int, int], default (5, 15)
        Range of marker sizes (min, max) in pixels
    title : str, default "Topic Distribution Over Time and Space"
        Title of the plot
    height : int, default 600
        Height of the plot in pixels
    width : int, default 800
        Width of the plot in pixels
    mapbox_style : str, default "open-street-map"
        Map style to use: "open-street-map", "carto-positron", "carto-darkmatter",
        "stamen-terrain", "stamen-toner", "stamen-watercolor"
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the animated map visualization
        
    Notes
    -----
    This visualization creates an animated map that shows how topic distribution
    changes across geographic locations over time. It requires both time and
    location data for each document.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, time_column, lat_column, lon_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Create a time frame column for animation
    if time_interval == 'D':
        plot_df['time_frame'] = plot_df[time_column].dt.strftime('%Y-%m-%d')
    elif time_interval == 'W':
        plot_df['time_frame'] = plot_df[time_column].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')
    elif time_interval == 'M':
        plot_df['time_frame'] = plot_df[time_column].dt.strftime('%Y-%m')
    elif time_interval == 'Q':
        plot_df['time_frame'] = plot_df[time_column].dt.to_period('Q').astype(str)
    else:  # 'Y'
        plot_df['time_frame'] = plot_df[time_column].dt.strftime('%Y')
    
    # Determine marker size
    if value_column:
        if value_column not in plot_df.columns:
            raise ValueError(f"Value column '{value_column}' not found in dataframe")
        
        # Group by time frame, topic, and location
        agg_df = plot_df.groupby(['time_frame', topic_column, lat_column, lon_column])[value_column].sum().reset_index()
        
        # Normalize sizes for each time frame
        frames = []
        for time_frame in agg_df['time_frame'].unique():
            frame_df = agg_df[agg_df['time_frame'] == time_frame].copy()
            min_val = frame_df[value_column].min()
            max_val = frame_df[value_column].max()
            
            if min_val == max_val:
                frame_df['marker_size'] = marker_size_range[1]
            else:
                norm = (frame_df[value_column] - min_val) / (max_val - min_val)
                size_range = marker_size_range[1] - marker_size_range[0]
                frame_df['marker_size'] = marker_size_range[0] + (norm * size_range)
            
            frames.append(frame_df)
        
        agg_df = pd.concat(frames)
        
    else:
        # Count occurrences for each time frame, topic, and location
        agg_df = plot_df.groupby(['time_frame', topic_column, lat_column, lon_column]).size().reset_index(name='count')
        value_column = 'count'
        
        # Normalize sizes for each time frame
        frames = []
        for time_frame in agg_df['time_frame'].unique():
            frame_df = agg_df[agg_df['time_frame'] == time_frame].copy()
            min_val = frame_df[value_column].min()
            max_val = frame_df[value_column].max()
            
            if min_val == max_val:
                frame_df['marker_size'] = marker_size_range[1]
            else:
                norm = (frame_df[value_column] - min_val) / (max_val - min_val)
                size_range = marker_size_range[1] - marker_size_range[0]
                frame_df['marker_size'] = marker_size_range[0] + (norm * size_range)
            
            frames.append(frame_df)
        
        agg_df = pd.concat(frames)
    
    # Create figure
    fig = px.scatter_mapbox(
        agg_df,
        lat=lat_column,
        lon=lon_column,
        color=topic_column,
        size='marker_size',
        animation_frame='time_frame',
        size_max=marker_size_range[1],
        hover_name=topic_column,
        hover_data=[col for col in [value_column] if col is not None],
        title=title,
        height=height,
        width=width,
    )
    
    # Update map style
    fig.update_layout(mapbox_style=mapbox_style)
    
    # Set map center if provided
    if center:
        fig.update_layout(
            mapbox=dict(
                center=center,
                zoom=zoom
            )
        )
    else:
        # Auto-center on data
        center_lat = agg_df[lat_column].mean()
        center_lon = agg_df[lon_column].mean()
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            )
        )
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    
    # Configure animation
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
    
    return fig

def create_space_time_heatmap(
    df: pd.DataFrame,
    topic_column: str,
    time_column: str,
    region_column: str,
    topic_to_plot: Optional[str] = None,
    value_column: Optional[str] = None,
    time_interval: str = "M",
    normalize: bool = False,
    title: Optional[str] = None,
    color_scale: str = "Viridis",
    height: int = 600,
    width: int = 900,
) -> go.Figure:
    """
    Create a heatmap showing topic intensity across regions over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    time_column : str
        Name of column containing datetime values
    region_column : str
        Name of column containing region identifiers
    topic_to_plot : str, optional
        Specific topic to visualize (if None, user can select from dropdown)
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    time_interval : str, default "M"
        Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    normalize : bool, default False
        If True, normalize values to show percentage distribution within each time period
    title : str, optional
        Title of the plot (if None, auto-generates based on topic)
    color_scale : str, default "Viridis"
        Colorscale for the heatmap
    height : int, default 600
        Height of the plot in pixels
    width : int, default 900
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the space-time heatmap visualization
        
    Notes
    -----
    This visualization shows how topics evolve across different regions over time,
    making it easy to identify regional trends and patterns.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, time_column, region_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Filter for specific topic if requested
    if topic_to_plot:
        plot_df = plot_df[plot_df[topic_column] == topic_to_plot]
        if len(plot_df) == 0:
            raise ValueError(f"No data found for topic '{topic_to_plot}'")
    
    # Create time period column
    if time_interval == 'D':
        plot_df['time_period'] = plot_df[time_column].dt.strftime('%Y-%m-%d')
    elif time_interval == 'W':
        plot_df['time_period'] = plot_df[time_column].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')
    elif time_interval == 'M':
        plot_df['time_period'] = plot_df[time_column].dt.strftime('%Y-%m')
    elif time_interval == 'Q':
        plot_df['time_period'] = plot_df[time_column].dt.to_period('Q').astype(str)
    else:  # 'Y'
        plot_df['time_period'] = plot_df[time_column].dt.strftime('%Y')
    
    # Aggregate data
    if topic_to_plot:
        # Single topic heatmap
        if value_column:
            if value_column not in plot_df.columns:
                raise ValueError(f"Value column '{value_column}' not found in dataframe")
            agg_df = plot_df.groupby(['time_period', region_column])[value_column].sum().reset_index()
        else:
            agg_df = plot_df.groupby(['time_period', region_column]).size().reset_index(name='count')
            value_column = 'count'
        
        # Create pivot table
        pivot_df = agg_df.pivot(index='time_period', columns=region_column, values=value_column).fillna(0)
        
        # Normalize if requested
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Auto-generate title if not provided
        if title is None:
            title = f"Distribution of '{topic_to_plot}' Across Regions Over Time"
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=color_scale,
            colorbar=dict(
                title=dict(
                    text="Percentage (%)" if normalize else value_column,
                    side="right"
                )
            ),
            hovertemplate=(
                "<b>Time:</b> %{y}<br>" +
                "<b>Region:</b> %{x}<br>" +
                "<b>Value:</b> %{z:.2f}" + ("%" if normalize else "") + "<extra></extra>"
            )
        ))
    else:
        # Multi-topic with dropdown selection
        topics = plot_df[topic_column].unique()
        
        # Create figure
        fig = go.Figure()
        
        # Create a heatmap for each topic
        for i, topic in enumerate(topics):
            topic_data = plot_df[plot_df[topic_column] == topic]
            
            if value_column:
                if value_column not in topic_data.columns:
                    raise ValueError(f"Value column '{value_column}' not found in dataframe")
                agg_df = topic_data.groupby(['time_period', region_column])[value_column].sum().reset_index()
            else:
                agg_df = topic_data.groupby(['time_period', region_column]).size().reset_index(name='count')
                value_column = 'count'
            
            # Create pivot table
            pivot_df = agg_df.pivot(index='time_period', columns=region_column, values=value_column).fillna(0)
            
            # Normalize if requested
            if normalize:
                pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
            
            # Add trace
            visible = True if i == 0 else False
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    colorscale=color_scale,
                    colorbar=dict(
                        title=dict(
                            text="Percentage (%)" if normalize else value_column,
                            side="right"
                        )
                    ),
                    hovertemplate=(
                        "<b>Time:</b> %{y}<br>" +
                        "<b>Region:</b> %{x}<br>" +
                        "<b>Value:</b> %{z:.2f}" + ("%" if normalize else "") + "<extra></extra>"
                    ),
                    visible=visible
                )
            )
        
        # Create dropdown menu
        buttons = []
        for i, topic in enumerate(topics):
            visibility = [i == j for j in range(len(topics))]
            
            buttons.append(
                dict(
                    label=topic,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"Distribution of '{topic}' Across Regions Over Time"}
                    ]
                )
            )
        
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        # Auto-generate title if not provided
        if title is None:
            title = f"Distribution of Topics Across Regions Over Time"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Region",
        yaxis_title="Time Period",
        height=height,
        width=width,
    )
    
    return fig

def create_category_time_plot(
    df: pd.DataFrame,
    topic_column: str,
    time_column: str,
    category_column: str,
    value_column: Optional[str] = None,
    time_interval: str = "M",
    plot_type: str = "line",
    normalize: bool = False,
    stacked: bool = True,
    top_n_topics: Optional[int] = None,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 900,
) -> go.Figure:
    """
    Create a visualization showing topic trends over time, faceted by categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    time_column : str
        Name of column containing datetime values
    category_column : str
        Name of column containing category labels for faceting
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    time_interval : str, default "M"
        Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    plot_type : str, default "line"
        Type of plot: "line", "area", "bar"
    normalize : bool, default False
        If True, normalize values to show percentage distribution
    stacked : bool, default True
        If True, stack areas/bars; if False, overlay/group them
    top_n_topics : int, optional
        If provided, only show top N topics by overall frequency
    title : str, optional
        Title of the plot (if None, auto-generates)
    height : int, default 500
        Height of the plot in pixels
    width : int, default 900
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the category-time visualization
        
    Notes
    -----
    This visualization shows how topics evolve over time within different categories,
    making it easy to compare trends across different subsets of data.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, time_column, category_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Create time period column
    if time_interval == 'D':
        plot_df['time_period'] = plot_df[time_column].dt.strftime('%Y-%m-%d')
    elif time_interval == 'W':
        plot_df['time_period'] = plot_df[time_column].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')
    elif time_interval == 'M':
        plot_df['time_period'] = plot_df[time_column].dt.strftime('%Y-%m')
    elif time_interval == 'Q':
        plot_df['time_period'] = plot_df[time_column].dt.to_period('Q').astype(str)
    else:  # 'Y'
        plot_df['time_period'] = plot_df[time_column].dt.strftime('%Y')
    
    # Aggregate data
    if value_column:
        if value_column not in plot_df.columns:
            raise ValueError(f"Value column '{value_column}' not found in dataframe")
        agg_df = plot_df.groupby(['time_period', category_column, topic_column])[value_column].sum().reset_index()
    else:
        agg_df = plot_df.groupby(['time_period', category_column, topic_column]).size().reset_index(name='count')
        value_column = 'count'
    
    # If requested, get only top N topics
    if top_n_topics:
        # Calculate overall topic frequency
        topic_totals = agg_df.groupby(topic_column)[value_column].sum()
        top_topics = topic_totals.nlargest(top_n_topics).index.tolist()
        agg_df = agg_df[agg_df[topic_column].isin(top_topics)]
    
    # Create pivot table for each category
    categories = agg_df[category_column].unique()
    
    # Create subplot figure
    if plot_type == "line":
        fig = create_line_subplots(agg_df, categories, time_interval, category_column, topic_column, value_column, normalize, stacked, title, height, width)
    elif plot_type == "area":
        fig = create_area_subplots(agg_df, categories, time_interval, category_column, topic_column, value_column, normalize, stacked, title, height, width)
    elif plot_type == "bar":
        fig = create_bar_subplots(agg_df, categories, time_interval, category_column, topic_column, value_column, normalize, stacked, title, height, width)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    return fig

def create_line_subplots(agg_df, categories, time_interval, category_column, topic_column, value_column, normalize, stacked, title, height, width):
    """Helper function to create line subplot for each category"""
    # Create figure with subplots
    fig = go.Figure()
    
    # Calculate number of rows and columns
    n_categories = len(categories)
    
    # Add a trace for each category-topic combination
    for i, category in enumerate(categories):
        # Filter for this category
        cat_df = agg_df[agg_df[category_column] == category]
        
        # Pivot table for this category
        pivot_df = cat_df.pivot_table(
            index='time_period', 
            columns=topic_column, 
            values=value_column,
            aggfunc='sum'
        ).fillna(0)
        
        # Sort time periods correctly
        if time_interval == 'M':
            # Convert to datetime for proper sorting
            pivot_df.index = pd.to_datetime(pivot_df.index, format='%Y-%m')
            pivot_df = pivot_df.sort_index()
            # Convert back to string
            pivot_df.index = pivot_df.index.strftime('%Y-%m')
        elif time_interval == 'D' or time_interval == 'W':
            # Convert to datetime for proper sorting
            pivot_df.index = pd.to_datetime(pivot_df.index)
            pivot_df = pivot_df.sort_index()
            # Convert back to string
            pivot_df.index = pivot_df.index.strftime('%Y-%m-%d')
        else:
            # For 'Q' and 'Y', lexicographic sorting should work
            pivot_df = pivot_df.sort_index()
        
        # Normalize if requested
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Add a line for each topic in this category
        for topic in pivot_df.columns:
            # Create display name with category
            display_name = f"{topic} ({category})"
            
            fig.add_trace(
                go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[topic],
                    mode='lines+markers',
                    name=display_name,
                    hovertemplate=(
                        f"<b>{category}</b><br>" +
                        f"Time: %{{x}}<br>" +
                        f"Topic: {topic}<br>" +
                        f"Value: %{{y:.2f}}{'%' if normalize else ''}<extra></extra>"
                    )
                )
            )
    
    # Title
    if title is None:
        title = "Topic Trends Over Time by Category"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Percentage (%)" if normalize else value_column,
        height=height,
        width=width,
        hovermode="closest",
        legend=dict(
            groupclick="toggleitem"
        ),
    )
    
    return fig

def create_area_subplots(agg_df, categories, time_interval, category_column, topic_column, value_column, normalize, stacked, title, height, width):
    """Helper function to create area subplot for each category"""
    # Create subplot figure
    fig = make_subplots(
        rows=len(categories), 
        cols=1,
        subplot_titles=[str(cat) for cat in categories],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add a subplot for each category
    for i, category in enumerate(categories):
        # Filter for this category
        cat_df = agg_df[agg_df[category_column] == category]
        
        # Pivot table for this category
        pivot_df = cat_df.pivot_table(
            index='time_period', 
            columns=topic_column, 
            values=value_column,
            aggfunc='sum'
        ).fillna(0)
        
        # Sort time periods correctly
        if time_interval == 'M':
            # Convert to datetime for proper sorting
            pivot_df.index = pd.to_datetime(pivot_df.index, format='%Y-%m')
            pivot_df = pivot_df.sort_index()
            # Convert back to string
            pivot_df.index = pivot_df.index.strftime('%Y-%m')
        elif time_interval == 'D' or time_interval == 'W':
            # Convert to datetime for proper sorting
            pivot_df.index = pd.to_datetime(pivot_df.index)
            pivot_df = pivot_df.sort_index()
            # Convert back to string
            pivot_df.index = pivot_df.index.strftime('%Y-%m-%d')
        else:
            # For 'Q' and 'Y', lexicographic sorting should work
            pivot_df = pivot_df.sort_index()
        
        # Normalize if requested
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Add area for each topic
        for topic in pivot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[topic],
                    mode='lines',
                    name=f"{topic} ({category})",
                    stackgroup='category' + str(i) if stacked else None,
                    hovertemplate=(
                        f"<b>{category}</b><br>" +
                        f"Time: %{{x}}<br>" +
                        f"Topic: {topic}<br>" +
                        f"Value: %{{y:.2f}}{'%' if normalize else ''}<extra></extra>"
                    )
                ),
                row=i+1, 
                col=1
            )
    
    # Title
    if title is None:
        title = "Topic Composition Over Time by Category"
    
    # Update layout
    fig.update_layout(
        title=title,
        height=100 * len(categories) + 150,  # Adjust height based on number of subplots
        width=width,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            groupclick="toggleitem"
        ),
    )
    
    # Update y-axis titles
    for i in range(len(categories)):
        fig.update_yaxes(
            title_text="Percentage (%)" if normalize else value_column,
            row=i+1,
            col=1
        )
    
    # Update x-axis title for bottom subplot
    fig.update_xaxes(
        title_text="Time Period",
        row=len(categories),
        col=1
    )
    
    return fig

def create_bar_subplots(agg_df, categories, time_interval, category_column, topic_column, value_column, normalize, stacked, title, height, width):
    """Helper function to create bar subplot for each category"""
    from plotly.subplots import make_subplots
    
    # Create subplot figure
    fig = make_subplots(
        rows=len(categories), 
        cols=1,
        subplot_titles=[str(cat) for cat in categories],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add a subplot for each category
    for i, category in enumerate(categories):
        # Filter for this category
        cat_df = agg_df[agg_df[category_column] == category]
        
        # Pivot table for this category
        pivot_df = cat_df.pivot_table(
            index='time_period', 
            columns=topic_column, 
            values=value_column,
            aggfunc='sum'
        ).fillna(0)
        
        # Sort time periods correctly
        if time_interval == 'M':
            # Convert to datetime for proper sorting
            pivot_df.index = pd.to_datetime(pivot_df.index, format='%Y-%m')
            pivot_df = pivot_df.sort_index()
            # Convert back to string
            pivot_df.index = pivot_df.index.strftime('%Y-%m')
        elif time_interval == 'D' or time_interval == 'W':
            # Convert to datetime for proper sorting
            pivot_df.index = pd.to_datetime(pivot_df.index)
            pivot_df = pivot_df.sort_index()
            # Convert back to string
            pivot_df.index = pivot_df.index.strftime('%Y-%m-%d')
        else:
            # For 'Q' and 'Y', lexicographic sorting should work
            pivot_df = pivot_df.sort_index()
        
        # Normalize if requested
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Add bars for each topic
        for topic in pivot_df.columns:
            fig.add_trace(
                go.Bar(
                    x=pivot_df.index,
                    y=pivot_df[topic],
                    name=f"{topic} ({category})",
                    hovertemplate=(
                        f"<b>{category}</b><br>" +
                        f"Time: %{{x}}<br>" +
                        f"Topic: {topic}<br>" +
                        f"Value: %{{y:.2f}}{'%' if normalize else ''}<extra></extra>"
                    )
                ),
                row=i+1, 
                col=1
            )
    
    # Update bar mode
    fig.update_layout(barmode='stack' if stacked else 'group')
    
    # Title
    if title is None:
        title = "Topic Distribution Over Time by Category"
    
    # Update layout
    fig.update_layout(
        title=title,
        height=150 * len(categories) + 150,  # Adjust height based on number of subplots
        width=width,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            groupclick="toggleitem"
        ),
    )
    
    # Update y-axis titles
    for i in range(len(categories)):
        fig.update_yaxes(
            title_text="Percentage (%)" if normalize else value_column,
            row=i+1,
            col=1
        )
    
    # Update x-axis title for bottom subplot
    fig.update_xaxes(
        title_text="Time Period",
        row=len(categories),
        col=1
    )
    
    return fig
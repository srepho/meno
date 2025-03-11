"""
Time Series Visualization Module for Meno Topic Modeling.

This module provides time-based visualization of topic trends and distributions
over time. It enables analysis of how topics evolve and change in frequency
across temporal dimensions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

def create_topic_trend_plot(
    df: pd.DataFrame,
    time_column: str,
    topic_column: str,
    value_column: Optional[str] = None,
    time_interval: str = "M",
    normalize: bool = False,
    cumulative: bool = False,
    top_n_topics: Optional[int] = None,
    color_map: Optional[Dict[str, str]] = None,
    title: str = "Topic Trends Over Time",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Create an interactive line chart showing topic trends over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    time_column : str
        Name of column containing datetime values
    topic_column : str
        Name of column containing topic labels
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    time_interval : str, default "M"
        Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    normalize : bool, default False
        If True, normalize values to show percentage distribution
    cumulative : bool, default False
        If True, show cumulative values over time
    top_n_topics : int, optional
        If provided, only show top N topics by overall frequency
    color_map : Dict[str, str], optional
        Mapping of topic names to colors
    title : str, default "Topic Trends Over Time"
        Title of the plot
    height : int, default 600
        Height of the plot in pixels
    width : int, default 800
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the time series visualization
        
    Notes
    -----
    This visualization is useful for understanding how topics evolve over time
    and identifying temporal patterns or shifts in topic prevalence.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Group by time interval and topic, then aggregate
    if value_column:
        # Aggregate provided values
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ])[value_column].sum().reset_index()
    else:
        # Count occurrences
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ]).size().reset_index(name='count')
        value_column = 'count'
    
    # If requested, get only top N topics
    if top_n_topics:
        top_topics = agg_df.groupby(topic_column)[value_column].sum().nlargest(top_n_topics).index.tolist()
        agg_df = agg_df[agg_df[topic_column].isin(top_topics)]
    
    # Pivot the data for easier plotting
    pivot_df = agg_df.pivot(index=time_column, columns=topic_column, values=value_column).fillna(0)
    
    # Apply normalization if requested
    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Apply cumulative sum if requested
    if cumulative:
        pivot_df = pivot_df.cumsum()
    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each topic
    for topic in pivot_df.columns:
        color = color_map.get(topic) if color_map else None
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[topic],
                mode='lines',
                name=topic,
                line=dict(width=2, color=color),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>" +
                    f"{topic}: %{{y:.2f}}{'%' if normalize else ''}<extra></extra>"
                )
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Frequency" if not normalize else "Percentage (%)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=height,
        width=width,
    )
    
    return fig

def create_topic_heatmap(
    df: pd.DataFrame,
    time_column: str,
    topic_column: str,
    value_column: Optional[str] = None,
    time_interval: str = "M",
    normalize: bool = False,
    top_n_topics: Optional[int] = None,
    color_scale: Optional[str] = "Viridis",
    title: str = "Topic Intensity Heatmap",
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """
    Create a heatmap showing topic intensity over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    time_column : str
        Name of column containing datetime values
    topic_column : str
        Name of column containing topic labels
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    time_interval : str, default "M"
        Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    normalize : bool, default False
        If True, normalize values to show percentage distribution
    top_n_topics : int, optional
        If provided, only show top N topics by overall frequency
    color_scale : str, default "Viridis"
        Colorscale for the heatmap
    title : str, default "Topic Intensity Heatmap"
        Title of the plot
    height : int, default 700
        Height of the plot in pixels
    width : int, default 900
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the heatmap visualization
        
    Notes
    -----
    The heatmap visualization makes it easy to spot "hot" periods for specific topics
    and compare multiple topics simultaneously across time.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Group by time interval and topic, then aggregate
    if value_column:
        # Aggregate provided values
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ])[value_column].sum().reset_index()
    else:
        # Count occurrences
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ]).size().reset_index(name='count')
        value_column = 'count'
    
    # If requested, get only top N topics
    if top_n_topics:
        top_topics = agg_df.groupby(topic_column)[value_column].sum().nlargest(top_n_topics).index.tolist()
        agg_df = agg_df[agg_df[topic_column].isin(top_topics)]
    
    # Pivot the data for heatmap
    pivot_df = agg_df.pivot(index=time_column, columns=topic_column, values=value_column).fillna(0)
    
    # Apply normalization if requested
    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Format time labels based on interval
    if time_interval == 'D':
        time_labels = [d.strftime('%Y-%m-%d') for d in pivot_df.index]
    elif time_interval == 'W':
        time_labels = [d.strftime('%Y-%m-%d') for d in pivot_df.index]
    elif time_interval == 'M':
        time_labels = [d.strftime('%Y-%m') for d in pivot_df.index]
    elif time_interval == 'Q':
        time_labels = [f"{d.year} Q{(d.month-1)//3+1}" for d in pivot_df.index]
    else:  # 'Y'
        time_labels = [str(d.year) for d in pivot_df.index]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values.T,
        x=time_labels,
        y=pivot_df.columns.tolist(),
        colorscale=color_scale,
        colorbar=dict(
            title=dict(
                text="Percentage (%)" if normalize else "Count",
                side="right"
            )
        ),
        hovertemplate=(
            "<b>Time:</b> %{x}<br>" +
            "<b>Topic:</b> %{y}<br>" +
            "<b>Value:</b> %{z:.2f}{'%' if normalize else ''}<extra></extra>"
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Topic",
        height=height,
        width=width,
    )
    
    return fig

def create_topic_stacked_area(
    df: pd.DataFrame,
    time_column: str,
    topic_column: str,
    value_column: Optional[str] = None,
    time_interval: str = "M",
    normalize: bool = True,
    top_n_topics: Optional[int] = None,
    color_map: Optional[Dict[str, str]] = None,
    title: str = "Topic Composition Over Time",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Create a stacked area chart showing relative or absolute topic composition over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    time_column : str
        Name of column containing datetime values
    topic_column : str
        Name of column containing topic labels
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    time_interval : str, default "M"
        Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    normalize : bool, default True
        If True, normalize values to show percentage distribution (100% stacked)
    top_n_topics : int, optional
        If provided, only show top N topics by overall frequency
    color_map : Dict[str, str], optional
        Mapping of topic names to colors
    title : str, default "Topic Composition Over Time"
        Title of the plot
    height : int, default 600
        Height of the plot in pixels
    width : int, default 800
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the stacked area visualization
        
    Notes
    -----
    Stacked area charts are particularly useful for showing how the composition
    of topics changes over time, especially when normalized to show percentages.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Group by time interval and topic, then aggregate
    if value_column:
        # Aggregate provided values
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ])[value_column].sum().reset_index()
    else:
        # Count occurrences
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ]).size().reset_index(name='count')
        value_column = 'count'
    
    # If requested, get only top N topics
    if top_n_topics:
        top_topics = agg_df.groupby(topic_column)[value_column].sum().nlargest(top_n_topics).index.tolist()
        agg_df = agg_df[agg_df[topic_column].isin(top_topics)]
    
    # Pivot the data for easier plotting
    pivot_df = agg_df.pivot(index=time_column, columns=topic_column, values=value_column).fillna(0)
    
    # Apply normalization if requested
    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add area for each topic
    for topic in pivot_df.columns:
        color = color_map.get(topic) if color_map else None
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[topic],
                mode='lines',
                name=topic,
                stackgroup='one',
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>" +
                    f"{topic}: %{{y:.2f}}{'%' if normalize else ''}<extra></extra>"
                )
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Percentage (%)" if normalize else "Count",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=height,
        width=width,
    )
    
    return fig

def create_topic_ridge_plot(
    df: pd.DataFrame,
    time_column: str,
    topic_column: str,
    value_column: Optional[str] = None,
    time_interval: str = "M",
    normalize: bool = False,
    top_n_topics: Optional[int] = None,
    sort_topics: bool = True,
    color_map: Optional[Dict[str, str]] = None,
    title: str = "Topic Ridge Plot Over Time",
    height: int = 800,
    width: int = 800,
) -> go.Figure:
    """
    Create a ridge plot showing topic distributions over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    time_column : str
        Name of column containing datetime values
    topic_column : str
        Name of column containing topic labels
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    time_interval : str, default "M"
        Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), 'Y' (yearly)
    normalize : bool, default False
        If True, normalize values to show percentage distribution
    top_n_topics : int, optional
        If provided, only show top N topics by overall frequency
    sort_topics : bool, default True
        If True, sort topics by overall frequency
    color_map : Dict[str, str], optional
        Mapping of topic names to colors
    title : str, default "Topic Ridge Plot Over Time"
        Title of the plot
    height : int, default 800
        Height of the plot in pixels
    width : int, default 800
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the ridge plot visualization
        
    Notes
    -----
    Ridge plots provide a way to visualize changes in distributions over
    time or categories, making it easy to spot trends and patterns.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Group by time interval and topic, then aggregate
    if value_column:
        # Aggregate provided values
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ])[value_column].sum().reset_index()
    else:
        # Count occurrences
        agg_df = plot_df.groupby([
            pd.Grouper(key=time_column, freq=time_interval),
            topic_column
        ]).size().reset_index(name='count')
        value_column = 'count'
    
    # If requested, get only top N topics
    if top_n_topics:
        top_topics = agg_df.groupby(topic_column)[value_column].sum().nlargest(top_n_topics).index.tolist()
        agg_df = agg_df[agg_df[topic_column].isin(top_topics)]
    
    # Sort topics if requested
    if sort_topics:
        topic_order = agg_df.groupby(topic_column)[value_column].sum().sort_values(ascending=False).index.tolist()
    else:
        topic_order = sorted(agg_df[topic_column].unique())
    
    # Pivot the data for ridge plot
    pivot_df = agg_df.pivot(index=topic_column, columns=time_column, values=value_column).fillna(0)
    
    # Apply normalization if requested
    if normalize:
        pivot_df = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 100
    
    # Reorder rows based on topic_order
    pivot_df = pivot_df.reindex(topic_order)
    
    # Create figure
    fig = go.Figure()
    
    # Spacing between traces
    spacing = 0.05 * pivot_df.max().max()
    
    # Add area for each topic
    for i, topic in enumerate(pivot_df.index):
        base = i * spacing
        color = color_map.get(topic) if color_map else None
        
        # Add filled area 
        fig.add_trace(
            go.Scatter(
                x=pivot_df.columns,
                y=pivot_df.loc[topic] + base,
                mode='lines',
                fill='tozeroy',
                name=topic,
                line=dict(color=color, width=1),
                fillcolor=f'rgba{(*px.colors.hex_to_rgb(color if color else px.colors.qualitative.Plotly[i % 10]), 0.6)}',
                hovertemplate=(
                    f"<b>{topic}</b><br>" +
                    f"Time: %{{x}}<br>" +
                    f"Value: %{{y:.2f}}<extra></extra>"
                )
            )
        )
        
        # Add a horizontal line for the base of each ridge
        fig.add_trace(
            go.Scatter(
                x=pivot_df.columns,
                y=[base] * len(pivot_df.columns),
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="",
        yaxis_showticklabels=False,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=height,
        width=width,
        plot_bgcolor='white',
    )
    
    # Add annotations for topic names on y-axis
    for i, topic in enumerate(pivot_df.index):
        base = i * spacing
        fig.add_annotation(
            x=0,
            y=base,
            xref="paper",
            yref="y",
            text=topic,
            showarrow=False,
            xanchor="right",
            xshift=-10,
            font=dict(size=10)
        )
    
    return fig

def create_topic_calendar_heatmap(
    df: pd.DataFrame,
    time_column: str,
    topic_column: str,
    topic_to_plot: str,
    value_column: Optional[str] = None,
    year: Optional[int] = None,
    title: Optional[str] = None,
    color_scale: Optional[str] = "Viridis",
    height: int = 250,
    width: int = 900,
) -> go.Figure:
    """
    Create a calendar heatmap showing intensity of a specific topic by day.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    time_column : str
        Name of column containing datetime values
    topic_column : str
        Name of column containing topic labels
    topic_to_plot : str
        The specific topic to visualize in the calendar
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    year : int, optional
        Specific year to plot (if None, uses the most recent year in the data)
    title : str, optional
        Title of the plot (if None, generates based on topic name)
    color_scale : str, default "Viridis"
        Colorscale for the heatmap
    height : int, default 250
        Height of the plot in pixels
    width : int, default 900
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the calendar heatmap
        
    Notes
    -----
    Calendar heatmaps are useful for identifying day-of-week patterns
    and seasonal variations within a specific topic.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure time column is datetime
    if not pd.core.dtypes.common.is_datetime64_dtype(plot_df[time_column]):
        plot_df[time_column] = pd.to_datetime(plot_df[time_column])
    
    # Filter for the specific topic
    plot_df = plot_df[plot_df[topic_column] == topic_to_plot].copy()
    
    # Determine year if not specified
    if year is None:
        year = plot_df[time_column].dt.year.max()
    
    # Filter for the specified year
    plot_df = plot_df[plot_df[time_column].dt.year == year]
    
    # Group by date
    if value_column:
        # Aggregate provided values
        daily_df = plot_df.groupby(plot_df[time_column].dt.date)[value_column].sum().reset_index()
    else:
        # Count occurrences
        daily_df = plot_df.groupby(plot_df[time_column].dt.date).size().reset_index(name='count')
        value_column = 'count'
    
    # Ensure we have all days of the year
    date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
    all_days_df = pd.DataFrame({time_column: date_range})
    all_days_df['date'] = all_days_df[time_column].dt.date
    
    # Merge with our data
    merged_df = pd.merge(all_days_df, daily_df, left_on='date', right_on=time_column, how='left')
    merged_df[value_column] = merged_df[value_column].fillna(0)
    
    # Extract components for the calendar
    merged_df['month'] = merged_df[time_column].dt.month
    merged_df['day'] = merged_df[time_column].dt.day
    merged_df['weekday'] = merged_df[time_column].dt.weekday
    merged_df['week'] = merged_df[time_column].dt.isocalendar().week
    
    # Create a unique week identifier for better visualization
    merged_df['monthweek'] = merged_df['month'] * 100 + merged_df['week']
    
    # Create auto title if none provided
    if title is None:
        title = f"Daily Frequency of '{topic_to_plot}' in {year}"
    
    # Get months for labels
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create calendar layout
    fig = go.Figure(data=go.Heatmap(
        z=merged_df[value_column],
        x=merged_df['day'],
        y=merged_df['monthweek'],
        colorscale=color_scale,
        showscale=True,
        colorbar=dict(title=dict(text=value_column, side='right')),
        hovertemplate=(
            "<b>Date:</b> %{text}<br>" +
            f"<b>{value_column}:</b> %{{z}}<extra></extra>"
        ),
        text=[d.strftime('%Y-%m-%d') for d in merged_df[time_column]]
    ))
    
    # Add month labels on the y-axis
    month_positions = []
    for month_num in range(1, 13):
        month_data = merged_df[merged_df['month'] == month_num]
        if not month_data.empty:
            first_monthweek = month_data['monthweek'].iloc[0]
            month_positions.append((first_monthweek, months[month_num-1]))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        xaxis=dict(
            title="Day of Month",
            dtick=1,
            tickmode='linear',
            tick0=1,
            tickangle=0
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickmode='array',
            tickvals=[pos[0] for pos in month_positions],
            ticktext=[pos[1] for pos in month_positions]
        ),
        plot_bgcolor='white',
    )
    
    return fig
"""
Geospatial Visualization Module for Meno Topic Modeling.

This module provides visualizations for displaying topic distributions and 
patterns across geographic locations, supporting various types of spatial 
data including coordinates, regions, and postcodes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any

def create_topic_map(
    df: pd.DataFrame,
    topic_column: str,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    value_column: Optional[str] = None,
    color_by_topic: bool = True,
    zoom: int = 3,
    center: Optional[Dict[str, float]] = None,
    marker_size_range: Tuple[int, int] = (5, 15),
    opacity: float = 0.7,
    title: str = "Topic Distribution Map",
    height: int = 600,
    width: int = 800,
    mapbox_style: str = "open-street-map",
) -> go.Figure:
    """
    Create an interactive map showing topic distribution across geographic coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    lat_column : str, default "latitude"
        Name of column containing latitude values
    lon_column : str, default "longitude"
        Name of column containing longitude values
    value_column : str, optional
        Name of column containing values for sizing markers (if None, all markers are equal size)
    color_by_topic : bool, default True
        If True, color markers by topic; if False, color by value_column
    zoom : int, default 3
        Initial zoom level for the map
    center : Dict[str, float], optional
        Center coordinates for the map, e.g. {"lat": -33.8688, "lon": 151.2093}
        If None, automatically centers based on data
    marker_size_range : Tuple[int, int], default (5, 15)
        Range of marker sizes (min, max) in pixels
    opacity : float, default 0.7
        Opacity of markers (0.0 to 1.0)
    title : str, default "Topic Distribution Map"
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
        Plotly figure object containing the geospatial visualization
        
    Notes
    -----
    This visualization requires latitude and longitude coordinates for each data point.
    It's useful for understanding geographic patterns in topic distribution.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, lat_column, lon_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Determine marker size
    if value_column:
        if value_column not in plot_df.columns:
            raise ValueError(f"Value column '{value_column}' not found in dataframe")
        # Normalize to marker size range
        min_val = plot_df[value_column].min()
        max_val = plot_df[value_column].max()
        if min_val == max_val:
            plot_df['marker_size'] = marker_size_range[1]
        else:
            norm = (plot_df[value_column] - min_val) / (max_val - min_val)
            size_range = marker_size_range[1] - marker_size_range[0]
            plot_df['marker_size'] = marker_size_range[0] + (norm * size_range)
    else:
        plot_df['marker_size'] = marker_size_range[1]
    
    # Determine marker color
    if color_by_topic:
        # Create figure using express for automatic color assignment
        fig = px.scatter_mapbox(
            plot_df,
            lat=lat_column,
            lon=lon_column,
            color=topic_column,
            size='marker_size',
            size_max=marker_size_range[1],
            opacity=opacity,
            zoom=zoom,
            title=title,
            height=height,
            width=width,
            hover_name=topic_column,
            hover_data=[col for col in [value_column] if col is not None],
        )
    else:
        # Color by value
        if value_column:
            fig = px.scatter_mapbox(
                plot_df,
                lat=lat_column,
                lon=lon_column,
                color=value_column,
                size='marker_size',
                size_max=marker_size_range[1],
                opacity=opacity,
                zoom=zoom,
                title=title,
                height=height,
                width=width,
                hover_name=topic_column,
                color_continuous_scale=px.colors.sequential.Viridis,
            )
        else:
            # No value column, fall back to coloring by topic
            fig = px.scatter_mapbox(
                plot_df,
                lat=lat_column,
                lon=lon_column,
                color=topic_column,
                size='marker_size',
                size_max=marker_size_range[1],
                opacity=opacity,
                zoom=zoom,
                title=title,
                height=height,
                width=width,
                hover_name=topic_column,
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
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    
    return fig

def create_region_choropleth(
    df: pd.DataFrame,
    topic_column: str,
    region_column: str,
    geojson: Any,
    feature_id_property: str,
    value_column: Optional[str] = None,
    aggregation: str = "count",
    color_continuous_scale: str = "Viridis",
    title: str = "Topic Distribution by Region",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Create a choropleth map showing topic distribution across regions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    region_column : str
        Name of column containing region identifiers (must match geojson features)
    geojson : Any
        GeoJSON object containing region boundaries
    feature_id_property : str
        Property in GeoJSON features that matches values in region_column
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    aggregation : str, default "count"
        Aggregation method: "count", "sum", "mean", "median", "max", "min"
    color_continuous_scale : str, default "Viridis"
        Colorscale for the choropleth map
    title : str, default "Topic Distribution by Region"
        Title of the plot
    height : int, default 600
        Height of the plot in pixels
    width : int, default 800
        Width of the plot in pixels
        
    Returns
    -------
    go.Figure
        Plotly figure object containing the choropleth map
        
    Notes
    -----
    This visualization requires a GeoJSON file with region boundaries
    and a column in the dataframe that matches region identifiers in the GeoJSON.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, region_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Aggregate data by region and topic
    if aggregation == "count":
        if value_column:
            agg_df = plot_df.groupby([region_column, topic_column]).count()[value_column].reset_index()
            agg_df.rename(columns={value_column: "count"}, inplace=True)
            value_col = "count"
        else:
            agg_df = plot_df.groupby([region_column, topic_column]).size().reset_index(name="count")
            value_col = "count"
    else:
        if not value_column:
            raise ValueError(f"Value column must be provided for aggregation method '{aggregation}'")
        
        if aggregation == "sum":
            agg_df = plot_df.groupby([region_column, topic_column])[value_column].sum().reset_index()
        elif aggregation == "mean":
            agg_df = plot_df.groupby([region_column, topic_column])[value_column].mean().reset_index()
        elif aggregation == "median":
            agg_df = plot_df.groupby([region_column, topic_column])[value_column].median().reset_index()
        elif aggregation == "max":
            agg_df = plot_df.groupby([region_column, topic_column])[value_column].max().reset_index()
        elif aggregation == "min":
            agg_df = plot_df.groupby([region_column, topic_column])[value_column].min().reset_index()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
        
        value_col = value_column
    
    # Get unique topics
    topics = plot_df[topic_column].unique()
    
    # Create a figure for each topic
    fig = go.Figure()
    
    # Add choropleth trace for each topic
    for i, topic in enumerate(topics):
        topic_df = agg_df[agg_df[topic_column] == topic]
        
        visible = True if i == 0 else "legendonly"
        
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=topic_df[region_column],
                z=topic_df[value_col],
                featureidkey=f"properties.{feature_id_property}",
                colorscale=color_continuous_scale,
                marker_line_width=0.5,
                colorbar=dict(
                    title=dict(text=value_col),
                    len=0.75,
                ),
                name=topic,
                visible=visible,
                hovertemplate=(
                    f"<b>%{{location}}</b><br>" +
                    f"Topic: {topic}<br>" +
                    f"{value_col}: %{{z}}<extra></extra>"
                )
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend_title_text="Topics",
    )
    
    # Update geos
    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )
    
    return fig

def create_topic_density_map(
    df: pd.DataFrame,
    topic_column: str,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    radius: int = 10,
    topic_to_plot: Optional[str] = None,
    zoom: int = 9,
    center: Optional[Dict[str, float]] = None,
    colorscale: str = "Viridis",
    opacity: float = 0.7,
    title: str = "Topic Density Heatmap",
    height: int = 600,
    width: int = 800,
    mapbox_style: str = "open-street-map",
) -> go.Figure:
    """
    Create a density heatmap showing concentrations of topics across geographic coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    lat_column : str, default "latitude"
        Name of column containing latitude values
    lon_column : str, default "longitude"
        Name of column containing longitude values
    radius : int, default 10
        Radius of influence for each point in pixels
    topic_to_plot : str, optional
        Specific topic to visualize (if None, creates a multi-layer map with all topics)
    zoom : int, default 9
        Initial zoom level for the map
    center : Dict[str, float], optional
        Center coordinates for the map, e.g. {"lat": -33.8688, "lon": 151.2093}
        If None, automatically centers based on data
    colorscale : str, default "Viridis"
        Colorscale for the heatmap
    opacity : float, default 0.7
        Opacity of heatmap layer (0.0 to 1.0)
    title : str, default "Topic Density Heatmap"
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
        Plotly figure object containing the density heatmap visualization
        
    Notes
    -----
    This visualization shows areas of high topic concentration, making it
    easy to identify hotspots for specific topics.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, lat_column, lon_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Create figure
    fig = go.Figure()
    
    # Process for single topic or all topics
    if topic_to_plot:
        # Filter for the specific topic
        topic_df = plot_df[plot_df[topic_column] == topic_to_plot]
        
        # Add a heatmap layer for the topic
        fig.add_trace(
            go.Densitymapbox(
                lat=topic_df[lat_column],
                lon=topic_df[lon_column],
                radius=radius,
                colorscale=colorscale,
                opacity=opacity,
                name=topic_to_plot,
                hovertemplate=(
                    f"Topic: {topic_to_plot}<br>" +
                    f"Density: High<extra></extra>"
                )
            )
        )
        
        # Update title to include topic name
        title = f"{title} - {topic_to_plot}"
        
    else:
        # Create a heatmap layer for each topic
        topics = plot_df[topic_column].unique()
        
        for i, topic in enumerate(topics):
            topic_df = plot_df[plot_df[topic_column] == topic]
            
            # Skip if no data for this topic
            if len(topic_df) == 0:
                continue
            
            # Generate a different colorscale for each topic
            if i == 0:
                topic_colorscale = "Viridis"
                visible = True
            elif i == 1:
                topic_colorscale = "Plasma"
                visible = False
            elif i == 2:
                topic_colorscale = "Inferno"
                visible = False
            elif i == 3:
                topic_colorscale = "Magma"
                visible = False
            elif i == 4:
                topic_colorscale = "Cividis"
                visible = False
            else:
                topic_colorscale = "Turbo"
                visible = False
            
            fig.add_trace(
                go.Densitymapbox(
                    lat=topic_df[lat_column],
                    lon=topic_df[lon_column],
                    radius=radius,
                    colorscale=topic_colorscale,
                    opacity=opacity,
                    name=topic,
                    visible=visible,
                    hovertemplate=(
                        f"Topic: {topic}<br>" +
                        f"Density: High<extra></extra>"
                    )
                )
            )
    
    # Create buttons for topic selection (if multiple topics)
    if not topic_to_plot and len(plot_df[topic_column].unique()) > 1:
        topics = plot_df[topic_column].unique()
        buttons = []
        
        # Add a button for each topic
        for i, topic in enumerate(topics):
            # Create visibility list
            visibility = [i == j for j in range(len(topics))]
            
            buttons.append(
                dict(
                    label=topic,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"{title} - {topic}"}
                    ]
                )
            )
        
        # Add dropdown menu
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
    
    # Update map style
    fig.update_layout(
        mapbox_style=mapbox_style,
        title=title,
        height=height,
        width=width,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    
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
        if not plot_df.empty:
            center_lat = plot_df[lat_column].mean()
            center_lon = plot_df[lon_column].mean()
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom
                )
            )
    
    return fig

def create_postcode_map(
    df: pd.DataFrame, 
    topic_column: str,
    postcode_column: str,
    value_column: Optional[str] = None,
    country_code: str = "AU",
    color_by_topic: bool = True,
    title: str = "Topic Distribution by Postcode",
    height: int = 600,
    width: int = 800,
    mapbox_style: str = "open-street-map",
) -> go.Figure:
    """
    Create a map showing topic distribution by postal/zip codes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    topic_column : str
        Name of column containing topic labels
    postcode_column : str
        Name of column containing postal/zip codes
    value_column : str, optional
        Name of column containing values to aggregate (if None, counts occurrences)
    country_code : str, default "AU"
        ISO country code for postcode mapping
    color_by_topic : bool, default True
        If True, color markers by topic; if False, color by value_column
    title : str, default "Topic Distribution by Postcode"
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
        Plotly figure object containing the postcode visualization
        
    Notes
    -----
    This function converts postcodes to approximate geographic coordinates
    based on country-specific lookups.
    """
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Check if required columns exist
    for col in [topic_column, postcode_column]:
        if col not in plot_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # This would typically use a geocoding service or database to convert postcodes to coordinates
    # For this example, we'll create dummy coordinates based on the postcode values
    # In a real application, you would replace this with actual geocoding
    
    # Create dummy coordinates for visualization (this is a placeholder)
    # In a real application, you would use a proper geocoding lookup for postcodes
    if country_code == "AU":
        # Simplified approximation for Australian postcodes
        # This is just for demonstration - real applications should use proper geocoding
        plot_df["postcode_numeric"] = pd.to_numeric(plot_df[postcode_column], errors="coerce")
        
        # Create approximate coordinates from postcode value
        # Australian postcodes: 0xxx = NT, 1xxx = NSW, 2xxx = NSW/ACT, 3xxx = VIC,
        # 4xxx = QLD, 5xxx = SA, 6xxx = WA, 7xxx = TAS
        def get_approx_au_coords(postcode):
            try:
                pc = int(postcode)
                if 800 <= pc <= 899:  # NT (0800-0899)
                    base_lat, base_lon = -12.5, 131.0
                    offset = (pc - 800) / 100
                elif 900 <= pc <= 999:  # SA/NT remote (0900-0999)
                    base_lat, base_lon = -25.0, 133.0
                    offset = (pc - 900) / 100
                elif 1000 <= pc <= 1999:  # NSW (1000-1999)
                    base_lat, base_lon = -33.8, 151.0
                    offset = (pc - 1000) / 1000
                elif 2000 <= pc <= 2999:  # NSW/ACT (2000-2999)
                    base_lat, base_lon = -35.0, 149.0
                    offset = (pc - 2000) / 1000
                elif 3000 <= pc <= 3999:  # VIC (3000-3999)
                    base_lat, base_lon = -37.8, 145.0
                    offset = (pc - 3000) / 1000
                elif 4000 <= pc <= 4999:  # QLD (4000-4999)
                    base_lat, base_lon = -27.5, 153.0
                    offset = (pc - 4000) / 1000
                elif 5000 <= pc <= 5999:  # SA (5000-5999)
                    base_lat, base_lon = -34.9, 138.6
                    offset = (pc - 5000) / 1000
                elif 6000 <= pc <= 6999:  # WA (6000-6999)
                    base_lat, base_lon = -31.9, 115.9
                    offset = (pc - 6000) / 1000
                elif 7000 <= pc <= 7999:  # TAS (7000-7999)
                    base_lat, base_lon = -42.9, 147.3
                    offset = (pc - 7000) / 1000
                else:
                    return None, None
                
                # Add some randomness to prevent exact overlaps
                import random
                random.seed(pc)
                lat_jitter = random.uniform(-0.1, 0.1)
                lon_jitter = random.uniform(-0.1, 0.1)
                
                return base_lat + offset + lat_jitter, base_lon + offset + lon_jitter
            except:
                return None, None
        
        # Apply the function to each postcode
        coords = plot_df[postcode_column].apply(get_approx_au_coords)
        plot_df["latitude"] = coords.apply(lambda x: x[0])
        plot_df["longitude"] = coords.apply(lambda x: x[1])
        
    else:
        # For other countries, generate random coordinates (placeholder)
        # In a real application, use proper geocoding service
        import random
        random.seed(42)  # For reproducibility
        
        # Generate random coordinates within reasonable bounds
        plot_df["latitude"] = [random.uniform(-90, 90) for _ in range(len(plot_df))]
        plot_df["longitude"] = [random.uniform(-180, 180) for _ in range(len(plot_df))]
    
    # Remove rows with missing coordinates
    plot_df = plot_df.dropna(subset=["latitude", "longitude"])
    
    # Aggregate data by postcode and topic
    if value_column:
        agg_df = plot_df.groupby([postcode_column, topic_column, "latitude", "longitude"])[value_column].sum().reset_index()
    else:
        # Count occurrences
        agg_df = plot_df.groupby([postcode_column, topic_column, "latitude", "longitude"]).size().reset_index(name="count")
        value_column = "count"
    
    # Now create the map using the coordinates
    if color_by_topic:
        fig = px.scatter_mapbox(
            agg_df,
            lat="latitude",
            lon="longitude",
            color=topic_column,
            size=value_column if value_column else None,
            hover_name=postcode_column,
            hover_data=[topic_column, value_column],
            title=title,
            height=height,
            width=width,
            opacity=0.7,
        )
    else:
        fig = px.scatter_mapbox(
            agg_df,
            lat="latitude",
            lon="longitude",
            color=value_column,
            size=value_column if value_column else None,
            hover_name=postcode_column,
            hover_data=[topic_column],
            color_continuous_scale=px.colors.sequential.Viridis,
            title=title,
            height=height,
            width=width,
            opacity=0.7,
        )
    
    # Update layout
    fig.update_layout(
        mapbox_style=mapbox_style,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    
    # Auto-center on data
    center_lat = agg_df["latitude"].mean()
    center_lon = agg_df["longitude"].mean()
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4  # Zoomed out to show country level
        )
    )
    
    return fig
"""Tests for the extended visualization capabilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from meno.visualization import (
    # Time series visualizations
    create_topic_trend_plot,
    create_topic_heatmap,
    create_topic_stacked_area,
    create_topic_ridge_plot,
    create_topic_calendar_heatmap,
    # Geospatial visualizations
    create_topic_map,
    create_region_choropleth,
    create_topic_density_map,
    create_postcode_map,
    # Time-space visualizations
    create_animated_map,
    create_space_time_heatmap,
    create_category_time_plot,
)


@pytest.fixture
def sample_time_data():
    """Create a sample DataFrame with time and topic data."""
    # Create dates for the past 30 days
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    
    # Create a DataFrame with random topic assignments
    data = {
        "text": [f"Document {i}" for i in range(100)],
        "topic": np.random.choice(["Topic A", "Topic B", "Topic C"], 100),
        "timestamp": np.random.choice(dates, 100),
        "value": np.random.randint(1, 100, 100),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_geo_data():
    """Create a sample DataFrame with geospatial and topic data."""
    # Create a DataFrame with random topic assignments and coordinates
    # Using Australia as an example region
    data = {
        "text": [f"Document {i}" for i in range(100)],
        "topic": np.random.choice(["Topic A", "Topic B", "Topic C"], 100),
        "latitude": np.random.uniform(-39, -33, 100),  # Approximate latitudes for Australia
        "longitude": np.random.uniform(140, 153, 100),  # Approximate longitudes for Australia
        "value": np.random.randint(1, 100, 100),
        "region": np.random.choice(["NSW", "VIC", "QLD", "SA", "WA"], 100),
        "postcode": np.random.choice(["2000", "3000", "4000", "5000", "6000"], 100),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_geo_data(sample_time_data, sample_geo_data):
    """Create a combined DataFrame with time, geospatial, and topic data."""
    # Combine the time and geospatial data
    geo_data = sample_geo_data.drop("text", axis=1)
    time_data = sample_time_data.drop("value", axis=1)
    
    # Reset indices to ensure correct alignment when merging
    geo_data = geo_data.reset_index(drop=True)
    time_data = time_data.reset_index(drop=True)
    
    # Add a category column
    geo_data["category"] = np.random.choice(["Category X", "Category Y", "Category Z"], len(geo_data))
    
    # Combine the data
    combined_data = pd.concat([time_data, geo_data], axis=1)
    
    return combined_data


def test_topic_trend_plot(sample_time_data):
    """Test creating a line chart of topic trends over time."""
    fig = create_topic_trend_plot(
        df=sample_time_data,
        topic_column="topic",
        time_column="timestamp",
        value_column="value",
        time_interval="D",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)


def test_topic_heatmap(sample_time_data):
    """Test creating a heatmap of topic intensity over time."""
    fig = create_topic_heatmap(
        df=sample_time_data,
        topic_column="topic",
        time_column="timestamp",
        value_column="value",
        time_interval="D",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)


def test_topic_stacked_area(sample_time_data):
    """Test creating a stacked area chart of topic composition over time."""
    fig = create_topic_stacked_area(
        df=sample_time_data,
        topic_column="topic",
        time_column="timestamp",
        value_column="value",
        time_interval="D",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)
    assert all(trace.stackgroup == "one" for trace in fig.data)


def test_topic_ridge_plot(sample_time_data):
    """Test creating a ridge plot of topic distributions over time."""
    fig = create_topic_ridge_plot(
        df=sample_time_data,
        topic_column="topic",
        time_column="timestamp",
        value_column="value",
        time_interval="D",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)


def test_topic_calendar_heatmap(sample_time_data):
    """Test creating a calendar heatmap for a specific topic."""
    # Select a specific topic for the calendar
    topic = sample_time_data["topic"].unique()[0]
    
    fig = create_topic_calendar_heatmap(
        df=sample_time_data,
        topic_column="topic",
        time_column="timestamp",
        topic_to_plot=topic,
        value_column="value",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)


def test_topic_map(sample_geo_data):
    """Test creating a map of topic distribution across coordinates."""
    fig = create_topic_map(
        df=sample_geo_data,
        topic_column="topic",
        lat_column="latitude",
        lon_column="longitude",
        value_column="value",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scattermapbox) for trace in fig.data)


def test_topic_density_map(sample_geo_data):
    """Test creating a density heatmap of topics."""
    fig = create_topic_density_map(
        df=sample_geo_data,
        topic_column="topic",
        lat_column="latitude",
        lon_column="longitude",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Densitymapbox) for trace in fig.data)


def test_postcode_map(sample_geo_data):
    """Test creating a map of topics by postcode."""
    fig = create_postcode_map(
        df=sample_geo_data,
        topic_column="topic",
        postcode_column="postcode",
        value_column="value",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scattermapbox) for trace in fig.data)


def test_animated_map(sample_time_geo_data):
    """Test creating an animated map of topics over time."""
    fig = create_animated_map(
        df=sample_time_geo_data,
        topic_column="topic",
        time_column="timestamp",
        lat_column="latitude",
        lon_column="longitude",
        value_column="value",
        time_interval="D",
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.frames is not None
    assert len(fig.frames) > 0


def test_space_time_heatmap(sample_time_geo_data):
    """Test creating a heatmap of topics across regions over time."""
    fig = create_space_time_heatmap(
        df=sample_time_geo_data,
        topic_column="topic",
        time_column="timestamp",
        region_column="region",
        value_column="value",
        time_interval="D",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Heatmap)


def test_category_time_plot(sample_time_geo_data):
    """Test creating a visualization of topics over time by category."""
    fig = create_category_time_plot(
        df=sample_time_geo_data,
        topic_column="topic",
        time_column="timestamp",
        category_column="category",
        value_column="value",
        time_interval="D",
        plot_type="line",
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)
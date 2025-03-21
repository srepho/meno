"""Tests for BERTopic visualization integration in Meno."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import from conftest
from conftest import BERTOPIC_AVAILABLE

# Import dependencies only if available
if BERTOPIC_AVAILABLE:
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from bertopic.representation import KeyBERTInspired
        from plotly.graph_objects import Figure
    except ImportError as e:
        print(f"Error importing BERTopic visualization dependencies: {e}")
        BERTOPIC_AVAILABLE = False

from meno.modeling.bertopic_model import BERTopicModel
from meno.visualization.bertopic_viz import (
    create_bertopic_topic_similarity,
    create_bertopic_hierarchy,
    create_bertopic_barchart,
    create_bertopic_topic_distribution,
    create_bertopic_over_time
)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        "This is a document about artificial intelligence and machine learning.",
        "Natural language processing is a subfield of artificial intelligence.",
        "Deep learning is a type of machine learning based on neural networks.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning finds patterns without labeled data.",
        "Reinforcement learning involves agents taking actions to maximize rewards.",
        "Transfer learning uses knowledge from one task to help with another.",
        "Computer vision is used for image recognition and object detection.",
        "Recurrent neural networks are good for sequential data like text.",
        "Generative adversarial networks can create realistic synthetic data."
    ]


@pytest.fixture
def sample_timestamps():
    """Create sample timestamps for time-based visualizations."""
    import pandas as pd
    start_date = pd.Timestamp('2023-01-01')
    dates = [start_date + pd.Timedelta(days=i*7) for i in range(10)]
    return dates


@pytest.fixture
def bertopic_model(sample_documents):
    """Create a fitted BERTopic model."""
    if not BERTOPIC_AVAILABLE:
        pytest.skip("BERTopic not installed")
    
    # Create and fit BERTopic model
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    keybert_model = KeyBERTInspired()
    
    model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        vectorizer_model=ctfidf_model,
        representation_model=keybert_model,
        nr_topics=3,
        calculate_probabilities=True
    )
    
    topics, probs = model.fit_transform(sample_documents)
    return model, topics, probs


@pytest.fixture
def meno_bertopic_model(sample_documents):
    """Create a fitted Meno BERTopicModel."""
    if not BERTOPIC_AVAILABLE:
        pytest.skip("BERTopic not installed")
    
    model = BERTopicModel(
        n_topics=3,
        embedding_model=None,  # Will create default
        min_topic_size=2,
        verbose=False
    )
    
    model.fit(sample_documents)
    return model


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_create_bertopic_topic_similarity(bertopic_model):
    """Test creating topic similarity visualization from BERTopic model."""
    model, _, _ = bertopic_model
    
    # Create topic similarity visualization
    fig = create_bertopic_topic_similarity(model)
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_create_bertopic_hierarchy(bertopic_model):
    """Test creating topic hierarchy visualization from BERTopic model."""
    model, _, _ = bertopic_model
    
    # Create topic hierarchy visualization
    fig = create_bertopic_hierarchy(model)
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_create_bertopic_barchart(bertopic_model):
    """Test creating topic barchart visualization from BERTopic model."""
    model, _, _ = bertopic_model
    
    # Create topic barchart visualization
    fig = create_bertopic_barchart(model, top_n_topics=3)
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0
    
    # Test with custom parameters
    fig = create_bertopic_barchart(
        model, 
        top_n_topics=2, 
        n_words=3,
        title="Custom Barchart Title"
    )
    
    assert fig.layout.title.text == "Custom Barchart Title"


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_create_bertopic_topic_distribution(bertopic_model):
    """Test creating topic distribution visualization from BERTopic model."""
    model, topics, _ = bertopic_model
    
    # Create topic distribution visualization
    fig = create_bertopic_topic_distribution(model, topics)
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0
    
    # Test with custom parameters
    fig = create_bertopic_topic_distribution(
        model, 
        topics,
        title="Custom Distribution Title",
        width=800,
        height=500
    )
    
    assert fig.layout.title.text == "Custom Distribution Title"
    assert fig.layout.width == 800
    assert fig.layout.height == 500


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_create_bertopic_over_time(bertopic_model, sample_timestamps):
    """Test creating topics over time visualization from BERTopic model."""
    model, topics, _ = bertopic_model
    
    # Create a DataFrame with timestamps
    df = pd.DataFrame({
        "topic": topics,
        "timestamp": sample_timestamps
    })
    
    # Create topics over time visualization
    fig = create_bertopic_over_time(
        model,
        df,
        timestamp_col="timestamp",
        topic_col="topic"
    )
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0
    
    # Test with custom parameters
    fig = create_bertopic_over_time(
        model,
        df,
        timestamp_col="timestamp",
        topic_col="topic",
        title="Topics Over Time",
        width=900,
        height=600,
        frequency="D"  # Daily frequency
    )
    
    assert fig.layout.title.text == "Topics Over Time"
    assert fig.layout.width == 900
    assert fig.layout.height == 600


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_meno_bertopic_visualize_topics(meno_bertopic_model):
    """Test visualize_topics method of Meno BERTopicModel."""
    model = meno_bertopic_model
    
    # Create visualization
    fig = model.visualize_topics()
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_meno_bertopic_visualize_hierarchy(meno_bertopic_model):
    """Test visualize_hierarchy method of Meno BERTopicModel."""
    model = meno_bertopic_model
    
    # Create visualization
    fig = model.visualize_hierarchy()
    
    # Check that a figure was returned
    assert isinstance(fig, Figure)
    assert fig.layout.title.text is not None
    
    # Check that the figure has data
    assert len(fig.data) > 0


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_bertopic_model_save_load(meno_bertopic_model, sample_documents):
    """Test saving and loading a BERTopicModel."""
    model = meno_bertopic_model
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        save_path = os.path.join(tmpdir, "bertopic_model")
        model.save(save_path)
        
        # Check that files were created
        assert os.path.exists(os.path.join(save_path, "metadata.json"))
        assert os.path.exists(os.path.join(save_path, "bertopic_model"))
        
        # Load model
        loaded_model = BERTopicModel.load(save_path)
        
        # Check that loaded model works
        topics, probs = loaded_model.transform(sample_documents[:3])
        assert len(topics) == 3
        assert probs.shape[0] == 3
        
        # Test visualization with loaded model
        fig = loaded_model.visualize_topics()
        assert isinstance(fig, Figure)


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_find_similar_topics(meno_bertopic_model):
    """Test finding similar topics with BERTopicModel."""
    model = meno_bertopic_model
    
    # Find similar topics
    query = "artificial intelligence and deep learning"
    similar_topics = model.find_similar_topics(query, n_topics=2)
    
    # Check results
    assert len(similar_topics) <= 2  # May be fewer if not enough topics
    
    for topic_id, description, score in similar_topics:
        assert isinstance(topic_id, int)
        assert isinstance(description, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Similarity score should be between 0 and 1
"""Tests for lightweight visualization module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Tuple, Any
from pathlib import Path

# Mock the base class to avoid dependency issues
class BaseTopicModel(ABC):
    """Mock base class for testing."""
    
    def __init__(self, num_topics=10, **kwargs):
        self.num_topics = num_topics
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, documents, embeddings=None, **kwargs):
        pass
        
    @abstractmethod
    def transform(self, documents, embeddings=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass
        
    @abstractmethod
    def get_topic_info(self) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def visualize_topics(self, width=800, height=600, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        pass
        
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseTopicModel":
        pass

# Mock the models
class SimpleTopicModel(MagicMock):
    pass
    
class TFIDFTopicModel(MagicMock):
    pass

from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape,
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)


@pytest.fixture
def sample_documents():
    """Create a sample dataset for testing."""
    return [
        "Customer service was excellent and the product arrived on time.",
        "The package was damaged during shipping but customer service resolved the issue.",
        "Product quality is outstanding and exceeded my expectations.",
        "The software has a steep learning curve but powerful features.",
        "Technical support was helpful in resolving my installation issues.",
        "User interface is intuitive and easy to navigate.",
        "The documentation lacks examples and could be improved.",
        "Performance is excellent even with large datasets.",
        "Pricing is competitive compared to similar products.",
        "Regular updates keep adding valuable new features.",
        "The mobile app lacks some features available in the desktop version.",
        "Setup was straightforward and took only a few minutes.",
        "The product is reliable and hasn't crashed in months of use.",
        "Customer service response time could be improved.",
        "Training materials are comprehensive and well-structured.",
        "The hardware integration works seamlessly with our existing systems.",
        "Battery life is impressive compared to previous models.",
        "The AI features are innovative but sometimes unpredictable.",
        "Security features are robust and meet our compliance requirements.",
        "The community forum is active and helpful for troubleshooting."
    ]


@pytest.fixture
def mock_embedding_model():
    """Create a mock document embedding model."""
    model = MagicMock()
    model.embed_documents.return_value = np.random.rand(20, 384)  # Random embeddings
    return model


@pytest.fixture
def mock_simple_model():
    """Create a mock SimpleTopicModel."""
    model = MagicMock()
    model.is_fitted = True
    
    # Mock get_topic_info
    topic_info = pd.DataFrame({
        'Topic': [0, 1, 2],
        'Size': [7, 8, 5],
        'Count': [7, 8, 5],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2'],
        'Words': [
            ['service', 'customer', 'excellent', 'support', 'response'],
            ['product', 'quality', 'features', 'excellent', 'performance'],
            ['software', 'documentation', 'interface', 'user', 'training']
        ]
    })
    model.get_topic_info.return_value = topic_info
    
    # Mock get_topic
    topics = {
        0: [('service', 0.8), ('customer', 0.7), ('excellent', 0.6)],
        1: [('product', 0.8), ('quality', 0.7), ('features', 0.6)],
        2: [('software', 0.8), ('interface', 0.7), ('user', 0.6)]
    }
    model.get_topic.side_effect = lambda topic_id: topics.get(topic_id, [])
    
    # Mock transform
    model.transform.return_value = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.2, 0.2]
    ])
    
    # Mock get_topics and get_topic_labels
    model.get_topics.return_value = {
        0: ['service', 'customer', 'excellent', 'support', 'response'],
        1: ['product', 'quality', 'features', 'excellent', 'performance'],
        2: ['software', 'documentation', 'interface', 'user', 'training']
    }
    model.get_topic_labels.return_value = {
        0: 'Service: customer, excellent, support',
        1: 'Product: quality, features, excellent',
        2: 'Software: documentation, interface, user'
    }
    
    return model


@pytest.fixture
def mock_tfidf_model():
    """Create a mock TFIDFTopicModel."""
    model = MagicMock()
    model.is_fitted = True
    
    # Mock get_topic_info
    topic_info = pd.DataFrame({
        'Topic': [0, 1, 2],
        'Size': [6, 9, 5],
        'Count': [6, 9, 5],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2'],
        'Words': [
            ['customer', 'service', 'support', 'response', 'time'],
            ['product', 'quality', 'features', 'performance', 'reliable'],
            ['interface', 'user', 'documentation', 'examples', 'navigate']
        ]
    })
    model.get_topic_info.return_value = topic_info
    
    # Mock get_topic
    topics = {
        0: [('customer', 0.9), ('service', 0.8), ('support', 0.7)],
        1: [('product', 0.9), ('quality', 0.8), ('features', 0.7)],
        2: [('interface', 0.9), ('user', 0.8), ('documentation', 0.7)]
    }
    model.get_topic.side_effect = lambda topic_id: topics.get(topic_id, [])
    
    # Mock transform
    model.transform.return_value = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.7, 0.2, 0.1]
    ])
    
    # Mock get_topics and get_topic_labels
    model.get_topics.return_value = {
        0: ['customer', 'service', 'support', 'response', 'time'],
        1: ['product', 'quality', 'features', 'performance', 'reliable'],
        2: ['interface', 'user', 'documentation', 'examples', 'navigate']
    }
    model.get_topic_labels.return_value = {
        0: 'Customer: service, support, response',
        1: 'Product: quality, features, performance',
        2: 'Interface: user, documentation, examples'
    }
    
    return model


class TestLightweightVisualization:
    """Test the lightweight visualization functions."""
    
    def test_plot_model_comparison(self, sample_documents, mock_simple_model, mock_tfidf_model):
        """Test the model comparison plot function."""
        # Set up inputs
        document_lists = [sample_documents, sample_documents]
        model_names = ["Simple", "TF-IDF"]
        models = [mock_simple_model, mock_tfidf_model]
        
        # Call plot function
        fig = plot_model_comparison(document_lists, model_names, models)
        
        # Basic validation
        assert fig is not None
        assert len(fig.data) == 4  # 2 models * 2 plots per model
        
    def test_plot_topic_landscape(self, sample_documents, mock_simple_model):
        """Test the topic landscape plot function."""
        # Call plot function
        fig = plot_topic_landscape(mock_simple_model, sample_documents)
        
        # Basic validation
        assert fig is not None
        
        # Try with different method
        fig_direct = plot_topic_landscape(mock_simple_model, sample_documents, method="direct")
        assert fig_direct is not None
        
        # Test with invalid method
        with pytest.raises(ValueError):
            plot_topic_landscape(mock_simple_model, sample_documents, method="invalid")
            
    def test_plot_multi_topic_heatmap_two_models(self, sample_documents, mock_simple_model, mock_tfidf_model):
        """Test the multi-topic heatmap with two models."""
        # Set up inputs
        document_lists = [sample_documents, sample_documents]
        model_names = ["Simple", "TF-IDF"]
        models = [mock_simple_model, mock_tfidf_model]
        
        # Call plot function
        fig = plot_multi_topic_heatmap(models, model_names, document_lists)
        
        # Basic validation
        assert fig is not None
        assert len(fig.data) == 1  # One heatmap
        
    def test_plot_multi_topic_heatmap_three_models(self, sample_documents, mock_simple_model, mock_tfidf_model):
        """Test the multi-topic heatmap with three models."""
        # Create third model
        mock_third_model = MagicMock()
        mock_third_model.is_fitted = True
        mock_third_model.get_topic_info.return_value = mock_simple_model.get_topic_info.return_value
        mock_third_model.get_topic.side_effect = mock_simple_model.get_topic.side_effect
        
        # Set up inputs
        document_lists = [sample_documents, sample_documents, sample_documents]
        model_names = ["Simple", "TF-IDF", "Third"]
        models = [mock_simple_model, mock_tfidf_model, mock_third_model]
        
        # Call plot function
        fig = plot_multi_topic_heatmap(models, model_names, document_lists)
        
        # Basic validation
        assert fig is not None
        assert len(fig.data) == 3  # Three heatmaps (one for each pair)
        
    def test_plot_comparative_document_analysis(self, sample_documents, mock_simple_model):
        """Test the comparative document analysis plot."""
        # Call plot function with default parameters
        fig = plot_comparative_document_analysis(mock_simple_model, sample_documents[:5])
        
        # Basic validation
        assert fig is not None
        assert len(fig.data) == 1  # One heatmap
        
        # Test with document labels and highlighting
        doc_labels = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]
        highlight_docs = [1, 3]
        
        fig = plot_comparative_document_analysis(
            mock_simple_model, 
            sample_documents[:5],
            document_labels=doc_labels,
            highlight_docs=highlight_docs
        )
        
        # Basic validation
        assert fig is not None
        assert len(fig.data) == 1  # One heatmap
        assert len(fig.layout.shapes) == 2  # Two highlighted docs
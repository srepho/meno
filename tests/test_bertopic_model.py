"""Tests for the BERTopic model module."""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from meno.modeling.bertopic_model import BERTopicModel

# Skip tests if BERTopic is not available
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not installed")


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
def mock_bertopic():
    """Create a mock BERTopic instance."""
    model = MagicMock()
    
    # Mock get_topic_info
    topic_info = pd.DataFrame({
        'Topic': [-1, 0, 1, 2],
        'Count': [2, 6, 7, 5],
        'Name': ['Outlier', 'Topic 0', 'Topic 1', 'Topic 2']
    })
    model.get_topic_info.return_value = topic_info
    
    # Mock get_topics
    topics = {
        -1: [],
        0: [('service', 0.8), ('customer', 0.7), ('support', 0.6), ('response', 0.5), ('excellent', 0.4)],
        1: [('product', 0.8), ('quality', 0.7), ('features', 0.6), ('performance', 0.5), ('reliable', 0.4)],
        2: [('software', 0.8), ('interface', 0.7), ('user', 0.6), ('documentation', 0.5), ('training', 0.4)]
    }
    model.get_topics.return_value = topics
    
    # Mock get_topic
    model.get_topic.side_effect = lambda topic_id: topics.get(topic_id, [])
    
    # Mock transform
    model.transform.return_value = (np.array([0, 0, 1, 2, 0, 2, 2, 1, 1, 1, 1, 0, 1, 0, 2, 0, 1, 2, 2, 0]), 
                                    np.random.rand(20, 4))  # Random probabilities
    
    # Mock fit_transform
    model.fit_transform.return_value = model.transform.return_value
    
    # Mock visualize_topics
    model.visualize_topics.return_value = MagicMock()
    model.visualize_hierarchy.return_value = MagicMock()
    
    return model


class TestBERTopicModel:
    """Test the BERTopicModel class."""
    
    def test_init(self):
        """Test initializing the model."""
        model = BERTopicModel(
            n_topics=10,
            min_topic_size=5,
            use_gpu=False,
            n_neighbors=15,
            n_components=5,
            verbose=False
        )
        
        assert model.n_topics == 10
        assert model.min_topic_size == 5
        assert model.use_gpu is False
        assert model.n_neighbors == 15
        assert model.n_components == 5
        assert model.verbose is False
        assert model.is_fitted is False
        assert hasattr(model, 'model')
        assert hasattr(model, 'embedding_model')
    
    @patch('bertopic.BERTopic')
    def test_fit(self, mock_bertopic_class, sample_documents, mock_bertopic):
        """Test fitting the model."""
        # Setup mock
        mock_bertopic_class.return_value = mock_bertopic
        
        # Create model and fit
        model = BERTopicModel(n_topics=3, verbose=False)
        model.fit(sample_documents)
        
        # Verify
        assert mock_bertopic.fit_transform.called
        assert model.is_fitted is True
        assert len(model.topics) > 0
        assert len(model.topic_sizes) > 0
        
        # Check if topic embeddings were computed
        assert hasattr(model, 'topic_embeddings')
    
    @patch('bertopic.BERTopic')
    def test_transform(self, mock_bertopic_class, sample_documents, mock_bertopic):
        """Test transforming documents."""
        # Setup mock
        mock_bertopic_class.return_value = mock_bertopic
        
        # Create model
        model = BERTopicModel(verbose=False)
        model.is_fitted = True  # Simulate fitted model
        
        # Transform
        topics, probs = model.transform(sample_documents)
        
        # Verify
        assert mock_bertopic.transform.called
        assert topics.shape == (20,)
        assert probs.shape[0] == 20
    
    @patch('bertopic.BERTopic')
    def test_transform_not_fitted(self, mock_bertopic_class, sample_documents):
        """Test transforming with an unfitted model."""
        # Create model
        model = BERTopicModel(verbose=False)
        
        # Attempt to transform without fitting
        with pytest.raises(ValueError, match="Model must be fitted before transform can be called"):
            model.transform(sample_documents)
    
    @patch('bertopic.BERTopic')
    def test_compute_topic_embeddings(self, mock_bertopic_class, mock_bertopic):
        """Test computing topic embeddings."""
        # Setup mocks
        mock_bertopic_class.return_value = mock_bertopic
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = np.random.rand(3, 384)
        
        # Create model
        model = BERTopicModel(verbose=False)
        model.embedding_model = mock_embedding_model
        model.model = mock_bertopic
        
        # Compute embeddings
        model._compute_topic_embeddings()
        
        # Verify
        assert mock_embedding_model.embed_documents.called
        assert hasattr(model, 'topic_embeddings')
        assert hasattr(model, 'topic_id_mapping')
    
    @patch('bertopic.BERTopic')
    def test_find_similar_topics(self, mock_bertopic_class, mock_bertopic):
        """Test finding similar topics."""
        # Setup mocks
        mock_bertopic_class.return_value = mock_bertopic
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = np.random.rand(1, 384)
        
        # Create model
        model = BERTopicModel(verbose=False)
        model.embedding_model = mock_embedding_model
        model.model = mock_bertopic
        model.is_fitted = True
        model.topic_embeddings = np.random.rand(3, 384)
        model.topic_id_mapping = {0: 0, 1: 1, 2: 2}
        model.topics = {0: "Topic 0", 1: "Topic 1", 2: "Topic 2"}
        
        # Find similar topics
        similar_topics = model.find_similar_topics("customer service", n_topics=2)
        
        # Verify
        assert mock_embedding_model.embed_documents.called
        assert len(similar_topics) == 2
        assert similar_topics[0][0] in [0, 1, 2]  # Topic ID
        assert isinstance(similar_topics[0][2], float)  # Similarity score
    
    @patch('bertopic.BERTopic')
    def test_find_similar_topics_not_fitted(self, mock_bertopic_class):
        """Test finding similar topics with an unfitted model."""
        # Create model
        model = BERTopicModel(verbose=False)
        
        # Attempt to find similar topics without fitting
        with pytest.raises(ValueError, match="Model must be fitted before finding similar topics"):
            model.find_similar_topics("customer service")
    
    @patch('bertopic.BERTopic')
    def test_visualize_topics(self, mock_bertopic_class, mock_bertopic):
        """Test visualizing topics."""
        # Setup mock
        mock_bertopic_class.return_value = mock_bertopic
        
        # Create model
        model = BERTopicModel(verbose=False)
        model.model = mock_bertopic
        model.is_fitted = True
        
        # Visualize topics
        fig = model.visualize_topics()
        
        # Verify
        assert mock_bertopic.visualize_topics.called
        assert fig is not None
    
    @patch('bertopic.BERTopic')
    def test_visualize_hierarchy(self, mock_bertopic_class, mock_bertopic):
        """Test visualizing topic hierarchy."""
        # Setup mock
        mock_bertopic_class.return_value = mock_bertopic
        
        # Create model
        model = BERTopicModel(verbose=False)
        model.model = mock_bertopic
        model.is_fitted = True
        
        # Visualize hierarchy
        fig = model.visualize_hierarchy()
        
        # Verify
        assert mock_bertopic.visualize_hierarchy.called
        assert fig is not None
    
    @patch('bertopic.BERTopic')
    def test_visualize_not_fitted(self, mock_bertopic_class):
        """Test visualizing with an unfitted model."""
        # Create model
        model = BERTopicModel(verbose=False)
        
        # Attempt to visualize without fitting
        with pytest.raises(ValueError, match="Model must be fitted before visualization"):
            model.visualize_topics()
    
    @patch('bertopic.BERTopic')
    def test_save_and_load(self, mock_bertopic_class, mock_bertopic, sample_documents):
        """Test saving and loading the model."""
        # Setup mocks
        mock_bertopic_class.return_value = mock_bertopic
        mock_bertopic.load.return_value = mock_bertopic
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and fit model
            model = BERTopicModel(n_topics=3, verbose=False)
            model.fit(sample_documents)
            
            # Save model
            model.save(tmpdir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(tmpdir, "bertopic_model"))
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            
            # Load model
            loaded_model = BERTopicModel.load(tmpdir)
            
            # Verify
            assert mock_bertopic.load.called
            assert loaded_model.n_topics == model.n_topics
            assert loaded_model.min_topic_size == model.min_topic_size
            assert loaded_model.n_neighbors == model.n_neighbors
            assert loaded_model.n_components == model.n_components
            assert loaded_model.is_fitted is True
            assert loaded_model.topics == model.topics
            assert loaded_model.topic_sizes == model.topic_sizes
    
    def test_pandas_series_handling(self, sample_documents):
        """Test handling pandas Series as input."""
        with patch('bertopic.BERTopic') as mock_bertopic_class:
            # Setup mock
            mock_model = MagicMock()
            mock_model.fit_transform.return_value = (np.zeros(20), np.zeros((20, 2)))
            mock_bertopic_class.return_value = mock_model
            
            # Convert sample to pandas Series
            series = pd.Series(sample_documents)
            
            # Create model
            model = BERTopicModel(verbose=False)
            
            # Fit with Series
            model.fit(series)
            
            # Verify that Series was converted to list
            args, kwargs = mock_model.fit_transform.call_args
            assert isinstance(args[0], list)
            
            # Reset mock
            mock_model.reset_mock()
            mock_model.transform.return_value = (np.zeros(20), np.zeros((20, 2)))
            
            # Transform with Series
            model.is_fitted = True
            model.transform(series)
            
            # Verify that Series was converted to list
            args, kwargs = mock_model.transform.call_args
            assert isinstance(args[0], list)
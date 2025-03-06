"""Tests for the unified topic modeling API."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from meno.modeling.unified_topic_modeling import UnifiedTopicModeler, create_topic_modeler

# Skip tests if BERTopic is not available
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Skip tests if Top2Vec is not available
try:
    from top2vec import Top2Vec
    TOP2VEC_AVAILABLE = True
except ImportError:
    TOP2VEC_AVAILABLE = False


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
def mock_bertopic_model():
    """Create a mock BERTopic model."""
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
    
    return model


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not installed")
class TestUnifiedTopicModelerBertopic:
    """Test the UnifiedTopicModeler class with BERTopic backend."""
    
    @patch('meno.modeling.bertopic_model.BERTopicModel')
    def test_init_bertopic(self, mock_bertopic_model_class):
        """Test initializing with BERTopic."""
        mock_bertopic_model_class.return_value = MagicMock()
        
        modeler = UnifiedTopicModeler(
            method="bertopic",
            n_topics=10,
            min_topic_size=5,
            verbose=False
        )
        
        assert modeler.method == "bertopic"
        assert modeler.n_topics == 10
        assert modeler.min_topic_size == 5
        assert modeler.verbose is False
        assert mock_bertopic_model_class.called
    
    @patch('meno.modeling.bertopic_model.BERTopicModel')
    def test_fit_bertopic(self, mock_bertopic_model_class, sample_documents):
        """Test fitting with BERTopic."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.topics = {0: "Topic 0", 1: "Topic 1"}
        mock_model.topic_sizes = {0: 10, 1: 10}
        mock_model.is_fitted = True
        mock_bertopic_model_class.return_value = mock_model
        
        # Create modeler and fit
        modeler = UnifiedTopicModeler(method="bertopic", verbose=False)
        modeler.fit(sample_documents)
        
        # Verify
        mock_model.fit.assert_called_once()
        assert modeler.topics == {0: "Topic 0", 1: "Topic 1"}
        assert modeler.topic_sizes == {0: 10, 1: 10}
        assert modeler.is_fitted is True
    
    @patch('meno.modeling.bertopic_model.BERTopicModel')
    def test_transform_bertopic(self, mock_bertopic_model_class, sample_documents):
        """Test transforming with BERTopic."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.transform.return_value = (np.array([0, 1]), np.array([[0.8, 0.2], [0.3, 0.7]]))
        mock_model.is_fitted = True
        mock_bertopic_model_class.return_value = mock_model
        
        # Create modeler
        modeler = UnifiedTopicModeler(method="bertopic", verbose=False)
        modeler.is_fitted = True
        
        # Transform
        topics, probs = modeler.transform(sample_documents[:2])
        
        # Verify
        mock_model.transform.assert_called_once()
        assert topics.shape == (2,)
        assert probs.shape == (2, 2)
    
    @patch('meno.modeling.bertopic_model.BERTopicModel')
    def test_find_similar_topics_bertopic(self, mock_bertopic_model_class):
        """Test finding similar topics with BERTopic."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.find_similar_topics.return_value = [
            (0, "Topic 0: service, customer", 0.8),
            (1, "Topic 1: product, quality", 0.6)
        ]
        mock_model.is_fitted = True
        mock_bertopic_model_class.return_value = mock_model
        
        # Create modeler
        modeler = UnifiedTopicModeler(method="bertopic", verbose=False)
        modeler.is_fitted = True
        
        # Find similar topics
        similar_topics = modeler.find_similar_topics("customer service", n_topics=2)
        
        # Verify
        mock_model.find_similar_topics.assert_called_once_with("customer service", 2)
        assert len(similar_topics) == 2
        assert similar_topics[0][0] == 0
        assert similar_topics[0][2] == 0.8
    
    @patch('meno.modeling.bertopic_model.BERTopicModel')
    def test_get_topic_words_bertopic(self, mock_bertopic_model_class, mock_bertopic_model):
        """Test getting topic words with BERTopic."""
        # Setup mock
        mock_bertopic_model_class.return_value = MagicMock()
        type(mock_bertopic_model_class.return_value).model = PropertyMock(return_value=mock_bertopic_model)
        mock_bertopic_model_class.return_value.is_fitted = True
        
        # Create modeler
        modeler = UnifiedTopicModeler(method="bertopic", verbose=False)
        modeler.is_fitted = True
        
        # Get topic words
        words = modeler.get_topic_words(0, n_words=3)
        
        # Verify
        mock_bertopic_model.get_topic.assert_called_with(0)
        assert len(words) == 3
        assert words[0][0] == 'service'
        assert words[1][0] == 'customer'
        assert words[2][0] == 'support'


@pytest.mark.skipif(not TOP2VEC_AVAILABLE, reason="Top2Vec not installed")
class TestUnifiedTopicModelerTop2Vec:
    """Test the UnifiedTopicModeler class with Top2Vec backend."""
    
    @patch('meno.modeling.top2vec_model.Top2VecModel')
    def test_init_top2vec(self, mock_top2vec_model_class):
        """Test initializing with Top2Vec."""
        mock_top2vec_model_class.return_value = MagicMock()
        
        modeler = UnifiedTopicModeler(
            method="top2vec",
            n_topics=10,
            min_topic_size=5,
            verbose=False
        )
        
        assert modeler.method == "top2vec"
        assert modeler.n_topics == 10
        assert modeler.min_topic_size == 5
        assert modeler.verbose is False
        assert mock_top2vec_model_class.called
    
    @patch('meno.modeling.top2vec_model.Top2VecModel')
    def test_fit_top2vec(self, mock_top2vec_model_class, sample_documents):
        """Test fitting with Top2Vec."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.topics = {0: "Topic 0", 1: "Topic 1"}
        mock_model.topic_sizes = {0: 10, 1: 10}
        mock_model.is_fitted = True
        mock_top2vec_model_class.return_value = mock_model
        
        # Create modeler and fit
        modeler = UnifiedTopicModeler(method="top2vec", verbose=False)
        modeler.fit(sample_documents)
        
        # Verify
        mock_model.fit.assert_called_once()
        assert modeler.topics == {0: "Topic 0", 1: "Topic 1"}
        assert modeler.topic_sizes == {0: 10, 1: 10}
        assert modeler.is_fitted is True
    
    @patch('meno.modeling.top2vec_model.Top2VecModel')
    def test_transform_top2vec(self, mock_top2vec_model_class, sample_documents):
        """Test transforming with Top2Vec."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.transform.return_value = (np.array([[0], [1]]), np.array([[0.8], [0.7]]))
        mock_model.is_fitted = True
        mock_top2vec_model_class.return_value = mock_model
        
        # Create modeler
        modeler = UnifiedTopicModeler(method="top2vec", verbose=False)
        modeler.is_fitted = True
        
        # Transform
        topics, probs = modeler.transform(sample_documents[:2], top_n=1)
        
        # Verify
        mock_model.transform.assert_called_once_with(sample_documents[:2], None, top_n=1)
        assert topics.shape == (2, 1)
        assert probs.shape == (2, 1)
    
    @patch('meno.modeling.top2vec_model.Top2VecModel')
    def test_find_similar_topics_top2vec(self, mock_top2vec_model_class):
        """Test finding similar topics with Top2Vec."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.search_topics.return_value = [
            (0, "Topic 0: service, customer", 0.8),
            (1, "Topic 1: product, quality", 0.6)
        ]
        mock_model.is_fitted = True
        mock_top2vec_model_class.return_value = mock_model
        
        # Create modeler
        modeler = UnifiedTopicModeler(method="top2vec", verbose=False)
        modeler.is_fitted = True
        
        # Find similar topics
        similar_topics = modeler.find_similar_topics("customer service", n_topics=2)
        
        # Verify
        mock_model.search_topics.assert_called_once_with("customer service", 2)
        assert len(similar_topics) == 2
        assert similar_topics[0][0] == 0
        assert similar_topics[0][2] == 0.8


def test_create_topic_modeler():
    """Test the create_topic_modeler function."""
    with patch('meno.modeling.unified_topic_modeling.UnifiedTopicModeler') as mock_modeler_class:
        mock_modeler_class.return_value = MagicMock()
        
        modeler = create_topic_modeler(
            method="bertopic",
            n_topics=10,
            min_topic_size=5,
            verbose=False
        )
        
        mock_modeler_class.assert_called_once_with(
            method="bertopic",
            n_topics=10,
            embedding_model=None,
            min_topic_size=5,
            use_gpu=False,
            advanced_config=None,
            optimizer_config=None,
            verbose=False
        )
        
        assert modeler is not None
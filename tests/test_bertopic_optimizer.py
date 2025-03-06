"""Tests for the BERTopic optimizer module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from meno.modeling.bertopic_optimizer import BERTopicOptimizer, optimize_bertopic

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


class TestBERTopicOptimizer:
    """Test the BERTopicOptimizer class."""
    
    def test_init(self):
        """Test initializing the optimizer."""
        optimizer = BERTopicOptimizer(
            embedding_model="all-MiniLM-L6-v2",
            n_trials=5,
            random_state=42,
            metric="combined",
            verbose=False
        )
        
        assert optimizer.embedding_model == "all-MiniLM-L6-v2"
        assert optimizer.n_trials == 5
        assert optimizer.random_state == 42
        assert optimizer.metric == "combined"
        assert optimizer.verbose is False
        assert optimizer.best_params is None
        assert optimizer.best_model is None
        assert optimizer.best_score == float('-inf')
    
    def test_set_param_grid(self):
        """Test setting a custom parameter grid."""
        optimizer = BERTopicOptimizer(n_trials=5)
        
        custom_grid = {
            "n_neighbors": [5, 15],
            "n_components": [5, 10],
            "min_cluster_size": [5, 10]
        }
        
        optimizer.set_param_grid(custom_grid)
        assert optimizer.param_grid == custom_grid
    
    def test_is_better_score_combined(self):
        """Test the _is_better_score method with combined metric."""
        optimizer = BERTopicOptimizer(metric="combined")
        optimizer.best_score = 10.0
        
        assert optimizer._is_better_score(15.0) is True
        assert optimizer._is_better_score(5.0) is False
    
    def test_is_better_score_outlier_percentage(self):
        """Test the _is_better_score method with outlier_percentage metric."""
        optimizer = BERTopicOptimizer(metric="outlier_percentage")
        optimizer.best_score = 10.0
        
        # For outlier_percentage, lower is better
        assert optimizer._is_better_score(5.0) is True
        assert optimizer._is_better_score(15.0) is False
    
    def test_generate_random_combinations(self):
        """Test generating random parameter combinations."""
        optimizer = BERTopicOptimizer(n_trials=3, random_state=42)
        
        # Simplify the param grid for testing
        optimizer.param_grid = {
            "n_neighbors": [5, 15],
            "n_components": [5, 10],
            "min_cluster_size": [5, 10]
        }
        
        combinations = optimizer._generate_random_combinations()
        
        assert len(combinations) == 3
        for combo in combinations:
            assert set(combo.keys()) == {"n_neighbors", "n_components", "min_cluster_size"}
            assert combo["n_neighbors"] in [5, 15]
            assert combo["n_components"] in [5, 10]
            assert combo["min_cluster_size"] in [5, 10]
    
    @patch('bertopic.BERTopic')
    def test_create_model(self, mock_bertopic_class):
        """Test creating a BERTopic model with parameters."""
        mock_bertopic_class.return_value = MagicMock()
        
        optimizer = BERTopicOptimizer(embedding_model="all-MiniLM-L6-v2")
        
        params = {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.1,
            "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "eom",
            "reduce_frequent_words": True,
            "bm25_weighting": True,
            "representation_model": "Both",
            "diversity": 0.3,
            "nr_topics": "auto"
        }
        
        model = optimizer._create_model(params)
        
        assert mock_bertopic_class.called
        assert model is not None
    
    def test_evaluate_model(self, mock_bertopic_model, sample_documents):
        """Test evaluating a BERTopic model."""
        optimizer = BERTopicOptimizer(metric="combined")
        
        score = optimizer._evaluate_model(mock_bertopic_model, sample_documents)
        
        assert isinstance(score, float)
        assert score > 0
    
    def test_evaluate_model_metrics(self, mock_bertopic_model, sample_documents):
        """Test evaluating a BERTopic model with different metrics."""
        # Test n_topics metric
        optimizer = BERTopicOptimizer(metric="n_topics")
        score = optimizer._evaluate_model(mock_bertopic_model, sample_documents)
        assert score == 3.0  # 3 topics excluding outliers
        
        # Test outlier_percentage metric
        optimizer = BERTopicOptimizer(metric="outlier_percentage")
        score = optimizer._evaluate_model(mock_bertopic_model, sample_documents)
        assert score == 10.0  # 2/20 = 10%
        
        # Test diversity metric
        optimizer = BERTopicOptimizer(metric="diversity")
        score = optimizer._evaluate_model(mock_bertopic_model, sample_documents)
        assert 0.0 <= score <= 1.0  # Diversity score is between 0 and 1
    
    @patch('meno.modeling.bertopic_optimizer.BERTopicOptimizer._create_model')
    @patch('meno.modeling.bertopic_optimizer.BERTopicOptimizer._evaluate_model')
    def test_optimize(self, mock_evaluate, mock_create_model, mock_bertopic_model, sample_documents):
        """Test the optimize method."""
        # Setup mocks
        mock_create_model.return_value = mock_bertopic_model
        mock_evaluate.return_value = 20.0
        
        optimizer = BERTopicOptimizer(n_trials=2, verbose=False)
        
        # Simplify the param grid for testing
        optimizer.param_grid = {
            "n_neighbors": [15],
            "n_components": [5],
            "min_dist": [0.1],
            "min_cluster_size": [10],
            "min_samples": [5],
            "cluster_selection_method": ["eom"],
            "reduce_frequent_words": [True],
            "bm25_weighting": [True],
            "representation_model": ["Both"],
            "diversity": [0.3],
            "nr_topics": ["auto"]
        }
        
        best_params, best_model, best_score = optimizer.optimize(
            documents=sample_documents,
            search_method="random"
        )
        
        assert mock_create_model.call_count == 2
        assert mock_evaluate.call_count == 2
        assert best_params is not None
        assert best_model is mock_bertopic_model
        assert best_score == 20.0


@patch('meno.modeling.bertopic_optimizer.BERTopicOptimizer')
def test_optimize_bertopic_function(mock_optimizer_class, sample_documents):
    """Test the optimize_bertopic function."""
    # Setup mock
    mock_optimizer = MagicMock()
    mock_optimizer.optimize.return_value = ({"n_neighbors": 15}, MagicMock(), 20.0)
    mock_optimizer_class.return_value = mock_optimizer
    
    best_params, best_model, best_score = optimize_bertopic(
        documents=sample_documents,
        embedding_model="all-MiniLM-L6-v2",
        n_trials=5,
        search_method="random",
        metric="combined",
        random_state=42,
        verbose=False
    )
    
    mock_optimizer_class.assert_called_once_with(
        embedding_model="all-MiniLM-L6-v2",
        n_trials=5,
        random_state=42,
        metric="combined",
        verbose=False
    )
    
    mock_optimizer.optimize.assert_called_once_with(
        documents=sample_documents,
        search_method="random"
    )
    
    assert best_params == {"n_neighbors": 15}
    assert best_score == 20.0
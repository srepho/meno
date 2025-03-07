"""Test API standardization for Meno topic modeling."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from typing import List, Tuple, Any

from meno.modeling.base import BaseTopicModel
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler

class TestAPIStandardization(unittest.TestCase):
    """Test the API standardization updates."""

    def test_unified_topic_modeler_inheritance(self):
        """Test that UnifiedTopicModeler inherits from BaseTopicModel."""
        self.assertTrue(issubclass(UnifiedTopicModeler, BaseTopicModel),
                        "UnifiedTopicModeler should inherit from BaseTopicModel")

    def test_unified_topic_modeler_parameters(self):
        """Test that UnifiedTopicModeler uses standardized parameter names."""
        # Create modeler with standardized parameter names
        modeler = UnifiedTopicModeler(
            method="embedding_cluster",
            num_topics=15,
            config_overrides={"min_topic_size": 5},
            embedding_model="test-model"
        )
        
        # Check that parameters were set correctly
        self.assertEqual(modeler.method, "embedding_cluster")
        self.assertEqual(modeler.num_topics, 15)
        self.assertEqual(modeler.config_overrides.get("min_topic_size"), 5)
        self.assertEqual(modeler.embedding_model, "test-model")

    @patch('meno.modeling.unified_topic_modeling.BERTopicModel')
    def test_parameter_conversion(self, mock_bertopic):
        """Test parameter conversion between num_topics and n_topics."""
        # Set up mock
        mock_bertopic_instance = MagicMock()
        mock_bertopic.return_value = mock_bertopic_instance
        
        # Create modeler
        modeler = UnifiedTopicModeler(method="bertopic", num_topics=10)
        
        # Check that n_topics was passed to BERTopicModel
        mock_bertopic.assert_called_once()
        kwargs = mock_bertopic.call_args[1]
        self.assertEqual(kwargs["n_topics"], 10)
        
        # Test fit with num_topics parameter
        mock_documents = ["doc1", "doc2"]
        modeler.fit(mock_documents, num_topics=20)
        
        # Check that n_topics was passed to the underlying model's fit method
        mock_bertopic_instance.fit.assert_called_once()
        kwargs = mock_bertopic_instance.fit.call_args[1]
        self.assertEqual(kwargs["n_topics"], 20)

    @patch('meno.modeling.unified_topic_modeling.BERTopicModel')
    def test_transform_return_type(self, mock_bertopic):
        """Test standardized return type for transform method."""
        # Set up mock
        mock_bertopic_instance = MagicMock()
        mock_topic_assignments = np.array([0, 1])
        mock_probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_bertopic_instance.transform.return_value = (mock_topic_assignments, mock_probabilities)
        mock_bertopic.return_value = mock_bertopic_instance
        
        # Create and "fit" modeler
        modeler = UnifiedTopicModeler(method="bertopic")
        modeler.is_fitted = True
        
        # Test transform
        documents = ["doc1", "doc2"]
        result = modeler.transform(documents)
        
        # Check return type and values
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        topics, probs = result
        self.assertIsInstance(topics, np.ndarray)
        self.assertIsInstance(probs, np.ndarray)
        
        np.testing.assert_array_equal(topics, mock_topic_assignments)
        np.testing.assert_array_equal(probs, mock_probabilities)
        
if __name__ == "__main__":
    unittest.main()

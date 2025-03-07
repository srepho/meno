#\!/usr/bin/env python
"""
Direct unit tests for the base and unified_topic_modeling modules.

This script can be run directly without importing the full Meno package.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow direct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports from the modules we want to test
from modeling.base import BaseTopicModel
from modeling.unified_topic_modeling import UnifiedTopicModeler

class TestAPIStandardization(unittest.TestCase):
    """Direct tests for API standardization."""

    def test_unified_topic_modeler_inheritance(self):
        """Test that UnifiedTopicModeler inherits from BaseTopicModel."""
        self.assertTrue(issubclass(UnifiedTopicModeler, BaseTopicModel),
                       "UnifiedTopicModeler should inherit from BaseTopicModel")

    @patch('modeling.unified_topic_modeling.BERTopicModel')
    def test_model_creation(self, mock_bertopic):
        """Test model creation with standardized parameters."""
        # Setup mock
        mock_bertopic_instance = MagicMock()
        mock_bertopic.return_value = mock_bertopic_instance
        
        # Test with standardized num_topics parameter
        modeler = UnifiedTopicModeler(
            method="bertopic",
            num_topics=15
        )
        
        # Check that the parameter was properly converted for the underlying model
        mock_bertopic.assert_called_once()
        kwargs = mock_bertopic.call_args[1]
        self.assertEqual(kwargs.get('n_topics'), 15)

if __name__ == "__main__":
    unittest.main()

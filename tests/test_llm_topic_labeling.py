"""Tests for the LLM topic labeler module."""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# Skip tests if transformers or openai is not available
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from meno.modeling.llm_topic_labeling import LLMTopicLabeler


@pytest.fixture
def sample_keywords():
    """Create sample keywords for testing."""
    return [
        "technology", "computer", "software", "hardware", "device",
        "system", "application", "developer", "code", "program"
    ]


@pytest.fixture
def sample_example_docs():
    """Create sample example documents for testing."""
    return [
        "Computer software developers create programs that power our devices.",
        "Hardware systems require proper maintenance and updates.",
        "Modern technology enables remote work and collaboration tools."
    ]


@pytest.fixture
def mock_topic_model():
    """Create a mock topic model for testing."""
    model = MagicMock()
    
    # Mock get_topic_info
    topic_info = pd.DataFrame({
        'Topic': [-1, 0, 1, 2],
        'Count': [3, 7, 5, 4],
        'Name': ['Outlier', 'Topic 0', 'Topic 1', 'Topic 2']
    })
    model.get_topic_info.return_value = topic_info
    
    # Mock get_topics
    topics = {
        -1: [],
        0: [("technology", 0.9), ("computer", 0.8), ("software", 0.7), ("program", 0.6), ("developer", 0.5)],
        1: [("health", 0.9), ("medical", 0.8), ("doctor", 0.7), ("patient", 0.6), ("hospital", 0.5)],
        2: [("finance", 0.9), ("money", 0.8), ("banking", 0.7), ("investment", 0.6), ("market", 0.5)]
    }
    model.topics = topics
    
    # Mock get_topic method
    model.get_topic.side_effect = lambda topic_id: topics.get(topic_id, [])
    
    return model


class TestLLMTopicLabeler:
    """Test the LLMTopicLabeler class."""
    
    def test_init_defaults(self):
        """Test initializing with default settings."""
        with patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True):
            with patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", False):
                with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
                    with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                        with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                            # Mock tokenizer and model
                            mock_tokenizer.from_pretrained.return_value = MagicMock()
                            mock_model.from_pretrained.return_value = MagicMock()
                            mock_pipeline.return_value = MagicMock()
                            
                            # Initialize labeler
                            labeler = LLMTopicLabeler()
                            
                            # Check default attributes
                            assert labeler.model_type == "local"
                            assert labeler.model_name == "google/flan-t5-small"
                            assert labeler.max_new_tokens == 50
                            assert labeler.temperature == 0.7
                            assert labeler.enable_fallback is True
                            assert labeler.verbose is False
    
    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_init_openai(self, mock_openai):
        """Test initializing with OpenAI settings."""
        mock_openai.OpenAI.return_value = MagicMock()
        
        # Initialize labeler with OpenAI
        labeler = LLMTopicLabeler(model_type="openai", model_name="gpt-3.5-turbo")
        
        # Check attributes
        assert labeler.model_type == "openai"
        assert labeler.model_name == "gpt-3.5-turbo"
        assert hasattr(labeler, "client")
        
        # Check that OpenAI client was created
        mock_openai.OpenAI.assert_called_once()
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    def test_init_auto_selection(self):
        """Test auto selection of model type."""
        with patch("meno.modeling.llm_topic_labeling.openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            
            # Initialize labeler with auto type
            labeler = LLMTopicLabeler(model_type="auto")
            
            # Should prefer OpenAI if available
            assert labeler.model_type == "openai"
            assert labeler.model_name == "gpt-3.5-turbo"
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", False)
    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", False)
    def test_init_no_backends(self):
        """Test initializing when no backends are available."""
        # Should raise ImportError
        with pytest.raises(ImportError, match="No LLM backend is available"):
            LLMTopicLabeler(model_type="auto")
    
    def test_build_prompt(self, sample_keywords, sample_example_docs):
        """Test building prompts for the LLM."""
        with patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True):
            with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
                with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                    with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                        # Mock tokenizer and model
                        mock_tokenizer.from_pretrained.return_value = MagicMock()
                        mock_model.from_pretrained.return_value = MagicMock()
                        mock_pipeline.return_value = MagicMock()
                        
                        # Initialize labeler
                        labeler = LLMTopicLabeler()
                        
                        # Test basic prompt
                        basic_prompt = labeler._build_prompt(sample_keywords)
                        assert "Keywords: technology, computer, software" in basic_prompt
                        assert "Topic name:" in basic_prompt
                        
                        # Test detailed prompt
                        detailed_prompt = labeler._build_prompt(sample_keywords, detailed=True)
                        assert "generate a descriptive and specific topic name" in detailed_prompt
                        
                        # Test with example documents
                        docs_prompt = labeler._build_prompt(sample_keywords, sample_example_docs)
                        assert "Example documents:" in docs_prompt
                        assert "Computer software developers" in docs_prompt
    
    @patch("meno.modeling.llm_topic_labeling.OPENAI_AVAILABLE", True)
    @patch("meno.modeling.llm_topic_labeling.openai")
    def test_generate_openai(self, mock_openai, sample_keywords):
        """Test generating topic names with OpenAI."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Technology and Software Development"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        # Initialize labeler
        labeler = LLMTopicLabeler(model_type="openai")
        
        # Test generating topic name
        topic_name = labeler._generate_openai("Test prompt")
        
        # Verify result
        assert topic_name == "Technology and Software Development"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_generate_local(self, sample_keywords):
        """Test generating topic names with local model."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock tokenizer, model and pipeline
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    # Mock pipeline output
                    mock_pipeline_instance = MagicMock()
                    mock_pipeline_instance.return_value = [{"generated_text": "Test prompt Computer Technology"}]
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    # Initialize labeler
                    labeler = LLMTopicLabeler(model_type="local")
                    
                    # Test generating topic name
                    topic_name = labeler._generate_local("Test prompt")
                    
                    # Verify result
                    assert topic_name == "Computer Technology"
                    mock_pipeline_instance.assert_called_once()
    
    def test_fallback_labeling(self, sample_keywords):
        """Test fallback labeling mechanism."""
        with patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True):
            with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
                with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                    with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                        # Mock tokenizer and model
                        mock_tokenizer.from_pretrained.return_value = MagicMock()
                        mock_model.from_pretrained.return_value = MagicMock()
                        mock_pipeline.return_value = MagicMock()
                        
                        # Initialize labeler
                        labeler = LLMTopicLabeler()
                        
                        # Test with full keyword list
                        fallback_name = labeler._fallback_labeling(sample_keywords)
                        assert fallback_name.startswith("Technology")
                        assert "computer" in fallback_name
                        
                        # Test with single keyword
                        single_fallback = labeler._fallback_labeling(["Technology"])
                        assert single_fallback == "Technology"
                        
                        # Test with empty list
                        empty_fallback = labeler._fallback_labeling([])
                        assert empty_fallback == "Unknown Topic"
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_generate_topic_name(self, sample_keywords, sample_example_docs):
        """Test the main generate_topic_name method."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock pipeline output
                    mock_pipeline_instance = MagicMock()
                    mock_pipeline_instance.return_value = [{"generated_text": "Computer Technology and Software Development"}]
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    # Mock tokenizer and model
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    # Initialize labeler
                    labeler = LLMTopicLabeler(model_type="local")
                    
                    # Test generating topic name
                    topic_name = labeler.generate_topic_name(sample_keywords, sample_example_docs)
                    
                    # Verify result is from the mocked pipeline
                    assert "Computer Technology" in topic_name
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_generate_topic_name_with_fallback(self, sample_keywords):
        """Test fallback in generate_topic_name when LLM fails."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock tokenizer and model
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    # Mock pipeline to raise an exception
                    mock_pipeline_instance = MagicMock()
                    mock_pipeline_instance.side_effect = RuntimeError("Model failed")
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    # Initialize labeler with fallback enabled
                    labeler = LLMTopicLabeler(model_type="local", enable_fallback=True)
                    
                    # Generate topic name should use fallback
                    topic_name = labeler.generate_topic_name(sample_keywords)
                    
                    # Verify result is from fallback
                    assert topic_name.startswith("Technology")
                    assert "computer" in topic_name.lower()
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_generate_topic_name_without_fallback(self, sample_keywords):
        """Test generate_topic_name with fallback disabled."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock tokenizer and model
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    # Mock pipeline to raise an exception
                    mock_pipeline_instance = MagicMock()
                    mock_pipeline_instance.side_effect = RuntimeError("Model failed")
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    # Initialize labeler with fallback disabled
                    labeler = LLMTopicLabeler(model_type="local", enable_fallback=False)
                    
                    # Should raise the original exception
                    with pytest.raises(RuntimeError, match="Model failed"):
                        labeler.generate_topic_name(sample_keywords)
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_label_topics(self, mock_topic_model):
        """Test labeling all topics in a model."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock pipeline output to return different names for different topics
                    mock_pipeline_instance = MagicMock()
                    mock_pipeline_instance.side_effect = [
                        [{"generated_text": "Technology and Software Development"}],
                        [{"generated_text": "Healthcare and Medical Services"}],
                        [{"generated_text": "Finance and Investment Banking"}]
                    ]
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    # Mock tokenizer and model
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    # Initialize labeler
                    labeler = LLMTopicLabeler(model_type="local")
                    
                    # Label all topics
                    topic_names = labeler.label_topics(mock_topic_model, progress_bar=False)
                    
                    # Verify results
                    assert 0 in topic_names
                    assert 1 in topic_names
                    assert 2 in topic_names
                    assert -1 not in topic_names  # Should skip outlier topic
                    assert "Technology" in topic_names[0]
                    assert "Healthcare" in topic_names[1]
                    assert "Finance" in topic_names[2]
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_update_model_topic_names(self, mock_topic_model):
        """Test updating topic names in a model."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock pipeline output
                    mock_pipeline_instance = MagicMock()
                    mock_pipeline_instance.side_effect = [
                        [{"generated_text": "Technology and Software Development"}],
                        [{"generated_text": "Healthcare and Medical Services"}],
                        [{"generated_text": "Finance and Investment Banking"}]
                    ]
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    # Mock tokenizer and model
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    # Initialize labeler
                    labeler = LLMTopicLabeler(model_type="local")
                    
                    # Update model topic names
                    updated_model = labeler.update_model_topic_names(mock_topic_model, progress_bar=False)
                    
                    # Verify model was updated
                    assert "Technology" in updated_model.topics[0]
                    assert "Healthcare" in updated_model.topics[1]
                    assert "Finance" in updated_model.topics[2]
                    assert updated_model.topics[-1] == "Other/Outlier"
    
    @patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True)
    def test_save_and_load(self):
        """Test saving and loading the labeler configuration."""
        with patch("meno.modeling.llm_topic_labeling.AutoTokenizer") as mock_tokenizer:
            with patch("meno.modeling.llm_topic_labeling.AutoModelForCausalLM") as mock_model:
                with patch("meno.modeling.llm_topic_labeling.pipeline") as mock_pipeline:
                    # Mock tokenizer and model
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    mock_pipeline.return_value = MagicMock()
                    
                    # Create a temporary directory
                    with tempfile.TemporaryDirectory() as tmpdir:
                        config_path = Path(tmpdir) / "labeler_config.json"
                        
                        # Create and save labeler
                        labeler = LLMTopicLabeler(
                            model_type="local",
                            model_name="google/flan-t5-base",
                            temperature=0.5,
                            max_new_tokens=100
                        )
                        
                        labeler.save(config_path)
                        
                        # Verify config file was created
                        assert config_path.exists()
                        
                        # Check config contents
                        with open(config_path, "r") as f:
                            config = json.load(f)
                            assert config["model_type"] == "local"
                            assert config["model_name"] == "google/flan-t5-base"
                            assert config["temperature"] == 0.5
                            assert config["max_new_tokens"] == 100
                        
                        # Load labeler from config
                        with patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True):
                            loaded_labeler = LLMTopicLabeler.load(config_path)
                            
                            # Verify loaded configuration
                            assert loaded_labeler.model_type == "local"
                            assert loaded_labeler.model_name == "google/flan-t5-base"
                            assert loaded_labeler.temperature == 0.5
                            assert loaded_labeler.max_new_tokens == 100
                        
                        # Test overriding with kwargs
                        with patch("meno.modeling.llm_topic_labeling.TRANSFORMERS_AVAILABLE", True):
                            custom_labeler = LLMTopicLabeler.load(
                                config_path,
                                temperature=0.8,
                                model_name="google/flan-t5-small"
                            )
                            
                            # Verify configuration was overridden
                            assert custom_labeler.model_type == "local"
                            assert custom_labeler.model_name == "google/flan-t5-small"
                            assert custom_labeler.temperature == 0.8
                            assert custom_labeler.max_new_tokens == 100


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
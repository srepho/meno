"""Tests for the multi-provider LLM integration functionality."""

import os
import unittest
from unittest.mock import patch, MagicMock

import pytest

from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi


class TestMultiProviderLLM(unittest.TestCase):
    """Test the multi-provider LLM integration functionality."""

    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HUGGINGFACE_API_KEY": "test-hf-key",
            "AWS_ACCESS_KEY_ID": "test-aws-key",
            "AWS_SECRET_ACCESS_KEY": "test-aws-secret"
        })
        self.env_patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()

    @patch("meno.modeling.llm_topic_labeling_extended.openai")
    def test_openai_integration(self, mock_openai):
        """Test OpenAI integration."""
        # Configure mock
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response from OpenAI"
        mock_client.chat.completions.create.return_value = mock_response

        # Call the function
        result = generate_text_with_llm_multi(
            text="Test prompt",
            provider="openai",
            model_name="gpt-3.5-turbo"
        )
        
        # Check results
        assert result == "Test response from OpenAI"
        mock_client.chat.completions.create.assert_called_once()
        
    @patch("meno.modeling.llm_topic_labeling_extended.requests")
    def test_anthropic_integration_with_requests(self, mock_requests):
        """Test Anthropic integration using requests library."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Test response from Anthropic"}]
        }
        mock_requests.post.return_value = mock_response
        
        # Call the function
        result = generate_text_with_llm_multi(
            text="Test prompt",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            library="requests"
        )
        
        # Check results
        assert result == "Test response from Anthropic"
        mock_requests.post.assert_called_once()
        
    @patch("meno.modeling.llm_topic_labeling_extended.anthropic")
    def test_anthropic_integration_with_sdk(self, mock_anthropic):
        """Test Anthropic integration using official SDK."""
        # Configure mock
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [{"text": "Test response from Anthropic SDK"}]
        mock_client.messages.create.return_value = mock_message
        
        # Call the function
        result = generate_text_with_llm_multi(
            text="Test prompt",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            library="sdk"
        )
        
        # Check results
        assert result == "Test response from Anthropic SDK"
        mock_client.messages.create.assert_called_once()

    @patch("meno.modeling.llm_topic_labeling_extended.google.generativeai")
    def test_google_integration(self, mock_genai):
        """Test Google Gemini integration."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.text = "Test response from Google"
        mock_genai.generate_text.return_value = mock_response
        
        # Call the function
        result = generate_text_with_llm_multi(
            text="Test prompt",
            provider="google",
            model_name="gemini-pro"
        )
        
        # Check results
        assert result == "Test response from Google"
        mock_genai.generate_text.assert_called_once()
        
    @patch("meno.modeling.llm_topic_labeling_extended.requests")
    def test_huggingface_integration(self, mock_requests):
        """Test Hugging Face integration."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Test response from Hugging Face"}]
        mock_requests.post.return_value = mock_response
        
        # Call the function
        result = generate_text_with_llm_multi(
            text="Test prompt",
            provider="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.2"
        )
        
        # Check results
        assert result == "Test response from Hugging Face"
        mock_requests.post.assert_called_once()
        
    @patch("meno.modeling.llm_topic_labeling_extended.boto3")
    def test_bedrock_integration(self, mock_boto3):
        """Test AWS Bedrock integration."""
        # Configure mock
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_response = {
            "completion": "Test response from Bedrock",
            "stop_reason": "stop"
        }
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=MagicMock(return_value=b'{"completion": "Test response from Bedrock", "stop_reason": "stop"}'))
        }
        
        # Call the function
        result = generate_text_with_llm_multi(
            text="Test prompt",
            provider="bedrock",
            model_name="anthropic.claude-3-sonnet-20240229",
            region_name="us-east-1"
        )
        
        # Check results
        assert result == "Test response from Bedrock"
        mock_client.invoke_model.assert_called_once()
        
    def test_invalid_provider(self):
        """Test error handling for invalid provider."""
        with pytest.raises(ValueError):
            generate_text_with_llm_multi(
                text="Test prompt",
                provider="invalid_provider",
                model_name="some-model"
            )
            
    @patch("meno.modeling.llm_topic_labeling_extended.generate_text_with_llm_multi.cache_get")
    @patch("meno.modeling.llm_topic_labeling_extended.generate_text_with_llm_multi.cache_set")
    @patch("meno.modeling.llm_topic_labeling_extended.openai")
    def test_caching(self, mock_openai, mock_cache_set, mock_cache_get):
        """Test caching functionality."""
        # Configure mocks
        mock_cache_get.return_value = None  # First call: cache miss
        
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response from OpenAI"
        mock_client.chat.completions.create.return_value = mock_response
        
        # First call should use API
        result1 = generate_text_with_llm_multi(
            text="Test prompt",
            provider="openai",
            model_name="gpt-3.5-turbo",
            enable_cache=True
        )
        
        # Configure cache hit for second call
        mock_cache_get.return_value = "Cached response"
        
        # Second call should use cache
        result2 = generate_text_with_llm_multi(
            text="Test prompt",
            provider="openai", 
            model_name="gpt-3.5-turbo",
            enable_cache=True
        )
        
        # Check results
        assert result1 == "Test response from OpenAI"
        assert result2 == "Cached response"
        mock_client.chat.completions.create.assert_called_once()  # API called only once
        mock_cache_set.assert_called_once()  # Cache set called once


if __name__ == "__main__":
    unittest.main()
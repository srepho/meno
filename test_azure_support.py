"""
Test script for Azure OpenAI support and multi-library implementation in Meno.

This tests:
1. Azure OpenAI support in the LLMTopicLabeler class
2. The enhanced generate_text_with_llm function with both OpenAI SDK and direct requests support
3. Caching functionality in the requests implementation
"""

import sys
import logging
import importlib.util
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for openai
if importlib.util.find_spec("openai") is None:
    logger.error("OpenAI package is not installed. Please install it with 'pip install openai'")
    sys.exit(1)

# Check for requests
if importlib.util.find_spec("requests") is None:
    logger.error("Requests package is not installed. Please install it with 'pip install requests'")
    sys.exit(1)

# Import required modules
import openai
import requests
from meno.modeling.llm_topic_labeling import LLMTopicLabeler, generate_text_with_llm

def test_azure_openai_integration():
    """
    Test that the Azure OpenAI integration works correctly.
    """
    logger.info("Testing Azure OpenAI integration")
    
    # Create a labeler with Azure settings
    labeler = LLMTopicLabeler(
        model_type="openai",
        model_name="test-deployment",
        api_key="test-api-key",
        api_endpoint="https://test-endpoint.openai.azure.com",
        api_version="2023-05-15",
        use_azure=True,
        verbose=True
    )
    
    # Verify the client is initialized as AzureOpenAI
    assert isinstance(labeler.client, openai.AzureOpenAI)
    logger.info("✅ Client properly initialized as AzureOpenAI")
    
    # Create a mock response for the client's create method
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    
    # Replace the client with a mock to avoid making actual API calls
    labeler.client = MagicMock()
    labeler.client.chat.completions.create.return_value = mock_response
    
    # Try to generate text
    result = labeler._generate_openai("Test prompt")
    logger.info(f"Generated result: {result}")
    
    # Verify the result and the API call
    assert result == "Test response"
    assert labeler.client.chat.completions.create.called
    
    # Check the arguments passed to the API call
    call_args = labeler.client.chat.completions.create.call_args
    kwargs = call_args[1]  # Get keyword arguments
    
    # Print what was received for debugging
    logger.info(f"API call arguments: {list(kwargs.keys())}")
    
    # Check the messages
    assert len(kwargs['messages']) == 2  # Should have system and user messages
    
    logger.info("✅ Text generation API call uses the correct parameters")
    logger.info("All Azure OpenAI integration tests passed!")
    
    return True

@patch('meno.modeling.llm_topic_labeling.openai')
def test_generate_text_with_llm_openai_sdk(mock_openai):
    """Test the generate_text_with_llm function with OpenAI SDK."""
    logger.info("Testing generate_text_with_llm with OpenAI SDK")
    
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    
    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "SDK Test Response"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Use the function with "openai" library setting
    result = generate_text_with_llm(
        text="Test prompt",
        api_key="test-key",
        api_endpoint=None,
        model_name="gpt-4",
        use_azure=False,
        library="openai",
        temperature=0.5
    )
    
    # Verify the result
    assert result == "SDK Test Response"
    
    # Verify the SDK was used correctly
    mock_openai.OpenAI.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()
    
    # Check parameters
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args['model'] == "gpt-4"
    assert call_args['temperature'] == 0.5
    
    logger.info("✅ generate_text_with_llm works correctly with OpenAI SDK")
    return True

@patch('meno.modeling.llm_topic_labeling.requests')
def test_generate_text_with_llm_requests(mock_requests):
    """Test the generate_text_with_llm function with requests library."""
    logger.info("Testing generate_text_with_llm with requests library")
    
    # Setup mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Requests Test Response"
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_requests.post.return_value = mock_response
    
    # Use the function with "requests" library setting
    result = generate_text_with_llm(
        text="Test prompt",
        api_key="test-key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-4",
        use_azure=False,
        library="requests",
        temperature=0.7,
        timeout=30
    )
    
    # Verify the result
    assert result == "Requests Test Response"
    
    # Verify requests was used correctly
    mock_requests.post.assert_called_once()
    
    # Check parameters
    call_args = mock_requests.post.call_args
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
    assert call_args[1]["json"]["model"] == "gpt-4"
    assert call_args[1]["json"]["temperature"] == 0.7
    assert call_args[1]["timeout"] == 30
    
    logger.info("✅ generate_text_with_llm works correctly with requests library")
    return True

@patch('meno.modeling.llm_topic_labeling.requests')
def test_azure_with_requests(mock_requests):
    """Test using Azure OpenAI with the requests library."""
    logger.info("Testing Azure OpenAI with requests library")
    
    # Setup mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Azure Requests Test Response"
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_requests.post.return_value = mock_response
    
    # Azure endpoint format
    deployment_id = "test-deployment"
    api_endpoint = f"https://test.openai.azure.com/openai/deployments/{deployment_id}/chat/completions"
    api_version = "2023-05-15"
    
    # Use the function with Azure settings
    result = generate_text_with_llm(
        text="Test prompt",
        api_key="test-key",
        api_endpoint=api_endpoint,
        deployment_id=deployment_id,
        api_version=api_version,
        use_azure=True,
        library="requests"
    )
    
    # Verify the result
    assert result == "Azure Requests Test Response"
    
    # Verify requests was used correctly with Azure headers
    mock_requests.post.assert_called_once()
    
    # Check Azure-specific headers
    call_args = mock_requests.post.call_args
    headers = call_args[1]["headers"]
    assert "api-key" in headers
    assert headers["api-key"] == "test-key"
    assert "api-version" in call_args[1]["params"]
    assert call_args[1]["params"]["api-version"] == api_version
    
    logger.info("✅ Azure OpenAI works correctly with requests library")
    return True

@patch('meno.modeling.llm_topic_labeling.requests')
@patch('meno.modeling.llm_topic_labeling.os.path.exists')
@patch('meno.modeling.llm_topic_labeling.os.makedirs')
@patch('builtins.open', new_callable=MagicMock)
def test_caching_functionality(mock_open, mock_makedirs, mock_exists, mock_requests):
    """Test the caching functionality of the requests implementation."""
    logger.info("Testing caching functionality")
    
    # Setup cache directory checks
    mock_exists.return_value = True
    
    # Setup pickle load/dump mocks
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # First call - should miss cache
    mock_exists.side_effect = [True, False]  # Cache dir exists, file doesn't
    
    # Setup mock response for the API call
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Cached Response"
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_requests.post.return_value = mock_response
    
    # First call should hit the API
    result1 = generate_text_with_llm(
        text="Cache test prompt",
        api_key="test-key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-4",
        library="requests",
        enable_cache=True
    )
    
    assert result1 == "Cached Response"
    assert mock_requests.post.call_count == 1
    
    # Reset mocks for second call
    mock_requests.post.reset_mock()
    mock_exists.side_effect = [True, True]  # Cache dir exists, file exists too
    
    # Setup mock pickle load
    mock_open.return_value.__enter__.return_value.read.return_value = b'{"content": "Cached Response", "timestamp": 9999999999.9}'
    
    # Second call with same parameters should hit cache
    result2 = generate_text_with_llm(
        text="Cache test prompt",
        api_key="test-key", 
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-4",
        library="requests",
        enable_cache=True
    )
    
    assert result2 == "Cached Response"
    assert mock_requests.post.call_count == 0  # No API call
    
    logger.info("✅ Caching functionality works correctly")
    return True

@patch('meno.modeling.llm_topic_labeling.requests')
def test_error_handling(mock_requests):
    """Test error handling in the requests implementation."""
    logger.info("Testing error handling")
    
    # Setup mock to raise an exception
    mock_requests.post.side_effect = Exception("Test API Error")
    
    # Call should return an error string instead of raising
    result = generate_text_with_llm(
        text="Error test prompt",
        api_key="test-key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-4",
        library="requests"
    )
    
    # Verify error is captured
    assert "[Error:" in result
    assert "Test API Error" in result
    
    # Test with invalid library parameter
    result = generate_text_with_llm(
        text="Invalid library test",
        api_key="test-key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        library="invalid_library"
    )
    
    # Verify error is captured
    assert "[Error:" in result
    assert "Unsupported library" in result
    
    logger.info("✅ Error handling works correctly")
    return True

def run_all_tests():
    """Run all test functions."""
    results = {}
    
    tests = [
        test_azure_openai_integration,
        test_generate_text_with_llm_openai_sdk,
        test_generate_text_with_llm_requests,
        test_azure_with_requests,
        test_caching_functionality,
        test_error_handling
    ]
    
    for test_func in tests:
        try:
            logger.info(f"\n{'='*50}\nRunning {test_func.__name__}")
            result = test_func()
            results[test_func.__name__] = "PASS" if result else "FAIL"
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            results[test_func.__name__] = f"ERROR: {str(e)}"
    
    # Print summary
    logger.info("\n\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result == "PASS" else f"❌ {result}"
        logger.info(f"{test_name}: {status}")
        if result == "PASS":
            passed += 1
    
    logger.info(f"\nPassed {passed}/{len(tests)} tests")
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
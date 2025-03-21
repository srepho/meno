# Enhanced LLM API Integration in Meno

This document provides details on the enhanced LLM API integration in Meno, which now supports both the OpenAI SDK and direct requests-based approaches for interacting with language models.

## Overview

The `generate_text_with_llm` function has been enhanced to provide:

1. Support for both OpenAI SDK and direct requests-based API access
2. Consistent interface for both standard OpenAI and Azure OpenAI APIs
3. Caching mechanism for the requests implementation to avoid redundant API calls
4. Proper error handling and timeout configuration
5. Extensible design that can be adapted for other LLM providers

## Key Features

### Multiple Library Support

Choose between two implementation libraries:

- **OpenAI SDK**: Uses the official OpenAI Python client
- **Requests**: Uses direct HTTP requests with the `requests` library, ideal for environments where installing the full SDK is not feasible

### Caching

The requests implementation includes an optional caching mechanism that:

- Stores API responses on disk to avoid redundant calls
- Includes TTL (time-to-live) settings for cache expiration
- Uses a hash-based cache key system for accurate retrieval

### Azure OpenAI Support

Both implementations support:

- Standard OpenAI API
- Azure OpenAI API with proper authentication and parameter handling

### Error Handling

Comprehensive error handling that:

- Catches and formats API errors
- Provides informative error messages
- Handles connectivity issues gracefully

## Usage Examples

### Basic Usage with OpenAI SDK

```python
from meno.modeling.llm_topic_labeling import generate_text_with_llm

response = generate_text_with_llm(
    text="What are three interesting facts about machine learning?",
    api_key="OPENAI_API_KEY_PLACEHOLDER",
    model_name="gpt-4",
    library="openai"  # Use OpenAI SDK (default)
)

print(response)
```

### Using Direct Requests with Caching

```python
from meno.modeling.llm_topic_labeling import generate_text_with_llm

response = generate_text_with_llm(
    text="What are three interesting facts about machine learning?",
    api_key="OPENAI_API_KEY_PLACEHOLDER",
    api_endpoint="https://api.openai.com/v1/chat/completions",
    model_name="gpt-3.5-turbo",
    library="requests",  # Use direct requests
    enable_cache=True,   # Enable caching
    timeout=30           # 30 second timeout
)

print(response)
```

### Azure OpenAI with OpenAI SDK

```python
from meno.modeling.llm_topic_labeling import generate_text_with_llm

response = generate_text_with_llm(
    text="What are three interesting facts about machine learning?",
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_endpoint="https://RESOURCE_NAME.openai.azure.com",
    deployment_id="DEPLOYMENT_NAME_PLACEHOLDER",
    api_version="2023-05-15",
    use_azure=True,
    library="openai"  # Use OpenAI SDK
)

print(response)
```

### Azure OpenAI with Direct Requests

```python
from meno.modeling.llm_topic_labeling import generate_text_with_llm

# Construct Azure endpoint
endpoint = "https://RESOURCE_NAME.openai.azure.com"
deployment = "DEPLOYMENT_NAME_PLACEHOLDER"
full_endpoint = f"{endpoint}/openai/deployments/{deployment}/chat/completions"

response = generate_text_with_llm(
    text="What are three interesting facts about machine learning?",
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_endpoint=full_endpoint,
    deployment_id=deployment,
    api_version="2023-05-15",
    use_azure=True,
    library="requests",  # Use direct requests
    enable_cache=True    # Enable caching
)

print(response)
```

## Function Parameters

The enhanced `generate_text_with_llm` function accepts the following parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `text` | str | Text prompt to send to the LLM | Required |
| `api_key` | str | API key for OpenAI or Azure OpenAI | Required |
| `api_endpoint` | str | API endpoint URL | None (uses default OpenAI endpoint) |
| `deployment_id` | str | Azure OpenAI deployment name | None |
| `model_name` | str | OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo") | "gpt-4o" |
| `api_version` | str | API version (required for Azure) | "2023-05-15" |
| `use_azure` | bool | Whether to use Azure OpenAI | False |
| `system_prompt` | str | System prompt for the LLM | "You are a helpful assistant." |
| `user_prompt_prefix` | str | Text to prefix to the user's input | "" |
| `temperature` | float | Temperature setting for generation | 0.7 |
| `max_tokens` | int | Maximum tokens to generate | 1000 |
| `library` | str | Library to use ("openai" or "requests") | "openai" |
| `timeout` | int | Request timeout in seconds (requests only) | 60 |
| `enable_cache` | bool | Whether to enable caching (requests only) | True |
| `cache_dir` | str | Directory to store cache files | None (uses ~/.meno/llm_cache) |

## Extending to Other LLM Providers

The architecture allows for easy extension to support other LLM providers:

1. Copy the pattern used for OpenAI with two implementations (SDK and requests)
2. Implement provider-specific authentication and request formatting
3. Add the provider as an option to the library parameter or create a new parameter

See the `extending_llm_providers.py` example for a demonstration of extending the functionality to support:

- Google's Gemini
- Anthropic's Claude
- Cohere's models
- Other LLM providers

## Caching Technical Details

The caching mechanism works by:

1. Generating a unique cache key based on the input text, model, and parameters
2. Storing responses in JSON files in the cache directory
3. Including a timestamp for TTL (time-to-live) calculations
4. Checking for cache hits before making API calls

Cache files are stored in `~/.meno/llm_cache` by default, but this can be customized with the `cache_dir` parameter.

## Error Handling

The function returns formatted error messages rather than raising exceptions, making it more robust for production use:

- API errors: `[Error: API error message]`
- Network errors: `[Error: Network error details]`
- Invalid parameters: `[Error: Invalid parameter details]`

## Testing

You can use the provided test scripts to verify functionality:

- `test_azure_support.py`: Tests both OpenAI SDK and requests implementations with mocked responses
- `test_llm_api_integration.py`: Manual testing script for real API interaction

## Best Practices

1. **Use caching** for repeated calls to save on API costs
2. **Set appropriate timeouts** based on your application's requirements
3. **Use the `requests` implementation** when the OpenAI SDK cannot be installed
4. **Specify the API version** when using Azure OpenAI to ensure compatibility
5. **Check error messages** for troubleshooting API interactions

## Limitations

1. The requests implementation does not support streaming responses
2. Caching is only available with the requests implementation
3. Azure OpenAI requires additional parameters (deployment_id, api_version)
# Multi-Provider LLM Integration in Meno

This document provides details on the extended LLM API integration in Meno, which now supports multiple providers including Google Gemini, Anthropic Claude, Hugging Face, and AWS Bedrock, in addition to the original OpenAI support.

## Overview

The `generate_text_with_llm_multi` function has been introduced to provide:

1. Support for multiple LLM providers beyond OpenAI
2. Consistent interface across all providers
3. Option to use either SDK libraries or direct HTTP requests
4. Caching mechanism for all providers to avoid redundant API calls
5. Proper error handling and timeout configuration
6. Extensible design that can be adapted for future LLM providers

## Key Features

### Multiple Provider Support

Choose between five major LLM providers:

- **OpenAI**: Original implementation with both SDK and requests options
- **Google Gemini**: Supports Gemini Pro and other Google generative AI models
- **Anthropic Claude**: Supports Claude 3 and other Anthropic models
- **Hugging Face**: Access to thousands of open models via Inference API
- **AWS Bedrock**: Managed access to multiple foundation models including those from Anthropic, Cohere, AI21, and Amazon

### Multiple Library Support

For each provider, choose between two implementation approaches:

- **SDK**: Uses the official provider-specific Python client
- **Requests**: Uses direct HTTP requests with the `requests` library, ideal for environments where installing the full SDKs is not feasible

### Caching

All implementations include an optional caching mechanism that:

- Stores API responses on disk to avoid redundant calls
- Includes TTL (time-to-live) settings for cache expiration
- Uses a hash-based cache key system for accurate retrieval

## Usage Examples

### Basic Usage with OpenAI (Original Implementation)

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

response = generate_text_with_llm_multi(
    text="What are three interesting facts about machine learning?",
    api_key="your-openai-api-key",
    provider="openai",
    model_name="gpt-4",
    library="sdk"
)

print(response)
```

### Using Google Gemini

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

response = generate_text_with_llm_multi(
    text="What are three interesting facts about machine learning?",
    api_key="your-google-api-key",
    provider="google",
    model_name="gemini-pro",
    library="sdk",
    temperature=0.7,
    max_tokens=800
)

print(response)
```

### Using Anthropic Claude

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

response = generate_text_with_llm_multi(
    text="What are three interesting facts about machine learning?",
    api_key="your-anthropic-api-key",
    provider="anthropic",
    model_name="claude-3-sonnet-20240229",
    library="requests",
    temperature=0.7,
    max_tokens=800,
    enable_cache=True,
    api_version="2023-06-01"
)

print(response)
```

### Using Hugging Face Inference API

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

response = generate_text_with_llm_multi(
    text="What are three interesting facts about machine learning?",
    api_key="your-huggingface-api-key",
    provider="huggingface",
    model_name="meta-llama/Llama-2-70b-chat-hf",
    library="sdk",
    temperature=0.7,
    max_tokens=800
)

print(response)
```

### Using AWS Bedrock

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

response = generate_text_with_llm_multi(
    text="What are three interesting facts about machine learning?",
    api_key="your-aws-access-key",
    api_secret="your-aws-secret-key",
    provider="bedrock",
    model_name="anthropic.claude-3-sonnet-20240229",
    library="sdk",
    temperature=0.7,
    max_tokens=800,
    region_name="us-east-1"
)

print(response)
```

## Function Parameters

The enhanced `generate_text_with_llm_multi` function accepts the following parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `text` | str | Text prompt to send to the LLM | Required |
| `api_key` | str | API key for the LLM provider | Required |
| `api_endpoint` | str | API endpoint URL | None |
| `deployment_id` | str | Azure OpenAI deployment name | None |
| `model_name` | str | Model name to use | "gpt-4o" |
| `api_version` | str | API version | "2023-05-15" |
| `use_azure` | bool | Whether to use Azure OpenAI | False |
| `system_prompt` | str | System prompt for the LLM | "You are a helpful assistant." |
| `user_prompt_prefix` | str | Text to prefix to the user's input | "" |
| `temperature` | float | Temperature setting for generation | 0.7 |
| `max_tokens` | int | Maximum tokens to generate | 1000 |
| `library` | str | Library to use ("sdk" or "requests") | "openai" |
| `timeout` | int | Request timeout in seconds | 60 |
| `enable_cache` | bool | Whether to enable caching | True |
| `cache_dir` | str | Directory to store cache files | None (uses ~/.meno/llm_cache) |
| `provider` | str | LLM provider to use | "openai" |
| `api_secret` | str | Secondary API credential | None |
| `region_name` | str | Region for AWS Bedrock | "us-east-1" |
| `additional_params` | dict | Additional provider-specific parameters | None |

## Provider-Specific Model Names

Each provider has its own model naming convention:

### OpenAI
- `gpt-4o`: Latest GPT-4 model
- `gpt-4-turbo`: GPT-4 Turbo
- `gpt-3.5-turbo`: More cost-effective model

### Google Gemini
- `gemini-pro`: Flagship text model
- `gemini-pro-vision`: Multimodal model

### Anthropic Claude
- `claude-3-opus-20240229`: Most powerful Claude model
- `claude-3-sonnet-20240229`: Balanced performance and cost
- `claude-3-haiku-20240307`: Fastest, most cost-effective Claude model

### Hugging Face
- `meta-llama/Llama-2-70b-chat-hf`: Llama 2 70B chat model
- `mistralai/Mistral-7B-Instruct-v0.2`: Mistral 7B instruction-tuned model
- Many other models available

### AWS Bedrock
- `anthropic.claude-3-sonnet-20240229`: Claude on Bedrock
- `amazon.titan-text-express-v1`: Amazon's Titan model
- `cohere.command-text-v14`: Cohere Command
- `meta.llama2-70b-chat-v1`: Llama 2 on Bedrock

## Provider-Specific Additional Parameters

Each provider supports custom parameters through the `additional_params` dictionary:

### OpenAI
```python
additional_params = {
    "presence_penalty": 0.6,
    "frequency_penalty": 0.0,
}
```

### Google Gemini
```python
additional_params = {
    "top_k": 40,
    "top_p": 0.95,
    "stop_sequences": ["User:"],
}
```

### Anthropic Claude
```python
additional_params = {
    "top_p": 0.9,
    "stop_sequences": ["Human:"],
}
```

### Hugging Face
```python
additional_params = {
    "parameters": {
        "top_p": 0.95,
        "repetition_penalty": 1.1,
    },
}
```

### AWS Bedrock
```python
additional_params = {
    "top_p": 0.9,
    "stop_sequences": ["Human:"],
}
```

## Installation Requirements

To use the multi-provider LLM integration, you'll need to install the relevant packages for each provider:

```bash
# For OpenAI
pip install openai

# For Google Gemini
pip install google-generativeai

# For Anthropic Claude
pip install anthropic

# For Hugging Face
pip install huggingface_hub

# For AWS Bedrock
pip install boto3
```

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

## Best Practices

1. **Use caching** for repeated calls to save on API costs
2. **Set appropriate timeouts** based on your application's requirements
3. **Use the SDK implementation** when available for better error handling
4. **Use the `requests` implementation** when SDKs cannot be installed
5. **Check error messages** for troubleshooting API interactions

## Limitations

1. The requests implementation does not support streaming responses
2. Some advanced features may only be available through the SDK implementations
3. Each provider has different parameter naming conventions and capabilities
# Multi-Provider LLM Integration in Meno v1.3.4

## Overview

Version 1.3.4 expands Meno's LLM integration capabilities to support multiple providers beyond OpenAI. The new `generate_text_with_llm_multi` function provides a unified interface for interacting with:

- OpenAI (GPT models)
- Google Gemini
- Anthropic Claude
- Hugging Face Inference API
- AWS Bedrock

This guide explains how to use these new capabilities in your projects.

## Installation

To use the multi-provider LLM integration, install Meno with the `llm_multi` extra:

```bash
pip install meno[llm_multi]
```

Or selectively install only the providers you need:

```bash
# For Google Gemini
pip install meno google-generativeai

# For Anthropic Claude
pip install meno anthropic

# For Hugging Face
pip install meno huggingface_hub

# For AWS Bedrock
pip install meno boto3
```

## Basic Usage

The function signature is similar to the original `generate_text_with_llm` but adds support for selecting different providers:

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

# Generate text with OpenAI (original functionality)
response = generate_text_with_llm_multi(
    text="What are three key benefits of topic modeling?",
    api_key="OPENAI_API_KEY_PLACEHOLDER",
    provider="openai",
    model_name="gpt-3.5-turbo"
)

# Generate text with Google Gemini
response = generate_text_with_llm_multi(
    text="What are three key benefits of topic modeling?",
    api_key="GOOGLE_API_KEY_PLACEHOLDER",
    provider="google",
    model_name="gemini-pro"
)

# Generate text with Anthropic Claude
response = generate_text_with_llm_multi(
    text="What are three key benefits of topic modeling?",
    api_key="ANTHROPIC_API_KEY_PLACEHOLDER",
    provider="anthropic",
    model_name="claude-3-haiku-20240307"
)
```

## Choosing Between SDK and Direct Requests

For each provider, you can choose between using the official SDK or direct HTTP requests:

```python
# Using Google Gemini with SDK
response = generate_text_with_llm_multi(
    text="Compare and contrast different topic modeling approaches.",
    api_key="GOOGLE_API_KEY_PLACEHOLDER",
    provider="google",
    library="sdk",
    model_name="gemini-pro"
)

# Using Google Gemini with direct requests
response = generate_text_with_llm_multi(
    text="Compare and contrast different topic modeling approaches.",
    api_key="GOOGLE_API_KEY_PLACEHOLDER",
    provider="google",
    library="requests",
    model_name="gemini-pro"
)
```

## Provider-Specific Parameters

Each provider has unique parameters that can be configured:

### OpenAI
```python
response = generate_text_with_llm_multi(
    text="Summarize the key benefits of BERTopic.",
    api_key="OPENAI_API_KEY_PLACEHOLDER",
    provider="openai",
    model_name="gpt-4",
    use_azure=False,  # Set to True to use Azure OpenAI
    api_endpoint=None,  # Custom endpoint if needed
    api_version="2023-05-15"  # Required for Azure
)
```

### Google Gemini
```python
response = generate_text_with_llm_multi(
    text="Summarize the key benefits of BERTopic.",
    api_key="GOOGLE_API_KEY_PLACEHOLDER",
    provider="google",
    model_name="gemini-pro",
    additional_params={
        "top_k": 40,
        "top_p": 0.95
    }
)
```

### Anthropic Claude
```python
response = generate_text_with_llm_multi(
    text="Summarize the key benefits of BERTopic.",
    api_key="ANTHROPIC_API_KEY_PLACEHOLDER",
    provider="anthropic",
    model_name="claude-3-sonnet-20240229",
    api_version="2023-06-01"
)
```

### Hugging Face
```python
response = generate_text_with_llm_multi(
    text="Summarize the key benefits of BERTopic.",
    api_key="HUGGINGFACE_API_KEY_PLACEHOLDER",
    provider="huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    additional_params={
        "parameters": {
            "do_sample": True,
            "top_p": 0.95
        }
    }
)
```

### AWS Bedrock
```python
response = generate_text_with_llm_multi(
    text="Summarize the key benefits of BERTopic.",
    api_key="AWS_ACCESS_KEY_PLACEHOLDER",
    api_secret="AWS_SECRET_KEY_PLACEHOLDER",
    provider="bedrock",
    model_name="anthropic.claude-3-sonnet-20240229",
    region_name="us-east-1"
)
```

## Caching for All Providers

The caching mechanism works for all providers, helping to save on API costs and improve response times:

```python
response = generate_text_with_llm_multi(
    text="Explain how UMAP dimensionality reduction works.",
    api_key="API_KEY_PLACEHOLDER",
    provider="anthropic",
    model_name="claude-3-haiku-20240307",
    enable_cache=True,
    cache_dir="/path/to/custom/cache"  # Optional, defaults to ~/.meno/llm_cache
)
```

## Integration with the Topic Labeler

The multi-provider function can be used with the LLMTopicLabeler class for enhanced topic naming:

```python
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

# Example function to use Claude for topic labeling
def generate_with_claude(text, api_key):
    return generate_text_with_llm_multi(
        text=text,
        api_key=api_key,
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=100
    )

# Use a custom function for topic labeling
labeler = LLMTopicLabeler(
    model_type="custom",
    api_key="ANTHROPIC_API_KEY_PLACEHOLDER",
    custom_generation_function=generate_with_claude
)

# Generate topic names
keywords = ["neural", "networks", "deep", "learning", "gradient"]
topic_name = labeler.generate_topic_name(keywords)
```

## Examples

For detailed examples of using different providers, see the example script:
```
examples/multi_provider_llm_example.py
```

This example demonstrates:
- Using all supported providers
- Both SDK and requests implementations
- Side-by-side comparison of results
- Performance benchmarking

## Best Practices

1. **Choose the appropriate provider based on your needs:**
   - OpenAI: General purpose, strong performance
   - Google Gemini: Strong for multimodal content
   - Anthropic Claude: Safety and long-context processing
   - Hugging Face: Access to open models and customization
   - AWS Bedrock: Enterprise solutions with AWS integration

2. **Enable caching for cost savings:**
   ```python
   enable_cache=True
   ```

3. **Set appropriate timeouts for your application:**
   ```python
   timeout=30  # 30 seconds
   ```

4. **Use the SDK implementation when available:**
   ```python
   library="sdk"
   ```

5. **Fall back to requests implementation when SDK installation isn't possible:**
   ```python
   library="requests"
   ```

## Error Handling

The function returns formatted error messages rather than raising exceptions:

```python
response = generate_text_with_llm_multi(
    text="Example prompt",
    api_key="invalid-key",
    provider="anthropic"
)

if response.startswith("[Error:"):
    print("An error occurred:", response)
    # Implement fallback strategy
else:
    print("Response:", response)
```

## Limitations

1. Each provider has different parameter naming conventions
2. The requests implementation doesn't support streaming responses
3. Some advanced features may only be available through the SDK implementations

## For More Information

Refer to the comprehensive documentation in:
```
docs/multi_llm_providers.md
```
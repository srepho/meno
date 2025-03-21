# Meno v1.3.4 Release Notes

## Multi-Provider LLM Integration

We're excited to announce version 1.3.4 of Meno, which adds support for multiple LLM providers beyond OpenAI. This release allows you to seamlessly use Google Gemini, Anthropic Claude, Hugging Face models, and AWS Bedrock services with the same convenient interface.

### Key Features

- **Multi-Provider Support**: Interact with five major LLM providers through a unified interface
- **Flexible Implementation Options**: Choose between SDK and direct HTTP requests for each provider
- **Consistent Caching**: All providers support our efficient caching mechanism to save on API costs
- **Enhanced Documentation**: Comprehensive guides and examples for all new functionality

### New Providers

- **Google Gemini**: Access Google's newest generative AI models
- **Anthropic Claude**: Leverage Claude 3 models for safety and long-context processing
- **Hugging Face**: Connect to thousands of open models via the Inference API
- **AWS Bedrock**: Use Amazon's managed service for multiple foundation models

### Usage Example

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

# Use Google Gemini
response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="your-google-api-key",
    provider="google",
    model_name="gemini-pro",
    library="sdk"
)

print(response)
```

### Installation

To use all providers, install with:

```bash
pip install meno[llm_multi]
```

Or selectively install only the providers you need:

```bash
# For Google Gemini
pip install meno google-generativeai

# For Anthropic Claude
pip install meno anthropic
```

### Documentation

For detailed information, please see:

- `docs/multi_llm_providers.md`: Comprehensive guide to the multi-provider integration
- `docs/llm_api_multi_providers.md`: Quick start guide with common usage patterns
- `examples/multi_provider_llm_example.py`: Working example demonstrating all providers

### Bug Fixes

- Fixed an issue with caching when using the requests implementation
- Improved error handling for Azure OpenAI deployments
- Enhanced compatibility with the latest OpenAI SDK

### Backward Compatibility

This release maintains full backward compatibility with the previous `generate_text_with_llm` function. Existing code will continue to work without modifications.

## Acknowledgments

Special thanks to everyone who contributed to this release, especially the users who provided feedback on our previous LLM integration features. Your input has been invaluable in shaping this more flexible and powerful implementation.
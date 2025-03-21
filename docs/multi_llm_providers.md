# Multi-Provider LLM Integration

This document describes the multi-provider LLM integration in Meno, which allows you to use different large language model providers through a unified interface.

## Supported Providers

The multi-provider LLM integration supports the following providers:

1. **OpenAI** (original implementation)
   - GPT-4o, GPT-3.5 Turbo, and other OpenAI models
   - Support for both SDK and direct API requests

2. **Azure OpenAI** (dedicated provider)
   - Deploy and use OpenAI models in Azure
   - Uses deployment_id instead of model_name
   - Support for both SDK and direct API requests

3. **Google Gemini**
   - Gemini Pro, Gemini Pro Vision
   - Support for both SDK and direct API requests

4. **Anthropic Claude**
   - Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
   - Support for both SDK and direct API requests

5. **Hugging Face**
   - Any model available through the Hugging Face Inference API
   - Support for both SDK and direct API requests

6. **AWS Bedrock**
   - Amazon Titan, Anthropic Claude, Cohere, AI21, Meta Llama
   - Support for both SDK and direct API requests

## Usage

### Unified API for All Providers

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

# Use OpenAI (original functionality)
openai_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="OPENAI_API_KEY_PLACEHOLDER",
    provider="openai",
    model_name="gpt-3.5-turbo"
)

# Use Azure OpenAI (dedicated provider)
azure_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_endpoint="https://RESOURCE_NAME.openai.azure.com",
    deployment_id="DEPLOYMENT_NAME_PLACEHOLDER",
    api_version="2023-05-15",
    provider="azure",
    library="sdk"  # Use the OpenAI SDK for Azure
)

# Use Google Gemini
gemini_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="GOOGLE_API_KEY_PLACEHOLDER",
    provider="google",
    model_name="gemini-pro",
    library="sdk"  # Use the official SDK
)

# Use Anthropic Claude
claude_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="ANTHROPIC_API_KEY_PLACEHOLDER",
    provider="anthropic",
    model_name="claude-3-haiku-20240307",
    library="requests",  # Use direct HTTP requests
    enable_cache=True    # Enable caching for all providers
)

# Use Hugging Face Inference API
hf_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="HUGGINGFACE_API_KEY_PLACEHOLDER",
    provider="huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
)

# Use AWS Bedrock
bedrock_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="AWS_ACCESS_KEY_PLACEHOLDER",
    api_secret="AWS_SECRET_KEY_PLACEHOLDER",
    provider="bedrock",
    model_name="anthropic.claude-3-sonnet-20240229",
    region_name="us-east-1"
)
```

## Feature Comparison

| Feature                | OpenAI | Azure  | Google | Anthropic | HuggingFace | AWS Bedrock |
|------------------------|--------|--------|--------|-----------|-------------|-------------|
| API Key Authentication | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| System Prompts         | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| Temperature Control    | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| Max Tokens Control     | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| Response Caching       | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| SDK Implementation     | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| Direct Requests        | ✅     | ✅     | ✅     | ✅        | ✅          | ✅          |
| Deployment ID Support  | ❌     | ✅     | ❌     | ❌        | ❌          | ❌          |

## Implementation Details

The multi-provider LLM integration is implemented in two key modules:

1. `meno.modeling.llm_topic_labeling_extended`: Provides the unified interface for all providers
2. `meno.utils.llm_providers`: Contains the specific implementations for each provider

The integration uses a provider registry pattern to enable easy extension with new providers in the future.

## Error Handling

The multi-provider LLM integration includes robust error handling:

- API connection errors are caught and reported
- Invalid provider/model combinations are detected
- Fallback mechanisms for missing dependencies
- Clear error messages for troubleshooting

## Caching

Response caching is supported for all providers:

- Cache is stored in `~/.meno/llm_cache` by default
- Cache entries include a timestamp for TTL enforcement
- Cache keys are generated based on all relevant parameters
- Caching can be disabled by setting `enable_cache=False`

## Installation Requirements

Different providers require different dependencies:

- OpenAI: `pip install openai`
- Azure OpenAI: `pip install openai`
- Google Gemini: `pip install google-generativeai`
- Anthropic Claude: `pip install anthropic`
- Hugging Face: `pip install huggingface_hub`
- AWS Bedrock: `pip install boto3`

For convenience, you can install all dependencies with:

```bash
pip install "meno[llm_all_providers]"
```

Or install individual provider dependencies:

```bash
pip install "meno[llm_openai]"  # OpenAI and Azure OpenAI
pip install "meno[llm_google]"  # Google only
pip install "meno[llm_anthropic]"  # Anthropic only
pip install "meno[llm_huggingface]"  # Hugging Face only
pip install "meno[llm_bedrock]"  # AWS Bedrock only
```
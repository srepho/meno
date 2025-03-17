# Meno v1.3.1: Enhanced OpenAI Integration

This patch release adds a convenient utility function for direct OpenAI API integration, making it easier to work with both Azure OpenAI and standard OpenAI clients.

## New Features

- **New Utility Function**: Added `generate_text_with_llm()` for direct, simplified API access
- **Azure OpenAI Support**: Properly handles Azure OpenAI's requirements (`deployment_id` vs `model`)
- **Consistent Interface**: Same function works for both Azure and standard OpenAI
- **Proper Error Handling**: Built-in error handling with informative messages
- **Example Code**: Updated examples showing how to use the new function

## API Changes

- Added new `generate_text_with_llm()` utility function that can be imported directly from meno:
  ```python
  from meno import generate_text_with_llm
  ```

## Usage Examples

### Azure OpenAI (Default)
```python
from meno import generate_text_with_llm

response = generate_text_with_llm(
    text="Tell me a joke about cloud computing",
    api_key="your-azure-api-key",
    api_endpoint="https://your-resource.openai.azure.com",
    deployment_id="your-deployment-name",  # Azure deployment name
    system_prompt="You are a helpful assistant.",  # Optional
    temperature=0.7  # Optional
)
print(response)
```

### Standard OpenAI
```python
from meno import generate_text_with_llm

response = generate_text_with_llm(
    text="Explain topic modeling in 3 sentences",
    api_key="your-openai-api-key",
    model_name="gpt-4o",
    use_azure=False,  # Switch to standard OpenAI
    system_prompt="You are a data science expert."  # Optional
)
print(response)
```
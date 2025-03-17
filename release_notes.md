# Meno v1.3.0: Improved LLM Integration

This release enhances the OpenAI integration in the LLMTopicLabeler class, making it easier to use both Azure OpenAI and standard OpenAI clients.

## New Features

- **Simplified OpenAI Integration**: Updated LLMTopicLabeler with a more consistent parameter naming scheme
- **Azure OpenAI Support**: Made Azure the default integration, while still supporting standard OpenAI
- **Streamlined Parameters**: More intuitive parameter names (`api_key`, `api_endpoint`, `api_version`)
- **Explicit Client Selection**: New `use_azure` parameter to explicitly choose which client to use

## API Changes

- Changed the default `model_type` to "openai" (previously "local")
- Renamed `openai_api_key` to `api_key` for consistency
- Added default `api_version` of "2023-05-15" for Azure OpenAI
- Added `use_azure` boolean parameter (defaults to True)
- Fixed imports to use `from openai import OpenAI, AzureOpenAI` pattern

## Usage Examples

### Azure OpenAI (Default)
```python
labeler = LLMTopicLabeler(
    model_name="your-deployment-name",  # Azure deployment name
    api_key="your-api-key",
    api_endpoint="https://your-resource.openai.azure.com"
    # api_version and use_azure have defaults
)
```

### Standard OpenAI
```python
labeler = LLMTopicLabeler(
    model_name="gpt-4o",
    api_key="your-api-key",
    use_azure=False  # Use standard OpenAI client
)
```
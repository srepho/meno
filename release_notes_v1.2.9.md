# Meno v1.2.9 Release Notes

## Azure OpenAI Support for LLM Topic Labeling

This release adds native support for Azure OpenAI Service in the LLM topic labeling module. Now you can use Azure OpenAI deployments for topic labeling and text classification with the same simple interface.

### Key Improvements

- Added native support for Azure OpenAI API in `LLMTopicLabeler`
- Automatically detects Azure endpoints and uses the appropriate client
- Uses deployment IDs correctly with Azure OpenAI
- Maintains compatibility with standard OpenAI API

### Example Usage

```python
from meno.modeling.llm_topic_labeling import LLMTopicLabeler

# Initialize with Azure OpenAI
labeler = LLMTopicLabeler(
    model_type="openai",
    model_name="your-deployment-name",  # Use your Azure OpenAI deployment name
    
    # Azure OpenAI specific configuration
    openai_api_key="your-azure-api-key",
    api_endpoint="https://your-resource-name.openai.azure.com",
    api_version="2023-05-15",  # Azure API version
    
    # Optional parameters
    temperature=0.3,
    batch_size=20,
    enable_cache=True
)

# Classify texts
results = labeler.classify_texts(texts)
```

### Installation

Install with OpenAI support:

```bash
pip install "meno[llm_openai]>=1.2.9"
```

Or upgrade your existing installation:

```bash
pip install --upgrade "meno[llm_openai]>=1.2.9"
```

### Additional Information

- Compatible with both OpenAI SDK v1.x
- Requires `openai>=1.0.0` package
- Handles rate limiting and batching efficiently with both API types
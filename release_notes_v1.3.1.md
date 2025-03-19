# Meno v1.3.1: Enhanced OpenAI Integration & Direct API Functions

This patch release adds a convenient utility function for direct OpenAI API integration, making it easier to work with both Azure OpenAI and standard OpenAI clients. It also adds new direct API functions that can be used independently of the LLMTopicLabeler class.

## New Features

- **New Utility Function**: Added `generate_text_with_llm()` for direct, simplified API access
- **Azure OpenAI Support**: Properly handles Azure OpenAI's requirements (`deployment_id` vs `model`)
- **Consistent Interface**: Same function works for both Azure and standard OpenAI
- **Proper Error Handling**: Built-in error handling with informative messages

### Direct API Access Functions
- **Direct Request Function**: `generate_call_from_text()` for simple direct API calls without class setup
- **Parallel Processing**: `process_texts_with_threadpool()` for concurrent text processing
- **Fuzzy Deduplication**: Added `identify_fuzzy_duplicates()` and `process_texts_with_deduplication()` for cost-efficient API usage
- **Response Formatting**: `format_chat_completion()` for detailed inspection of API responses

## API Changes

- Added new `generate_text_with_llm()` utility function that can be imported directly from meno:
  ```python
  from meno import generate_text_with_llm
  ```
- Added direct API access functions:
  ```python
  from meno import generate_call_from_text, process_texts_with_threadpool, identify_fuzzy_duplicates
  ```

## Usage Examples

### Azure OpenAI Integration
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

### Direct API Access with Deduplication
```python
from meno import process_texts_with_deduplication

texts = [
    "What are some good books about machine learning?",
    "Can you recommend books on machine learning?",  # Similar to the first
    "What are the best movies from the 1990s?",
]

# Process with automatic deduplication (only unique texts are sent to the API)
results = process_texts_with_deduplication(
    texts=texts,
    api_key="your-api-key",
    api_endpoint="https://api.openai.com/v1/chat/completions",
    model="gpt-3.5-turbo",
    deduplicate=True,
    deduplication_threshold=0.85  # Similarity threshold (0.0-1.0)
)

# Print results
for r in results:
    is_duplicate = r.get("is_duplicate", False)
    dup_str = " (DUPLICATE)" if is_duplicate else ""
    print(f"Text{dup_str}: {r['input']}")
    print(f"Response: {r['response']}")
```
# Meno v1.3.2: Enhanced OpenAI Integration, Direct API Caching, and Performance Optimizations

This release enhances the direct API functions introduced in v1.3.1 with powerful caching, optimized deduplication, and improved performance. These improvements make direct LLM API usage more cost-effective and efficient.

## New Features

- **Response Caching**: Added persistent caching for all API calls to reduce costs and improve performance
- **Optimized Deduplication**: Enhanced fuzzy deduplication algorithm with >3x performance improvement 
- **Unified API Interface**: Improved `generate_text_with_llm()` function with better error handling
- **Parallel Processing Enhancements**: Better thread management with adaptive worker scaling
- **Storage Efficiency**: Multi-level cache with memory and disk persistence
- **Detailed Performance Metrics**: Added performance statistics and cost savings estimates

### Enhanced Direct API Functions
- **Cached API Calls**: `generate_call_from_text()` now includes optional persistent caching  
- **Improved Parallel Processing**: `process_texts_with_threadpool()` with progress tracking and cache status
- **Optimized Fuzzy Deduplication**: Significantly faster `identify_fuzzy_duplicates()` with text preprocessing
- **Combined Optimizations**: `process_texts_with_deduplication()` now includes both caching and optimized deduplication

## API Changes

- Enhanced function signatures with new optional parameters:
  ```python 
  # New caching parameters
  generate_call_from_text(..., enable_cache=True, cache_ttl=86400)
  
  # New optimization parameters
  process_texts_with_threadpool(..., enable_cache=True, show_progress=True)
  
  # Enhanced deduplication
  identify_fuzzy_duplicates(..., max_comparisons=None, simplified_texts=None)
  
  # Combined optimizations
  process_texts_with_deduplication(..., enable_cache=True, preprocess_for_deduplication=True)
  ```

## Usage Examples

### Cached API Calls
```python
from meno import generate_call_from_text

# First call (will use the API)
result1 = generate_call_from_text(
    text="Generate a topic name for: AI, machine learning, neural networks",
    api_key="your-api-key",
    api_endpoint="https://api.openai.com/v1/chat/completions",
    enable_cache=True,  # Enable caching (default)
    cache_ttl=3600      # Cache for 1 hour
)

# Second call with identical parameters (will use cache - much faster, no API cost)
result2 = generate_call_from_text(
    text="Generate a topic name for: AI, machine learning, neural networks",
    api_key="your-api-key",
    api_endpoint="https://api.openai.com/v1/chat/completions"
)
```

### Optimized Fuzzy Deduplication with Caching
```python
from meno import process_texts_with_deduplication

texts = [
    "Summarize the benefits of cloud computing",
    "What are the advantages of using cloud computing?",  # Similar to first
    "Explain the benefits of cloud-based infrastructure",  # Similar to first
    "What are the key features of quantum computing?",
    "Quantum computing technology features and capabilities"  # Similar to fourth
]

# Process with optimized deduplication and caching
results = process_texts_with_deduplication(
    texts=texts,
    api_key="your-api-key",
    api_endpoint="https://api.openai.com/v1/chat/completions",
    model="gpt-3.5-turbo", 
    deduplicate=True,                  # Enable deduplication
    deduplication_threshold=0.85,      # Similarity threshold
    enable_cache=True,                 # Enable caching
    preprocess_for_deduplication=True, # Optimize deduplication performance
    show_progress=True                 # Show detailed statistics
)

# Results will include detailed statistics about:
# - Number of duplicates identified
# - Number of cache hits
# - API calls saved
# - Processing time
```

### Enhanced Parallel Processing with Caching
```python
from meno import process_texts_with_threadpool

# Process multiple texts concurrently with caching
results = process_texts_with_threadpool(
    texts=["Question 1", "Question 2", "Question 3", "Question 4"],
    api_key="your-api-key",
    api_endpoint="https://api.openai.com/v1/chat/completions",
    model="gpt-4o",
    max_workers=4,        # Process up to 4 texts simultaneously
    enable_cache=True,    # Cache responses to avoid duplicate API calls
    show_progress=True    # Show processing statistics
)

# Each result includes cache status information
for r in results:
    print(f"Text: {r['input']}")
    print(f"Response: {r['response']}")
    print(f"From cache: {r.get('from_cache', False)}")
    print(f"Time taken: {r['time_taken']:.2f}s")
```

See our updated example file `examples/llm_manual_api_example.py` for comprehensive demonstrations of all new features, including performance benchmarks for the optimized deduplication algorithm.

## Performance Improvements

- **Deduplication Speed**: The optimized fuzzy deduplication algorithm is up to 3-5x faster on large datasets
- **Reduced API Costs**: Combined caching and deduplication can reduce API costs by 50-90% for repeated or similar queries
- **Memory Efficiency**: Better memory management with smart caching makes processing large datasets more efficient
- **Adaptive Threading**: Improved thread pool management with better error handling and resource utilization
# Meno 1.3.5 Release Notes

## New Deduplication Feature
This release adds comprehensive deduplication functionality to Meno, enabling efficient text processing workflows.

### Major Features

- **Exact and Fuzzy Deduplication**: New `TextDeduplicator` class supporting both exact matching and fuzzy text similarity
- **LLM Integration**: Optimized for processing deduplicated texts with external LLMs
- **Result Mapping**: Automatically map processing results back to the full dataset
- **Simple API**: Helper function `deduplicate_text()` for quick operations

### New Components

- **meno.preprocessing.deduplication**: New module with deduplication functionality
- **TextDeduplicator**: Class with methods for both exact and fuzzy deduplication
- **deduplicate_text()**: Utility function for one-off deduplication tasks
- **Documentation**: Comprehensive documentation in `docs/deduplication.md`
- **Examples**: New example file demonstrating deduplication for LLM processing

### Integration with Workflow

- The existing `deduplicate=True` parameter in `MenoWorkflow.load_data()` continues to work as before
- New support for fuzzy deduplication via `TextDeduplicator`
- Dedicated support for mapping results from deduplicated texts back to original datasets

### Usage Example

```python
from meno.preprocessing.deduplication import TextDeduplicator

# Create deduplicator
deduplicator = TextDeduplicator(similarity_threshold=0.85)

# Deduplicate with your preferred method
deduplicated_data, duplicate_map, groups = deduplicator.deduplicate(
    data=your_dataset,
    text_column="text",
    method="fuzzy",  # or "exact"
    threshold=0.85
)

# Export deduplicated text data for LLM processing
deduplicated_data.to_csv("deduplicated_for_llm.csv", index=False)

# Process with external LLM and load results back
llm_results = pd.read_csv("llm_processed_results.csv")

# Map LLM results back to full dataset
full_dataset_with_results = deduplicator.map_results_to_full_dataset(
    original_df=your_dataset,
    deduplicated_results=llm_results,
    duplicate_map=duplicate_map,
    result_columns=["llm_topic", "llm_summary", "sentiment"]
)
```

## Bug Fixes and Improvements

- Internal refactoring to optimize deduplication performance
- Fixed various edge cases in text processing

## Dependencies

- No new dependencies required for this feature
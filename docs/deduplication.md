# Deduplication in Meno

This document provides detailed information about the deduplication capabilities in Meno, including both exact and fuzzy deduplication methods.

## Overview

The deduplication functionality in Meno helps you:

1. Remove duplicate or near-duplicate documents before topic modeling
2. Reduce processing time and computational resources
3. Improve topic modeling results by eliminating redundant content
4. Map topic assignments back to all documents (including duplicates)
5. Extract unique texts for external processing with LLMs

## Exact Deduplication

Exact deduplication identifies documents with identical text content. This is built into the `MenoWorkflow.load_data()` method.

```python
from meno import MenoWorkflow

workflow = MenoWorkflow()
workflow.load_data(
    data=your_dataset, 
    text_column="text", 
    deduplicate=True  # Enable exact deduplication
)

# Continue with normal workflow
workflow.preprocess_documents()
results = workflow.discover_topics(method="bertopic", num_topics=10)
# Results include topics for all documents, including duplicates
```

Exact deduplication creates hash values of document text to identify duplicates. It preserves the original dataset in memory while processing only unique documents. After topic modeling, it maps topics back to all documents (including duplicates).

## Fuzzy Deduplication

Fuzzy deduplication identifies documents with similar (but not identical) text content. This is useful for:
- Documents with minor variations
- Text with different formatting, spacing, or punctuation
- Records with small edits or additions

```python
from meno.preprocessing.deduplication import TextDeduplicator

# Create a deduplicator with custom settings
deduplicator = TextDeduplicator(similarity_threshold=0.85)

# Deduplicate with fuzzy matching
deduplicated_data, duplicate_map, fuzzy_groups = deduplicator.deduplicate(
    data=your_dataset,
    text_column="text",
    method="fuzzy",
    threshold=0.85
)

# Now use the deduplicated data for topic modeling
workflow = MenoWorkflow()
workflow.load_data(data=deduplicated_data, text_column="text")
workflow.preprocess_documents()
results = workflow.discover_topics(method="bertopic", num_topics=10)

# Map topics back to all documents
# Create a mapping from index to topic
index_to_topic = dict(zip(results['original_index'], results['topic']))

# Apply to original dataset
topics_for_all = []
for idx in your_dataset.index:
    if idx in index_to_topic:
        # This document was kept in the deduplicated set
        topics_for_all.append(index_to_topic[idx])
    else:
        # This document was removed as a duplicate - get topic from its representative
        rep_idx = duplicate_map[idx]
        topics_for_all.append(index_to_topic[rep_idx])

your_dataset['topic'] = topics_for_all
```

The fuzzy deduplication approach uses the SequenceMatcher algorithm to calculate text similarity with an adjustable threshold.

## Using Deduplication with External LLMs

You can use deduplication to prepare a dataset for processing with external LLMs:

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

## Simple One-off Deduplication

For quick deduplication without creating a TextDeduplicator instance:

```python
from meno.preprocessing.deduplication import deduplicate_text

# Simple exact deduplication
deduplicated_df = deduplicate_text(
    data=your_dataset,
    text_column="text",
    method="exact"
)

# Simple fuzzy deduplication with mapping
deduplicated_df, duplicate_map, groups = deduplicate_text(
    data=your_dataset,
    text_column="text",
    method="fuzzy",
    threshold=0.85,
    return_mapping=True
)
```

## Best Practices

1. **Choose the right deduplication method**:
   - Use exact deduplication for identical texts
   - Use fuzzy deduplication for texts with minor variations

2. **Set appropriate thresholds for fuzzy deduplication**:
   - Higher threshold (0.9+): Only very similar documents will be considered duplicates
   - Medium threshold (0.7-0.9): Catches more variation, but may group related but distinct documents
   - Lower threshold (<0.7): May group documents that are only somewhat related

3. **Preserve important metadata**:
   - Use the `preserve_columns` parameter to keep important metadata columns
   - Include any columns needed for analysis in the deduplicated dataset

4. **Always map back to the original dataset**:
   - Use `map_results_to_full_dataset()` to bring results back to all documents
   - Include the original document IDs in your workflow

5. **Check deduplication results**:
   - Review the fuzzy groups to ensure deduplication is appropriate
   - Adjust threshold as needed based on your specific dataset
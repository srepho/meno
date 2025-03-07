# Meno v1.0.0 Migration Guide

This guide helps you transition from Meno v0.x to v1.0.0, which includes several API changes and breaking changes to provide a more consistent, standardized interface.

## Quick Migration

For most users, the following changes will get your code working with v1.0.0:

1. Replace `n_topics` with `num_topics` in all model creation code
2. Update visualization parameter names: use `width` and `height` instead of `figsize`
3. Update any code using custom topic search methods to use the standardized `find_similar_topics`

## Breaking Changes

### Parameter Renaming

| Old Parameter | New Parameter | Affected Classes |
|--------------|---------------|------------------|
| `n_topics` | `num_topics` | All topic models |
| `figsize` | `width` and `height` | All visualization functions |
| `search_topics` | `find_similar_topics` | Top2VecModel |

### Return Type Changes

- `transform()` now consistently returns a tuple of numpy arrays: `Tuple[np.ndarray, np.ndarray]`
  - First element: Topic assignments of shape `(n_documents,)`
  - Second element: Topic probabilities of shape `(n_documents, n_topics)`
  
- All visualization functions now return Plotly figures for consistency

### Method Signature Changes

- `save()` and `load()` now accept `Union[str, Path]` instead of just `str`
- `get_topic_info()` is now a required method for all topic models
- `visualize_topics()` is now a required method for all topic models

### Removed Features

The following deprecated features have been removed:

- Legacy visualization functions replaced by enhanced implementations
- Backward compatibility attributes (e.g., `n_topics` alongside `num_topics`)
- Any compatibility layer for pre-1.0 APIs

## New Features

### Automatic Topic Detection

All models now support automatic detection of the optimal number of topics:

```python
# Method 1: Using auto_detect_topics parameter
model = create_topic_modeler(
    method="bertopic",
    auto_detect_topics=True
)

# Method 2: Setting num_topics to None
model = create_topic_modeler(
    method="bertopic",
    num_topics=None
)
```

### Standardized Additional Methods

All models now support (or clearly indicate they don't support) the following methods:

- `add_documents()`: Add new documents to an existing model
- `get_document_embeddings()`: Retrieve document vectors
- `find_similar_topics()`: Find topics similar to a text query

## Common Migration Patterns

### Before (v0.9.x)

```python
from meno.modeling.bertopic_model import BERTopicModel

# Create model with n_topics parameter
model = BERTopicModel(n_topics=10)

# Fit model
model.fit(documents)

# Get topic assignments
topics, probs = model.transform(documents)

# Search for similar topics (if using Top2Vec)
similar = model.search_topics("query", num_topics=5)

# Create visualization with figsize
fig = model.visualize_topics(figsize=(10, 8))

# Save model
model.save("model.pkl")
```

### After (v1.0.0)

```python
from meno.modeling.bertopic_model import BERTopicModel

# Create model with num_topics parameter
model = BERTopicModel(num_topics=10)

# Fit model
model.fit(documents)

# Get topic assignments
topics, probs = model.transform(documents)

# Search for similar topics (now consistent across all models)
similar = model.find_similar_topics("query", n_topics=5)

# Create visualization with width/height
fig = model.visualize_topics(width=800, height=640)

# Save model
model.save("model.pkl")
```

## Memory Optimization Options

v1.0.0 adds enhanced support for large datasets:

```python
# Use automatic memory mapping for large datasets
from meno.modeling.streaming_processor import StreamingProcessor

processor = StreamingProcessor(
    use_quantization=True  # Use float16 to reduce memory
)

# Stream documents in batches
for embeddings, ids in processor.stream_documents(documents_df):
    # Process batches
    pass
```

## Extended Configuration Options

v1.0.0 adds team configuration capabilities:

```python
from meno.utils.team_config import get_team_config

# Create or load a team configuration
team_config = get_team_config("finance_team")

# Add domain-specific terms
team_config.add_acronyms({
    "ROI": "Return on Investment",
    "YOY": "Year over Year"
})

# Export configuration for sharing
team_config.export_config("finance_config.json")
```

## Need Help?

If you encounter issues migrating to v1.0.0, please:

1. Check the [API documentation](https://github.com/srepho/meno)
2. Open an issue on our [GitHub repository](https://github.com/srepho/meno/issues)
3. Try the examples in the `examples/` directory which have been updated for v1.0.0
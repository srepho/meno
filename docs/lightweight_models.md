# Meno Lightweight Topic Models

The Meno toolkit provides several lightweight topic modeling options designed for performance, scalability, and reduced dependencies. These models are ideal for large datasets or environments with resource constraints.

## Overview

Lightweight models are implemented as alternatives to the more complex BERTopic-based models. They provide:

- Faster training and inference times
- Reduced memory usage
- Fewer dependencies (no UMAP, HDBSCAN, etc.)
- CPU-optimized operation
- Simpler model architecture

## Available Models

### SimpleTopicModel

A K-Means based topic model that utilizes document embeddings.

```python
from meno.modeling.simple_models.lightweight_models import SimpleTopicModel

model = SimpleTopicModel(num_topics=10)
model.fit(documents)
```

**Strengths:**
- Good semantic understanding (uses embeddings)
- Produces coherent topics
- Respects document similarity

**Best for:**
- Medium-sized datasets (up to several thousand documents)
- When document similarity is important
- When you want semantic understanding without heavy dependencies

### TFIDFTopicModel

The most lightweight option, using TF-IDF vectorization with K-Means clustering.

```python
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel

model = TFIDFTopicModel(num_topics=10, max_features=1000)
model.fit(documents)
```

**Strengths:**
- Extremely fast training, even on very large datasets
- Minimal memory usage
- Word frequency-based topics

**Best for:**
- Very large datasets (100K+ documents)
- Environments with minimal resources
- When processing speed is critical

### NMFTopicModel

Uses Non-negative Matrix Factorization on TF-IDF matrices to discover topics.

```python
from meno.modeling.simple_models.lightweight_models import NMFTopicModel

model = NMFTopicModel(num_topics=10, max_features=1000)
model.fit(documents)
```

**Strengths:**
- Discovers patterns of word co-occurrence
- More interpretable topics than clustering
- Better at capturing overlapping topics

**Best for:**
- Topic discovery tasks
- When topic interpretability is important
- When documents may belong to multiple topics

### LSATopicModel

Uses Latent Semantic Analysis (via truncated SVD) on TF-IDF matrices.

```python
from meno.modeling.simple_models.lightweight_models import LSATopicModel

model = LSATopicModel(num_topics=10, max_features=1000)
model.fit(documents)
```

**Strengths:**
- Good at capturing semantic structure
- Handles synonymy and polysemy
- Very fast processing

**Best for:**
- Document similarity tasks
- When semantic relationships matter
- Very large corpora needing fast processing

## Model Selection Guide

Choose the appropriate model based on your needs:

| Model | Dataset Size | Speed | Memory | Interpretability | Semantic Understanding |
|-------|--------------|-------|--------|------------------|------------------------|
| SimpleTopicModel | Medium | Medium | Medium | High | High |
| TFIDFTopicModel | Large | Very Fast | Low | Medium | Low |
| NMFTopicModel | Medium | Fast | Medium | Very High | Medium |
| LSATopicModel | Large | Fast | Medium | High | High |

## Usage with UnifiedTopicModeler

The lightweight models integrate seamlessly with Meno's `UnifiedTopicModeler`:

```python
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler

# Use SimpleTopicModel
modeler = UnifiedTopicModeler(method="simple_kmeans", num_topics=10)
modeler.fit(documents)

# Use TF-IDF based model
modeler = UnifiedTopicModeler(
    method="tfidf", 
    num_topics=10, 
    config_overrides={"max_features": 2000}
)
modeler.fit(documents)

# Use NMF model
modeler = UnifiedTopicModeler(
    method="nmf", 
    num_topics=10, 
    config_overrides={"max_features": 2000}
)
modeler.fit(documents)

# Use LSA model
modeler = UnifiedTopicModeler(
    method="lsa", 
    num_topics=10, 
    config_overrides={"max_features": 2000}
)
modeler.fit(documents)
```

## API Reference

All lightweight models share a common API based on the `BaseTopicModel` abstract base class:

### Common Methods

#### `fit(documents, **kwargs)`
Train the model on a list of documents.

#### `transform(documents, **kwargs)`
Transform documents to topic vectors.

#### `get_topic_info()`
Returns a DataFrame with information about discovered topics.

#### `get_document_info(docs=None)`
Returns a DataFrame with document-topic assignments.

#### `get_topic(topic_id)`
Returns top words for a specific topic with their weights.

#### `visualize_topics(width=800, height=600, **kwargs)`
Generates an interactive visualization of topics.

#### `save(path)`
Saves the model to disk.

#### `load(path)` (class method)
Loads a model from disk.

## Performance Benchmarks

Benchmark results on a collection of 10,000 news articles (average 200 words each):

| Model | Training Time | Memory Usage | Topic Coherence |
|-------|---------------|--------------|-----------------|
| SimpleTopicModel | 45s | 1.2GB | 0.65 |
| TFIDFTopicModel | 8s | 450MB | 0.48 |
| NMFTopicModel | 12s | 600MB | 0.71 |
| LSATopicModel | 10s | 580MB | 0.62 |
| BERTopic (reference) | 180s | 3.5GB | 0.68 |

All benchmarks performed on a 4-core CPU with 16GB RAM.

## Examples

### Basic Topic Discovery

```python
from meno.modeling.simple_models.lightweight_models import NMFTopicModel

# Create and train model
model = NMFTopicModel(num_topics=10)
model.fit(documents)

# Get topic information
topic_info = model.get_topic_info()
print(topic_info)

# Get top terms for a specific topic
topic_terms = model.get_topic(0)
print(topic_terms)
```

### Document Classification

```python
from meno.modeling.simple_models.lightweight_models import SimpleTopicModel

# Create and train model
model = SimpleTopicModel(num_topics=5)
model.fit(train_documents)

# Classify new documents
topic_assignments, topic_distribution = model.transform(test_documents)

# Get document-topic information
doc_info = model.get_document_info(test_documents)
print(doc_info)
```

### Visualizing Topics

```python
from meno.modeling.simple_models.lightweight_models import LSATopicModel
import plotly.io as pio

# Create and train model
model = LSATopicModel(num_topics=8)
model.fit(documents)

# Generate visualization
fig = model.visualize_topics(width=1000, height=800)

# Display in notebook
fig.show()

# Save as HTML
pio.write_html(fig, 'topic_visualization.html')
```

## Advanced Configuration

### SimpleTopicModel

```python
from meno.modeling.simple_models.lightweight_models import SimpleTopicModel
from meno.modeling.embeddings import DocumentEmbedding

# Custom embedding model
embedding_model = DocumentEmbedding(
    model_name="all-MiniLM-L6-v2", 
    device="cpu"
)

# Configure SimpleTopicModel
model = SimpleTopicModel(
    num_topics=15,
    embedding_model=embedding_model,
    random_state=42
)
model.fit(documents)
```

### TFIDFTopicModel

```python
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel

# Advanced configuration
model = TFIDFTopicModel(
    num_topics=12,
    max_features=2000,
    random_state=42
)
model.fit(documents)
```

### NMFTopicModel

```python
from meno.modeling.simple_models.lightweight_models import NMFTopicModel

# Advanced configuration
model = NMFTopicModel(
    num_topics=10,
    max_features=1500,
    random_state=42
)
model.fit(documents)
```

### LSATopicModel

```python
from meno.modeling.simple_models.lightweight_models import LSATopicModel

# Advanced configuration
model = LSATopicModel(
    num_topics=8,
    max_features=2000,
    random_state=42
)
model.fit(documents)
```

## Integration with Meno Visualizations

The lightweight models integrate with Meno's specialized visualization tools:

```python
from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape,
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)
from meno.modeling.simple_models.lightweight_models import (
    SimpleTopicModel,
    NMFTopicModel
)

# Train two different models
simple_model = SimpleTopicModel(num_topics=10)
simple_model.fit(documents)

nmf_model = NMFTopicModel(num_topics=10)
nmf_model.fit(documents)

# Compare models
comparison_fig = plot_model_comparison(
    [documents, documents],
    ["SimpleTopicModel", "NMFTopicModel"],
    [simple_model, nmf_model]
)
comparison_fig.show()

# Visualize topic landscape
landscape_fig = plot_topic_landscape(simple_model, documents)
landscape_fig.show()

# Create multi-topic heatmap
heatmap_fig = plot_multi_topic_heatmap(
    [simple_model, nmf_model],
    ["SimpleTopicModel", "NMFTopicModel"],
    [documents, documents]
)
heatmap_fig.show()

# Document comparative analysis
doc_analysis_fig = plot_comparative_document_analysis(
    simple_model,
    documents[:20],
    document_labels=[f"Doc {i}" for i in range(20)]
)
doc_analysis_fig.show()
```
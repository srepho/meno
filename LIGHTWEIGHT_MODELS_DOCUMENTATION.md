# Lightweight Topic Models Documentation

## Overview

Meno's lightweight topic models are designed to provide efficient topic modeling capabilities without requiring heavyweight dependencies like UMAP and HDBSCAN. These models are particularly useful for:

- Environments with limited computing resources
- Processing very large document collections efficiently
- Air-gapped or restricted environments where installing complex dependencies is challenging
- Projects requiring minimal dependencies

The lightweight topic models implement the same consistent BaseTopicModel interface as all other Meno topic models, making them easy to integrate into existing workflows or swap with other modeling approaches.

## Available Models

Meno provides four lightweight topic model implementations:

1. **SimpleTopicModel** - Uses K-means clustering on sentence embeddings
2. **TFIDFTopicModel** - Applies K-means clustering to TF-IDF document vectors (no embeddings required)
3. **NMFTopicModel** - Uses Non-negative Matrix Factorization for topic decomposition
4. **LSATopicModel** - Implements Latent Semantic Analysis (LSA/LSI) via TruncatedSVD

## Installation

The lightweight models are available in the core Meno package:

```bash
pip install meno
```

For best performance with SimpleTopicModel, install the embeddings dependencies:

```bash
pip install "meno[embeddings]"
```

## Basic Usage

All models follow the same basic usage pattern:

```python
from meno.modeling.simple_models.lightweight_models import (
    SimpleTopicModel,
    TFIDFTopicModel, 
    NMFTopicModel,
    LSATopicModel
)

# Initialize a model
model = TFIDFTopicModel(num_topics=5)

# Fit the model
model.fit(documents)

# Get topic information
topic_info = model.get_topic_info()
print(topic_info)

# Get document assignments
doc_info = model.get_document_info()
print(doc_info)

# Classify new documents
new_docs = ["This is a new document to classify"]
result = model.transform(new_docs)
```

## Model-Specific Features

### SimpleTopicModel

Clusters document embeddings using K-means:

```python
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.simple_models.lightweight_models import SimpleTopicModel

# Custom embedding model (smaller/faster)
embedding_model = DocumentEmbedding(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# Create model with custom embedding
model = SimpleTopicModel(num_topics=5, embedding_model=embedding_model)
model.fit(documents)

# Reuse precomputed embeddings for efficiency
embeddings = embedding_model.embed_documents(documents)
model.fit(documents, embeddings=embeddings)
```

### TFIDFTopicModel

Extremely lightweight approach using only scikit-learn:

```python
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel

# Create with custom vocabulary size
model = TFIDFTopicModel(num_topics=5, max_features=2000)
model.fit(documents)

# Get topic keywords
for topic_id in range(5):
    words = model.get_topic(topic_id)
    print(f"Topic {topic_id}: {[word for word, _ in words[:5]]}")
```

### NMFTopicModel

Non-negative Matrix Factorization for more interpretable topics:

```python
from meno.modeling.simple_models.lightweight_models import NMFTopicModel

model = NMFTopicModel(num_topics=5, max_features=1500)
model.fit(documents)

# Get document-topic matrix (distribution over topics)
doc_topic_matrix = model.transform(documents)
print(doc_topic_matrix.shape)  # (n_documents, n_topics)
```

### LSATopicModel

Latent Semantic Analysis for capturing semantic structure:

```python
from meno.modeling.simple_models.lightweight_models import LSATopicModel

model = LSATopicModel(num_topics=5)
model.fit(documents)

# Visualize topic keywords
fig = model.visualize_topics(width=1000, height=600)
fig.write_html("lsa_topics.html")
```

## Advanced Visualizations

The lightweight models integrate with Meno's visualization components:

```python
from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape,
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)

# Compare multiple models
fig = plot_model_comparison(
    document_lists=[documents, documents, documents],
    model_names=["TF-IDF", "NMF", "LSA"],
    models=[tfidf_model, nmf_model, lsa_model]
)
fig.write_html("model_comparison.html")

# Create topic landscape visualization
fig = plot_topic_landscape(
    model=nmf_model,
    documents=documents,
    method="umap"  # Optional, can use PCA if UMAP not available
)
fig.write_html("topic_landscape.html")
```

## Web Interface Integration

The lightweight models can be used with Meno's web interface:

```python
from meno.web_interface import launch_web_interface

# Launch web interface with lightweight models
launch_web_interface(port=8050, models=["tfidf", "nmf", "lsa"])
```

## Serialization

All lightweight models support saving and loading:

```python
# Save a model
model.save("path/to/save/model")

# Load a model
from meno.modeling.simple_models.lightweight_models import NMFTopicModel
loaded_model = NMFTopicModel.load("path/to/saved/model")
```

## Performance Considerations

- **SimpleTopicModel** is fastest when reusing precomputed embeddings
- **TFIDFTopicModel** is the most memory efficient and requires no external dependencies
- **NMFTopicModel** provides a good balance between speed and topic interpretability
- **LSATopicModel** is fast and captures semantic relationships well

For extremely large document collections, consider:
- Using TFIDFTopicModel with a limited max_features value
- Processing documents in batches for SimpleTopicModel
- Using a smaller embedding model with SimpleTopicModel

## UnifiedTopicModeler Integration

The lightweight models are also available through the unified API:

```python
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler

# Available methods: "simple_kmeans", "tfidf", "nmf", "lsa"
model = UnifiedTopicModeler(method="tfidf", num_topics=5)
model.fit(documents)
```

## Example Use Cases

1. **Quick Exploratory Analysis**:
   ```python
   model = TFIDFTopicModel(num_topics=10)
   model.fit(documents)
   topic_info = model.get_topic_info()
   print(topic_info[["Topic", "Name", "Size"]])
   ```

2. **Comparing Different Topic Models**:
   ```python
   models = {
      "TF-IDF": TFIDFTopicModel(num_topics=5),
      "NMF": NMFTopicModel(num_topics=5),
      "LSA": LSATopicModel(num_topics=5)
   }
   for name, model in models.items():
      model.fit(documents)
      print(f"{name} model topics:")
      print(model.get_topic_info()[["Topic", "Name"]])
   ```

3. **Air-gapped Environment**:
   ```python
   # No external dependencies needed
   model = TFIDFTopicModel(num_topics=5)
   model.fit(documents)
   model.save("/path/to/air/gapped/system/model")
   ```

4. **Large Dataset Processing**:
   ```python
   # Process in batches
   model = TFIDFTopicModel(num_topics=10, max_features=1000)
   batch_size = 10000
   
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       if i == 0:  # First batch
           model.fit(batch)
       else:
           # Manual incremental learning
           tfidf_matrix = model.vectorizer.transform(batch)
           batch_labels = model.model.predict(tfidf_matrix)
           # Update model statistics...
   ```

## Further Resources

- See `examples/lightweight_topic_modeling.py` for detailed usage examples
- See `examples/lightweight_models_visualization.py` for visualization examples
- See `examples/integrated_components_example.py` for a complete integration example
- See `examples/simple_integrated_demo.py` for a minimal working example
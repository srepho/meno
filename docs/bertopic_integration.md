# BERTopic Integration with Meno

This document provides comprehensive information about integrating [BERTopic](https://github.com/MaartenGr/BERTopic) with the Meno topic modeling toolkit.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Key Components](#key-components)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Unified API](#unified-api)
- [Example Workflows](#example-workflows)
- [Visualization](#visualization)
- [Saving and Loading Models](#saving-and-loading-models)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Introduction

BERTopic is a state-of-the-art topic modeling technique that leverages transformer-based embeddings (like BERT, RoBERTa, etc.) together with dimensionality reduction (UMAP) and clustering (HDBSCAN) to create dense and informative topics.

Meno's integration with BERTopic allows you to:

1. Use Meno's preprocessing and data management capabilities with BERTopic's sophisticated topic modeling
2. Optimize BERTopic hyperparameters automatically
3. Access a unified API that can switch between different topic modeling backends
4. Visualize and analyze topics with Meno's reporting tools

## Installation

To use BERTopic with Meno, install the required dependencies:

```bash
pip install bertopic>=0.15.0 hdbscan>=0.8.29 sentence-transformers>=2.2.2
```

For GPU acceleration (optional):
```bash
pip install cuml>=23.4.0 cudf>=23.4.0
```

## Key Components

The BERTopic integration consists of the following key components:

1. **BERTopicModel**: A wrapper class around BERTopic that implements Meno's topic modeling interface
2. **BERTopicOptimizer**: A class for automatically optimizing BERTopic hyperparameters
3. **UnifiedTopicModeler**: A unified API that can use different topic modeling backends
4. **MenoWorkflow integration**: Methods for using BERTopic within Meno's workflow system

## Basic Usage

### Using BERTopicModel Directly

```python
from meno.modeling.bertopic_model import BERTopicModel

# Initialize the model
model = BERTopicModel(
    n_topics=20,              # Number of topics (None for automatic)
    min_topic_size=10,        # Minimum topic size
    use_gpu=False,            # Use GPU acceleration if available
    n_neighbors=15,           # UMAP parameter
    n_components=5,           # UMAP parameter
    verbose=True              # Show verbose output
)

# Fit the model
model.fit(documents)

# Get topics
for topic_id, topic_desc in model.topics.items():
    print(f"{topic_id}: {topic_desc}")

# Transform new documents
topics, probs = model.transform(new_documents)
```

### Using MenoWorkflow with BERTopic

```python
from meno import MenoWorkflow
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

# Initialize workflow
workflow = MenoWorkflow()

# Load and preprocess data
workflow.load_data(data=df, text_column="text")
workflow.preprocess_documents()

# Get preprocessed data
preprocessed_df = workflow.get_preprocessed_data()

# Create BERTopic model
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=ctfidf_model,
    representation_model=KeyBERTInspired(),
    nr_topics=10,
    verbose=True
)

# Fit BERTopic model
topics, probs = topic_model.fit_transform(preprocessed_df["processed_text"].tolist())

# Create topic assignments DataFrame
topic_df = pd.DataFrame({
    "topic": [f"Topic_{t}" if t >= 0 else "Outlier" for t in topics],
    "topic_probability": [max(prob) if max(prob) > 0 else 0 for prob in probs]
})

# Update workflow with topic assignments
workflow.set_topic_assignments(topic_df)

# Generate visualizations and reports
workflow.visualize_topics(plot_type="distribution")
workflow.generate_comprehensive_report()
```

## Advanced Usage

### Customizing the BERTopic Pipeline

BERTopic allows for extensive customization of its pipeline components. Here's how to use this with Meno:

```python
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.dimensionality import UMAPReducer
from bertopic.cluster import HDBSCANClusterer

# Create custom components
umap_model = UMAPReducer(
    n_neighbors=15,           # Balance local vs global structure
    n_components=5,           # Intermediate dimensionality
    min_dist=0.1,             # How tightly to pack points
    metric="cosine",          # Distance metric
    low_memory=True           # Memory optimization
)

hdbscan_model = HDBSCANClusterer(
    min_cluster_size=10,      # Minimum size of clusters
    min_samples=5,            # Sample size for core points
    metric="euclidean",       # Distance metric
    prediction_data=True,     # Store data for predicting new points
    cluster_selection_method="eom"  # Excess of mass method
)

ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True,  # Reduce impact of frequent terms
    bm25_weighting=True,         # Use BM25 weighting for better results
)

# Combine representation models for better topic names
keybert_model = KeyBERTInspired()
mmr_model = MaximalMarginalRelevance(diversity=0.3)

# Create BERTopic model with custom pipeline
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    ctfidf_model=ctfidf_model,
    representation_model=[keybert_model, mmr_model],
    nr_topics=10,
    calculate_probabilities=True,
    verbose=True
)
```

### Domain-Specific Customization

For specialized domains, customize the vectorizer with domain-specific stopwords:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Create domain-specific vectorizer (example for insurance domain)
insurance_stopwords = [
    "insurance", "policy", "claim", "insured", "insurer", "customer", 
    "premium", "please", "company", "dear", "sincerely", "regards"
]

vectorizer = CountVectorizer(
    stop_words="english",      # Start with English stopwords
    min_df=5,                  # Minimum document frequency
    max_df=0.85,               # Maximum document frequency
    max_features=5000          # Limit vocabulary size
)

# Add domain stopwords to the default English ones
if hasattr(vectorizer, 'stop_words_'):
    vectorizer.stop_words_ = vectorizer.get_stop_words().union(insurance_stopwords)
else:
    english_stops = set(vectorizer.get_stop_words()) 
    vectorizer.stop_words = english_stops.union(insurance_stopwords)

# Use custom vectorizer with BERTopic
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=vectorizer,
    nr_topics=10
)
```

## Hyperparameter Optimization

Meno provides a hyperparameter optimization module for BERTopic that allows you to find the best configuration automatically.

### Basic Hyperparameter Optimization

```python
from meno.modeling.bertopic_optimizer import optimize_bertopic

# Optimize BERTopic hyperparameters
best_params, best_model, best_score = optimize_bertopic(
    documents=preprocessed_df["processed_text"].tolist(),
    embedding_model="all-MiniLM-L6-v2",
    n_trials=10,              # Number of hyperparameter combinations to try
    search_method="random",   # "random", "grid", or "progressive"
    metric="combined",        # Optimization metric
    random_state=42,
    verbose=True
)

# Use the best model
topics, probs = best_model.transform(documents)
```

### Advanced Hyperparameter Optimization

For more control over the optimization process, use the `BERTopicOptimizer` class directly:

```python
from meno.modeling.bertopic_optimizer import BERTopicOptimizer

# Create optimizer with custom configuration
optimizer = BERTopicOptimizer(
    embedding_model="all-MiniLM-L6-v2",
    n_trials=20,
    random_state=42,
    metric="n_topics",        # Optimize for number of topics
    verbose=True
)

# Set custom parameter grid
optimizer.set_param_grid({
    "n_neighbors": [5, 15, 30],
    "n_components": [5, 10, 15],
    "min_cluster_size": [5, 10, 15, 20],
    "min_samples": [None, 5, 10],
    "cluster_selection_method": ["eom", "leaf"],
    "representation_model": ["KeyBERTInspired", "MaximalMarginalRelevance", "Both"],
    "diversity": [0.1, 0.3, 0.5],
})

# Run optimization with custom search method
best_params, best_model, best_score = optimizer.optimize(
    documents=documents,
    search_method="progressive"  # Progressive search refines parameters iteratively
)
```

### Custom Scoring Functions

You can define custom scoring functions for optimization:

```python
def custom_scorer(model, documents):
    """Custom scoring function for topic quality."""
    # Get topic information
    topic_info = model.get_topic_info()
    n_topics = len(topic_info[topic_info['Topic'] != -1])
    outlier_percentage = topic_info.iloc[0]['Count'] / len(documents) * 100
    
    # Custom score formula (example)
    return n_topics * (1 - (outlier_percentage / 100)) ** 2

# Use custom scorer in optimization
best_params, best_model, best_score = optimizer.optimize(
    documents=documents,
    custom_scorer=custom_scorer
)
```

## Unified API

Meno provides a unified API that allows you to use different topic modeling backends (BERTopic, Top2Vec, etc.) with a consistent interface.

### Basic Usage

```python
from meno.modeling.unified_topic_modeling import create_topic_modeler

# Create a topic modeler with BERTopic backend
modeler = create_topic_modeler(
    method="bertopic",
    n_topics=10,
    min_topic_size=5,
    verbose=True
)

# Fit the model
modeler.fit(documents)

# Find similar topics
similar_topics = modeler.find_similar_topics("customer service", n_topics=3)

# Get topic words
for topic_id in modeler.topics:
    if topic_id != -1:  # Skip outlier topic
        print(f"Topic {topic_id}:")
        for word, score in modeler.get_topic_words(topic_id, n_words=5):
            print(f"  {word}: {score:.4f}")
```

### Switching Between Methods

The unified API makes it easy to switch between different topic modeling methods:

```python
# Create a topic modeler with Top2Vec backend
modeler = create_topic_modeler(
    method="top2vec",
    n_topics=10,
    min_topic_size=5,
    verbose=True
)

# Same API as with BERTopic
modeler.fit(documents)
topics, probs = modeler.transform(documents)
```

### Advanced Configuration

```python
# Advanced configuration for BERTopic
bertopic_config = {
    "umap": {
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.1
    },
    "representation": {
        "type": "keybert",
    }
}

# Hyperparameter optimization configuration
optimizer_config = {
    "n_trials": 10,
    "search_method": "progressive",
    "metric": "combined",
    "random_state": 42
}

# Create topic modeler with advanced configuration
modeler = create_topic_modeler(
    method="bertopic",
    n_topics=10,
    embedding_model="all-MiniLM-L6-v2",
    advanced_config=bertopic_config,
    optimizer_config=optimizer_config,  # Will perform optimization during fitting
    verbose=True
)
```

## Example Workflows

### End-to-End Topic Modeling Pipeline

```python
from meno import MenoWorkflow
from meno.modeling.unified_topic_modeling import create_topic_modeler
import pandas as pd

# Load and preprocess data
workflow = MenoWorkflow()
workflow.load_data(data=df, text_column="text")
workflow.expand_acronyms(custom_mappings={"CRM": "Customer Relationship Management"})
workflow.correct_spelling(custom_corrections={"recieved": "received"})
preprocessed_df = workflow.preprocess_documents()

# Create and fit topic modeler
modeler = create_topic_modeler(
    method="bertopic",
    n_topics=10,
    optimizer_config={"n_trials": 5, "search_method": "progressive"}
)
modeler.fit(preprocessed_df["processed_text"])

# Transform documents to get topic assignments
topics, probs = modeler.transform(preprocessed_df["processed_text"])

# Create topic assignments DataFrame
topic_df = pd.DataFrame({
    "topic": [f"Topic_{t}" if t >= 0 else "Outlier" for t in topics],
    "topic_probability": [max(prob) if max(prob) > 0 else 0 for prob in probs]
})

# Update workflow with topic assignments
workflow.set_topic_assignments(topic_df)

# Generate visualizations and reports
workflow.visualize_topics(plot_type="distribution")
workflow.generate_comprehensive_report(
    title="BERTopic Analysis Results",
    include_interactive=True,
    include_raw_data=True
)
```

### Topic Analysis and Exploration

```python
from meno.modeling.bertopic_model import BERTopicModel

# Create and fit model
model = BERTopicModel(n_topics=15)
model.fit(documents)

# Analyze topic sizes
for topic_id, size in model.topic_sizes.items():
    if topic_id != -1:  # Skip outlier topic
        print(f"Topic {topic_id}: {size} documents ({size/len(documents)*100:.1f}%)")

# Find similar topics to a query
similar_topics = model.find_similar_topics("customer complaints", n_topics=3)
for topic_id, desc, score in similar_topics:
    print(f"{topic_id}: {desc} (score: {score:.2f})")

# Visualize topics
model.visualize_topics().write_html("topic_visualization.html")
model.visualize_hierarchy().write_html("topic_hierarchy.html")
```

## Visualization

Meno provides enhanced visualization capabilities for BERTopic models, adding improvements to BERTopic's native visualizations and additional visualization types.

### Standard BERTopic Visualizations

```python
# Visualize topic similarity
fig = model.visualize_topics()
fig.write_html("topic_similarity.html")

# Visualize topic hierarchy
fig = model.visualize_hierarchy()
fig.write_html("topic_hierarchy.html")
```

### Enhanced BERTopic Visualizations

Meno's enhanced visualizations provide better readability, interactivity, and customization:

```python
from meno.visualization.bertopic_viz import (
    create_enhanced_topic_visualization,
    create_topic_timeline,
    create_topic_comparison,
    create_topic_network
)

# Enhanced topic similarity visualization
fig = create_enhanced_topic_visualization(
    bertopic_model=model.model,  # Use the underlying BERTopic model
    width=900,
    height=700,
    title="Topic Similarity Map",
    color_by="size",  # Color by topic size
    theme="plotly_white"
)
fig.write_html("enhanced_topic_similarity.html")

# Topic timeline visualization (requires timestamps)
fig = create_topic_timeline(
    bertopic_model=model.model,
    timestamps=df["date"],  # Assuming df has a date column
    topics=None,  # None for top 10 largest topics
    width=1000,
    height=600,
    title="Topic Trends Over Time",
    time_format="monthly"  # Aggregate by month
)
fig.write_html("topic_timeline.html")

# Compare specific topics side by side
fig = create_topic_comparison(
    bertopic_model=model.model,
    topic_ids=[0, 1, 2, 3],  # Compare these topics
    width=1000,
    height=500,
    title="Comparison of Insurance-Related Topics"
)
fig.write_html("topic_comparison.html")

# Create a network visualization of topic relationships
fig = create_topic_network(
    bertopic_model=model.model,
    min_similarity=0.3,  # Minimum similarity threshold for connections
    width=800,
    height=800,
    title="Topic Relationship Network"
)
fig.write_html("topic_network.html")
```

### Using Meno's Workflow Visualization Tools

```python
# Initialize workflow and assign topics
workflow = MenoWorkflow()
workflow.load_data(data=df, text_column="text")
workflow.preprocess_documents()
workflow.set_topic_assignments(topic_df)  # From BERTopic results

# Visualize topic embeddings
embedding_viz = workflow.visualize_topics(plot_type="embeddings")
embedding_viz.write_html("topic_embeddings.html")

# Visualize topic distribution
distribution_viz = workflow.visualize_topics(plot_type="distribution")
distribution_viz.write_html("topic_distribution.html")

# Additional visualization types for temporal or geospatial data
if workflow.time_column:
    time_viz = workflow.visualize_topics(plot_type="trends")
    time_viz.write_html("topic_trends.html")
    
if workflow.geo_column:
    geo_viz = workflow.visualize_topics(plot_type="map")
    geo_viz.write_html("topic_map.html")
    
if workflow.time_column and workflow.geo_column:
    time_space_viz = workflow.visualize_topics(plot_type="timespace")
    time_space_viz.write_html("topic_timespace.html")
```

### Customizing Visualizations

You can customize all visualizations with various parameters:

```python
# Customize enhanced topic visualization
fig = create_enhanced_topic_visualization(
    bertopic_model=model.model,
    width=1000,
    height=800,
    title="Custom Topic Map",
    color_by="id",           # Color by topic ID
    theme="plotly_dark",     # Dark theme
    background_color="#222"  # Custom background color
)

# Add annotations or custom layout elements
fig.add_annotation(
    x=0.5, y=1.05,
    text="Generated with Meno + BERTopic",
    xref="paper", yref="paper",
    showarrow=False,
    font=dict(size=14, color="white")
)

# Update layout
fig.update_layout(
    font=dict(family="Arial", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Save high-resolution image
fig.write_image("high_res_topic_map.png", scale=2)  # 2x resolution
```

## Saving and Loading Models

### Saving Models

```python
# Save BERTopicModel
model = BERTopicModel()
model.fit(documents)
model.save("models/bertopic_model")

# Save UnifiedTopicModeler
modeler = create_topic_modeler(method="bertopic")
modeler.fit(documents)
modeler.save("models/unified_model")
```

### Loading Models

```python
# Load BERTopicModel
from meno.modeling.bertopic_model import BERTopicModel
loaded_model = BERTopicModel.load("models/bertopic_model")

# Load UnifiedTopicModeler
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler
loaded_modeler = UnifiedTopicModeler.load("models/unified_model")
```

## Best Practices

### Embedding Model Selection

BERTopic can use various embedding models. Here are some recommendations:

- For general-purpose text: `all-MiniLM-L6-v2` (fast and good quality)
- For multilingual text: `paraphrase-multilingual-MiniLM-L12-v2`
- For domain-specific text: Consider fine-tuning a model on your domain

### Preprocessing Recommendations

- Expand acronyms using Meno's acronym detection and expansion
- Correct spelling errors with Meno's spelling correction
- Remove domain-specific stopwords
- For longer documents, consider splitting into smaller chunks

### Hyperparameter Guidelines

- **n_neighbors** (UMAP): Higher for more global structure (15-30), lower for more local structure (5-15)
- **n_components** (UMAP): 5-15 is usually sufficient
- **min_cluster_size** (HDBSCAN): Adjust based on your dataset size (5-20)
- **representation_model**: Use both KeyBERTInspired and MaximalMarginalRelevance for diverse topic representations

### Performance Optimization

- For large datasets, use a smaller embedding model like `all-MiniLM-L6-v2`
- Use GPU acceleration when available
- For extremely large datasets, consider sampling or chunking

## Troubleshooting

### Common Issues

**Issue**: No topics found (all documents assigned to topic -1)

**Solution**:
- Decrease `min_cluster_size` 
- Try different UMAP parameters (`n_neighbors`, `min_dist`)
- Check if documents are too short or too similar

**Issue**: Too many topics found

**Solution**:
- Increase `min_cluster_size`
- Use topic reduction with `nr_topics` parameter
- Try different HDBSCAN parameters

**Issue**: Poor topic quality

**Solution**:
- Use a better embedding model
- Add domain-specific stopwords
- Use `bm25_weighting=True` in ClassTfidfTransformer
- Try combining representation models

**Issue**: "CUDA out of memory" errors

**Solution**:
- Reduce batch size or use a smaller dataset
- Use a smaller embedding model
- Set `use_gpu=False` to use CPU instead

**Issue**: "ModuleNotFoundError" for hdbscan or umap-learn

**Solution**:
- Install the required dependencies: `pip install hdbscan umap-learn`
- For GPU support: `pip install cuml`
# Meno: Topic Modeling Toolkit

<p align="center">
  <img src="meno.webp" alt="Meno Logo" width="250"/>
</p>

[![PyPI version](https://img.shields.io/pypi/v/meno.svg)](https://pypi.org/project/meno/)
[![Python Version](https://img.shields.io/pypi/pyversions/meno.svg)](https://pypi.org/project/meno/)
[![License](https://img.shields.io/github/license/srepho/meno.svg)](https://github.com/srepho/meno/blob/main/LICENSE)
[![Tests](https://github.com/srepho/meno/workflows/tests/badge.svg)](https://github.com/srepho/meno/actions?query=workflow%3Atests)
[![Downloads](https://img.shields.io/pypi/dm/meno.svg)](https://pypi.org/project/meno/)

Meno is a toolkit for topic modeling on messy text data, featuring an interactive workflow system that guides users from raw text to insights through acronym detection, spelling correction, topic modeling, and visualization.

## Installation

```bash
# Basic installation with core dependencies
pip install "numpy<2.0.0"  # NumPy 1.x is required for compatibility
pip install meno

# Recommended: Minimal installation with essential topic modeling dependencies
pip install "numpy<2.0.0"
pip install "meno[minimal]"

# CPU-optimized installation without NVIDIA packages
pip install "numpy<2.0.0"
pip install "meno[embeddings]" -f https://download.pytorch.org/whl/torch_stable.html
```

### Offline/Air-gapped Environment Installation

For environments with limited internet access:

1. Download required models on a connected machine:
   ```python
   from sentence_transformers import SentenceTransformer
   # Download and cache model
   model = SentenceTransformer("all-MiniLM-L6-v2")
   # Note the model path (usually in ~/.cache/huggingface)
   ```

2. Transfer the downloaded model files to the offline machine in the same directory structure

3. Use the local_files_only option when initializing:
   ```python
   from meno.modeling.embeddings import DocumentEmbedding
   
   # Option 1: Direct path to downloaded model
   embedding_model = DocumentEmbedding(
       local_model_path="/path/to/local/model",
       use_gpu=False
   )
   
   # Option 2: Using standard HuggingFace cache location
   embedding_model = DocumentEmbedding(
       model_name="all-MiniLM-L6-v2",
       local_files_only=True,
       use_gpu=False
   )
   ```

See `examples/local_model_example.py` for detailed offline usage examples.

## Quick Start

### Basic Topic Modeling

```python
from meno import MenoTopicModeler
import pandas as pd

# Load your data
df = pd.read_csv("documents.csv")

# Initialize and run basic topic modeling
modeler = MenoTopicModeler()
processed_docs = modeler.preprocess(df, text_column="text")
topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=5)

# Visualize results
fig = modeler.visualize_embeddings()
fig.write_html("topic_embeddings.html")

# Generate comprehensive HTML report
report_path = modeler.generate_report(output_path="topic_report.html")
```

### Interactive Workflow

```python
from meno import MenoWorkflow
import pandas as pd

# Load your data
data = pd.DataFrame({
    "text": [
        "The CEO and CFO met to discuss the AI implementation in our CRM system.",
        "Customer submitted a claim for their vehical accident on HWY 101.",
        "The CTO presented the ML strategy for improving cust retention.",
        "Policyholder recieved the EOB and was confused about the CPT codes."
    ]
})

# Initialize and run workflow
workflow = MenoWorkflow()
workflow.load_data(data=data, text_column="text")

# Generate interactive acronym report
workflow.generate_acronym_report(output_path="acronyms.html", open_browser=True)

# Apply acronym expansions
workflow.expand_acronyms({"CRM": "Customer Relationship Management", "CTO": "Chief Technology Officer"})

# Generate interactive misspelling report
workflow.generate_misspelling_report(output_path="misspellings.html", open_browser=True)

# Apply spelling corrections
workflow.correct_spelling({"vehical": "vehicle", "recieved": "received"})

# Preprocess and model topics
workflow.preprocess_documents()
workflow.discover_topics(num_topics=2)

# Generate comprehensive report
workflow.generate_comprehensive_report("final_report.html", open_browser=True)
```

## What's New in v1.0.0

- **Standardized API** - Consistent parameter names and method signatures across all models
- **Automatic Topic Detection** - Models can discover the optimal number of topics automatically
- **Enhanced Memory Efficiency** - Process larger datasets with streaming and quantization
- **Path Object Support** - Better file handling with pathlib integration
- **Return Type Standardization** - Consistent return values across all methods

## Overview

Meno streamlines topic modeling on messy text data, with a special focus on datasets like insurance claims and customer correspondence. It combines traditional methods (LDA) with modern techniques using large language models, dimensionality reduction with UMAP, and interactive visualizations.

## Key Features

- **Interactive Workflow System**
  - Guided process from raw data to insights
  - Acronym detection and expansion
  - Spelling correction with contextual examples
  - Topic discovery and visualization
  - Interactive HTML reports

- **Versatile Topic Modeling**
  - Unsupervised discovery with embedding-based clustering
  - Supervised matching against predefined topics
  - Automatic topic detection
  - Integration with BERTopic and other advanced models

- **Team Configuration System**
  - Share domain-specific dictionaries across teams
  - Import/export terminology (JSON, YAML)
  - CLI tools for configuration management

- **Performance Optimizations**
  - Memory-efficient processing for large datasets
  - Quantized embedding models
  - Streaming processing for larger-than-memory data
  - CPU-first design with optional GPU acceleration

- **Visualization & Reporting**
  - Interactive embedding visualizations
  - Topic distribution and similarity analysis
  - Time series and geospatial visualizations
  - Comprehensive HTML reports

## Installation Options

```bash
# For additional topic modeling approaches (BERTopic, Top2Vec)
pip install "meno[additional_models]"

# For embeddings with GPU acceleration
pip install "meno[embeddings-gpu]"

# For LDA topic modeling
pip install "meno[lda]"

# For visualization capabilities
pip install "meno[viz]"

# For NLP processing capabilities
pip install "meno[nlp]"

# For large dataset optimization using Polars
pip install "meno[optimization]"

# For memory-efficient embeddings
pip install "meno[memory_efficient]"

# For all features (CPU only)
pip install "meno[full]"

# For all features with GPU acceleration
pip install "meno[full-gpu]"

# For development
pip install "meno[dev,test]"
```

## Examples

### Advanced Topic Discovery

```python
from meno import MenoTopicModeler
import pandas as pd

# Initialize modeler
modeler = MenoTopicModeler()

# Load and preprocess data
df = pd.read_csv("documents.csv")
processed_docs = modeler.preprocess(
    df, 
    text_column="text",
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    additional_stopwords=["specific", "custom", "words"]
)

# Discover topics (automatic detection with HDBSCAN)
topics_df = modeler.discover_topics(
    method="embedding_cluster",
    clustering_algorithm="hdbscan",
    min_cluster_size=10,
    min_samples=5
)

print(f"Discovered {len(topics_df['topic'].unique())} topics")

# Visualize results
fig = modeler.visualize_embeddings(
    plot_3d=True,
    include_topic_centers=True
)
fig.write_html("3d_topic_visualization.html")

# Generate report
report_path = modeler.generate_report(
    output_path="topic_report.html",
    include_interactive=True
)
```

### BERTopic Integration

```python
from meno import MenoWorkflow
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

# Load data and initialize workflow
df = pd.read_csv("documents.csv")
workflow = MenoWorkflow()
workflow.load_data(data=df, text_column="text")
workflow.preprocess_documents()

# Get preprocessed data from workflow
preprocessed_df = workflow.get_preprocessed_data()

# Configure and fit BERTopic model
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
keybert_model = KeyBERTInspired()

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=ctfidf_model,
    representation_model=keybert_model,
    calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(
    preprocessed_df["processed_text"].tolist()
)

# Update workflow with BERTopic results
preprocessed_df["topic"] = [f"Topic_{t}" if t >= 0 else "Outlier" for t in topics]
preprocessed_df["topic_probability"] = probs
workflow.set_topic_assignments(preprocessed_df[["topic", "topic_probability"]])

# Generate visualizations and report
topic_model.visualize_topics().write_html("bertopic_similarity.html")
workflow.generate_comprehensive_report(
    output_path="bertopic_report.html",
    open_browser=True
)
```

### Matching Documents to Predefined Topics

```python
from meno import MenoTopicModeler
import pandas as pd

# Initialize and load data
modeler = MenoTopicModeler()
df = pd.read_csv("support_tickets.csv")
processed_docs = modeler.preprocess(df, text_column="description")

# Define topics and descriptions
predefined_topics = [
    "Account Access",
    "Billing Issue",
    "Technical Problem",
    "Feature Request",
    "Product Feedback"
]

topic_descriptions = [
    "Issues related to logging in, password resets, or account security",
    "Problems with payments, invoices, or subscription changes",
    "Technical issues, bugs, crashes, or performance problems",
    "Requests for new features or enhancements to existing functionality",
    "General feedback about the product, including compliments and complaints"
]

# Match documents to topics
matched_df = modeler.match_topics(
    topics=predefined_topics,
    descriptions=topic_descriptions,
    threshold=0.6,
    assign_multiple=True,
    max_topics_per_doc=2
)

# View topic assignments
print(matched_df[["description", "topic", "topic_probability"]].head())
```

### Large Dataset Processing

```python
from meno import MenoWorkflow
import pandas as pd

# Create optimized configuration
config_overrides = {
    "modeling": {
        "embeddings": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "quantize": True,
            "low_memory": True
        }
    }
}

# Initialize workflow with optimized settings
workflow = MenoWorkflow(config_overrides=config_overrides)

# Process in batches
data = pd.read_csv("large_dataset.csv")
batch_size = 10000

for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    
    if i == 0:  # First batch
        workflow.load_data(batch, text_column="text")
    else:  # Update with subsequent batches
        workflow.update_data(batch)

# Process with memory-efficient settings
workflow.preprocess_documents()
workflow.discover_topics(method="embedding_cluster")
workflow.generate_comprehensive_report("large_dataset_report.html")
```

## Team Configuration CLI

```bash
# Create a new team configuration
meno-config create "Healthcare" \
    --acronyms-file healthcare_acronyms.json \
    --corrections-file medical_spelling.json \
    --output-path healthcare_config.yaml

# Update an existing configuration
meno-config update healthcare_config.yaml \
    --acronyms-file new_acronyms.json

# Compare configurations from different teams
meno-config compare healthcare_config.yaml insurance_config.yaml \
    --output-path comparison.json
```

## Architecture

The package follows a modular design:

- **Data Preprocessing:** Spelling correction, acronym resolution, text normalization
- **Topic Modeling:** Unsupervised discovery, supervised matching, multiple model support
- **Visualization:** Interactive embeddings, topic distributions, time series
- **Report Generation:** HTML reports with Plotly and Jinja2
- **Team Configuration:** Domain knowledge sharing, CLI tools

## Dependencies

- **Python:** 3.8-3.12 (primary target: 3.10)
- **Core Libraries:** pandas, scikit-learn, thefuzz, pydantic, PyYAML
- **Optional Libraries:** sentence-transformers, transformers, torch, umap-learn, hdbscan, plotly, bertopic

## Testing

```bash
# Run basic tests
python -m pytest -xvs tests/

# Run with coverage reporting
python -m pytest --cov=meno
```

## Documentation

For detailed usage information, see the [full documentation](https://github.com/srepho/meno/wiki).

## Future Development

With v1.0.0 complete, our focus is shifting to:

1. **Cloud Integration** - Native support for cloud-based services
2. **Multilingual Support** - Expand beyond English
3. **Domain-Specific Fine-Tuning** - Adapt models to specific industries
4. **Explainable AI Features** - Better interpret topic assignments
5. **Interactive Dashboards** - More powerful visualization tools

See our [detailed roadmap](ROADMAP.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# Meno: Topic Modeling Toolkit

[![PyPI version](https://img.shields.io/pypi/v/meno.svg)](https://pypi.org/project/meno/)
[![Python Version](https://img.shields.io/pypi/pyversions/meno.svg)](https://pypi.org/project/meno/)
[![License](https://img.shields.io/github/license/srepho/meno.svg)](https://github.com/srepho/meno/blob/main/LICENSE)
[![Tests](https://github.com/srepho/meno/workflows/tests/badge.svg)](https://github.com/srepho/meno/actions?query=workflow%3Atests)
[![Downloads](https://img.shields.io/pypi/dm/meno.svg)](https://pypi.org/project/meno/)

## Installation

### Basic Installation

Install the basic package with core dependencies:

```bash
pip install meno
```

### CPU-Optimized Installation (Recommended)

Install with embeddings for CPU-only operation (recommended for most users):

```bash
pip install meno[embeddings]
```

For a truly CPU-only version with no NVIDIA packages:

```bash
pip install meno[embeddings] -f https://download.pytorch.org/whl/torch_stable.html
```

### Installation with Optional Components

```bash
# For additional topic modeling approaches (BERTopic, Top2Vec)
pip install meno[additional_models]

# For embeddings with GPU acceleration (only if needed)
pip install meno[embeddings-gpu]

# For LDA topic modeling
pip install meno[lda]

# For visualization capabilities
pip install meno[viz]

# For NLP processing capabilities
pip install meno[nlp]

# For large dataset optimization using Polars
pip install meno[optimization]

# For developers
pip install meno[dev,test]

# For all features (full installation, CPU only)
pip install meno[full]

# For all features with GPU acceleration
pip install meno[full-gpu]
```

### Development Installation

For development work, clone the repository and install in editable mode:

```bash
git clone https://github.com/srepho/meno.git
cd meno
pip install -e ".[dev,test]"
```

## Quick Start

```python
from meno import MenoTopicModeler
import pandas as pd

# Load your data
data = pd.DataFrame({
    "text": [
        "Customer's vehicle was damaged in a parking lot by a shopping cart.",
        "Claimant's home flooded due to heavy rain. Water damage to first floor.",
        "Vehicle collided with another car at an intersection. Front-end damage.",
        "Tree fell on roof during storm causing damage to shingles and gutters.",
        "Insured slipped on ice in parking lot and broke wrist requiring treatment."
    ]
})

# Initialize topic modeler
modeler = MenoTopicModeler()

# Preprocess documents
processed_docs = modeler.preprocess(data, text_column="text")

# Generate embeddings
embeddings = modeler.embed_documents()

# Discover topics
topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=3)

# Visualize results
fig = modeler.visualize_embeddings()
fig.show()

# Generate HTML report
report_path = modeler.generate_report(output_path="topics_report.html")
```

## Overview

Meno is designed to streamline topic modeling on free text data, with a special focus on messy datasets such as insurance claims notes and customer correspondence. The package combines classical methods like Latent Dirichlet Allocation (LDA) with modern techniques leveraging large language models (LLMs) via Hugging Face, dimensionality reduction with UMAP, and advanced visualizations. It is built to be primarily used in Jupyter environments while also being flexible enough for other settings.

## Key Features

*   **Unsupervised Topic Modeling:**
    *   Automatically discover topics when no pre-existing topics are available using LDA and LLM-based embedding and clustering techniques.
*   **Supervised Topic Matching:**
    *   Match free text against a user-provided list of topics using semantic similarity and classification techniques.
*   **Advanced Visualization:**
    *   Create interactive and static visualizations including topic distributions, embeddings (UMAP projections), cluster analyses, and topic coherence metrics (e.g., word clouds per topic).
*   **Interactive HTML Reports:**
    *   Generate standalone, interactive HTML reports to present topic analysis to less technical stakeholders, with options for customization and data export.
*   **Robust Data Preprocessing:**
    *   Tackle messy data challenges (misspellings, unknown acronyms) with integrated cleaning functionalities using NLP libraries (spaCy, fuzzy matching, context-aware spelling correction, and customizable stop words/lemmatization rules).
*   **Active Learning with Cleanlab:**
    *   Incorporate active learning loops and fine-tuning of labels using Cleanlab, facilitating hand-labeling and iterative improvements, with multiple sampling strategies (e.g., uncertainty sampling).
*   **Flexible Deployment Options:**
    *   CPU-first design with optional GPU acceleration through separate installation options.
    *   Load models from local files for use in environments without internet access or behind firewalls.
*   **Extensibility & Ease of Use:**
    *   Designed with modularity in mind so that users can plug in new cleaning, modeling, or visualization techniques without deep customization while still maintaining a simple interface.

## Example Usage

### Basic Topic Discovery

```python
from meno import MenoTopicModeler

# Initialize modeler
modeler = MenoTopicModeler()

# Load and preprocess data
df = pd.read_csv("my_documents.csv")
processed_docs = modeler.preprocess(df, text_column="document_text")

# Discover topics
topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=10)

# Visualize results
fig = modeler.visualize_embeddings()
fig.show()
```

### Matching Documents to Predefined Topics

```python
# Define topics and descriptions
predefined_topics = [
    "Vehicle Damage",
    "Water Damage",
    "Personal Injury",
    "Property Damage"
]

topic_descriptions = [
    "Damage to vehicles from collisions, parking incidents, or natural events",
    "Damage from water including floods, leaks, and burst pipes",
    "Injuries to people including slips, falls, and accidents",
    "Damage to property from fire, storms, or other causes"
]

# Match documents to topics
matched_df = modeler.match_topics(
    topics=predefined_topics,
    descriptions=topic_descriptions,
    threshold=0.5
)

# View the topic assignments
print(matched_df[["text", "topic", "topic_probability"]].head())
```

### Generating Reports

```python
# Generate an interactive HTML report
report_path = modeler.generate_report(
    output_path="topic_analysis.html",
    include_interactive=True,
    title="Document Topic Analysis"
)
```

## Documentation

For detailed usage information, see the [full documentation](https://github.com/srepho/meno/wiki).

## Examples

The package includes several example notebooks and scripts:

- `examples/basic_workflow.ipynb`: Basic topic modeling workflow in a Jupyter notebook
- `examples/cpu_only_example.py`: Demonstrates CPU-optimized topic modeling
- `examples/insurance_topic_modeling.py`: Topic modeling on insurance complaint dataset
- `examples/minimal_sample.py`: Simple script to generate visualizations
- `examples/sample_reports/`: Directory with pre-generated sample visualizations

### Insurance Complaint Analysis

The package includes an example that demonstrates topic modeling on the Australian Insurance PII Dataset from Hugging Face. This dataset contains over 1,500 insurance complaint letters with various types of insurance issues.

To run the insurance example:

```bash
# Install required dependencies
pip install -r requirements_insurance_example.txt

# Run the example script
python examples/insurance_topic_modeling.py
```

The results will be saved in the `output` directory.

## Architecture & Design

The package follows a modular design with clear separation of concerns:

### Data Preprocessing Module:
- Spelling correction using thefuzz
- Acronym resolution 
- Text normalization (lowercasing, punctuation removal, stemming/lemmatization)
- Customizable stop words and lemmatization

### Topic Modeling Module:
- Unsupervised modeling with LDA or LLM-based embeddings + clustering
- Supervised topic matching using semantic similarity
- CPU-first design with optional GPU acceleration

### Visualization Module:
- Static plots (topic distributions)
- Interactive embedding plots with UMAP projections
- Topic coherence visualizations

### Report Generation Module:
- Interactive HTML reports using Plotly and Jinja2
- Customizable appearance and content
- Data export options

## Dependencies & Requirements

*   **Python:** 3.8, 3.9, 3.10, 3.11, 3.12 (primary target: 3.10)
*   **Core Libraries** (always installed):
    *   Data Processing: `pandas`, `pyarrow`
    *   Machine Learning: `scikit-learn`
    *   Text Processing: `thefuzz`
    *   Configuration: `pydantic`, `PyYAML`, `jinja2`
*   **Optional Libraries** (install based on needs):
    *   Topic Modeling: `gensim` (for LDA)
    *   Additional Topic Models: `bertopic`, `top2vec`
    *   Embeddings (CPU): `transformers`, `sentence-transformers`, `torch`
    *   Embeddings (GPU): Additional `accelerate`, `bitsandbytes` 
    *   Dimensionality Reduction: `umap-learn`
    *   Clustering: `hdbscan`
    *   Data Cleaning & NLP: `spaCy`
    *   Visualization: `plotly`
    *   Active Learning: `cleanlab`
    *   Large Dataset Optimization: `polars` (for streaming and memory efficiency)

## Testing & Contribution

### Running Tests

```bash
# Run basic tests
python -m pytest -xvs tests/

# Run full tests including embedding model tests
python -m pytest -xvs tests/ --run-functional

# Run with coverage reporting
python -m pytest --cov=meno
```

## Contribution Guidelines

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/srepho/meno/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
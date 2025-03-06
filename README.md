# Topic Modeling Toolkit: meno

[![PyPI version](https://img.shields.io/pypi/v/meno.svg)](https://pypi.org/project/meno/)
[![Python Version](https://img.shields.io/pypi/pyversions/meno.svg)](https://pypi.org/project/meno/)
[![License](https://img.shields.io/github/license/srepho/meno.svg)](https://github.com/srepho/meno/blob/main/LICENSE)
[![Tests](https://github.com/srepho/meno/workflows/tests/badge.svg)](https://github.com/srepho/meno/actions?query=workflow%3Atests)
[![Downloads](https://img.shields.io/pypi/dm/meno.svg)](https://pypi.org/project/meno/)

## Overview

This Python package, meno, is designed to streamline topic modeling on free text data, with a special focus on messy datasets such as insurance claims notes and customer correspondence. The package combines classical methods like Latent Dirichlet Allocation (LDA) with modern techniques leveraging large language models (LLMs) via Hugging Face, dimensionality reduction with UMAP, and advanced visualizations. It is built to be primarily used in Jupyter environments while also being flexible enough for other settings.

## Key Features

*   **Unsupervised Topic Modeling:**
    *   Automatically discover topics when no pre-existing topics are available using LDA and LLM-based embedding and clustering techniques. **(See details in Architecture section below)**
*   **Supervised Topic Matching:**
    *   Match free text against a user-provided list of topics using semantic similarity and classification techniques. **(See details in Architecture section below)**
*   **Advanced Visualization:**
    *   Create interactive and static visualizations including topic distributions, embeddings (UMAP projections), cluster analyses, **and topic coherence metrics (e.g., word clouds per topic).**
*   **Interactive HTML Reports:**
    *   Generate standalone, interactive HTML reports to present topic analysis to less technical stakeholders, **with options for customization and data export.**
*   **Robust Data Preprocessing:**
    *   Tackle messy data challenges (misspellings, unknown acronyms) with integrated cleaning functionalities using NLP libraries (spaCy, fuzzy matching, **context-aware spelling correction, and customizable stop words/lemmatization rules.**). **(See details in Architecture section below)**
*   **Active Learning with Cleanlab:**
    *   Incorporate active learning loops and fine-tuning of labels using Cleanlab, facilitating hand-labeling and iterative improvements, **with multiple sampling strategies (e.g., uncertainty sampling).**
*   **Flexible Deployment Options:**
    *   CPU-first design with optional GPU acceleration through separate installation options.
    *   Load models from local files for use in environments without internet access or behind firewalls.
*   **Extensibility & Ease of Use:**
    *   Designed with modularity in mind so that users can plug in new cleaning, modeling, or visualization techniques without deep customization while still maintaining a simple interface.

## Architecture & Design

The package follows a modular design with clear separation of concerns, ensuring that each component can be developed, tested, and extended independently:

### Data Preprocessing Module:

Handles cleaning tasks such as:

*   **Spelling Correction:** Utilizes `fuzzywuzzy` for basic corrections and **incorporates context-aware spelling correction using a pre-trained language model (e.g., a smaller, faster model than the main LLMs used for topic modeling).**
*   **Acronym Resolution:**  Addresses acronyms using a **combination of a user-expandable predefined dictionary, rule-based pattern matching (e.g., identifying all-caps words), and, optionally, leveraging LLMs to disambiguate acronyms based on context.**
*   **Normalization:**  Handles tasks like lowercasing, punctuation removal, and stemming/lemmatization.
*    **Customizable Stop Words and Lemmatization:** Allows users to **easily define and apply custom stop word lists and lemmatization rules specific to their domain via configuration files.**

### Topic Modeling Module:

#### Unsupervised Modeling:

*   **LDA:** Implements Latent Dirichlet Allocation using `gensim`.
*   **LLM-based Topic Extraction:**
    *   **Embedding Generation:**  Uses pre-trained LLMs from Hugging Face Transformers (e.g., **`answerdotai/ModernBERT-base` as the default model, with options for users to specify other models or use local model files**) to generate document embeddings.
    *   **Clustering:** Applies clustering algorithms (e.g., HDBSCAN, K-Means) to the embeddings to identify topic clusters.  **Provides options for users to adjust clustering parameters.**
    *   **GPU Acceleration:** Optional GPU support for faster embedding generation with large datasets.
    *   **Optional LLM Fine-tuning:** *Consider adding this as a later feature.  It would involve allowing users to fine-tune the LLM on their data.*

#### Supervised Matching:

*   **Semantic Similarity:** Uses cosine similarity between document embeddings (generated by the same LLMs as in unsupervised modeling) and embeddings of user-provided topic descriptions.
*   **Classification:**  **(Optional, for later development):**  Trains a classifier (e.g., a Transformer model) on labeled data to directly predict topic labels.
*   **Topic Input Format:** Users provide topics as a **list of strings (topic names) and, optionally, short descriptions for each topic.  The system will generate embeddings for these descriptions.**
*   **Thresholding and "Other" Category:** Implements a **similarity threshold (configurable by the user) below which a document is assigned to an "Other" or "Unmatched" category.**

### Visualization Module:

Provides functions to create:

*   **Static Plots:**  Topic distribution histograms.
*   **Interactive Embedding Plots:** UMAP projections of document embeddings, colored by topic (either discovered or assigned).  **Interactivity includes zooming, panning, hovering for details (document text and topic probabilities), and selecting data points.**
*   **Topic Coherence Visualizations:** Word clouds for each topic, showing the most important words. **Top words can be ranked by probability (for LDA) or by some measure of relevance to the embedding centroid (for LLM-based topics).**
*    **UMAP Parameter Control:** Allows users to **customize UMAP parameters (e.g., `n_neighbors`, `min_dist`) via configuration files.**

### Report Generation Module:

Generates interactive, standalone HTML reports using Plotly or Bokeh alongside Jinja2 for templating.

*   **Customization:**  Allows users to **customize the report's appearance (e.g., colors, fonts) and content (e.g., adding a logo, introductory text, conclusions) via configuration files or a simple API.**
*   **Data Export:**  Provides options to **export the underlying data (topic assignments, probabilities, embeddings) in various formats (e.g., CSV, JSON).**

### Active Learning Module:

Integrates with Cleanlab to facilitate hand-labeling, iterative label fine-tuning, and quality control over topic assignments.

*   **Wrapper Functions:** Provides **simplified wrapper functions around Cleanlab's functionality to streamline the active learning process.  These functions handle presenting documents for labeling, updating the model (either LDA or the LLM-based similarity matching), and re-evaluating performance.**
*   **Labeling Interface:**  Provides a **basic labeling interface within Jupyter notebooks using `ipywidgets`. This allows users to view documents and assign topic labels without leaving the notebook environment.**  *(Consider linking to external labeling tools in the future.)*
*   **Sampling Strategies:**  Offers **different sampling strategies, primarily uncertainty sampling (selecting documents where the model is least confident).**

## Dependencies & Requirements

*   **Python:** 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
*   **Core Libraries** (always installed):
    *   Data Processing: `pandas`, `pyarrow`
    *   Machine Learning: `scikit-learn`
    *   Text Processing: `fuzzywuzzy`
    *   Configuration: `pydantic`, `PyYAML`, `jinja2`
*   **Optional Libraries** (install based on needs):
    *   Topic Modeling: `gensim` (for LDA)
    *   Additional Topic Models: `bertopic`, `top2vec`
    *   Embeddings (CPU): `transformers`, `sentence-transformers`, `torch`
    *   Embeddings (GPU): Additional `accelerate`, `bitsandbytes` 
    *   Dimensionality Reduction: `umap-learn`
    *   Clustering: `hdbscan`
    *   Data Cleaning & NLP: `spaCy`, `python-Levenshtein`
    *   Visualization: `plotly`
    *   Active Learning: `cleanlab`
    *   Large Dataset Optimization: `polars` (for streaming and memory efficiency)
*   Development and testing libraries: `pytest`, `hypothesis`, `black`, `ruff`, `mypy`, `sphinx`

## Installation & Setup

### Basic Installation

Install the basic package with core dependencies:

```bash
pip install meno
```

### Installation with Optional Components

Install with specific optional dependencies:

```bash
# For embeddings and LLM-based topic modeling (CPU only - recommended)
pip install meno[embeddings]

# For embeddings with GPU acceleration (only if needed)
pip install meno[embeddings-gpu]

# For additional topic modeling approaches (BERTopic, Top2Vec)
pip install meno[additional_models]

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

# For all features (full installation, CPU only - recommended for most users)
pip install meno[full]

# For all features with GPU acceleration (only if needed)
pip install meno[full-gpu]
```

> **Note:** The CPU-only installations (`meno[embeddings]` and `meno[full]`) are recommended for most users and will provide excellent performance for most workloads. To install a truly CPU-only version with no NVIDIA packages, use `pip install meno[embeddings] -f https://download.pytorch.org/whl/torch_stable.html`

### Development Installation

For development work, clone the repository and install in editable mode:

```bash
git clone https://github.com/srepho/meno.git
cd meno
pip install -e ".[dev,test]"
```

### Using with Conda

Create a new conda environment:

```bash
conda create -n meno python=3.10  # Primary supported version
conda activate meno
pip install meno[full]  # CPU-only version recommended
```

## Usage in Jupyter

The package is optimized for use within Jupyter notebooks.

Example notebooks demonstrating:

*   Loading and preprocessing messy text data.
*   Running both unsupervised and supervised topic models.
*   Visualizing topics and generating interactive reports.
*   Using Cleanlab for active learning and fine-tuning.

## Extensibility & Customization

*   **Modular API:** Each component (cleaning, modeling, visualization, reporting) can be replaced or extended.
*   **Configuration Files:**  Uses YAML/JSON configuration files (with **validation against a schema**) to allow users to customize pipelines without modifying code. **Provides example configuration files and clear documentation on the available options.**
*   **Plugin System:** *Consider a simple plugin architecture for users to add custom processing steps or models.*

## Testing & Continuous Integration

*   **Unit Tests:** Comprehensive tests for each module using `pytest`:
    * Mock-based tests for quick validation without heavy dependencies
    * Functional tests that test actual implementations with real models
    * Property-based tests using `hypothesis` for robust validation
*   **Integration Tests:** End-to-end pipelines tested on sample datasets
*   **Test Organization:** Tests can run at different levels:
    * Fast tests: Run without optional dependencies
    * Full tests: Test all functionality including embeddings and visualization
*   **CI Pipeline:** Automated testing on GitHub Actions for every push and pull request

### Running Tests

Run tests with different configuration levels:

```bash
# Run basic tests that don't require optional dependencies
python -m pytest -xvs tests/

# Run full tests including embedding model tests
python -m pytest -xvs tests/ --run-functional

# Run with coverage reporting
python -m pytest --cov=meno --cov-report=term

# Run specific test module
python -m pytest tests/test_preprocessing.py
```

## Documentation & Examples

*   **User Guide:** Detailed documentation on installation, API usage, and examples of typical workflows.
*   **Tutorial Notebooks:** Step-by-step Jupyter notebooks demonstrating real-world applications.
*   **API Reference:** Generated from docstrings to ensure up-to-date and detailed reference materials.

## Contribution Guidelines

*   **Code of Conduct & Contribution Guide:** Clear instructions for contributing, reporting issues, and submitting pull requests.
*   **Issue Tracker & Feature Requests:** Guidance on how to report bugs or request new features.

## Examples

### Insurance Complaint Analysis

The package includes an example that demonstrates topic modeling on the Australian Insurance PII Dataset from Hugging Face. This dataset contains over 1,500 insurance complaint letters with various types of insurance issues, making it ideal for testing topic modeling capabilities.

To run the insurance example:

```bash
# Install required dependencies
pip install -r requirements_insurance_example.txt

# Run the example script
python examples/insurance_topic_modeling.py
```

This example demonstrates:
- Loading data from Hugging Face Datasets
- Preprocessing insurance complaint letters
- Discovering topics using unsupervised clustering
- Matching documents to predefined insurance-related topics
- Visualizing topic distributions and document embeddings
- Generating an interactive HTML report

The results will be saved in the `output` directory.

## Roadmap & Future Enhancements

*   **Enhanced NLP Cleaning:** Integration of more advanced text normalization and context-aware corrections.
*   **Additional Topic Models:** Further integrate other topic modeling techniques (e.g., NMF, CTM, neural topic models) beyond the existing LDA, BERTopic, and Top2Vec implementations.
*   **Expanded Reporting Options:** Additional templates and customization options for HTML report generation.
*   **User Feedback Loop:** Streamlined integration with active learning systems for continuous model improvement.
*   **Scalability:**  **Implement support for larger datasets through techniques like mini-batch processing (for LDA) and potentially distributed processing (e.g., using Dask).**
*   **Multilingual Support:**  **Extend support beyond English to other languages by leveraging multilingual LLMs and language-specific NLP tools.**
*   **Fine-tuning LLMs:** Allow users to fine-tune LLMs for specific domains or datasets.
*   **Additional Datasets:** Add more example datasets from different domains to showcase the package's versatility.
# Meno: Topic Modeling Toolkit

[![PyPI version](https://img.shields.io/pypi/v/meno.svg)](https://pypi.org/project/meno/)
[![Python Version](https://img.shields.io/pypi/pyversions/meno.svg)](https://pypi.org/project/meno/)
[![License](https://img.shields.io/github/license/srepho/meno.svg)](https://github.com/srepho/meno/blob/main/LICENSE)
[![Tests](https://github.com/srepho/meno/workflows/tests/badge.svg)](https://github.com/srepho/meno/actions?query=workflow%3Atests)
[![Downloads](https://img.shields.io/pypi/dm/meno.svg)](https://pypi.org/project/meno/)

Meno is a toolkit for topic modeling on messy text data, featuring an interactive workflow system that guides users from raw text to insights through acronym detection, spelling correction, topic modeling, and visualization.

## What's New in v0.8.0

- **Interactive Guided Workflow** - Step-by-step analysis flow with interactive reports
- **Acronym Detection & Expansion** - Find and expand domain-specific acronyms
- **Spelling Correction** - Detect and fix misspellings with contextual examples
- **Team Configuration System** - Share domain knowledge across your organization
- **Minimal Dependencies Mode** - Core features work without full ML dependencies

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

### Quick Test

To quickly check if the core functionality works, run the minimal test:

```bash
# Create output directory
mkdir -p output

# Run the minimal test script
python examples/minimal_test.py
```

This will generate interactive HTML reports for acronyms and misspellings in a synthetic insurance dataset.

## Quick Start

### Basic Topic Modeling

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

### Interactive Workflow (New in 0.8.0)

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
    ],
    "date": pd.date_range(start="2023-01-01", periods=4, freq="W"),
    "department": ["Executive", "Claims", "Technology", "Claims"],
    "region": ["North", "West", "North", "East"]
})

# Initialize the workflow
workflow = MenoWorkflow()

# 1. Load the data with column mappings
workflow.load_data(
    data=data,
    text_column="text",
    time_column="date",
    category_column="department", 
    geo_column="region"
)

# 2. Generate interactive acronym report
acronym_report_path = workflow.generate_acronym_report(
    output_path="acronym_report.html",
    open_browser=True  # Opens in your default browser
)

# 3. Apply acronym expansions with custom mappings
workflow.expand_acronyms({
    "CRM": "Customer Relationship Management",
    "HWY": "Highway",
    "EOB": "Explanation of Benefits",
    "CPT": "Current Procedural Terminology"
})

# 4. Generate interactive misspelling report
misspelling_report_path = workflow.generate_misspelling_report(
    output_path="misspelling_report.html",
    open_browser=True
)

# 5. Apply spelling corrections
workflow.correct_spelling({
    "vehical": "vehicle",
    "cust": "customer",
    "recieved": "received"
})

# 6. Preprocess and model
workflow.preprocess_documents()
workflow.discover_topics(num_topics=3)

# 7. Generate visualizations
workflow.visualize_topics(plot_type="embeddings").write_html("topic_embeddings.html")
workflow.visualize_topics(plot_type="trends").write_html("topic_trends.html")

# 8. Create comprehensive report
report_path = workflow.generate_comprehensive_report(
    output_path="topic_report.html",
    include_interactive=True,
    open_browser=True
)
```

## Overview

Meno is designed to streamline topic modeling on free text data, with a special focus on messy datasets such as insurance claims notes and customer correspondence. The package combines classical methods like Latent Dirichlet Allocation (LDA) with modern techniques leveraging large language models (LLMs) via Hugging Face, dimensionality reduction with UMAP, and advanced visualizations. It is built to be primarily used in Jupyter environments while also being flexible enough for other settings.

## Key Features

*   **Interactive Workflow (New in v0.8.0):**
    *   Guided workflow that takes users from raw data to final visualization
    *   Interactive reporting of potential acronyms with expansion suggestions
    *   Spelling detection and correction with context examples
    *   Seamless connection to visualization and reporting
    
*   **Unsupervised Topic Modeling:**
    *   Automatically discover topics when no pre-existing topics are available using LDA and LLM-based embedding and clustering techniques.
*   **Supervised Topic Matching:**
    *   Match free text against a user-provided list of topics using semantic similarity and classification techniques.
*   **Advanced Visualization:**
    *   Create interactive and static visualizations including topic distributions, embeddings (UMAP projections), cluster analyses, and topic coherence metrics (e.g., word clouds per topic).
    *   Time series, geospatial, and combined time-space visualizations for deeper analysis.
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

Meno provides several ways to use the topic modeling functionalities, from simple usage patterns to more advanced configurations. Here are comprehensive examples showing different ways to use the package:

### Basic Topic Discovery

```python
import pandas as pd
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

# Generate a simple report
report_path = modeler.generate_report(output_path="basic_topic_report.html")
```

### Advanced Topic Discovery with Configuration

```python
import pandas as pd
from meno import MenoTopicModeler
from meno.utils.config import load_config

# Load custom configuration from YAML file
config = load_config("my_custom_config.yaml")

# Or create configuration programmatically with overrides
from meno.utils.config import MenoConfig, ClusteringConfig, UMAPConfig
custom_config = MenoConfig(
    modeling=ModelingConfig(
        clustering=ClusteringConfig(
            algorithm="hdbscan",
            min_cluster_size=20,
            min_samples=7
        )
    ),
    visualization=VisualizationConfig(
        umap=UMAPConfig(
            n_neighbors=20,
            min_dist=0.2
        )
    )
)

# Initialize modeler with custom config
modeler = MenoTopicModeler(config=custom_config)

# Load and preprocess data with custom parameters
df = pd.read_csv("my_documents.csv")
processed_docs = modeler.preprocess(
    df, 
    text_column="document_text",
    lowercase=True,
    remove_punctuation=True,
    lemmatize=True,
    custom_stopwords=["specific", "words", "to", "remove"]
)

# Discover topics with specific method and parameters
topics_df = modeler.discover_topics(
    method="embedding_cluster", 
    num_topics=10,
    clustering_algorithm="hdbscan",
    min_cluster_size=20
)

# Generate detailed embeddings visualization
fig = modeler.visualize_embeddings(
    plot_3d=True,
    include_topic_centers=True,
    marker_size=7
)
fig.show()
```

### Matching Documents to Predefined Topics

```python
import pandas as pd
from meno import MenoTopicModeler

# Initialize modeler
modeler = MenoTopicModeler()

# Load and preprocess data
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
    "Technical issues, bugs, crashes, or performance problems with the product",
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

# View the topic assignments
print(matched_df[["description", "topic", "topic_probability"]].head())

# Analyze distribution of topics
import matplotlib.pyplot as plt
topic_counts = matched_df["topic"].value_counts()
plt.figure(figsize=(10, 6))
topic_counts.plot(kind="bar")
plt.title("Distribution of Support Ticket Topics")
plt.tight_layout()
plt.show()
```

### Topic Evolution Over Time

```python
import pandas as pd
from meno import MenoTopicModeler
import matplotlib.pyplot as plt

# Initialize modeler
modeler = MenoTopicModeler()

# Load data with timestamps
df = pd.read_csv("news_articles.csv")
df["date"] = pd.to_datetime(df["date"])

# Preprocess and model
processed_docs = modeler.preprocess(df, text_column="article_text")
topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=8)

# Get topic distribution over time
df_with_topics = df.copy()
df_with_topics["topic"] = topics_df["topic"]

# Convert to monthly data
monthly_topics = df_with_topics.groupby([pd.Grouper(key="date", freq="M"), "topic"]).size().unstack().fillna(0)

# Plot topic evolution
plt.figure(figsize=(12, 6))
monthly_topics.plot(kind="line", stacked=False)
plt.title("Topic Evolution Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.legend(title="Topic")
plt.tight_layout()
plt.show()
```

### Generating Enhanced Interactive Reports

```python
import pandas as pd
import numpy as np
from meno import MenoTopicModeler

# Initialize and process data
modeler = MenoTopicModeler()
df = pd.read_csv("customer_feedback.csv")
processed_docs = modeler.preprocess(df, text_column="feedback")
topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=6)

# Create word frequency dictionaries for each topic
topic_words = {}
for topic in topics_df["topic"].unique():
    # Get documents for this topic
    topic_docs = df[topics_df["topic"] == topic]["feedback"]
    
    # Create a word frequency dictionary
    words = " ".join(topic_docs).lower().split()
    word_freq = {}
    for word in set(words):
        if len(word) > 3:  # Only include words with more than 3 characters
            word_freq[word] = words.count(word)
    
    topic_words[topic] = word_freq

# Create a similarity matrix between topics
topic_names = sorted(topics_df["topic"].unique())
n_topics = len(topic_names)
similarity_matrix = np.zeros((n_topics, n_topics))

# Fill with random similarities for demonstration
# In practice, you would calculate actual similarities between topic embeddings
for i in range(n_topics):
    for j in range(n_topics):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            # Random similarity between 0.1 and 0.6
            similarity_matrix[i, j] = 0.1 + 0.5 * np.random.random()

# Generate enhanced report with all visualizations
report_path = modeler.generate_report(
    output_path="enhanced_report.html",
    include_interactive=True,
    title="Customer Feedback Analysis",
    include_raw_data=True,
    similarity_matrix=similarity_matrix,
    topic_words=topic_words
)

print(f"Enhanced report generated at: {report_path}")
```

### Custom Topic Modeling Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from meno import MenoTopicModeler
from meno.preprocessing.text_processor import preprocess_text

# Load data
df = pd.read_csv("scientific_papers.csv")

# Custom preprocessing
texts = df["abstract"].tolist()
processed_texts = [preprocess_text(
    text, 
    lowercase=True,
    remove_punctuation=True,
    remove_numbers=True,
    lemmatize=True,
    language="en"
) for text in texts]

# Create dictionary and corpus for LDA
dictionary = Dictionary([text.split() for text in processed_texts])
corpus = [dictionary.doc2bow(text.split()) for text in processed_texts]

# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    passes=20,
    alpha="auto",
    eta="auto"
)

# Get topic assignments
topic_probs = [lda_model.get_document_topics(doc) for doc in corpus]
primary_topics = [max(probs, key=lambda x: x[1])[0] if probs else -1 for probs in topic_probs]
topic_probabilities = [max(probs, key=lambda x: x[1])[1] if probs else 0.0 for probs in topic_probs]

# Create dataframe with results
results_df = pd.DataFrame({
    "text": df["abstract"],
    "processed_text": processed_texts,
    "topic": [f"Topic_{t}" for t in primary_topics],
    "topic_probability": topic_probabilities
})

# Get top words for each topic
topic_words = {}
for i in range(lda_model.num_topics):
    topic_name = f"Topic_{i}"
    words = dict(lda_model.show_topic(i, topn=20))
    topic_words[topic_name] = words

# Initialize MenoTopicModeler for visualization and reporting
modeler = MenoTopicModeler()

# Set the documents and topic assignments
modeler.documents = results_df
modeler.topic_assignments = results_df[["topic", "topic_probability"]]

# Generate embeddings for visualization
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(results_df["text"].tolist())
modeler.document_embeddings = embeddings

# Generate UMAP projection
from umap import UMAP
umap_model = UMAP(n_components=2, random_state=42)
umap_projection = umap_model.fit_transform(embeddings)
modeler.umap_projection = umap_projection

# Generate report with custom topic words
report_path = modeler.generate_report(
    output_path="scientific_papers_topics.html",
    title="Scientific Paper Topics Analysis",
    topic_words=topic_words
)
```

### Working with Large Datasets Using Chunking

```python
import pandas as pd
from meno import MenoTopicModeler
from meno.utils.data_chunker import process_in_chunks

# Initialize modeler
modeler = MenoTopicModeler()

# Load a large dataset
df = pd.read_csv("large_dataset.csv")  # This could be millions of documents

# Define a function to process chunks
def process_chunk(chunk_df):
    # Preprocess the chunk
    processed = modeler.preprocess(chunk_df, text_column="content")
    return processed

# Process in chunks
chunk_size = 10000
processed_chunks = process_in_chunks(df, process_chunk, chunk_size)

# Combine processed chunks
processed_df = pd.concat(processed_chunks, ignore_index=True)

# Now discover topics on the combined processed data
topics_df = modeler.discover_topics(
    method="embedding_cluster", 
    num_topics=15,
    batch_size=5000  # Process embeddings in batches
)

# Generate report
report_path = modeler.generate_report(output_path="large_dataset_topics.html")
```

### Integrating with Active Learning

```python
import pandas as pd
import numpy as np
from meno import MenoTopicModeler
from meno.active_learning.uncertainty_sampler import UncertaintySampler

# Initialize modeler
modeler = MenoTopicModeler()

# Load data
df = pd.read_csv("unlabeled_data.csv")
processed_docs = modeler.preprocess(df, text_column="text")

# Initial topic discovery
topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=8)

# Identify uncertain samples for human labeling
sampler = UncertaintySampler(threshold=0.6)
uncertain_indices = sampler.get_uncertain_samples(
    topic_probs=topics_df["topic_probability"],
    n_samples=20
)

# Display samples that need human verification
uncertain_samples = df.iloc[uncertain_indices]
print("Please label these uncertain samples:")
for i, (idx, row) in enumerate(uncertain_samples.iterrows()):
    print(f"{i+1}. Text: {row['text'][:100]}...")
    print(f"   Current topic: {topics_df.iloc[idx]['topic']}")
    print(f"   Confidence: {topics_df.iloc[idx]['topic_probability']:.2f}")
    print("   Suggested topics: ", ", ".join(modeler.get_candidate_topics(idx, top_n=3)))
    print()

# After human labeling, update the model
# (This would be part of an interactive loop in a real application)
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
# Meno: Topic Modeling Toolkit (v1.1.1)

<p align="center">
  <img src="meno.webp" alt="Meno Logo" width="250"/>
</p>

[![License](https://img.shields.io/github/license/srepho/meno)](https://github.com/srepho/meno/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![GitHub Stars](https://img.shields.io/github/stars/srepho/meno?style=social)](https://github.com/srepho/meno)

Meno is a toolkit for topic modeling on messy text data, featuring an interactive workflow system that guides users from raw text to insights through acronym detection, spelling correction, topic modeling, and visualization. It includes both high-powered models and lightweight alternatives that work without heavy dependencies.

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

2. Manually download the necessary files for your chosen model. You can find these files on the model's Hugging Face page under the "Files and versions" tab. You need:
   - config.json
   - pytorch_model.bin
   - special_tokens_map.json
   - tokenizer.json
   - tokenizer_config.json
   - vocab.txt (if applicable)
   - modules.json (for Sentence Transformers models)
   
   Download these files and place them in a local directory.

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

## What's New in v1.1.0

- **Enhanced Lightweight Models** - Four CPU-optimized topic models with minimal dependencies
- **Interactive Feedback System** - Notebook-friendly interface for refining topic assignments
- **Feedback Visualization Tools** - Specialized visualizations to analyze feedback impact
- **Integrated Components** - Seamless integration between models, visualizations, and web interface
- **Improved Documentation** - Comprehensive guides for all components
- **New Example Scripts** - Demonstrations of all features working together
- **Advanced Visualizations** - New comparative visualization tools for lightweight models
- **Web Interface Improvements** - Better support for lightweight models in the interactive UI
- **Performance Enhancements** - Faster processing and reduced memory usage

## What's in v1.0.0

- **Standardized API** - Consistent parameter names and method signatures across all models
- **Automatic Topic Detection** - Models can discover the optimal number of topics automatically
- **Enhanced Memory Efficiency** - Process larger datasets with streaming and quantization
- **Path Object Support** - Better file handling with pathlib integration
- **Return Type Standardization** - Consistent return values across all methods
- **Advanced Preprocessing** - Context-aware spelling correction and acronym expansion
- **Domain-Specific Adapters** - Medical, technical, financial, and legal domain support
- **Cross-Document Learning** - Learns terminology and acronyms across multiple documents
- **Performance Optimizations** - Parallel and batch processing for large datasets
- **Evaluation Framework** - Metrics to measure correction quality and improvement
- **Lightweight Topic Models** - CPU-optimized models with minimal dependencies for large datasets
- **Advanced Visualizations** - New comparative visualization tools for topic models
- **Web Interface** - Interactive no-code UI for topic modeling exploration

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
  - Lightweight models optimized for performance (SimpleTopicModel, TFIDFTopicModel, NMFTopicModel, LSATopicModel)

- **Web Interface for No-Code Exploration**
  - Interactive data upload and preprocessing
  - Model configuration and training through UI
  - Topic exploration and visualization
  - Document search and filtering
  - Customizable and extensible Dash-based interface

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
  - Advanced topic comparison visualizations

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

# For web interface and interactive UI
pip install "meno[web]"

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

### Advanced Text Preprocessing with Domain Knowledge

```python
from meno.preprocessing.acronyms import AcronymExpander
from meno.preprocessing.spelling import SpellingCorrector
from meno.nlp.domain_adapters import get_domain_adapter
import pandas as pd

# Load data
df = pd.read_csv("medical_records.csv")

# Get domain-specific adapter for medical text
medical_adapter = get_domain_adapter("healthcare")

# Create enhanced spelling corrector and acronym expander
spelling_corrector = SpellingCorrector(
    domain="medical",
    min_word_length=3,
    use_keyboard_proximity=True,
    learn_corrections=True
)

acronym_expander = AcronymExpander(
    domain="healthcare",
    ignore_case=True,
    contextual_expansion=True
)

# Process text with domain knowledge
df["corrected_text"] = df["text"].apply(spelling_corrector.correct_text)
df["processed_text"] = df["corrected_text"].apply(acronym_expander.expand_acronyms)

# Initialize modeler with preprocessed text
modeler = MenoTopicModeler()
modeler.preprocess(df, text_column="processed_text")

# Continue with topic modeling...
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

### Using Lightweight Topic Models

```python
from meno.modeling.simple_models.lightweight_models import (
    SimpleTopicModel,
    TFIDFTopicModel,
    NMFTopicModel,
    LSATopicModel
)

# Create a TF-IDF based model (extremely fast and lightweight)
tfidf_model = TFIDFTopicModel(num_topics=10, max_features=2000)
tfidf_model.fit(documents)

# Get topic information and visualize
topic_info = tfidf_model.get_topic_info()
print(topic_info)

# Create an NMF model for more interpretable topics
nmf_model = NMFTopicModel(num_topics=8, max_features=1500)
nmf_model.fit(documents)

# Compare document-topic distributions
doc_topic_matrix = nmf_model.transform(test_documents)
print(f"Document-topic matrix shape: {doc_topic_matrix.shape}")

# Visualize topics
fig = nmf_model.visualize_topics(width=1000, height=600)
fig.write_html("nmf_topics.html")

# Simple K-means based model with embeddings
from meno.modeling.embeddings import DocumentEmbedding
embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
simple_model = SimpleTopicModel(num_topics=5, embedding_model=embedding_model)
simple_model.fit(documents)
```

For more detailed examples, see [LIGHTWEIGHT_MODELS_DOCUMENTATION.md](LIGHTWEIGHT_MODELS_DOCUMENTATION.md).

### Advanced Topic Visualizations

```python
from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape,
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)

# Create multiple models for comparison
tfidf_model = TFIDFTopicModel(num_topics=5)
nmf_model = NMFTopicModel(num_topics=5)
lsa_model = LSATopicModel(num_topics=5)

# Fit all models on the same data
for model in [tfidf_model, nmf_model, lsa_model]:
    model.fit(documents)

# Compare multiple models side-by-side
fig = plot_model_comparison(
    document_lists=[documents, documents, documents],
    model_names=["TF-IDF", "NMF", "LSA"],
    models=[tfidf_model, nmf_model, lsa_model]
)
fig.write_html("model_comparison.html")

# Create topic landscape visualization with dimensionality reduction
fig = plot_topic_landscape(
    model=nmf_model,
    documents=documents,
    method="umap"  # Can also use 'pca' if UMAP not available
)
fig.write_html("topic_landscape.html")

# Generate topic similarity heatmap between models
fig = plot_multi_topic_heatmap(
    models=[nmf_model, lsa_model],
    model_names=["NMF", "LSA"],
    document_lists=[documents, documents]
)
fig.write_html("topic_heatmap.html")

# Analyze how documents relate to different topics
fig = plot_comparative_document_analysis(
    model=nmf_model,
    documents=documents[:10],  # Show first 10 documents
    title="Document Topic Analysis"
)
fig.write_html("document_analysis.html")
```

For complete examples, see `examples/lightweight_models_visualization.py` and `examples/integrated_components_example.py`.

### Using the Web Interface

```python
from meno.web_interface import launch_web_interface
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel, NMFTopicModel

# Create and train some models
tfidf_model = TFIDFTopicModel(num_topics=5)
nmf_model = NMFTopicModel(num_topics=5)
tfidf_model.fit(documents)
nmf_model.fit(documents)

# Launch the web interface with pre-trained models
launch_web_interface(
    port=8050, 
    debug=True,
    models={
        "TF-IDF Model": tfidf_model,
        "NMF Model": nmf_model
    },
    data=df,  # Optional: pass a dataframe with your documents
    text_column="text"  # Specify which column contains the text
)
```

Or run from the command line:

```bash
# Basic launch
meno-web --port 8050

# Launch with debugging enabled
meno-web --port 8050 --debug

# Launch with specific model types
meno-web --port 8050 --models tfidf nmf lsa
```

See `examples/web_lightweight_example.py` for a complete example of using the web interface with lightweight models.

### Interactive Topic Feedback with Visualizations

```python
from meno import MenoTopicModeler
from meno import TopicFeedbackManager, plot_feedback_impact

# Run initial topic modeling
modeler = MenoTopicModeler()
modeler.preprocess(df, text_column="text")
modeler.discover_topics(method="embedding_cluster", num_topics=5)

# Create feedback manager
feedback_manager = TopicFeedbackManager(modeler)

# Set up with descriptive topic information
feedback_system = feedback_manager.setup_feedback(
    n_samples=20,  # Number of documents to review
    uncertainty_ratio=0.7,  # Focus on uncertain documents
    topic_descriptions=["Description for Topic 1", "Description for Topic 2", ...],
)

# Start interactive review (in a Jupyter notebook)
feedback_manager.start_review()

# After providing feedback, apply updates
feedback_system.apply_updates()

# Get the updated model
updated_modeler = feedback_manager.get_updated_model()

# Export feedback for collaboration
feedback_system.export_to_csv("topic_feedback.csv")

# Visualize the impact of feedback on topics
import matplotlib.pyplot as plt
fig = plot_feedback_impact(feedback_manager)
plt.figure(fig.number)
plt.savefig("feedback_impact.png")

# Analyze topic-specific changes
from meno import plot_topic_feedback_distribution
original_topics = []  # Stored from before feedback
current_topics = updated_modeler.get_document_topics()["topic"].tolist()
fig = plot_topic_feedback_distribution(
    updated_modeler,
    documents,
    original_topics,
    current_topics,
    show_wordclouds=True
)
plt.figure(fig.number)
plt.savefig("topic_distribution_changes.png")

# For web-based interactive dashboard (requires dash)
from meno import create_feedback_comparison_dashboard
app = create_feedback_comparison_dashboard(
    before_model=modeler,  # Before feedback
    after_model=updated_modeler,  # After feedback
    documents=documents,
    title="Feedback Impact Analysis"
)
app.run_server(debug=True)
```

See `examples/feedback_visualization_example.py`, `examples/feedback_visualization_notebook.ipynb`, and `examples/interactive_feedback_example.py` for complete examples of using the feedback system with visualizations.

See the example scripts in the [examples directory](examples/) for more detailed usage.

## Future Development

With v1.1.0 enhancing our lightweight components, we're now focusing on:

1. **Incremental Learning** - Support for streaming data and updating models
2. **Advanced Model Integration** - Better integration with external models
3. **Multilingual Support** - Expand beyond English
4. **Domain-Specific Fine-Tuning** - Adapt models to specific industries
5. **Explainable AI Features** - Better interpret topic assignments
6. **Interactive Dashboards** - More powerful visualization tools
7. **Cloud Integration** - Native support for cloud-based services
8. **Export/Import Format** - Standard format for sharing models and results
9. **Extension API** - Plugin system for custom models and visualizations

See our [detailed roadmap](ROADMAP.md) for more information and the [INTEGRATED_COMPONENTS_SUMMARY.md](INTEGRATED_COMPONENTS_SUMMARY.md) for details on our recent work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
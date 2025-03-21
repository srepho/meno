# Meno: Topic Modeling Toolkit (v1.3.4)

<p align="center">
  <img src="meno.webp" alt="Meno Logo" width="250"/>
</p>

[![License](https://img.shields.io/github/license/srepho/meno)](https://github.com/srepho/meno/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![PyPI version](https://img.shields.io/pypi/v/meno.svg)](https://pypi.org/project/meno/)
[![Build Status](https://github.com/srepho/meno/actions/workflows/ci-cpu.yml/badge.svg)](https://github.com/srepho/meno/actions)
[![Code Coverage](https://codecov.io/gh/srepho/meno/branch/main/graph/badge.svg)](https://codecov.io/gh/srepho/meno)
[![Downloads](https://static.pepy.tech/badge/meno)](https://pepy.tech/project/meno)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/srepho/meno/main?filepath=examples/notebooks)
[![GitHub Stars](https://img.shields.io/github/stars/srepho/meno?style=social)](https://github.com/srepho/meno)

Meno is a toolkit for topic modeling on messy text data, featuring an interactive workflow system that guides users from raw text to insights through acronym detection, spelling correction, topic modeling, and visualization. It includes both high-powered models and lightweight alternatives that work without heavy dependencies.

## Quick Start

### Minimal Example (5 Lines)

```python
from meno import MenoTopicModeler
import pandas as pd

# Load data and discover topics
modeler = MenoTopicModeler()
df = pd.read_csv("documents.csv")
modeler.preprocess(df, text_column="text")
topics_df = modeler.discover_topics(method="embedding_cluster", auto_detect_topics=True)
modeler.generate_report(output_path="topic_report.html")
```

### Lightweight CPU-Optimized Example (No Heavy Dependencies)

```python
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel
import pandas as pd

# Load data
df = pd.read_csv("documents.csv")
documents = df["text"].tolist()

# Create and fit a lightweight TF-IDF model
model = TFIDFTopicModel(auto_detect_topics=True, max_features=2000)
model.fit(documents)

# Print topics and save visualization
print(model.get_topic_info())
model.visualize_topics().write_html("tfidf_topics.html")
```

### Multi-Provider LLM Integration (NEW!)

```python
from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi

# Use OpenAI (original functionality)
openai_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="your-openai-api-key",
    provider="openai",
    model_name="gpt-3.5-turbo"
)

# Use Google Gemini
gemini_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="your-google-api-key",
    provider="google",
    model_name="gemini-pro",
    library="sdk"  # Use the official SDK
)

# Use Anthropic Claude
claude_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="your-anthropic-api-key",
    provider="anthropic",
    model_name="claude-3-haiku-20240307",
    library="requests",  # Use direct HTTP requests
    enable_cache=True    # Enable caching for all providers
)

# Use Hugging Face Inference API
hf_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="your-huggingface-api-key",
    provider="huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
)

# Use AWS Bedrock
bedrock_response = generate_text_with_llm_multi(
    text="Summarize the key benefits of topic modeling",
    api_key="your-aws-access-key",
    api_secret="your-aws-secret-key",
    provider="bedrock",
    model_name="anthropic.claude-3-sonnet-20240229",
    region_name="us-east-1"
)
```

### Direct LLM API Usage with Caching and Optimization

```python
from meno.modeling.llm_topic_labeling import (
    generate_call_from_text,
    process_texts_with_threadpool,
    identify_fuzzy_duplicates,
    process_texts_with_deduplication,
    generate_text_with_llm
)

# Simple direct API call with caching
api_key = "your_api_key_here"
api_endpoint = "https://api.openai.com/v1/chat/completions"
system_prompt = "You are a topic labeling assistant."
text = "Please analyze this document and suggest a topic"

result = generate_call_from_text(
    text=text, 
    api_key=api_key, 
    api_endpoint=api_endpoint,
    system_prompt=system_prompt,
    enable_cache=True,  # Cache responses to avoid duplicate API calls
    cache_ttl=3600      # Cache time-to-live in seconds (1 hour)
)
print(f"Generated topic: {result}")

# Process multiple texts concurrently with caching
texts = ["Sample document 1", "Sample document 2", "Sample document 3"]
results = process_texts_with_threadpool(
    texts=texts,
    api_key=api_key,
    api_endpoint=api_endpoint,
    max_workers=3,      # Process up to 3 texts simultaneously
    enable_cache=True,  # Automatically use cached responses when available
    show_progress=True  # Show progress updates
)

# Print results with cache status
for r in results:
    print(f"Document: {r['input'][:50]}...")
    print(f"Topic: {r['response']}")
    print(f"Processing time: {r['time_taken']:.2f} seconds")
    print(f"From cache: {r.get('from_cache', False)}")

# Process with optimized deduplication to save on API costs
texts_with_duplicates = [
    "Keywords: finance, investment, banking",
    "Keywords: banking, finance, investment",  # Similar to first text
    "Keywords: finance, markets, stocks",      # Similar to first text
    "Keywords: technology, computers, software",
    "Keywords: computer technology, software"  # Similar to fourth text
]

# Process with advanced deduplication and caching
results = process_texts_with_deduplication(
    texts=texts_with_duplicates,
    api_key=api_key,
    api_endpoint=api_endpoint,
    deduplicate=True,               # Enable deduplication
    deduplication_threshold=0.85,   # Similarity threshold
    enable_cache=True,              # Enable response caching
    preprocess_for_deduplication=True,  # Pre-process texts for faster comparison
    show_progress=True              # Show detailed progress and stats
)

# Generate text using simplified unified API (works with both OpenAI and Azure)
response = generate_text_with_llm(
    text="Classify this document by topic",
    api_key="your_api_key",
    
    # For standard OpenAI API:
    api_endpoint="https://api.openai.com/v1/chat/completions",
    model_name="gpt-4o",
    use_azure=False,
    
    # For Azure OpenAI API:
    # api_endpoint="https://your-resource.openai.azure.com",
    # deployment_id="your-deployment-name",  # Required for Azure
    # use_azure=True,
    
    # Common parameters:
    system_prompt="You are a topic classification expert.",
    temperature=0.7,
    max_tokens=200
)
print(f"Response: {response}")
```

### Complete Interactive Workflow

```python
from meno import MenoWorkflow
import pandas as pd

# Sample data with acronyms and misspellings
data = pd.DataFrame({
    "text": [
        "The CEO and CFO discussed AI implementation in our CRM system.",
        "Customer submitted a claim for their vehical accident on HWY 101.",
        "The CTO presented ML strategy for improving cust retention.",
        "Policyholder recieved the EOB and was confused about CPT codes."
    ]
})

# Initialize workflow
workflow = MenoWorkflow()
workflow.load_data(data=data, text_column="text")

# 1. Detect and expand acronyms
workflow.generate_acronym_report(output_path="acronyms.html", open_browser=True)
workflow.expand_acronyms({"CRM": "Customer Relationship Management", 
                          "CTO": "Chief Technology Officer"})

# 2. Detect and correct spelling
workflow.generate_misspelling_report(output_path="misspellings.html", open_browser=True)
workflow.correct_spelling({"vehical": "vehicle", "recieved": "received"})

# 3. Process documents and discover topics
workflow.preprocess_documents(lowercase=True, remove_stopwords=True)
workflow.discover_topics(auto_detect_topics=True)

# 4. Generate final report
workflow.generate_comprehensive_report("final_report.html", open_browser=True)
```

## Installation

Choose the installation option that best fits your needs:

```bash
# Lightweight (basic topic modeling, minimal dependencies)
pip install "meno[lightweight]"

# Standard (full-featured, CPU-optimized)
pip install "meno[cpu]" -f https://download.pytorch.org/whl/torch_stable.html

# GPU-accelerated (maximum performance)
pip install "meno[gpu]"
```

For more installation options, see our [Simplified Installation Guide](SIMPLIFIED_INSTALL.md).

### Continuous Integration and GPU Testing

Meno uses GitHub Actions for continuous integration and testing:

```bash
# The repository includes workflows for:
├── .github/workflows/
│   ├── secret-scanning.yml     # Scans code for accidentally committed credentials
│   ├── ci-cpu.yml              # Runs tests on CPU environments
│   ├── ci-gpu.yml              # Executes GPU-specific tests
│   └── scheduled-testing.yml   # Weekly tests across Python versions
```

The secret scanning workflow automatically detects and prevents:
- API keys and tokens
- Passwords and credentials
- AWS access keys
- Other sensitive information

GPU tests run on:
- Self-hosted runners with GPU (see `.github/self-hosted-runner-setup.md`)
- Azure ML GPU compute (see `.azure/gpu-test-job.yml`)
- GitHub-hosted runners with GPU support

For local GPU testing:

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Run GPU-specific tests
pytest tests/test_optimized_embedding.py -v
```

### Offline Installation (Air-gapped Environments)

For environments with limited internet access:

```python
from meno.modeling.embeddings import DocumentEmbedding

# Use a local model (pre-downloaded)
embedding_model = DocumentEmbedding(
    local_model_path="/path/to/local/model",  # Path to downloaded model files
    local_files_only=True
)

# Initialize modeler with offline mode
from meno import MenoTopicModeler
modeler = MenoTopicModeler(
    embedding_model=embedding_model,
    offline_mode=True
)
```

See `examples/specialized/offline_model.py` for detailed offline usage.

## Feature Examples

### Human-Readable Topic Names with LLM Labeling

```python
from meno import MenoTopicModeler
import pandas as pd

# Load data
df = pd.read_csv("documents.csv")

# Initialize with LLM labeling enabled
modeler = MenoTopicModeler(
    use_llm_labeling=True,  # Enable LLM labeling for human-readable names
    llm_model_type="local",  # Use local model (or "openai")
    llm_model_name="google/flan-t5-small"  # Small, fast model
)

# Preprocess and discover topics
modeler.preprocess(df, text_column="text")
topics_df = modeler.discover_topics(method="embedding_cluster", auto_detect_topics=True)

# Print topics with LLM-generated names
topic_info = modeler.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]])
```

### CPU-Optimized Topic Modeling

```python
from meno import MenoTopicModeler
import pandas as pd

# Define CPU-optimized configuration
CPU_CONFIG = {
    "preprocessing": {
        "normalization": {
            "lowercase": True,  # Convert text to lowercase
            "remove_punctuation": True,
            "lemmatize": True,
        },
    },
    "modeling": {
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",  # Small, fast model
            "device": "cpu",                   # Explicitly use CPU
            "batch_size": 32,                  # CPU-optimized batch size
            "quantize": True,                  # Reduce memory usage
        },
    },
}

# Load data and initialize modeler with CPU configuration
df = pd.read_csv("documents.csv")
modeler = MenoTopicModeler(config_overrides=CPU_CONFIG)

# Process and generate report
modeler.preprocess(df, text_column="text")
modeler.discover_topics(method="embedding_cluster", auto_detect_topics=True)
modeler.generate_report(output_path="cpu_optimized_report.html")
```

### Advanced Visualization

```python
from meno import MenoTopicModeler
import pandas as pd

# Load data and run topic modeling
modeler = MenoTopicModeler()
df = pd.read_csv("documents.csv")
modeler.preprocess(df, text_column="text")
modeler.discover_topics(method="embedding_cluster", num_topics=8)

# Create 3D interactive visualization
fig_3d = modeler.visualize_embeddings(
    plot_3d=True,               # Create 3D visualization
    include_topic_centers=True, # Show topic centers
    width=1000,                 # Set figure dimensions
    height=800
)
fig_3d.write_html("3d_topic_visualization.html")

# Create topic similarity heatmap
heatmap = modeler.visualize_topic_similarity(return_figure=True)
heatmap.write_html("topic_similarity.html")

# Create wordcloud visualization
modeler.visualize_topic_words(topic_id=1, output_path="topic1_wordcloud.html")
```

## What's New in v1.3.1

- **Enhanced OpenAI Integration**
  - **New Utility Function** - Added `generate_text_with_llm` utility for simple direct API access
  - **Improved Parameter Handling** - Consistent API for both Azure and standard OpenAI endpoints
  - **Simplified API Interface** - Renamed `openai_api_key` to `api_key` for consistency
  - **Better Documentation** - Enhanced examples for both Azure and standard OpenAI usage

- **Bug Fixes**
  - Fixed Azure OpenAI integration with proper deployment_id parameter usage
  - Fixed client initialization for both Azure and standard OpenAI APIs
  - Improved error messages and parameter validation

## What's New in v1.2.10

- **Bug Fixes**
  - Fixed critical OpenAI API integration for classification_texts method
  - Improved API parameter handling for OpenAI chat completions
  - Fixed message formatting for both Azure and standard OpenAI endpoints

- **Enhanced Features**
  - **Deduplication** - Remove duplicate documents for cleaner topic modeling results
  - **Fuzzy Deduplication** - Identify and filter near-duplicate content
  - **Memory Optimization** - Process larger datasets with limited memory resources
  - **Topic Drift Visualization** - Track how topics evolve over time
  - **Incremental Topic Updates** - Update existing models with new data efficiently
  
- **Performance Improvements**
  - Optimized embedding generation for CPU environments
  - Reduced memory usage for large dataset processing
  - Improved speed for lightweight topic models

- **New Documentation & Examples**
  - Reorganized example files by category:
    - `examples/basic/` - Simple examples to get started
    - `examples/advanced/` - Advanced features like deduplication and incremental learning
    - `examples/models/` - Different topic modeling backends
    - `examples/visualization/` - Specialized visualizations
    - `examples/specialized/` - Domain-specific use cases
    - `examples/notebooks/` - Jupyter notebook tutorials
  - Added new examples for all recent features
  - Improved documentation for specialized use cases

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

| Installation Method | Command | Features Included |
|---------------------|---------|------------------|
| Basic | `pip install meno` | Core functionality, basic preprocessing, simple topic models |
| Minimal | `pip install "meno[minimal]"` | Essential topic modeling dependencies, lightweight models |
| LLM Topic Labeling | `pip install "meno[llm]"` | Local HuggingFace models for topic naming |
| OpenAI Integration | `pip install "meno[llm_openai]"` | OpenAI API for topic naming |
| CPU-optimized | `pip install "meno[embeddings]"` | Optimized for CPU-only environments |
| BERTopic & Top2Vec | `pip install "meno[additional_models]"` | Advanced topic modeling approaches |
| GPU Acceleration | `pip install "meno[embeddings-gpu]"` | GPU-accelerated embeddings |
| LDA Models | `pip install "meno[lda]"` | Traditional LDA topic modeling |
| Visualization | `pip install "meno[viz]"` | Enhanced visualization capabilities |
| NLP Processing | `pip install "meno[nlp]"` | Advanced NLP preprocessing capabilities |
| Large Datasets | `pip install "meno[optimization]"` | Polars for large dataset optimization |
| Memory Efficiency | `pip install "meno[memory_efficient]"` | Quantized models, reduced memory usage |
| Web Interface | `pip install "meno[web]"` | Interactive web UI for exploration |
| Complete (CPU) | `pip install "meno[full]"` | All features (CPU optimized) |
| Complete (GPU) | `pip install "meno[full-gpu]"` | All features with GPU acceleration |
| Development | `pip install "meno[dev,test]"` | Development and testing tools |

```bash
# Example: Install with LLM topic labeling support
pip install "meno[llm]"  # For local HuggingFace models
pip install "meno[llm_openai]"  # For OpenAI API integration

# Example: Install with CPU optimization
pip install "meno[embeddings]" -f https://download.pytorch.org/whl/torch_stable.html
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

### Advanced BERTopic Features

```python
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.embeddings import DocumentEmbedding
import pandas as pd

# Load data
df = pd.read_csv("documents.csv")
documents = df["text"].tolist()

# Create embedding model
embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")

# Create BERTopic model with LLM topic labeling
model = BERTopicModel(
    auto_detect_topics=True,  # Let the model determine the optimal number of topics
    embedding_model=embedding_model,
    min_topic_size=5,
    use_llm_labeling=True,  # Enable LLM labeling
    llm_model_type="local",  # Use local model (or "openai")
    llm_model_name="google/flan-t5-small"  # Or any other HuggingFace model
)

# Fit model
model.fit(documents)

# Print topics with LLM-generated names
topic_info = model.get_topic_info()
print(topic_info[["Topic", "Count", "Name"]])

# Topic manipulation: merge similar topics
topics_to_merge = [[0, 1], [2, 3, 4]]  # Merge topics 0&1 and 2&3&4
model.merge_topics(topics_to_merge, documents=documents)

# Reduce to a specific number of topics
model.reduce_topics(documents, nr_topics=5)

# Create a second model for a different dataset
second_model = BERTopicModel(num_topics=6, embedding_model=embedding_model)
second_model.fit(other_documents)

# Merge models
merged_model = model.merge_models(
    models=[second_model],
    documents=documents + other_documents,
    min_similarity=0.7
)

# Dynamic topic modeling with timestamps
topics, probs, timestamps = model.fit_transform_with_timestamps(
    documents=documents_with_time,
    timestamps=timestamp_list,
    global_tuning=True
)

# Visualize topics
model.visualize_topics().write_html("topic_similarity.html")
model.visualize_topics_over_time().write_html("topics_over_time.html")
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

## Common Troubleshooting

- **Lowercase not working:** Ensure you're explicitly setting `lowercase=True` when calling `preprocess_documents()` or passing it in your config
- **Missing visualizations:** Install the visualization dependencies with `pip install "meno[viz]"`
- **Memory errors:** Try setting `quantize=True` and `low_memory=True` in your config
- **Slow processing:** Use batch processing or try the lightweight models for faster results

## Dependencies

- **Python:** 3.9-3.13 (primary target: 3.10)
- **Core Libraries:** pandas, scikit-learn, thefuzz, pydantic, PyYAML
- **Optional Libraries:** sentence-transformers, transformers, torch, umap-learn, hdbscan, plotly, bertopic

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
tfidf_model = TFIDFTopicModel(auto_detect_topics=True, max_features=2000)
tfidf_model.fit(documents)

# Get topic information and visualize
topic_info = tfidf_model.get_topic_info()
print(topic_info)

# Create an NMF model for more interpretable topics
nmf_model = NMFTopicModel(auto_detect_topics=True, max_features=1500)
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
simple_model = SimpleTopicModel(auto_detect_topics=True, embedding_model=embedding_model)
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
tfidf_model = TFIDFTopicModel(auto_detect_topics=True)
nmf_model = NMFTopicModel(auto_detect_topics=True)
lsa_model = LSATopicModel(auto_detect_topics=True)

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

For complete examples, see `examples/models/lightweight_models.py` and `examples/visualization/interactive_plots.py`.

### Using the Web Interface

```python
from meno.web_interface import launch_web_interface
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel, NMFTopicModel

# Create and train some models
tfidf_model = TFIDFTopicModel(auto_detect_topics=True)
nmf_model = NMFTopicModel(auto_detect_topics=True)
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

See `examples/specialized/web_interface.py` for a complete example of using the web interface with lightweight models.

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

See `examples/visualization/interactive_plots.py`, `examples/notebooks/topic_feedback.ipynb`, and `examples/advanced/feedback_system.py` for complete examples of using the feedback system with visualizations.

See the example scripts in the [examples directory](examples/) organized by category:
- **[Basic Examples](examples/basic/)** - Getting started with Meno
- **[Advanced Features](examples/advanced/)** - Examples of more complex capabilities
- **[Model Integration](examples/models/)** - Using different topic modeling backends
- **[Visualization & Reporting](examples/visualization/)** - Creating visualizations and reports
- **[Specialized Use Cases](examples/specialized/)** - Domain-specific examples
- **[Jupyter Notebooks](examples/notebooks/)** - Interactive tutorials

### LLM Topic Labeling

```python
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno import generate_text_with_llm
import pandas as pd

# Load your data
df = pd.read_csv("documents.csv")
documents = df["text"].tolist()

# Method 1: Automatic labeling during model fitting
model = BERTopicModel(
    auto_detect_topics=True,
    embedding_model="all-MiniLM-L6-v2",
    use_llm_labeling=True,
    llm_model_type="local",
    llm_model_name="google/flan-t5-small"
)
model.fit(documents)

# Method 2: Apply labeling after model fitting
model = BERTopicModel(auto_detect_topics=True, embedding_model="all-MiniLM-L6-v2")
model.fit(documents)

# Get original topic info
print("Original topic names:")
print(model.get_topic_info()[["Topic", "Name"]])

# Apply LLM labeling
model.apply_llm_labeling(
    documents=documents,
    model_type="local",
    model_name="google/flan-t5-small",
    detailed=True
)

# Get updated topic info
print("LLM-generated topic names:")
print(model.get_topic_info()[["Topic", "Name"]])

# Method 3: Standalone labeler for any topic model
topic_model = BERTopicModel(auto_detect_topics=True)
topic_model.fit(documents)

# Create LLM labeler with Azure OpenAI
labeler = LLMTopicLabeler(
    model_name="your-deployment-name",  # Azure deployment name
    api_key="your-api-key",             # Azure API key
    api_endpoint="https://your-resource.openai.azure.com",
    use_azure=True,                     # Use Azure OpenAI
    temperature=0.7
)

# Or with standard OpenAI
standard_labeler = LLMTopicLabeler(
    model_name="gpt-4o",                # Standard OpenAI model name
    api_key="your-api-key",             # OpenAI API key
    use_azure=False,                    # Use standard OpenAI
    temperature=0.7
)

# Generate topic names
topic_names = labeler.label_topics(
    topic_model=topic_model,
    example_docs_per_topic=None,  # Optional document examples
    detailed=True
)

for topic_id, name in topic_names.items():
    print(f"Topic {topic_id}: {name}")

# Method 4: Direct API interaction using utility function
response = generate_text_with_llm(
    text="Summarize these keywords into a topic name: technology, computer, software, AI",
    api_key="your-api-key",
    api_endpoint="https://your-resource.openai.azure.com",
    deployment_id="your-deployment-name",  # Required for Azure
    use_azure=True,
    system_prompt="You are a topic naming expert."
)
print(f"Generated topic name: {response}")

# With standard OpenAI
standard_response = generate_text_with_llm(
    text="Summarize these keywords into a topic name: finance, stocks, investing, market",
    api_key="your-openai-api-key",
    api_endpoint=None,
    model_name="gpt-4o",
    use_azure=False,
    system_prompt="You are a topic naming expert."
)
print(f"Generated topic name: {standard_response}")
```

## Future Development

With v1.2.0 adding advanced BERTopic features and LLM topic labeling, we're now focusing on:

1. **Incremental Learning** - Support for streaming data and updating models
2. **Multilingual Support** - Expand beyond English with better language handling
3. **Domain-Specific Fine-Tuning** - Adapt models to specific industries
4. **Explainable AI Features** - Better interpret topic assignments
5. **Interactive Dashboards** - More powerful visualization tools
6. **Cloud Integration** - Native support for cloud-based services
7. **Export/Import Format** - Standard format for sharing models and results
8. **Extension API** - Plugin system for custom models and visualizations
9. **Enhanced LLM Integration** - More language model options and applications

See our [detailed roadmap](ROADMAP.md) for more information and the [INTEGRATED_COMPONENTS_SUMMARY.md](INTEGRATED_COMPONENTS_SUMMARY.md) for details on our recent work.

## CPU-Optimized Usage (No LLM Required)

For CPU-bound systems without LLM integration needs, here's how to get the best performance:

### Installation
```bash
# Install with CPU-optimized dependencies
pip install "meno[embeddings,minimal]" -f https://download.pytorch.org/whl/torch_stable.html
```

### CPU-Optimized Topic Modeling
```python
import pandas as pd
from meno import MenoTopicModeler

# Define CPU-optimized configuration
CPU_CONFIG = {
    "preprocessing": {
        "normalization": {
            "lowercase": True,
            "remove_punctuation": True,
            "lemmatize": True,
        },
    },
    "modeling": {
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",  # Small, fast model
            "device": "cpu",                   # Explicitly use CPU
            "use_gpu": False,                  # Disable GPU
            "batch_size": 32,                  # CPU-optimized batch size
            "quantize": True,                  # Reduce memory usage
        },
    },
    "visualization": {
        "umap": {
            "n_neighbors": 15,
            "min_dist": 0.1,
        },
    },
}

# Load your data
df = pd.read_csv("your_documents.csv")

# Initialize modeler with CPU configuration
modeler = MenoTopicModeler(config_overrides=CPU_CONFIG)

# Preprocess documents
processed_docs = modeler.preprocess(
    df, 
    text_column="text",
    remove_stopwords=True
)

# Generate embeddings and discover topics
modeler.embed_documents()
topics_df = modeler.discover_topics(
    method="embedding_cluster",
    auto_detect_topics=True,
    modeling_approach="lightweight"  # Use NMF or TF-IDF-based approaches
)

# Generate comprehensive HTML report
report_path = modeler.generate_report(
    output_path="topic_report.html",
    include_interactive=True,
    title="Topic Modeling Report"
)

print(f"Report generated at {report_path}")
```

### Lightweight Model Options
For even better CPU performance, try the direct model interfaces:

```python
from meno.modeling.simple_models.lightweight_models import TFIDFTopicModel, NMFTopicModel
from meno.visualization.lightweight_viz import plot_topic_landscape

# Load data
documents = df["text"].tolist()

# Create TF-IDF model (extremely CPU-efficient)
model = TFIDFTopicModel(
    auto_detect_topics=True,
    max_features=2000,  # Limit vocabulary size
    random_state=42
)

# Fit the model and get topic info
model.fit(documents)
topic_info = model.get_topic_info()

# Create visualization (using PCA instead of UMAP for speed)
fig = plot_topic_landscape(
    model=model,
    documents=documents,
    method="pca"
)
fig.write_html("topic_landscape.html")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
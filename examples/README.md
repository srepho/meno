# Meno Examples

This directory contains example scripts and notebooks demonstrating how to use the Meno Topic Modeling Toolkit. Examples are organized by category to help you quickly find what you need.

## Getting Started

- [Basic Workflow Notebook](notebooks/basic_workflow.ipynb) - Step-by-step tutorial in Jupyter notebook format
- [Minimal Example](basic/minimal_example.py) - Simplest implementation of topic modeling (5 lines of code)
- [End-to-End Example](basic/end_to_end_workflow.py) - Complete topic modeling pipeline from data loading to visualization

## Examples by Category

### Basic Usage

Simple examples to get you started with Meno's core functionality:

- **[Minimal Example](basic/minimal_example.py)** - Basic usage with minimal code
- **[Workflow Example](basic/workflow_example.py)** - Using the MenoWorkflow for guided topic modeling
- **[CPU-Optimized Example](basic/cpu_optimized.py)** - Topic modeling optimized for CPU environments

### Advanced Features

Examples showcasing Meno's more advanced capabilities:

- **[LLM Topic Labeling](advanced/llm_topic_labeling.py)** - Using language models to generate human-readable topic names
- **[Deduplication](advanced/deduplication.py)** - Removing duplicate documents for cleaner topic modeling
- **[Fuzzy Deduplication](advanced/fuzzy_deduplication.py)** - Finding and removing near-duplicate content
- **[Memory Optimization](advanced/memory_optimization.py)** - Processing larger datasets with limited memory
- **[Incremental Learning](advanced/incremental_topic_update.py)** - Updating topic models with new data

### Model Integration

Examples showing how to integrate different topic modeling backends:

- **[BERTopic Basic](models/bertopic_basic.py)** - Simple BERTopic integration
- **[BERTopic Advanced](models/bertopic_advanced.py)** - Advanced BERTopic features like dynamic topic modeling
- **[Lightweight Models](models/lightweight_models.py)** - Using CPU-efficient alternatives to transformer models
- **[Top2Vec Integration](models/top2vec_example.py)** - Using Top2Vec for topic modeling

### Visualization & Reporting

Examples focused on creating visualizations and reports:

- **[Enhanced Reports](visualization/enhanced_reports.py)** - Creating comprehensive HTML reports
- **[Interactive Visualizations](visualization/interactive_plots.py)** - Creating interactive Plotly visualizations
- **[Time Series Analysis](visualization/time_series.py)** - Visualizing how topics change over time
- **[Geospatial Visualization](visualization/geospatial.py)** - Mapping topics with geographic data

### Specialized Use Cases

Examples for specific domains or scenarios:

- **[Insurance Topic Modeling](specialized/insurance_modeling.py)** - Topic modeling for insurance industry data
- **[Local Model Usage](specialized/offline_model.py)** - Using Meno in environments without internet access
- **[Team Configuration](specialized/team_config.py)** - Sharing configurations across teams
- **[Web Interface](specialized/web_interface.py)** - Using the Meno web interface for exploration

### Jupyter Notebooks

Interactive tutorials in notebook format:

- **[Basic Workflow](notebooks/basic_workflow.ipynb)** - End-to-end tutorial from preprocessing to visualization
- **[BERTopic Integration](notebooks/bertopic_integration.ipynb)** - Complete guide to BERTopic
- **[CPU Quality First](notebooks/cpu_quality_first.ipynb)** - High-quality results with CPU optimization
- **[Topic Feedback](notebooks/topic_feedback.ipynb)** - Interactive feedback and refinement for topics

## Usage

Most examples can be run directly from the command line:

```bash
python examples/basic/minimal_example.py
python examples/visualization/enhanced_reports.py
```

## Sample Reports

- [Enhanced HTML Report Samples](sample_reports/enhanced/index.html) - Explore interactive HTML reports
- [Sample Reports Directory](sample_reports/) - View various example reports and visualizations

## Feature Documentation

For more detailed documentation on specific features:

- [LLM Topic Labeling](../docs/llm_topic_labeling.md)
- [Deduplication](../docs/deduplication.md)
- [Fuzzy Deduplication](../docs/fuzzy_deduplication.md)
- [Memory Optimization](../docs/memory_optimization.md)
- [Topic Drift Visualization](../docs/topic_drift_visualization.md)
- [Lightweight Models](../docs/lightweight_models.md)
- [Offline Usage](../docs/offline_usage.md)
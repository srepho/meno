# Meno Examples

This directory contains example scripts and notebooks demonstrating how to use the Meno Topic Modeling Toolkit.

## Quick Links

- [Enhanced HTML Report Samples](sample_reports/enhanced/index.html) - Explore the new interactive HTML reports
- [Sample Reports Directory](sample_reports/) - View various example reports and visualizations
- [Basic Workflow Notebook](basic_workflow.ipynb) - Jupyter notebook with step-by-step tutorial
- [BERTopic Integration Notebook](bertopic_integration_notebook.ipynb) - Complete tutorial on BERTopic integration
- [Minimal Sample Script](minimal_sample.py) - Simple script to generate visualizations

## Example Scripts

- **[generate_enhanced_report.py](generate_enhanced_report.py)** - Creates sample reports showcasing all the enhanced visualization features
- **[minimal_sample.py](minimal_sample.py)** - Basic usage of Meno for topic modeling
- **[cpu_only_example.py](cpu_only_example.py)** - Demonstrates CPU-optimized topic modeling
- **[insurance_topic_modeling.py](insurance_topic_modeling.py)** - Topic modeling on insurance complaint dataset

### BERTopic Integration Examples
- **[simple_bertopic_example.py](simple_bertopic_example.py)** - BERTopic example with basic configuration
- **[bertopic_example.py](bertopic_example.py)** - Advanced topic modeling using BERTopic with KeyBERTInspired 
- **[bertopic_custom_pipeline.py](bertopic_custom_pipeline.py)** - Advanced BERTopic with full custom pipeline
- **[workflow_bertopic_example.py](workflow_bertopic_example.py)** - Integration of MenoWorkflow with BERTopic
- **[bertopic_integration_notebook.ipynb](bertopic_integration_notebook.ipynb)** - Comprehensive Jupyter notebook tutorial

### Advanced Examples
- **[workflow_with_optimizations.py](workflow_with_optimizations.py)** - Optimized workflow for large datasets
- **[interactive_workflow_demo.py](interactive_workflow_demo.py)** - Step-by-step guided workflow with interactive components

## Usage

Most examples can be run directly from the command line:

```bash
python examples/minimal_sample.py
python examples/generate_enhanced_report.py
```

## Enhanced HTML Reports

The latest version of Meno includes significant improvements to HTML report generation, including:

- Modern card-based design
- Interactive tabs for different visualizations
- Topic similarity heatmap
- Interactive word clouds
- Export functionality for data tables
- Responsive design for all devices

To explore these features, check out the [Enhanced Report Samples](sample_reports/enhanced/index.html) or run the `generate_enhanced_report.py` script.
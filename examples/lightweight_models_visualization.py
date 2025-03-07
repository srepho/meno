#!/usr/bin/env python
"""Example script to demonstrate the enhanced visualization options for lightweight models.

This script shows how to use the various visualization capabilities of Meno's
lightweight topic models, including model comparison, topic landscape, and
document-topic analysis.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from meno.modeling.simple_models import (
    SimpleTopicModel,
    TFIDFTopicModel,
    NMFTopicModel,
    LSATopicModel
)
from meno.visualization.lightweight_viz import (
    plot_model_comparison,
    plot_topic_landscape,
    plot_multi_topic_heatmap,
    plot_comparative_document_analysis
)


def load_sample_data(file_path=None):
    """Load sample data for the example.
    
    Parameters
    ----------
    file_path : str, optional
        Path to a text file with documents (one per line), by default None
        
    Returns
    -------
    list
        List of documents
    """
    if file_path and Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]
        return documents
    
    # Default sample data if no file provided
    return [
        "Customer service was excellent and the product arrived on time.",
        "The package was damaged during shipping but customer service resolved the issue.",
        "Product quality is outstanding and exceeded my expectations.",
        "The software has a steep learning curve but powerful features.",
        "Technical support was helpful in resolving my installation issues.",
        "User interface is intuitive and easy to navigate.",
        "The documentation lacks examples and could be improved.",
        "Performance is excellent even with large datasets.",
        "Pricing is competitive compared to similar products.",
        "Regular updates keep adding valuable new features.",
        "The mobile app lacks some features available in the desktop version.",
        "Setup was straightforward and took only a few minutes.",
        "The product is reliable and hasn't crashed in months of use.",
        "Customer service response time could be improved.",
        "Training materials are comprehensive and well-structured.",
        "The hardware integration works seamlessly with our existing systems.",
        "Battery life is impressive compared to previous models.",
        "The AI features are innovative but sometimes unpredictable.",
        "Security features are robust and meet our compliance requirements.",
        "The community forum is active and helpful for troubleshooting."
    ]


def visualize_model_comparison(documents, output_dir):
    """Visualize a comparison of different topic models.
    
    Parameters
    ----------
    documents : list
        List of documents to analyze
    output_dir : str
        Directory to save visualization outputs
    """
    print("Training multiple topic models for comparison...")
    
    # Train different models
    models = []
    model_names = []
    
    # TF-IDF K-Means model
    print("Training TF-IDF K-Means model...")
    tfidf_model = TFIDFTopicModel(num_topics=5, random_state=42)
    tfidf_model.fit(documents)
    models.append(tfidf_model)
    model_names.append("TF-IDF K-Means")
    
    # NMF model
    print("Training NMF model...")
    nmf_model = NMFTopicModel(num_topics=5, random_state=42)
    nmf_model.fit(documents)
    models.append(nmf_model)
    model_names.append("NMF")
    
    # LSA model
    print("Training LSA model...")
    lsa_model = LSATopicModel(num_topics=5, random_state=42)
    lsa_model.fit(documents)
    models.append(lsa_model)
    model_names.append("LSA")
    
    # Create model comparison visualization
    print("Creating model comparison visualization...")
    fig = plot_model_comparison(
        document_lists=[documents] * len(models),
        model_names=model_names,
        models=models,
        title="Topic Model Comparison"
    )
    
    # Save the visualization
    output_path = Path(output_dir) / "model_comparison.html"
    fig.write_html(str(output_path))
    print(f"Model comparison visualization saved to {output_path}")
    
    return models, model_names


def visualize_topic_landscape(model, documents, output_dir):
    """Visualize the topic landscape for a model.
    
    Parameters
    ----------
    model : BaseTopicModel
        Fitted topic model
    documents : list
        List of documents analyzed
    output_dir : str
        Directory to save visualization outputs
    """
    print("Creating topic landscape visualization...")
    
    # Create topic landscape visualization
    fig = plot_topic_landscape(
        model=model,
        documents=documents,
        title=f"Topic Landscape ({model.__class__.__name__})"
    )
    
    # Save the visualization
    output_path = Path(output_dir) / f"topic_landscape_{model.__class__.__name__}.html"
    fig.write_html(str(output_path))
    print(f"Topic landscape visualization saved to {output_path}")


def visualize_multi_topic_heatmap(models, model_names, documents, output_dir):
    """Visualize topic similarity heatmap across multiple models.
    
    Parameters
    ----------
    models : list
        List of fitted topic models
    model_names : list
        Names of the models
    documents : list
        List of documents analyzed
    output_dir : str
        Directory to save visualization outputs
    """
    print("Creating multi-topic heatmap visualization...")
    
    # Create multi-topic heatmap visualization
    fig = plot_multi_topic_heatmap(
        models=models,
        model_names=model_names,
        document_lists=[documents] * len(models),
        title="Topic Similarity Across Models"
    )
    
    # Save the visualization
    output_path = Path(output_dir) / "multi_topic_heatmap.html"
    fig.write_html(str(output_path))
    print(f"Multi-topic heatmap visualization saved to {output_path}")


def visualize_document_topic_analysis(model, documents, output_dir):
    """Visualize document-topic relationship analysis.
    
    Parameters
    ----------
    model : BaseTopicModel
        Fitted topic model
    documents : list
        List of documents analyzed
    output_dir : str
        Directory to save visualization outputs
    """
    print("Creating document-topic analysis visualization...")
    
    # Create document-topic analysis visualization
    fig = plot_comparative_document_analysis(
        model=model,
        documents=documents,
        title=f"Document-Topic Analysis ({model.__class__.__name__})"
    )
    
    # Save the visualization
    output_path = Path(output_dir) / f"document_topic_analysis_{model.__class__.__name__}.html"
    fig.write_html(str(output_path))
    print(f"Document-topic analysis visualization saved to {output_path}")


def main():
    """Run the lightweight models visualization example."""
    parser = argparse.ArgumentParser(
        description="Demonstrate enhanced visualization options for lightweight models."
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        help="Path to input text file with documents (one per line)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./visualization_outputs",
        help="Directory to save visualization outputs"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample data
    documents = load_sample_data(args.input)
    print(f"Loaded {len(documents)} documents")
    
    # Run visualizations
    models, model_names = visualize_model_comparison(documents, output_dir)
    
    # Create individual visualizations for each model
    for model in models:
        visualize_topic_landscape(model, documents, output_dir)
        visualize_document_topic_analysis(model, documents, output_dir)
    
    # Create multi-model visualization
    visualize_multi_topic_heatmap(models, model_names, documents, output_dir)
    
    print("\nAll visualizations completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
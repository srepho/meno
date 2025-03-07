"""Example script showing how to use the web interface with lightweight topic models.

This example demonstrates how to launch Meno's web interface with lightweight
topic models. It can run without heavyweight dependencies like UMAP and HDBSCAN.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Import Meno components
from meno.modeling.simple_models.lightweight_models import (
    TFIDFTopicModel, 
    NMFTopicModel,
    LSATopicModel
)
from meno.modeling.embeddings import DocumentEmbedding
from meno.web_interface import launch_web_interface


def create_sample_data():
    """Create a sample dataset for demonstration."""
    
    # Sample documents across different topics
    documents = [
        # Technology documents
        "Machine learning algorithms can analyze large datasets to identify patterns and make predictions.",
        "Artificial intelligence systems use neural networks to mimic human decision-making processes.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Blockchain technology creates secure, decentralized ledgers for recording transactions.",
        "The Internet of Things connects everyday devices to the internet for data collection and control.",
        
        # Healthcare documents
        "Medical imaging techniques like MRI and CT scans help diagnose internal conditions non-invasively.",
        "Electronic health records store patient information digitally for easier access and coordination.",
        "Telemedicine allows healthcare providers to consult with patients remotely via video calls.",
        "Preventive medicine focuses on maintaining health and preventing disease before symptoms appear.",
        "Personalized medicine tailors treatments based on an individual's genetic profile and health data.",
        
        # Environmental documents
        "Renewable energy sources like solar and wind power help reduce carbon emissions.",
        "Conservation efforts protect biodiversity by preserving natural habitats and ecosystems.",
        "Sustainable development balances economic growth with environmental protection.",
        "Climate change causes rising temperatures, extreme weather events, and shifting ecosystems.",
        "Recycling programs convert waste materials into new products to reduce landfill usage."
    ]
    
    # Create dataframe with metadata
    categories = ["Technology"] * 5 + ["Healthcare"] * 5 + ["Environment"] * 5
    df = pd.DataFrame({
        "text": documents,
        "category": categories,
        "doc_id": [f"doc_{i}" for i in range(len(documents))]
    })
    
    print(f"Created sample dataset with {len(df)} documents across {len(set(categories))} categories")
    return df


def create_models(documents):
    """Create and train lightweight topic models.
    
    Args:
        documents: List of text documents
    
    Returns:
        dict: Dictionary of trained topic models
    """
    print("Training lightweight topic models...")
    
    # Create models
    models = {
        "TF-IDF": TFIDFTopicModel(num_topics=3, random_state=42),
        "NMF": NMFTopicModel(num_topics=3, random_state=42),
        "LSA": LSATopicModel(num_topics=3, random_state=42)
    }
    
    # Train models
    for name, model in models.items():
        print(f"  Training {name} model...")
        model.fit(documents)
        
        # Print topic summary
        topic_info = model.get_topic_info()
        print(f"    Topics discovered:")
        for _, row in topic_info.iterrows():
            print(f"      Topic {row['Topic']}: {row['Name']}")
    
    return models


def launch_web_app(models, df, port=8050, debug=False):
    """Launch the web interface with the trained models.
    
    Args:
        models: Dictionary of trained topic models
        df: DataFrame with documents
        port: Port to run the web interface on
        debug: Whether to run in debug mode
    """
    print(f"\nLaunching web interface on port {port}...")
    print("Use Ctrl+C to stop the server when finished")
    
    try:
        # Convert models to format expected by web interface
        web_models = {
            "TF-IDF K-Means": models["TF-IDF"],
            "NMF Topic Model": models["NMF"],
            "LSA Topic Model": models["LSA"]
        }
        
        # Launch web interface
        launch_web_interface(
            port=port,
            debug=debug,
            models=web_models,
            data=df,
            text_column="text"
        )
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nWeb interface requires additional dependencies.")
        print("Install them with: pip install 'meno[web]'")


def main():
    """Main function to run the web interface example."""
    parser = argparse.ArgumentParser(description="Meno Web Interface with Lightweight Models")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the web interface on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MENO WEB INTERFACE WITH LIGHTWEIGHT MODELS")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data()
    
    # Train models
    models = create_models(df["text"].tolist())
    
    # Launch web interface
    launch_web_app(models, df, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
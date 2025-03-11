"""End-to-end workflow for topic modeling with LLM topic labeling.

This example demonstrates a complete workflow for topic discovery and 
visualization using LLM-based topic labeling for more descriptive topic names.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Any
import os
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if scikit-learn datasets is available for sample data
try:
    from sklearn.datasets import fetch_20newsgroups
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Using synthetic data instead.")

# Import Meno components
from meno.preprocessing.normalization import normalize_text
from meno.preprocessing.spelling import correct_misspellings
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.bertopic_model import BERTopicModel
from meno.visualization.bertopic_viz import plot_topic_hierarchy, plot_topic_similarity_network
from meno.reporting.html_generator import generate_topic_report
from meno.workflow import MenoWorkflow
from meno.modeling.unified_topic_modeling import create_topic_modeler

# Load or generate sample data
def get_sample_data(n_samples: int = 1000) -> List[str]:
    """Load or generate sample data for topic modeling.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to load, by default 1000
    
    Returns
    -------
    List[str]
        List of document texts
    """
    if SKLEARN_AVAILABLE:
        logger.info("Loading 20 Newsgroups dataset...")
        newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        # Take a subset of the data
        data = newsgroups.data[:n_samples]
        # Filter out empty documents
        data = [doc for doc in data if doc.strip()]
        return data
    else:
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        topics = [
            ["technology", "computer", "software", "hardware", "program", "code", "system"],
            ["science", "research", "study", "experiment", "theory", "scientist", "data"],
            ["politics", "government", "policy", "election", "president", "party", "vote"],
            ["sports", "team", "player", "game", "score", "win", "championship"],
            ["health", "medical", "doctor", "disease", "treatment", "patient", "hospital"],
        ]
        
        data = []
        for _ in range(n_samples):
            topic_idx = np.random.randint(0, len(topics))
            topic_words = topics[topic_idx]
            # Generate a simple document with topic words
            n_words = np.random.randint(5, 15)
            words = np.random.choice(topic_words, size=n_words, replace=True)
            doc = " ".join(words)
            data.append(doc)
            
        return data

# Define custom preprocessing function
def preprocess_documents(texts: List[str]) -> List[str]:
    """Apply preprocessing to document texts.
    
    Parameters
    ----------
    texts : List[str]
        List of document texts
    
    Returns
    -------
    List[str]
        List of preprocessed document texts
    """
    # Apply normalization to clean the text
    normalized_texts = [normalize_text(doc, lowercase=True, remove_html=True) for doc in texts]
    
    # Apply spelling correction
    corrected_texts = correct_misspellings(normalized_texts, max_edit_distance=2)
    
    # Additional preprocessing steps could be added here
    
    return corrected_texts

# Manual workflow approach using separate steps
def manual_workflow(data: List[str], output_dir: str = "./output") -> None:
    """Run a manual workflow with separate steps.
    
    Parameters
    ----------
    data : List[str]
        List of document texts
    output_dir : str, optional
        Directory to save outputs, by default "./output"
    """
    logger.info("Running manual workflow with separate steps...")
    start_time = time.time()
    
    # Step 1: Preprocessing
    logger.info("Step 1: Preprocessing documents...")
    processed_docs = preprocess_documents(data)
    
    # Step 2: Create embedding model
    logger.info("Step 2: Creating embedding model...")
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",
        use_gpu=False
    )
    
    # Step 3: Generate document embeddings
    logger.info("Step 3: Generating document embeddings...")
    embeddings = embedding_model.embed_documents(processed_docs)
    
    # Step 4: Create topic model with LLM labeling
    logger.info("Step 4: Creating topic model with LLM labeling...")
    topic_model = BERTopicModel(
        num_topics=12,  # Set to None for automatic detection
        embedding_model=embedding_model,
        min_topic_size=10,
        use_llm_labeling=True,  # Enable LLM labeling
        llm_model_type="local",
        llm_model_name="google/flan-t5-small",
        verbose=True
    )
    
    # Step 5: Fit the topic model
    logger.info("Step 5: Fitting topic model...")
    topic_model.fit(processed_docs, embeddings=embeddings)
    
    # Step 6: Get topic assignments
    logger.info("Step 6: Getting topic assignments...")
    topics, probs = topic_model.transform(processed_docs, embeddings=embeddings)
    
    # Step 7: Generate visualizations
    logger.info("Step 7: Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Topic similarity network
    fig_network = plot_topic_similarity_network(topic_model)
    fig_network.write_html(f"{output_dir}/topic_similarity_network.html")
    
    # Topic hierarchy
    fig_hierarchy = plot_topic_hierarchy(topic_model)
    fig_hierarchy.write_html(f"{output_dir}/topic_hierarchy.html")
    
    # Step 8: Generate topic report
    logger.info("Step 8: Generating topic report...")
    report_path = f"{output_dir}/topic_report.html"
    
    # Create a DataFrame with document texts and topic assignments
    doc_data = pd.DataFrame({
        'text': processed_docs,
        'topic': topics,
    })
    
    # Generate the report
    generate_topic_report(
        topic_model=topic_model,
        documents=doc_data,
        output_file=report_path,
        plot_width=800,
        plot_height=600,
        include_sample_docs=True
    )
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Manual workflow completed in {elapsed_time:.2f} seconds")
    topic_info = topic_model.get_topic_info()
    print("\nDiscovered topics with LLM-generated names:")
    print(topic_info[["Topic", "Count", "Name"]])
    print(f"\nOutput files saved to: {output_dir}")

# Unified workflow approach using MenoWorkflow
def pipeline_workflow(data: List[str], output_dir: str = "./output") -> None:
    """Run a unified workflow using MenoWorkflow with create_topic_modeler.
    
    Parameters
    ----------
    data : List[str]
        List of document texts
    output_dir : str, optional
        Directory to save outputs, by default "./output"
    """
    logger.info("Running unified workflow with MenoWorkflow...")
    start_time = time.time()
    
    # Create embedding model
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",
        use_gpu=False
    )
    
    # Create topic model with LLM labeling
    topic_model = create_topic_modeler(
        method="bertopic",
        num_topics=12,
        embedding_model=embedding_model,
        use_llm_labeling=True,
        llm_model_type="local",
        llm_model_name="google/flan-t5-small",
        config_overrides={
            'min_topic_size': 10,
            'verbose': True
        }
    )
    
    # Create dataframe from text data
    df = pd.DataFrame({"text": data})
    
    # Create and initialize workflow
    workflow = MenoWorkflow()
    
    # Load data
    workflow.load_data(df, text_column="text")
    
    # Step 1: Preprocessing
    logger.info("Preprocessing documents...")
    workflow.preprocess_text(
        normalize=True,
        correct_spelling=True
    )
    
    # Get the preprocessed data
    processed_data = workflow.documents[workflow.text_column]
    
    # Step 2: Fit topic model
    logger.info("Fitting topic model...")
    topic_model.fit(processed_data)
    
    # Step 3: Generate visualizations
    logger.info("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Topic similarity network
    fig_network = plot_topic_similarity_network(topic_model)
    fig_network.write_html(f"{output_dir}/topic_similarity_network.html")
    
    # Topic hierarchy
    fig_hierarchy = plot_topic_hierarchy(topic_model)
    fig_hierarchy.write_html(f"{output_dir}/topic_hierarchy.html")
    
    # Step 4: Generate report
    logger.info("Generating topic report...")
    
    # Get topic assignments
    topics, _ = topic_model.transform(processed_data)
    
    # Create a DataFrame with document texts and topic assignments
    doc_data = pd.DataFrame({
        'text': processed_data,
        'topic': topics,
    })
    
    # Generate report
    report_path = f"{output_dir}/topic_report.html"
    generate_topic_report(
        topic_model=topic_model,
        documents=doc_data,
        output_file=report_path,
        plot_width=800,
        plot_height=600,
        include_sample_docs=True
    )
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline workflow completed in {elapsed_time:.2f} seconds")
    topic_info = topic_model.get_topic_info()
    print("\nDiscovered topics with LLM-generated names:")
    print(topic_info[["Topic", "Count", "Name"]])
    print(f"\nOutput files saved to: {output_dir}")

# Main function to demonstrate both approaches
def main():
    """Run the example workflow."""
    # Create output directory
    output_dir = "./output/llm_labeled_topics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample data
    data = get_sample_data(n_samples=1000)
    logger.info(f"Loaded {len(data)} documents")
    
    # Run manual workflow
    try:
        manual_dir = f"{output_dir}/manual"
        manual_workflow(data, output_dir=manual_dir)
    except Exception as e:
        logger.error(f"Error in manual workflow: {e}")
    
    # Run pipeline workflow
    try:
        pipeline_dir = f"{output_dir}/pipeline"
        pipeline_workflow(data, output_dir=pipeline_dir)
    except Exception as e:
        logger.error(f"Error in pipeline workflow: {e}")

if __name__ == "__main__":
    main()
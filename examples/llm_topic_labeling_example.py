"""Example of using LLM-based topic labeling with BERTopic.

This example demonstrates how to use LLM-based topic labeling to generate more 
descriptive and human-readable topic names for BERTopic models.

It also shows how to use the generate_text_with_llm utility function for
direct interaction with Azure OpenAI or standard OpenAI, which provides a simplified
interface for API interactions without having to create a custom function.

Key features demonstrated:
- LLM-based topic labeling during model fitting
- Applying LLM labeling after model fitting
- Using the unified topic modeler with LLM labeling
- Direct LLM API usage with the utility function
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if scikit-learn datasets is available
try:
    from sklearn.datasets import fetch_20newsgroups
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Using sample data instead.")

# Import Meno components
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.unified_topic_modeling import create_topic_modeler
from meno.visualization.bertopic_viz import visualize_topics_over_time, plot_topic_hierarchy
from meno import generate_text_with_llm  # Import the new utility function

# Load sample data
def load_sample_data(n_samples: int = 1000) -> List[str]:
    """Load sample data for topic modeling.
    
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
        # Generate some sample data
        logger.info("Generating sample data...")
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

# Example 1: BERTopic with automatic LLM labeling during fitting
def example_bertopic_with_llm(data: List[str]) -> None:
    """Run BERTopic with LLM topic labeling enabled during fitting.
    
    Parameters
    ----------
    data : List[str]
        List of document texts
    """
    logger.info("Example 1: BERTopic with LLM labeling during fitting")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Initialize BERTopic with LLM labeling
    model = BERTopicModel(
        num_topics=10,
        embedding_model=embedding_model,
        min_topic_size=5,
        use_llm_labeling=True,  # Enable LLM labeling
        llm_model_type="local",  # Use local HuggingFace model
        llm_model_name="google/flan-t5-small"  # Small model for example
    )
    
    # Fit the model - LLM labeling will be applied during fitting
    start_time = time.time()
    model.fit(data)
    end_time = time.time()
    
    logger.info(f"BERTopic fitting completed in {end_time - start_time:.2f} seconds")
    
    # Print topic information
    topic_info = model.get_topic_info()
    print("\nTopics with LLM-generated names:")
    print(topic_info[["Topic", "Count", "Name"]])
    
    # Visualize topics
    fig = model.visualize_topics()
    # You can save the figure or show it in a notebook
    # fig.write_html("bertopic_llm_topics.html")

# Example 2: Apply LLM labeling after model fitting
def example_post_labeling(data: List[str]) -> None:
    """Fit a BERTopic model first, then apply LLM labeling afterward.
    
    Parameters
    ----------
    data : List[str]
        List of document texts
    """
    logger.info("Example 2: Applying LLM labeling after model fitting")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Initialize BERTopic without LLM labeling
    model = BERTopicModel(
        num_topics=10,
        embedding_model=embedding_model,
        min_topic_size=5,
        use_llm_labeling=False  # No LLM labeling during fitting
    )
    
    # Fit the model with default keyword-based topic names
    model.fit(data)
    
    # Print original topic information
    topic_info_original = model.get_topic_info()
    print("\nOriginal topics with keyword-based names:")
    print(topic_info_original[["Topic", "Count", "Name"]])
    
    # Now apply LLM labeling after fitting
    logger.info("Applying LLM topic labeling...")
    model.apply_llm_labeling(
        documents=data,
        model_type="local",
        model_name="google/flan-t5-small",
        detailed=True
    )
    
    # Print updated topic information
    topic_info_llm = model.get_topic_info()
    print("\nUpdated topics with LLM-generated names:")
    print(topic_info_llm[["Topic", "Count", "Name"]])

# Example 3: Using the unified topic modeler with LLM labeling
def example_unified_modeler(data: List[str]) -> None:
    """Use the unified topic modeler with LLM labeling.
    
    Parameters
    ----------
    data : List[str]
        List of document texts
    """
    logger.info("Example 3: Using unified topic modeler with LLM labeling")
    
    # Create a topic modeler with LLM labeling
    modeler = create_topic_modeler(
        method="bertopic",
        num_topics=10,
        use_llm_labeling=True,
        llm_model_type="local",
        llm_model_name="google/flan-t5-small"
    )
    
    # Fit the model
    modeler.fit(data)
    
    # Print topic information
    topic_info = modeler.get_topic_info()
    print("\nTopics with LLM-generated names:")
    print(topic_info[["Topic", "Count", "Name"]])
    
    # You can also apply different LLM labeling after initial fitting
    logger.info("Applying different LLM labeling...")
    modeler.apply_llm_labeling(
        documents=data,
        model_type="local",
        model_name="facebook/opt-125m",  # Different model
        detailed=True
    )
    
    # Print updated topic information
    topic_info_updated = modeler.get_topic_info()
    print("\nTopics with updated LLM-generated names:")
    print(topic_info_updated[["Topic", "Count", "Name"]])

# Example 4: Using the generate_text_with_llm utility function directly
def example_direct_llm_usage():
    """Demonstrate the use of the generate_text_with_llm utility function."""
    logger.info("Example 4: Using generate_text_with_llm utility directly")
    
    # Example text to process
    text = "Explain the concept of topic modeling in 3 sentences."
    
    print("\nDirect LLM Usage Example:")
    print("-" * 50)
    print(f"Input: {text}")
    print("-" * 50)
    
    # For Azure OpenAI (you would replace these with your actual credentials)
    print("\nAzure OpenAI Usage:")
    print("Note: This is just example code, not executed without credentials")
    print("""
    # Azure OpenAI example (add your credentials to run this)
    response = generate_text_with_llm(
        text="Explain the concept of topic modeling in 3 sentences.",
        api_key="your-azure-api-key",
        api_endpoint="https://your-resource.openai.azure.com",
        deployment_id="your-deployment-name",  # Your Azure deployment name
        use_azure=True,
        system_prompt="You are a data science expert specializing in NLP.",
        temperature=0.7
    )
    print(f"Response: {response}")
    """)
    
    # For standard OpenAI (you would replace these with your actual credentials)
    print("\nStandard OpenAI Usage:")
    print("Note: This is just example code, not executed without credentials")
    print("""
    # Standard OpenAI example (add your credentials to run this)
    response = generate_text_with_llm(
        text="Explain the concept of topic modeling in 3 sentences.",
        api_key="your-openai-api-key",
        model_name="gpt-4o",
        use_azure=False,
        system_prompt="You are a data science expert specializing in NLP.",
        temperature=0.7
    )
    print(f"Response: {response}")
    """)
    
    # Example of direct usage with error handling
    print("\nUsage with Error Handling:")
    print("""
    try:
        response = generate_text_with_llm(
            text="Your input text here",
            api_key="your-api-key",
            api_endpoint="your-endpoint",
            deployment_id="your-deployment-id",
            system_prompt="Custom system prompt"
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")
    """)

# Main function to run all examples
def main():
    """Run all examples."""
    # Load sample data
    data = load_sample_data(n_samples=500)  # Using a small dataset for the example
    
    # Run example 1: BERTopic with automatic LLM labeling
    try:
        example_bertopic_with_llm(data)
    except Exception as e:
        logger.error(f"Error in example 1: {e}")
    
    # Run example 2: Apply LLM labeling after model fitting
    try:
        example_post_labeling(data)
    except Exception as e:
        logger.error(f"Error in example 2: {e}")
    
    # Run example 3: Using the unified topic modeler with LLM labeling
    try:
        example_unified_modeler(data)
    except Exception as e:
        logger.error(f"Error in example 3: {e}")
    
    # Run example 4: Using the generate_text_with_llm utility function
    try:
        example_direct_llm_usage()
    except Exception as e:
        logger.error(f"Error in example 4: {e}")
        
    logger.info("All examples completed")

if __name__ == "__main__":
    main()
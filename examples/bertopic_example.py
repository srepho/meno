"""
Example demonstrating topic modeling with BERTopic and KeyBERTInspired representation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Import BERTopic components
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import UMAPReducer

# Add path for meno imports
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# For visualization
from meno.visualization import plot_embeddings, create_umap_projection
import plotly.io as pio

def main():
    """Run the BERTopic example with KeyBERTInspired representation."""
    
    print("Loading insurance dataset from Hugging Face...")
    dataset = load_dataset("soates/australian-insurance-pii-dataset-corrected")
    df = pd.DataFrame({
        "text": dataset["train"]["original_text"],
        "id": dataset["train"]["id"]
    })
    print(f"Loaded {len(df)} documents")
    
    # Preprocess - remove very short documents
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    
    # Additional stopwords for insurance domain
    stopwords = ["insurance", "policy", "claim", "customer", "please", 
                "company", "dear", "sincerely", "writing", "regards",
                "insure", "insured", "complaint", "email"]
    
    print("\nInitializing models...")
    # === OPTION 1: Simplified approach with string-based embedding model ===
    print("\nOption 1: Using simplified BERTopic configuration...")
    # Simply specify the embedding model name, BERTopic handles the rest
    simple_topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",  # BERTopic will load this model automatically
        nr_topics=12,                        # Target number of topics
        min_topic_size=10,                   # Minimum size for a topic
        verbose=True
    )
    
    # === OPTION 2: Advanced configuration for more control ===
    print("\nOption 2: Using advanced BERTopic configuration...")
    # 1. Initialize sentence transformer model for document embedding
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Initialize dimensionality reduction with UMAP
    umap_model = UMAPReducer(
        n_neighbors=15,
        n_components=5,
        min_dist=0.1,
        metric="cosine"
    )
    
    # 3. Initialize vectorizer for topic representation
    vectorizer_model = ClassTfidfTransformer(reduce_frequent_words=True)
    
    # 4. Initialize KeyBERTInspired representation model for better topic representation
    representation_model = KeyBERTInspired()
    
    # 5. Create and configure the BERTopic model with advanced settings
    topic_model = BERTopic(
        embedding_model=embedding_model,      # Model to create document embeddings
        umap_model=umap_model,                # Dimensionality reduction
        vectorizer_model=vectorizer_model,    # Vectorizer for topic representation
        representation_model=representation_model,  # KeyBERT for topic representation
        nr_topics=12,                         # Target number of topics (adjust as needed)
        min_topic_size=10,                    # Minimum size for a topic
        verbose=True
    )
    
    # Choose which model to use for this demo
    # topic_model = simple_topic_model  # Uncomment to use the simplified approach
    
    # Create embeddings and fit topic model
    print("\nFitting topic model...")
    topics, probs = topic_model.fit_transform(df["text"].tolist())
    
    # Print topic information
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info[topic_info['Topic'] != -1])} topics")
    print(topic_info.head(10))
    
    # Print top words for each topic
    print("\nTop words per topic:")
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id != -1:  # Skip outlier topic
            topic_words = topic_model.get_topic(topic_id)
            words = [word for word, _ in topic_words[:5]]
            print(f"Topic {topic_id}: {', '.join(words)}")
    
    # Create a DataFrame with topic assignments
    results_df = pd.DataFrame({
        "text": df["text"],
        "topic": [f"Topic_{t}" if t >= 0 else "Outlier" for t in topics],
        "topic_probability": probs
    })
    
    # Generate topic visualization using BERTopic's built-in visualization
    print("\nGenerating visualizations...")
    try:
        # Topic similarity visualization
        fig_similarity = topic_model.visualize_topics()
        fig_similarity.write_html("bertopic_topic_similarity.html")
        print("Generated topic similarity visualization: bertopic_topic_similarity.html")
        
        # Hierarchical topic tree
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html("bertopic_hierarchy.html")
        print("Generated hierarchical topic visualization: bertopic_hierarchy.html")
        
        # Topic barchart
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
        fig_barchart.write_html("bertopic_barchart.html")
        print("Generated topic barchart: bertopic_barchart.html")
    except Exception as e:
        print(f"Error generating BERTopic visualizations: {e}")
    
    # Create UMAP projection for document visualization
    print("\nCreating UMAP projection for document visualization...")
    doc_embeddings = embedding_model.encode(df["text"].tolist(), show_progress_bar=True)
    
    # Generate 2D projection for visualization
    umap_2d = create_umap_projection(
        doc_embeddings,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2
    )
    
    # Create interactive embedding plot
    fig = plot_embeddings(
        umap_2d,
        results_df["topic"],
        document_texts=results_df["text"],
        width=1000,
        height=800,
        marker_size=5,
        opacity=0.7
    )
    
    # Save the interactive plot
    fig.write_html("bertopic_document_embeddings.html")
    print("Generated document embedding visualization: bertopic_document_embeddings.html")
    
    # Print document examples from each topic
    print("\nDocument examples from selected topics:")
    for topic_id in range(min(5, len(topic_model.get_topics()))):
        if topic_id in topic_model.get_topics():
            docs = results_df[results_df["topic"] == f"Topic_{topic_id}"]["text"].head(1).values
            if len(docs) > 0:
                print(f"\nTopic {topic_id} example:")
                print(docs[0][:200] + "...")
    
    print("\nBERTopic example completed successfully.")

if __name__ == "__main__":
    main()
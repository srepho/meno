"""
Advanced example of BERTopic with a custom pipeline combining multiple components.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# BERTopic components
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.dimensionality import UMAPReducer
from bertopic.cluster import HDBSCANClusterer

# Add path for meno imports
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def load_data():
    """Load and prepare the dataset."""
    print("Loading insurance dataset from Hugging Face...")
    dataset = load_dataset("soates/australian-insurance-pii-dataset-corrected")
    df = pd.DataFrame({
        "text": dataset["train"]["original_text"],
        "id": dataset["train"]["id"]
    })
    print(f"Loaded {len(df)} documents")
    
    # Remove very short documents
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    return df

def create_insurance_specific_vectorizer():
    """Create a custom vectorizer with insurance-specific stopwords."""
    insurance_stopwords = [
        "insurance", "policy", "claim", "insured", "insurer", "customer", 
        "premium", "please", "company", "dear", "sincerely", "regards",
        "complaint", "email", "call", "phone", "contact", "hello", "hi",
        "thanks", "thank", "write", "writing"
    ]
    
    # Create custom vectorizer with domain-specific stopwords
    vectorizer = CountVectorizer(
        stop_words="english",          # Start with English stopwords
        min_df=5,                     # Minimum document frequency
        max_df=0.85,                  # Maximum document frequency
        max_features=5000,            # Limit vocabulary size
    )
    
    # Add insurance stopwords to the default English ones
    if hasattr(vectorizer, 'stop_words_'):
        # For sklearn 1.0+
        vectorizer.stop_words_ = vectorizer.get_stop_words().union(insurance_stopwords)
    else:
        # Manual approach
        english_stops = set(vectorizer.get_stop_words()) 
        vectorizer.stop_words = english_stops.union(insurance_stopwords)
    
    return vectorizer

def main():
    """Build and run a custom BERTopic pipeline."""
    # Load data
    df = load_data()
    
    print("\nBuilding custom BERTopic pipeline...")
    
    # 1. Create custom components
    print("Creating custom components...")
    
    # Custom count vectorizer with industry-specific stopwords
    count_vectorizer = create_insurance_specific_vectorizer()
    
    # Enhanced c-TF-IDF transformer
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,  # Reduce impact of frequent terms
        bm25_weighting=True,         # Use BM25 weighting for better results
    )
    
    # Dimensionality reduction with UMAP
    umap_model = UMAPReducer(
        n_neighbors=15,              # Balance local vs global structure
        n_components=5,              # Intermediate dimensionality
        min_dist=0.1,                # How tightly to pack points
        metric="cosine",             # Distance metric
        low_memory=True              # Memory optimization
    )
    
    # Clustering with HDBSCAN
    hdbscan_model = HDBSCANClusterer(
        min_cluster_size=10,         # Minimum size of clusters
        min_samples=5,               # Sample size for core points
        metric="euclidean",          # Distance metric
        prediction_data=True,        # Store data for predicting new points
        cluster_selection_method="eom"  # Excess of mass method
    )
    
    # Topic representation model - combining two approaches
    # First with KeyBERTInspired for better keywords
    keybert_model = KeyBERTInspired(
        embedding_model="paraphrase-MiniLM-L3-v2"  # Small embedding model for keywords
    )
    
    # Then with MMR for diversity in representation
    mmr_model = MaximalMarginalRelevance(
        diversity=0.3  # Balance between relevance and diversity (0-1)
    )
    
    # 2. Create BERTopic model with custom pipeline
    topic_model = BERTopic(
        # Main embedding model
        embedding_model="all-MiniLM-L6-v2",
        
        # Custom components
        vectorizer_model=count_vectorizer,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=[keybert_model, mmr_model],  # Pipeline of representations
        
        # Model parameters
        nr_topics=15,                # Target topic count
        min_topic_size=10,           # Minimum topic size
        calculate_probabilities=True,# Calculate doc-topic probabilities
        verbose=True
    )
    
    # 3. Fit the model
    print("\nFitting the model to the data...")
    topics, probs = topic_model.fit_transform(df["text"].tolist())
    
    # 4. Analyze the results
    print("\nAnalyzing results...")
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(f"Discovered {len(topic_info[topic_info['Topic'] != -1])} topics")
    print(f"Outliers: {topic_info.iloc[0]['Count']} documents ({topic_info.iloc[0]['Count']/len(df)*100:.1f}%)")
    
    # Print topics with detailed information about each component's contribution
    print("\nTopic representations:")
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id != -1:  # Skip outlier topic
            # Get documents in this topic
            topic_docs = df.iloc[[i for i, t in enumerate(topics) if t == topic_id]]
            doc_count = len(topic_docs)
            
            # Get topic representation
            topic_words = topic_model.get_topic(topic_id)
            words = [word for word, _ in topic_words[:7]]
            
            print(f"\nTopic {topic_id} ({doc_count} documents):")
            print(f"  Keywords: {', '.join(words)}")
            
            # Print a sample document
            if not topic_docs.empty:
                print(f"  Sample: {topic_docs.iloc[0]['text'][:150]}...")
    
    # 5. Visualize the results
    print("\nGenerating visualizations...")
    
    # Topic similarity visualization
    topic_model.visualize_topics().write_html("custom_pipeline_topic_similarity.html")
    
    # Topic hierarchy
    topic_model.visualize_hierarchy().write_html("custom_pipeline_hierarchy.html")
    
    # Topic barchart
    topic_model.visualize_barchart(top_n_topics=10).write_html("custom_pipeline_barchart.html")
    
    # Topic distribution
    topic_model.visualize_distribution(topic_model.get_topic_freq()).write_html("custom_pipeline_distribution.html")
    
    # Topic heatmap
    topic_model.visualize_heatmap(n_clusters=None).write_html("custom_pipeline_heatmap.html")
    
    print("\nBERTopic custom pipeline example completed successfully.")
    print("Generated visualizations:")
    print("- custom_pipeline_topic_similarity.html")
    print("- custom_pipeline_hierarchy.html")
    print("- custom_pipeline_barchart.html")
    print("- custom_pipeline_distribution.html")
    print("- custom_pipeline_heatmap.html")

if __name__ == "__main__":
    main()
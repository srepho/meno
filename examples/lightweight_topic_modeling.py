"""Example of lightweight topic modeling with minimal dependencies.

This example demonstrates how to use the lightweight topic modeling approaches
in Meno that don't require heavy dependencies like UMAP and HDBSCAN.

Approaches shown:
1. SimpleTopicModel: K-Means clustering on document embeddings
2. TFIDFTopicModel: TF-IDF vectorization with K-Means clustering
3. NMFTopicModel: Non-negative Matrix Factorization
4. LSATopicModel: Latent Semantic Analysis
"""

import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Meno components
from meno.modeling.simple_models import (
    SimpleTopicModel,
    TFIDFTopicModel,
    NMFTopicModel,
    LSATopicModel,
)
from meno.modeling.embeddings import DocumentEmbedding


def load_sample_data(min_size=100):
    """Load sample data from the 20 newsgroups dataset.
    
    Parameters
    ----------
    min_size : int, optional
        Minimum number of documents per category, by default 100
        
    Returns
    -------
    List[str]
        List of documents
    List[str]
        List of categories
    """
    from sklearn.datasets import fetch_20newsgroups
    
    # Load a subset of the 20 newsgroups dataset
    categories = [
        'comp.graphics',
        'rec.sport.baseball',
        'sci.med',
        'soc.religion.christian',
        'talk.politics.guns'
    ]
    
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    docs = newsgroups.data
    labels = [categories[i] for i in newsgroups.target]
    
    # Ensure we have at least min_size documents per category
    df = pd.DataFrame({'text': docs, 'category': labels})
    df = df.groupby('category').filter(lambda x: len(x) >= min_size)
    
    # Take a sample for faster processing
    df = pd.concat([
        group.sample(min_size, random_state=42) for name, group in df.groupby('category')
    ])
    
    # Clean the texts a bit
    df['text'] = df['text'].str.replace('\n', ' ').str.replace('\t', ' ')
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    
    print(f"Loaded {len(df)} documents across {len(categories)} categories.")
    return df['text'].tolist(), df['category'].tolist()


def simple_kmeans_example(docs, labels):
    """Example with SimpleTopicModel (K-Means on embeddings).
    
    Parameters
    ----------
    docs : List[str]
        Documents to analyze
    labels : List[str]
        True categories
    """
    print("\n" + "="*50)
    print("SimpleTopicModel (K-Means on Embeddings)")
    print("="*50)
    
    # Create embedding model with a smaller, faster model
    embedding_model = DocumentEmbedding(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    
    # Create and fit topic model
    start_time = time.time()
    
    model = SimpleTopicModel(
        num_topics=5,  # Number of true categories
        embedding_model=embedding_model,
        random_state=42
    )
    model.fit(docs)
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # Get topic information
    topic_info = model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info[["Topic", "Name", "Size"]])
    
    # Get document information
    doc_info = model.get_document_info()
    
    # Add true labels
    doc_info['True Category'] = labels
    
    # Print sample
    print("\nSample Document Topics:")
    print(doc_info.head(10)[["Document", "Topic", "True Category"]])
    
    # Calculate adjusted mutual information
    from sklearn.metrics import adjusted_mutual_info_score
    ami = adjusted_mutual_info_score(labels, doc_info["Topic"])
    print(f"\nAdjusted Mutual Information: {ami:.4f}")


def tfidf_kmeans_example(docs, labels):
    """Example with TFIDFTopicModel (no embedding model required).
    
    Parameters
    ----------
    docs : List[str]
        Documents to analyze
    labels : List[str]
        True categories
    """
    print("\n" + "="*50)
    print("TFIDFTopicModel (TF-IDF + K-Means)")
    print("="*50)
    
    # Create and fit topic model
    start_time = time.time()
    
    model = TFIDFTopicModel(
        num_topics=5,  # Number of true categories
        max_features=2000,  # Vocabulary size
        random_state=42
    )
    model.fit(docs)
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # Get topic information
    topic_info = model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info[["Topic", "Name", "Size"]])
    
    # Get document information
    doc_info = model.get_document_info()
    
    # Add true labels
    doc_info['True Category'] = labels
    
    # Print sample
    print("\nSample Document Topics:")
    print(doc_info.head(10)[["Document", "Topic", "True Category"]])
    
    # Calculate adjusted mutual information
    from sklearn.metrics import adjusted_mutual_info_score
    ami = adjusted_mutual_info_score(labels, doc_info["Topic"])
    print(f"\nAdjusted Mutual Information: {ami:.4f}")


def nmf_example(docs, labels):
    """Example with NMFTopicModel (Non-negative Matrix Factorization).
    
    Parameters
    ----------
    docs : List[str]
        Documents to analyze
    labels : List[str]
        True categories
    """
    print("\n" + "="*50)
    print("NMFTopicModel (Non-negative Matrix Factorization)")
    print("="*50)
    
    # Create and fit topic model
    start_time = time.time()
    
    model = NMFTopicModel(
        num_topics=5,  # Number of true categories
        max_features=2000,  # Vocabulary size
        random_state=42
    )
    model.fit(docs)
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # Get topic information
    topic_info = model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info[["Topic", "Name", "Size"]])
    
    # Get document information
    doc_info = model.get_document_info()
    
    # Add true labels
    doc_info['True Category'] = labels
    
    # Print sample
    print("\nSample Document Topics:")
    print(doc_info.head(10)[["Document", "Topic", "True Category"]])
    
    # Calculate adjusted mutual information
    from sklearn.metrics import adjusted_mutual_info_score
    ami = adjusted_mutual_info_score(labels, doc_info["Topic"])
    print(f"\nAdjusted Mutual Information: {ami:.4f}")


def lsa_example(docs, labels):
    """Example with LSATopicModel (Latent Semantic Analysis).
    
    Parameters
    ----------
    docs : List[str]
        Documents to analyze
    labels : List[str]
        True categories
    """
    print("\n" + "="*50)
    print("LSATopicModel (Latent Semantic Analysis)")
    print("="*50)
    
    # Create and fit topic model
    start_time = time.time()
    
    model = LSATopicModel(
        num_topics=5,  # Number of true categories
        max_features=2000,  # Vocabulary size
        random_state=42
    )
    model.fit(docs)
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # Get topic information
    topic_info = model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info[["Topic", "Name", "Size"]])
    
    # Get document information
    doc_info = model.get_document_info()
    
    # Add true labels
    doc_info['True Category'] = labels
    
    # Print sample
    print("\nSample Document Topics:")
    print(doc_info.head(10)[["Document", "Topic", "True Category"]])
    
    # Calculate adjusted mutual information
    from sklearn.metrics import adjusted_mutual_info_score
    ami = adjusted_mutual_info_score(labels, doc_info["Topic"])
    print(f"\nAdjusted Mutual Information: {ami:.4f}")


def unified_modeling_example(docs, labels):
    """Example with UnifiedTopicModeler.
    
    Parameters
    ----------
    docs : List[str]
        Documents to analyze
    labels : List[str]
        True categories
    """
    print("\n" + "="*50)
    print("UnifiedTopicModeler with Simple Models")
    print("="*50)
    
    from meno.modeling.unified_topic_modeling import UnifiedTopicModeler
    
    methods = ["simple_kmeans", "tfidf", "nmf", "lsa"]
    
    for method in methods:
        print(f"\nMethod: {method}")
        
        # Create and fit topic model
        start_time = time.time()
        
        model = UnifiedTopicModeler(
            method=method,
            num_topics=5,  # Number of true categories
            random_state=42
        )
        model.fit(docs)
        
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")
        
        # Get topic information
        topic_info = model.get_topic_info()
        print(f"Topics discovered: {len(topic_info)}")
        
        # Get document information
        doc_info = model.get_document_info()
        
        # Calculate adjusted mutual information
        from sklearn.metrics import adjusted_mutual_info_score
        ami = adjusted_mutual_info_score(labels, doc_info["Topic"])
        print(f"Adjusted Mutual Information: {ami:.4f}")


def performance_benchmark(docs, labels):
    """Performance benchmark for different methods.
    
    Parameters
    ----------
    docs : List[str]
        Documents to analyze
    labels : List[str]
        True categories
    """
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    results = []
    
    # Create embedding model for embedding-based methods
    embedding_model = DocumentEmbedding(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    
    # Define models to benchmark
    models = [
        ("SimpleTopicModel", SimpleTopicModel(
            num_topics=5,
            embedding_model=embedding_model,
            random_state=42
        )),
        ("TFIDFTopicModel", TFIDFTopicModel(
            num_topics=5,
            max_features=2000,
            random_state=42
        )),
        ("NMFTopicModel", NMFTopicModel(
            num_topics=5,
            max_features=2000,
            random_state=42
        )),
        ("LSATopicModel", LSATopicModel(
            num_topics=5,
            max_features=2000,
            random_state=42
        )),
    ]
    
    # Compute document embeddings once to reuse
    print("Computing document embeddings...")
    doc_embeddings = embedding_model.embed_documents(docs)
    
    # Run benchmark
    for name, model in models:
        print(f"\nBenchmarking {name}...")
        
        # Fit model
        start_time = time.time()
        
        if name == "SimpleTopicModel":
            model.fit(docs, embeddings=doc_embeddings)
        else:
            model.fit(docs)
        
        fit_time = time.time() - start_time
        
        # Transform
        start_time = time.time()
        
        if name == "SimpleTopicModel":
            doc_topic_matrix = model.transform(docs, embeddings=doc_embeddings)
        else:
            doc_topic_matrix = model.transform(docs)
        
        transform_time = time.time() - start_time
        
        # Get document topics
        doc_info = model.get_document_info()
        
        # Calculate adjusted mutual information
        from sklearn.metrics import adjusted_mutual_info_score
        ami = adjusted_mutual_info_score(labels, doc_info["Topic"])
        
        # Store results
        results.append({
            "Model": name,
            "Fit Time (s)": fit_time,
            "Transform Time (s)": transform_time,
            "Total Time (s)": fit_time + transform_time,
            "AMI": ami
        })
    
    # Print results
    results_df = pd.DataFrame(results)
    print("\nPerformance Results:")
    print(results_df)
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time plot
        results_df.plot(
            x="Model", 
            y=["Fit Time (s)", "Transform Time (s)"], 
            kind="bar", 
            ax=ax1
        )
        ax1.set_title("Processing Time")
        ax1.set_ylabel("Time (seconds)")
        
        # AMI plot
        results_df.plot(
            x="Model", 
            y="AMI", 
            kind="bar", 
            ax=ax2,
            color="green"
        )
        ax2.set_title("Clustering Quality (AMI)")
        ax2.set_ylabel("Adjusted Mutual Information")
        
        plt.tight_layout()
        plt.savefig("lightweight_model_benchmark.png")
        print(f"\nBenchmark plot saved to {os.path.abspath('lightweight_model_benchmark.png')}")
    
    except ImportError:
        print("\nMatplotlib not available. Skipping plot generation.")


def main():
    """Run the examples."""
    print("Lightweight Topic Modeling Examples")
    print("==================================")
    
    # Load sample data
    docs, labels = load_sample_data(min_size=100)
    
    # Run examples one by one to avoid potential failures
    try:
        simple_kmeans_example(docs, labels)
    except Exception as e:
        print(f"Error in simple_kmeans_example: {e}")
        
    try:
        tfidf_kmeans_example(docs, labels)
    except Exception as e:
        print(f"Error in tfidf_kmeans_example: {e}")
        
    try:
        nmf_example(docs, labels)
    except Exception as e:
        print(f"Error in nmf_example: {e}")
        
    try:
        lsa_example(docs, labels)
    except Exception as e:
        print(f"Error in lsa_example: {e}")
        
    try:
        unified_modeling_example(docs, labels)
    except Exception as e:
        print(f"Error in unified_modeling_example: {e}")
        
    try:
        performance_benchmark(docs, labels)
    except Exception as e:
        print(f"Error in performance_benchmark: {e}")
    
    print("\nAll examples completed successfully!")
    

if __name__ == "__main__":
    main()
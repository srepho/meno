"""
End-to-End Topic Modeling Example

This script demonstrates a complete topic modeling pipeline using Meno:
1. Preprocessing text data
2. Creating embeddings with all-MiniLM-L6-v2 on CPU
3. Clustering to generate topics
4. Labeling topics with LLMs
5. Generating comprehensive reports

It's optimized for medium-sized datasets (~6500 documents) and leverages
caching, deduplication, and dynamic batching for efficiency.
"""

import pandas as pd
import numpy as np
import time
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from meno.preprocessing.normalization import normalize_text
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno.reporting.html_generator import generate_topic_report
from meno.workflow import MenoWorkflow


def preprocess_text(df, text_column="text"):
    """Preprocess text data for topic modeling."""
    print("Step 1: Preprocessing text...")
    
    # Apply normalization (lowercase, remove punctuation, etc.)
    df["preprocessed_text"] = df[text_column].apply(normalize_text)
    
    print(f"Preprocessed {len(df)} documents")
    return df


def create_embeddings(df, text_column="preprocessed_text", use_cache=True):
    """Create document embeddings."""
    print("Step 2: Creating document embeddings...")
    start_time = time.time()
    
    # Use all-MiniLM-L6-v2 embeddings on CPU (as specified in requirements)
    embedding_model = DocumentEmbedding(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        cache_dir="./.meno_cache" if use_cache else None
    )
    
    # Create embeddings (with caching for faster reuse)
    embeddings = embedding_model.embed_documents(
        df[text_column].tolist(),
        show_progress_bar=True,
        cache=use_cache
    )
    
    elapsed = time.time() - start_time
    print(f"Created embeddings in {elapsed:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings, embedding_model


def cluster_topics(df, embeddings, num_topics=15, min_topic_size=20):
    """Cluster embeddings into topics."""
    print("Step 3: Clustering into topics...")
    start_time = time.time()
    
    # Use BERTopic with efficient CPU settings for clustering
    topic_model = BERTopicModel(
        embedding_model=None,  # We'll provide embeddings directly
        nr_topics=num_topics,
        min_topic_size=min_topic_size,
        # Configure for CPU efficiency
        n_gram_range=(1, 2),
        top_n_words=10,
        umap_args={
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0
        },
        hdbscan_args={
            "min_cluster_size": min_topic_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        }
    )
    
    # Fit the model with pre-computed embeddings
    topic_model.fit(
        documents=df["preprocessed_text"].tolist(),
        embeddings=embeddings
    )
    
    # Assign topics to documents
    df["topic"], df["topic_prob"] = topic_model.transform(
        df["preprocessed_text"].tolist(),
        embeddings=embeddings
    )
    
    elapsed = time.time() - start_time
    print(f"Clustered topics in {elapsed:.2f} seconds")
    
    # Count topics
    topic_counts = topic_model.get_topic_info()
    print(f"Found {len(topic_counts[topic_counts['Topic'] >= 0])} topics")
    print(f"Largest topic has {topic_counts['Count'].max()} documents")
    print(f"Smallest topic has {topic_counts[topic_counts['Topic'] >= 0]['Count'].min()} documents")
    
    return topic_model, df


def label_topics_with_llm(topic_model, df, api_key=None, azure_endpoint=None):
    """Label topics using LLM with optimized batching."""
    print("Step 4: Labeling topics with LLM...")
    start_time = time.time()
    
    # Configure the LLM topic labeler
    labeler_args = {
        "model_type": "openai" if api_key or azure_endpoint else "local",
        "model_name": "gpt-3.5-turbo",
        "max_new_tokens": 50,
        "temperature": 0.2,
        "enable_fallback": True,
        "requests_per_minute": 60,
        "max_parallel_requests": 4,
        "batch_size": 10,
        "enable_cache": True,
        "cache_dir": "./.meno_cache",
        "system_prompt_template": "You are a topic modeling assistant that generates concise, descriptive topic names."
    }
    
    # Add API credentials if provided
    if api_key:
        labeler_args["openai_api_key"] = api_key
    
    if azure_endpoint:
        labeler_args["api_endpoint"] = azure_endpoint
        labeler_args["api_version"] = "2023-07-01-preview"
    
    # Create the labeler
    labeler = LLMTopicLabeler(**labeler_args)
    
    # Get example documents for each topic to improve labeling
    example_docs = {}
    for topic_id in topic_model.get_topics():
        if topic_id == -1:  # Skip outliers
            continue
            
        # Get documents for this topic
        topic_docs = df[df["topic"] == topic_id]["preprocessed_text"].tolist()
        
        # Select up to 5 documents with highest probability
        topic_probs = df[df["topic"] == topic_id]["topic_prob"].tolist()
        if topic_docs and topic_probs:
            # Sort by probability and take top 5
            doc_prob_pairs = sorted(zip(topic_docs, topic_probs), key=lambda x: x[1], reverse=True)
            example_docs[topic_id] = [doc for doc, _ in doc_prob_pairs[:5]]
    
    # Label topics with example documents for better context
    updated_model = labeler.update_model_topic_names(
        topic_model=topic_model,
        example_docs_per_topic=example_docs,
        detailed=True,
        progress_bar=True
    )
    
    elapsed = time.time() - start_time
    print(f"Labeled topics in {elapsed:.2f} seconds")
    
    # Print topic labels
    print("\nTopic Labels:")
    topic_info = updated_model.get_topic_info()
    for _, row in topic_info.iterrows():
        if row["Topic"] >= 0:
            print(f"  Topic {row['Topic']}: {row['Name']} ({row['Count']} documents)")
    
    return updated_model


def generate_report(topic_model, df, output_dir="./meno_output"):
    """Generate comprehensive topic modeling report."""
    print("Step 5: Generating topic report...")
    start_time = time.time()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate topic report
    report_path = output_dir / "topic_model_report.html"
    generate_topic_report(
        topic_model=topic_model,
        documents=df["preprocessed_text"].tolist(),
        document_ids=df.index.tolist(),
        output_path=str(report_path),
        metadata={
            "project": "Meno Example",
            "dataset_size": len(df),
            "embedding_model": "all-MiniLM-L6-v2",
            "clustering_method": "BERTopic (UMAP + HDBSCAN)",
            "date": time.strftime("%Y-%m-%d")
        }
    )
    
    # Save document topics for further analysis
    df[["topic", "topic_prob"]].to_csv(output_dir / "document_topics.csv")
    
    # Save topic model for later use
    topic_model.save(output_dir / "topic_model")
    
    elapsed = time.time() - start_time
    print(f"Generated report in {elapsed:.2f} seconds")
    print(f"Report saved to {report_path}")
    
    return report_path


def main(input_file=None, text_column="text", num_topics=15, output_dir="./meno_output", 
         api_key=None, azure_endpoint=None, use_workflow=False):
    """Run the complete end-to-end topic modeling pipeline."""
    print("\n==== Meno End-to-End Topic Modeling Example ====\n")
    
    # Start timing
    start_time = time.time()
    
    # Load data
    if input_file:
        print(f"Loading data from {input_file}...")
        file_extension = Path(input_file).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(input_file)
        elif file_extension == '.json':
            df = pd.read_json(input_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        print(f"Loaded {len(df)} documents")
    else:
        print("No input file provided. Using sample data...")
        
        # Create sample data with multiple topics for demonstration
        topics = {
            "technology": [
                "The latest smartphone features improved AI capabilities",
                "Software developers are embracing cloud computing technologies",
                "Quantum computing breakthrough announced by research team",
                "New programming language designed for distributed systems",
                "Artificial intelligence is transforming business operations"
            ],
            "healthcare": [
                "Medical researchers develop new treatment for chronic disease",
                "Hospital implements electronic health record system",
                "Study shows benefits of regular exercise for heart health",
                "New vaccine developed against infectious disease",
                "Mental health services expand telehealth options"
            ],
            "finance": [
                "Stock market rises amid positive economic indicators",
                "Investment firm recommends diversifying portfolios",
                "Banking regulations updated to address emerging risks",
                "Cryptocurrency values fluctuate in volatile market",
                "Financial advisors suggest retirement planning strategies"
            ],
            "entertainment": [
                "New movie breaks box office records in opening weekend",
                "Streaming service announces original content lineup",
                "Music festival attracts record number of attendees",
                "Video game industry continues growth trend",
                "Celebrity endorses new entertainment platform"
            ]
        }
        
        # Generate sample data by repeating and varying these base examples
        np.random.seed(42)
        sample_data = []
        for topic, examples in topics.items():
            for i in range(25):  # 25 variations of each example in each topic
                for example in examples:
                    # Add some variation to avoid exact duplicates
                    noise = np.random.choice([
                        "", 
                        " according to recent reports", 
                        " experts say",
                        " as announced yesterday",
                        " which surprised many"
                    ])
                    sample_data.append({
                        "text": example + noise,
                        "true_topic": topic
                    })
        
        df = pd.DataFrame(sample_data)
        print(f"Created sample dataset with {len(df)} documents")
    
    if use_workflow:
        # Use the integrated Meno workflow
        print("\nUsing integrated MenoWorkflow for end-to-end processing...")
        
        workflow = MenoWorkflow()
        
        # Configure workflow
        if api_key:
            workflow.set_openai_api_key(api_key)
            
        # Load and process data
        workflow.load_data(df, text_column=text_column)
        workflow.preprocess_documents(normalize=True)
        
        # Configure embedding for CPU
        workflow.set_embedding_model("all-MiniLM-L6-v2", device="cpu")
        
        # Discover topics
        results = workflow.discover_topics(
            method="bertopic",
            num_topics=num_topics,
            min_topic_size=20,
            top_n_words=10,
            verbose=True
        )
        
        # Label topics with LLM if API key provided
        if api_key or azure_endpoint:
            if azure_endpoint:
                # Configure for Azure
                workflow.llm_label_topics(
                    model_type="openai",
                    model_name="gpt-3.5-turbo",
                    api_endpoint=azure_endpoint,
                    api_version="2023-07-01-preview"
                )
            else:
                # Use standard OpenAI
                workflow.llm_label_topics()
        
        # Generate the report
        report_path = workflow.generate_report(output_dir=output_dir)
        
        # Get the topic model for evaluation
        topic_model = workflow.get_topic_model()
        
        # Calculate performance
        elapsed_time = time.time() - start_time
        print(f"\nCompleted end-to-end processing in {elapsed_time:.2f} seconds")
        print(f"Report saved to {report_path}")
    else:
        # Run each step individually (more control, same result)
        # Step 1: Preprocess text
        df = preprocess_text(df, text_column)
        
        # Step 2: Create embeddings
        embeddings, embedding_model = create_embeddings(df)
        
        # Step 3: Cluster into topics
        topic_model, df = cluster_topics(df, embeddings, num_topics)
        
        # Step 4: Label topics with LLM (if API key provided)
        if api_key or azure_endpoint:
            topic_model = label_topics_with_llm(topic_model, df, api_key, azure_endpoint)
        
        # Step 5: Generate report
        report_path = generate_report(topic_model, df, output_dir)
        
        # Calculate overall performance
        elapsed_time = time.time() - start_time
        print(f"\nCompleted end-to-end processing in {elapsed_time:.2f} seconds")
    
    return df, topic_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meno End-to-End Topic Modeling Example")
    parser.add_argument("--input", help="Input file path (CSV, JSON, or Excel)")
    parser.add_argument("--text-column", default="text", help="Column containing text data")
    parser.add_argument("--topics", type=int, default=15, help="Number of topics to extract")
    parser.add_argument("--output", default="./meno_output", help="Output directory for results")
    parser.add_argument("--api-key", help="OpenAI API key for topic labeling")
    parser.add_argument("--azure", help="Azure OpenAI endpoint URL")
    parser.add_argument("--workflow", action="store_true", help="Use integrated MenoWorkflow")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        text_column=args.text_column,
        num_topics=args.topics,
        output_dir=args.output,
        api_key=args.api_key,
        azure_endpoint=args.azure,
        use_workflow=args.workflow
    )
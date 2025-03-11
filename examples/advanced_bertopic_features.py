"""
Example demonstrating advanced BERTopic features in Meno.

This example showcases the extended BERTopic functionality in Meno, including:
1. Model merging - Combining multiple topic models into one
2. Topic merging - Merging similar topics within a model
3. Dynamic topic modeling - Analyzing how topics evolve over time
4. Semi-supervised topic modeling - Using seed topics to guide modeling
5. LLM-based topic labeling - Using LLMs to generate human-readable topic labels
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import os
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Meno components
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.embeddings import DocumentEmbedding
from meno.visualization.bertopic_viz import plot_topic_similarity_network, plot_topic_hierarchy
from meno.visualization.static_plots import plot_topic_distribution
from meno.reporting.html_generator import generate_topic_report
from meno.preprocessing.normalization import normalize_text

# Sample datasets
def get_ai_dataset(n_samples: int = 100) -> List[str]:
    """Generate sample AI-related documents."""
    ai_documents = [
        "Machine learning algorithms require significant computational resources",
        "Deep neural networks have revolutionized computer vision tasks",
        "Natural language processing enables machines to understand text",
        "Reinforcement learning is used to train game-playing AI",
        "Data science relies on statistics and domain knowledge",
        "Feature engineering improves model performance significantly",
        "Transformers have improved natural language understanding",
        "Computer vision systems can now recognize objects in images",
        "Statistical models help understand patterns in data",
        "Unsupervised learning finds patterns without labeled data",
        "Neural networks are inspired by biological brain structures",
        "GPT models generate human-like text with remarkable fluency",
        "Transfer learning applies knowledge from one domain to another",
        "Embeddings represent words or entities as dense vectors",
        "Attention mechanisms help models focus on relevant information",
    ]
    # Expand dataset with variations
    expanded_docs = []
    for _ in range(max(1, n_samples // len(ai_documents))):
        for doc in ai_documents:
            expanded_docs.append(doc)
            expanded_docs.append(doc + " in artificial intelligence research.")
            expanded_docs.append("Research shows that " + doc.lower())
    
    return expanded_docs[:n_samples]

def get_cloud_dataset(n_samples: int = 100) -> List[str]:
    """Generate sample cloud computing-related documents."""
    cloud_documents = [
        "Cloud computing provides scalable resources for AI workloads",
        "AWS offers a wide range of cloud services for businesses",
        "Azure integrates well with Microsoft's enterprise ecosystem",
        "Google Cloud Platform excels at machine learning services",
        "Serverless architectures reduce operational complexity",
        "Kubernetes orchestrates containerized applications efficiently",
        "Docker containers provide consistent development environments",
        "Microservices architecture improves system modularity",
        "DevOps practices streamline software delivery pipelines",
        "Infrastructure as code automates resource provisioning",
        "Auto-scaling adjusts resources based on demand",
        "Cloud security requires specialized expertise",
        "Multi-cloud strategies prevent vendor lock-in",
        "Edge computing processes data near its source",
        "Virtual machines provide isolated execution environments",
    ]
    # Expand dataset with variations
    expanded_docs = []
    for _ in range(max(1, n_samples // len(cloud_documents))):
        for doc in cloud_documents:
            expanded_docs.append(doc)
            expanded_docs.append(doc + " in modern infrastructure.")
            expanded_docs.append("Companies find that " + doc.lower())
    
    return expanded_docs[:n_samples]

def get_data_science_dataset(n_samples: int = 100) -> List[str]:
    """Generate sample data science-related documents."""
    data_science_documents = [
        "Pandas provides powerful data manipulation capabilities",
        "Data visualization helps identify patterns and outliers",
        "Exploratory data analysis reveals insights in datasets",
        "Feature selection improves model performance and interpretability",
        "Time series analysis examines data points collected over time",
        "Regression models predict continuous numerical values",
        "Classification algorithms categorize data into predefined classes",
        "Clustering groups similar items without predefined categories",
        "Cross-validation ensures models generalize to new data",
        "Hyperparameter tuning optimizes model performance",
        "Data preprocessing cleans and transforms raw data",
        "Statistical significance measures result reliability",
        "A/B testing compares different versions of products",
        "Dimensionality reduction simplifies high-dimensional data",
        "Ensemble methods combine multiple models for better predictions",
    ]
    # Expand dataset with variations
    expanded_docs = []
    for _ in range(max(1, n_samples // len(data_science_documents))):
        for doc in data_science_documents:
            expanded_docs.append(doc)
            expanded_docs.append(doc + " in data science workflows.")
            expanded_docs.append("Data scientists know that " + doc.lower())
    
    return expanded_docs[:n_samples]

def generate_timestamped_documents(n_samples: int = 300, days: int = 30) -> Tuple[List[str], List[datetime]]:
    """Generate documents with timestamps for dynamic topic modeling."""
    # Define topics for each time period
    early_topics = [
        "Cloud computing reduces infrastructure costs significantly",
        "Virtual machines provide isolated environments for applications",
        "Machine learning algorithms analyze data patterns",
        "Business intelligence dashboards visualize key metrics",
        "Data warehouses centralize information from multiple sources",
    ]
    
    mid_topics = [
        "Serverless computing eliminates server management needs",
        "Containerization improves deployment consistency across environments",
        "Deep learning models achieve remarkable accuracy on complex tasks",
        "Data lakes store vast amounts of raw structured and unstructured data",
        "Real-time analytics processes data as it's generated",
    ]
    
    late_topics = [
        "Multi-cloud strategies prevent vendor lock-in situations",
        "Kubernetes orchestrates containerized applications at scale",
        "Generative AI creates new content like images and text",
        "Streaming analytics processes continuous data flows",
        "Graph databases model complex relationships efficiently",
    ]
    
    # Generate timestamps and documents
    now = datetime.now()
    documents = []
    timestamps = []
    
    for i in range(n_samples):
        # Determine which time period and topic set to use
        day = (i % days) + 1
        period = day / days
        
        if period < 0.33:
            topics = early_topics
            offset = i % len(early_topics)
        elif period < 0.66:
            topics = mid_topics
            offset = i % len(mid_topics)
        else:
            topics = late_topics
            offset = i % len(late_topics)
        
        # Create document and timestamp
        doc = topics[offset]
        timestamp = now - timedelta(days=days-day)
        
        # Add variations
        if i % 3 == 0:
            doc = "Studies show that " + doc.lower()
        elif i % 3 == 1:
            doc = doc + " according to industry experts."
            
        documents.append(doc)
        timestamps.append(timestamp)
        
    return documents, timestamps

# 1. Model Merging Example
def model_merging_example() -> None:
    """Demonstrate merging of multiple BERTopic models."""
    logger.info("Example 1: Model Merging")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Get three different datasets
    ai_docs = get_ai_dataset(150)
    cloud_docs = get_cloud_dataset(150)
    data_science_docs = get_data_science_dataset(150)
    
    logger.info(f"Loaded data: {len(ai_docs)} AI docs, {len(cloud_docs)} cloud docs, "
               f"{len(data_science_docs)} data science docs")
    
    # Train three separate models on different domains
    logger.info("Training AI topics model...")
    ai_model = BERTopicModel(
        num_topics=5,
        embedding_model=embedding_model,
        min_topic_size=5,
        n_neighbors=10,
        n_components=5,
        verbose=True
    )
    ai_model.fit(ai_docs)
    
    logger.info("Training cloud computing topics model...")
    cloud_model = BERTopicModel(
        num_topics=5,
        embedding_model=embedding_model,
        min_topic_size=5,
        n_neighbors=10,
        n_components=5,
        verbose=True
    )
    cloud_model.fit(cloud_docs)
    
    logger.info("Training data science topics model...")
    data_science_model = BERTopicModel(
        num_topics=5,
        embedding_model=embedding_model,
        min_topic_size=5,
        n_neighbors=10,
        n_components=5,
        verbose=True
    )
    data_science_model.fit(data_science_docs)
    
    # Print topics from individual models
    logger.info("AI Topics:")
    for topic_id, topic_name in ai_model.topics.items():
        if topic_id != -1:
            logger.info(f"  Topic {topic_id}: {topic_name}")
    
    logger.info("Cloud Topics:")
    for topic_id, topic_name in cloud_model.topics.items():
        if topic_id != -1:
            logger.info(f"  Topic {topic_id}: {topic_name}")
    
    logger.info("Data Science Topics:")
    for topic_id, topic_name in data_science_model.topics.items():
        if topic_id != -1:
            logger.info(f"  Topic {topic_id}: {topic_name}")
    
    # Merge models
    logger.info("Merging models...")
    all_docs = ai_docs + cloud_docs + data_science_docs
    
    merged_model = ai_model.merge_models(
        models=[cloud_model, data_science_model],
        documents=all_docs,
        min_similarity=0.7
    )
    
    # Print topics from merged model
    logger.info("Merged Topics:")
    topic_info = merged_model.get_topic_info()
    for i, row in topic_info.iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    
    # Visualize topic similarity network
    try:
        os.makedirs("./output/advanced_examples", exist_ok=True)
        fig = plot_topic_similarity_network(merged_model)
        fig.write_html("./output/advanced_examples/merged_model_network.html")
        logger.info("Generated topic similarity network visualization")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")

# 2. Topic Merging and Reduction Example
def topic_merging_example() -> None:
    """Demonstrate merging and reduction of topics within a single model."""
    logger.info("Example 2: Topic Merging and Reduction")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Get combined dataset
    all_docs = get_ai_dataset(100) + get_cloud_dataset(100) + get_data_science_dataset(100)
    
    # Train a model with many topics
    logger.info("Training model with many fine-grained topics...")
    model = BERTopicModel(
        num_topics=15,  # Intentionally create many topics that will later be merged
        embedding_model=embedding_model,
        min_topic_size=3,  # Small size to create more topics
        n_neighbors=5,
        n_components=5,
        verbose=True
    )
    model.fit(all_docs)
    
    # Print original topics
    logger.info("Original Topics:")
    original_topic_info = model.get_topic_info()
    for i, row in original_topic_info.iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
            
    # Identify similar topics to merge
    logger.info("Analyzing topics for potential merging...")
    
    # For demonstration purposes, we'll merge some topics based on their IDs
    # In a real application, you would analyze topic content for similarity
    topics_to_merge = []
    remaining_topics = [t for t in model.topics.keys() if t != -1]
    
    # Create groups of 2-3 topics to merge
    while len(remaining_topics) >= 2:
        group_size = min(3, len(remaining_topics))
        group = remaining_topics[:group_size]
        topics_to_merge.append(group)
        remaining_topics = remaining_topics[group_size:]
    
    if topics_to_merge:
        logger.info(f"Merging topic groups: {topics_to_merge}")
        
        # Merge topics
        model.merge_topics(topics_to_merge, documents=all_docs)
        
        # Print merged topics
        logger.info("Topics after merging:")
        merged_topic_info = model.get_topic_info()
        for i, row in merged_topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    
    # Alternative approach: Reduce to a specific number of topics
    logger.info("Reducing to 5 topics...")
    model.reduce_topics(all_docs, nr_topics=5)
    
    # Print reduced topics
    logger.info("Topics after reduction:")
    reduced_topic_info = model.get_topic_info()
    for i, row in reduced_topic_info.iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    
    # Visualize topic hierarchy
    try:
        os.makedirs("./output/advanced_examples", exist_ok=True)
        fig = plot_topic_hierarchy(model)
        fig.write_html("./output/advanced_examples/topic_hierarchy.html")
        logger.info("Generated topic hierarchy visualization")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
    
    # Apply LLM labeling to get better topic names
    try:
        logger.info("Applying LLM labeling to topics...")
        model.apply_llm_labeling(
            documents=all_docs,
            model_type="local",
            model_name="google/flan-t5-small",
            detailed=True
        )
        
        # Print topics with LLM labels
        logger.info("Topics with LLM-generated names:")
        llm_topic_info = model.get_topic_info()
        for i, row in llm_topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    except Exception as e:
        logger.error(f"LLM labeling failed: {e}")

# 3. Dynamic Topic Modeling Example
def dynamic_topic_modeling_example() -> None:
    """Demonstrate dynamic topic modeling over time."""
    logger.info("Example 3: Dynamic Topic Modeling")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Generate timestamped documents
    documents, timestamps = generate_timestamped_documents(300, 30)
    
    logger.info(f"Generated {len(documents)} timestamped documents over {30} days")
    
    # Create BERTopic model
    model = BERTopicModel(
        embedding_model=embedding_model,
        min_topic_size=5,
        verbose=True
    )
    
    # Try to use dynamic topic modeling if supported
    try:
        logger.info("Applying dynamic topic modeling...")
        topics, probs, timestamps_array = model.fit_transform_with_timestamps(
            documents=documents,
            timestamps=timestamps,
            global_tuning=True
        )
        
        # Print discovered topics
        logger.info("Discovered topics:")
        topic_info = model.get_topic_info()
        for i, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
        
        # Extract topics over time data from the model or create our own
        if hasattr(model.model, 'topics_over_time_'):
            topics_over_time = model.model.topics_over_time_
        else:
            # Create a manual topics over time dataset
            logger.info("Creating manual topics over time analysis...")
            
            # Convert timestamps to string format for grouping
            timestamp_strs = [ts.strftime('%Y-%m-%d') for ts in timestamps]
            
            # Create DataFrame with all data
            df = pd.DataFrame({
                'Document': documents,
                'Topic': topics,
                'Probability': np.max(probs, axis=1) if len(probs.shape) > 1 else probs,
                'Timestamp': timestamp_strs
            })
            
            # Group by timestamp and topic
            topics_over_time = df.groupby(['Timestamp', 'Topic']).agg(
                Count=('Document', 'count'),
                Probability=('Probability', 'mean')
            ).reset_index()
            
            # Convert to datetime for proper ordering
            topics_over_time['Timestamp'] = pd.to_datetime(topics_over_time['Timestamp'])
            
            # Sort by timestamp
            topics_over_time = topics_over_time.sort_values('Timestamp')
        
        # Visualize topics over time
        try:
            os.makedirs("./output/advanced_examples", exist_ok=True)
            fig = model.visualize_topics_over_time(
                topics_over_time=topics_over_time,
                top_n_topics=5
            )
            fig.write_html("./output/advanced_examples/topics_over_time.html")
            logger.info("Generated topics over time visualization")
        except Exception as e:
            logger.error(f"Failed to generate time visualization: {e}")
            
    except (NotImplementedError, ValueError) as e:
        logger.warning(f"Dynamic topic modeling not fully supported: {e}")
        logger.info("Falling back to standard topic modeling...")
        
        # Fall back to normal topic modeling
        model.fit(documents)
        
        # Print topics
        topic_info = model.get_topic_info()
        for i, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")

# 4. Semi-supervised Topic Modeling Example
def semi_supervised_example() -> None:
    """Demonstrate semi-supervised topic modeling with seed topics."""
    logger.info("Example 4: Semi-supervised Topic Modeling")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Get combined dataset
    all_docs = get_ai_dataset(100) + get_cloud_dataset(100) + get_data_science_dataset(100)
    
    # Define seed topics
    seed_topic_list = [
        ["machine", "learning", "neural", "network", "algorithm"],
        ["cloud", "computing", "aws", "azure", "serverless"],
        ["data", "science", "analysis", "statistics", "visualization"]
    ]
    
    # Initialize BERTopic model
    model = BERTopicModel(
        embedding_model=embedding_model,
        min_topic_size=5,
        verbose=True
    )
    
    try:
        logger.info("Applying semi-supervised topic modeling with seed topics...")
        model.fit_with_seed_topics(
            documents=all_docs,
            seed_topic_list=seed_topic_list
        )
        
        # Print topics
        logger.info("Discovered topics with seed guidance:")
        topic_info = model.get_topic_info()
        for i, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
                
        # Visualize topic similarity
        try:
            os.makedirs("./output/advanced_examples", exist_ok=True)
            fig = plot_topic_similarity_network(model)
            fig.write_html("./output/advanced_examples/seeded_topic_network.html")
            logger.info("Generated seeded topic network visualization")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            
    except (NotImplementedError, ValueError) as e:
        logger.warning(f"Semi-supervised topic modeling not fully supported: {e}")
        logger.info("Falling back to standard topic modeling...")
        
        # Fall back to normal topic modeling
        model.fit(all_docs)
        
        # Print topics
        topic_info = model.get_topic_info()
        for i, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")

# 5. LLM Topic Labeling Example
def llm_topic_labeling_example() -> None:
    """Demonstrate LLM-based topic labeling."""
    logger.info("Example 5: LLM Topic Labeling")
    
    # Create embedding model
    embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Get combined dataset
    all_docs = get_ai_dataset(70) + get_cloud_dataset(70) + get_data_science_dataset(70)
    
    # Create a BERTopic model without LLM labeling first
    model = BERTopicModel(
        num_topics=8,
        embedding_model=embedding_model,
        min_topic_size=5,
        verbose=True,
        use_llm_labeling=False  # Initially disable LLM labeling
    )
    
    # Fit the model with default keyword-based labeling
    model.fit(all_docs)
    
    # Print original topic names
    logger.info("Original topics with keyword-based names:")
    original_topic_info = model.get_topic_info()
    for i, row in original_topic_info.iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    
    # Now apply LLM labeling
    try:
        logger.info("Applying LLM topic labeling...")
        model.apply_llm_labeling(
            documents=all_docs,
            model_type="local",
            model_name="google/flan-t5-small",
            detailed=True
        )
        
        # Print topics with LLM labels
        logger.info("Topics with LLM-generated names:")
        llm_topic_info = model.get_topic_info()
        for i, row in llm_topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
                
        # Try different LLM model if available
        logger.info("Applying alternative LLM labeling...")
        model.apply_llm_labeling(
            documents=all_docs,
            model_type="local",
            model_name="facebook/opt-125m",  # Try a different model if available
            detailed=False  # Less detailed labels
        )
        
        # Print topics with alternative LLM labels
        logger.info("Topics with alternative LLM-generated names:")
        alt_llm_topic_info = model.get_topic_info()
        for i, row in alt_llm_topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    
    except Exception as e:
        logger.error(f"LLM labeling failed: {e}")
    
    # Now create a model with LLM labeling enabled during fitting
    try:
        logger.info("Creating a new model with LLM labeling during fitting...")
        llm_model = BERTopicModel(
            num_topics=8,
            embedding_model=embedding_model,
            min_topic_size=5,
            verbose=True,
            use_llm_labeling=True,  # Enable LLM labeling during fitting
            llm_model_type="local",
            llm_model_name="google/flan-t5-small"
        )
        
        # Fit the model with LLM labeling
        llm_model.fit(all_docs)
        
        # Print topics with LLM labels
        logger.info("Topics with integrated LLM-generated names:")
        integrated_llm_topic_info = llm_model.get_topic_info()
        for i, row in integrated_llm_topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
                
        # Generate HTML report
        try:
            os.makedirs("./output/advanced_examples", exist_ok=True)
            
            # Get topic assignments
            topics, probs = llm_model.transform(all_docs)
            
            # Create a DataFrame with document texts and topic assignments
            doc_data = pd.DataFrame({
                'text': all_docs,
                'topic': topics,
            })
            
            # Generate report
            report_path = "./output/advanced_examples/llm_labeled_topics_report.html"
            generate_topic_report(
                topic_model=llm_model,
                documents=doc_data,
                output_file=report_path,
                plot_width=800,
                plot_height=600,
                include_sample_docs=True
            )
            logger.info(f"Generated HTML report at {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            
    except Exception as e:
        logger.error(f"Integrated LLM labeling failed: {e}")

# Main function
def main():
    """Run the advanced BERTopic examples."""
    logger.info("Running Advanced BERTopic Features Examples")
    
    # Create output directory
    os.makedirs("./output/advanced_examples", exist_ok=True)
    
    # Run examples
    try:
        model_merging_example()
    except Exception as e:
        logger.error(f"Model merging example failed: {e}")
        
    try:
        topic_merging_example()
    except Exception as e:
        logger.error(f"Topic merging example failed: {e}")
        
    try:
        dynamic_topic_modeling_example()
    except Exception as e:
        logger.error(f"Dynamic topic modeling example failed: {e}")
        
    try:
        semi_supervised_example()
    except Exception as e:
        logger.error(f"Semi-supervised example failed: {e}")
        
    try:
        llm_topic_labeling_example()
    except Exception as e:
        logger.error(f"LLM topic labeling example failed: {e}")
    
    logger.info("Advanced BERTopic examples completed")

if __name__ == "__main__":
    main()
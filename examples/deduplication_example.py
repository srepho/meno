"""
Deduplication Example

This example demonstrates how to use Meno's document deduplication feature 
to optimize topic modeling by processing only unique documents.
"""

import pandas as pd
import numpy as np
import time

from meno import MenoWorkflow


def create_dataset_with_duplicates(num_unique=500, duplicates_per_doc=3, seed=42):
    """Create a synthetic dataset with duplicated documents."""
    np.random.seed(seed)
    
    # Create unique documents
    unique_docs = []
    topics = ["technology", "health", "finance", "sports", "entertainment"]
    
    for i in range(num_unique):
        topic = topics[i % len(topics)]
        words = f"{topic} document with specific terms about {topic}."
        words += f" This is unique document number {i}."
        unique_docs.append({
            "text": words,
            "id": f"doc_{i}",
            "true_topic": topic
        })
    
    # Create duplicate documents with some randomization
    all_docs = []
    for doc in unique_docs:
        # Add the original
        all_docs.append(doc)
        
        # Add duplicates
        for j in range(duplicates_per_doc):
            # Small random variation in some duplicates (20% chance)
            if np.random.random() < 0.2:
                text = doc["text"] + f" Slight variation {j+1}."
            else:
                text = doc["text"]
                
            duplicate = {
                "text": text,
                "id": f"{doc['id']}_dup_{j+1}",
                "true_topic": doc["true_topic"]
            }
            all_docs.append(duplicate)
    
    # Shuffle the documents
    np.random.shuffle(all_docs)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_docs)
    
    return df


def run_with_deduplication():
    """Run topic modeling with deduplication."""
    print("Creating dataset with duplicates...")
    data = create_dataset_with_duplicates(num_unique=500, duplicates_per_doc=5)
    
    print(f"Dataset size: {len(data)} documents")
    
    # Count true duplicates (exact matches)
    duplicates = data.duplicated(subset=["text"]).sum()
    print(f"Exact duplicates: {duplicates} documents")
    
    # Run with deduplication
    print("\nRunning topic modeling WITH deduplication...")
    start_time = time.time()
    
    workflow_dedup = MenoWorkflow()
    workflow_dedup.load_data(
        data=data, 
        text_column="text",
        deduplicate=True  # Enable deduplication
    )
    workflow_dedup.preprocess_documents()
    dedup_results = workflow_dedup.discover_topics(method="bertopic", num_topics=5)
    
    dedup_time = time.time() - start_time
    
    print(f"Completed in {dedup_time:.2f} seconds")
    print(f"Result size: {len(dedup_results)} documents (should match original dataset size)")
    
    # Count topics
    dedup_topic_counts = dedup_results["topic"].value_counts().sort_index()
    print("\nTopic distribution with deduplication:")
    print(dedup_topic_counts)
    
    return dedup_results, dedup_time


def run_without_deduplication():
    """Run topic modeling without deduplication."""
    print("Creating dataset with duplicates...")
    data = create_dataset_with_duplicates(num_unique=500, duplicates_per_doc=5)
    
    # Run without deduplication
    print("\nRunning topic modeling WITHOUT deduplication...")
    start_time = time.time()
    
    workflow_standard = MenoWorkflow()
    workflow_standard.load_data(
        data=data, 
        text_column="text",
        deduplicate=False  # Disable deduplication (default)
    )
    workflow_standard.preprocess_documents()
    standard_results = workflow_standard.discover_topics(method="bertopic", num_topics=5)
    
    standard_time = time.time() - start_time
    
    print(f"Completed in {standard_time:.2f} seconds")
    print(f"Result size: {len(standard_results)} documents")
    
    # Count topics
    standard_topic_counts = standard_results["topic"].value_counts().sort_index()
    print("\nTopic distribution without deduplication:")
    print(standard_topic_counts)
    
    return standard_results, standard_time


def compare_results():
    """Compare topic modeling with and without deduplication."""
    # Run both approaches
    print("\n=== COMPARING TOPIC MODELING WITH AND WITHOUT DEDUPLICATION ===\n")
    
    # Create consistent dataset
    np.random.seed(42)
    data = create_dataset_with_duplicates(num_unique=500, duplicates_per_doc=5)
    print(f"Dataset size: {len(data)} documents ({len(data.drop_duplicates(subset=['text']))} unique)")
    
    # Run with deduplication
    print("\n--- WITH DEDUPLICATION ---")
    start_time = time.time()
    
    workflow_dedup = MenoWorkflow()
    workflow_dedup.load_data(data=data, text_column="text", deduplicate=True)
    workflow_dedup.preprocess_documents()
    dedup_results = workflow_dedup.discover_topics(method="bertopic", num_topics=5)
    
    dedup_time = time.time() - start_time
    
    # Run without deduplication
    print("\n--- WITHOUT DEDUPLICATION ---")
    start_time = time.time()
    
    workflow_standard = MenoWorkflow()
    workflow_standard.load_data(data=data, text_column="text", deduplicate=False)
    workflow_standard.preprocess_documents()
    standard_results = workflow_standard.discover_topics(method="bertopic", num_topics=5)
    
    standard_time = time.time() - start_time
    
    # Compare results
    print("\n=== RESULTS COMPARISON ===")
    print(f"Time with deduplication:    {dedup_time:.2f} seconds")
    print(f"Time without deduplication: {standard_time:.2f} seconds")
    print(f"Speedup: {(standard_time / dedup_time):.2f}x")
    
    # Compare topic distributions
    dedup_topics = dedup_results["topic"].value_counts().sort_index()
    standard_topics = standard_results["topic"].value_counts().sort_index()
    
    print("\nTopic distribution comparison:")
    comparison_df = pd.DataFrame({
        "With Deduplication": dedup_topics,
        "Without Deduplication": standard_topics
    })
    print(comparison_df)
    
    # Calculate consistency
    consistency = (dedup_results["id"] == standard_results["id"]).mean() * 100
    print(f"\nID consistency between methods: {consistency:.2f}%")
    
    # Calculate topic assignment consistency
    topic_consistency = (dedup_results["topic"] == standard_results["topic"]).mean() * 100
    print(f"Topic assignment consistency: {topic_consistency:.2f}%")


def main():
    """Main function demonstrating the deduplication feature."""
    print("==== MENO DEDUPLICATION EXAMPLE ====\n")
    print("This example demonstrates how to use Meno's document deduplication feature")
    print("to optimize topic modeling when your dataset contains duplicate documents.\n")
    
    # Compare with and without deduplication
    compare_results()
    
    print("\nExplanation of results:")
    print("1. Deduplication typically provides a significant speedup")
    print("2. The final dataset size is still the same (all documents get topic assignments)")
    print("3. Topic distributions may vary slightly due to duplicate documents")
    print("   having less influence on the model with deduplication")
    print("\nTo use deduplication in your workflow, simply add deduplicate=True to load_data():")
    print("workflow = MenoWorkflow()")
    print("workflow.load_data(data=your_data, text_column='text', deduplicate=True)")


if __name__ == "__main__":
    main()
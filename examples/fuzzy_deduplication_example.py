"""
Fuzzy Deduplication Example

This example demonstrates how to use fuzzy matching for document deduplication,
which can identify and handle near-duplicate documents in topic modeling.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import adjusted_rand_index
from difflib import SequenceMatcher

from meno import MenoWorkflow
from meno.modeling.embeddings import DocumentEmbedding


def create_dataset_with_fuzzy_duplicates(num_unique=500, duplicates_per_doc=3, 
                                        noise_levels=[0.0, 0.1, 0.2], seed=42):
    """Create a synthetic dataset with near-duplicate documents at different noise levels."""
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
            "true_topic": topic,
            "duplicate_type": "original"
        })
    
    # Create near-duplicate documents with varying levels of noise
    all_docs = []
    for doc in unique_docs:
        # Add the original
        all_docs.append(doc)
        
        # Add near-duplicates with different noise levels
        for j in range(duplicates_per_doc):
            # Select a noise level
            noise_level = noise_levels[j % len(noise_levels)]
            
            # Apply noise to the text
            if noise_level > 0:
                words = doc["text"].split()
                num_changes = int(len(words) * noise_level)
                
                # Randomly select words to modify
                for _ in range(num_changes):
                    # Types of modifications: replace, delete, insert
                    modification = np.random.choice(["replace", "delete", "insert"])
                    
                    if modification == "replace" and words:
                        idx = np.random.randint(0, len(words))
                        words[idx] = f"altered_{words[idx]}"
                    elif modification == "delete" and words:
                        idx = np.random.randint(0, len(words))
                        words.pop(idx)
                    elif modification == "insert":
                        idx = np.random.randint(0, len(words) + 1)
                        words.insert(idx, f"new_word_{j}")
                
                text = " ".join(words)
            else:
                text = doc["text"]
                
            duplicate = {
                "text": text,
                "id": f"{doc['id']}_dup_{j+1}",
                "true_topic": doc["true_topic"],
                "duplicate_type": f"fuzzy_{noise_level:.1f}"
            }
            all_docs.append(duplicate)
    
    # Shuffle the documents
    np.random.shuffle(all_docs)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_docs)
    
    return df


def calculate_similarity(text1, text2):
    """Calculate text similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


def identify_fuzzy_duplicates(data, text_column, threshold=0.85):
    """Identify fuzzy duplicates in a dataset based on similarity threshold."""
    print(f"Identifying fuzzy duplicates with threshold {threshold}...")
    start_time = time.time()
    
    # Initialize results
    similarity_groups = []
    processed = set()
    
    # Process each document
    for i, row1 in data.iterrows():
        if i in processed:
            continue
            
        text1 = row1[text_column]
        group = [i]
        processed.add(i)
        
        # Compare with remaining documents
        for j, row2 in data.iloc[i+1:].iterrows():
            if j in processed:
                continue
                
            text2 = row2[text_column]
            
            # Calculate similarity
            similarity = calculate_similarity(text1, text2)
            
            # If similar enough, add to group
            if similarity >= threshold:
                group.append(j)
                processed.add(j)
        
        # Only record groups with duplicates
        if len(group) > 1:
            similarity_groups.append(group)
    
    # Convert groups to a more usable format
    fuzzy_groups = []
    for group_indices in similarity_groups:
        group_docs = data.iloc[group_indices]
        fuzzy_groups.append(group_docs)
    
    elapsed = time.time() - start_time
    print(f"Found {len(fuzzy_groups)} groups with fuzzy duplicates in {elapsed:.2f} seconds")
    
    return fuzzy_groups


def fuzzy_deduplicate(data, text_column, threshold=0.85):
    """Apply fuzzy deduplication to a dataset."""
    # Find fuzzy duplicate groups
    fuzzy_groups = identify_fuzzy_duplicates(data, text_column, threshold)
    
    # Create a mapping to track which rows to keep
    keep_indices = set(range(len(data)))
    duplicate_map = {}  # Maps removed indices to their representative
    
    # For each group, keep only the first document and map others to it
    for group in fuzzy_groups:
        indices = group.index.tolist()
        representative_idx = indices[0]
        
        for idx in indices[1:]:
            keep_indices.discard(idx)
            duplicate_map[idx] = representative_idx
    
    # Create deduplicated dataset
    deduplicated = data.iloc[list(keep_indices)].copy()
    
    # Create a reference to map back later
    data['original_index'] = range(len(data))
    
    return deduplicated, duplicate_map, fuzzy_groups


def run_with_fuzzy_deduplication():
    """Run topic modeling with fuzzy deduplication."""
    print("Creating dataset with fuzzy duplicates...")
    data = create_dataset_with_fuzzy_duplicates(
        num_unique=300, 
        duplicates_per_doc=3, 
        noise_levels=[0.0, 0.1, 0.2]
    )
    
    print(f"Dataset size: {len(data)} documents")
    
    # Count exact duplicates (exact text matches)
    exact_duplicates = data.duplicated(subset=["text"]).sum()
    print(f"Exact duplicates: {exact_duplicates} documents")
    
    # Apply fuzzy deduplication
    deduped_data, duplicate_map, fuzzy_groups = fuzzy_deduplicate(
        data, "text", threshold=0.85
    )
    
    print(f"After fuzzy deduplication: {len(deduped_data)} documents " +
          f"(removed {len(data) - len(deduped_data)} near-duplicates)")
    
    # Run topic modeling on deduplicated data
    print("\nRunning topic modeling on deduplicated data...")
    start_time = time.time()
    
    workflow = MenoWorkflow()
    workflow.load_data(
        data=deduped_data, 
        text_column="text"
    )
    workflow.preprocess_documents()
    dedup_results = workflow.discover_topics(method="bertopic", num_topics=5)
    
    dedup_time = time.time() - start_time
    print(f"Completed in {dedup_time:.2f} seconds")
    
    # Map topics back to original dataset
    print("\nMapping topics back to all documents...")
    
    # Create a mapping from index to topic
    index_to_topic = dict(zip(dedup_results['original_index'], dedup_results['topic']))
    
    # Apply to original dataset
    topics_for_all = []
    for idx in data['original_index']:
        if idx in index_to_topic:
            # This document was kept in the deduplicated set
            topics_for_all.append(index_to_topic[idx])
        else:
            # This document was removed as a duplicate - get topic from its representative
            rep_idx = duplicate_map[idx]
            topics_for_all.append(index_to_topic[rep_idx])
    
    data['assigned_topic'] = topics_for_all
    
    # Analyze results
    print("\nAnalyzing topic assignments for duplicates...")
    
    # Check if duplicates got the same topics
    consistency_scores = []
    
    for group in fuzzy_groups:
        topics = data.loc[group.index, 'assigned_topic'].values
        # Check if all topics in the group are the same
        if len(set(topics)) == 1:
            consistency_scores.append(1.0)
        else:
            # Calculate partial consistency
            main_topic = pd.Series(topics).value_counts().idxmax()
            consistency = (topics == main_topic).mean()
            consistency_scores.append(consistency)
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    print(f"Average topic consistency within duplicate groups: {avg_consistency:.2f}")
    
    # Analyze by noise level
    noise_levels = data['duplicate_type'].unique()
    noise_consistency = {}
    
    for noise_level in noise_levels:
        mask = data['duplicate_type'] == noise_level
        if mask.sum() == 0:
            continue
        
        # Group by original document
        orig_id = [id.split('_dup_')[0] if '_dup_' in id else id for id in data.loc[mask, 'id']]
        data.loc[mask, 'orig_id'] = orig_id
        
        # Check topic consistency
        topic_consistency = []
        for orig in pd.unique(data.loc[mask, 'orig_id']):
            topics = data.loc[(mask) & (data['orig_id'] == orig), 'assigned_topic'].values
            if len(topics) > 1:
                if len(set(topics)) == 1:
                    topic_consistency.append(1.0)
                else:
                    main_topic = pd.Series(topics).value_counts().idxmax()
                    consistency = (topics == main_topic).mean()
                    topic_consistency.append(consistency)
        
        noise_consistency[noise_level] = np.mean(topic_consistency) if topic_consistency else 0
    
    print("\nTopic consistency by duplicate type:")
    for noise_level, consistency in noise_consistency.items():
        print(f"  {noise_level}: {consistency:.2f}")
    
    return data, dedup_time, avg_consistency, noise_consistency


def compare_fuzzy_vs_exact():
    """Compare fuzzy deduplication with exact deduplication."""
    print("\n=== COMPARING FUZZY VS EXACT DEDUPLICATION ===\n")
    
    # Create dataset
    data = create_dataset_with_fuzzy_duplicates(
        num_unique=300, 
        duplicates_per_doc=3, 
        noise_levels=[0.0, 0.1, 0.2]
    )
    
    # Count document types
    duplicate_counts = data['duplicate_type'].value_counts()
    print("Dataset composition:")
    for dup_type, count in duplicate_counts.items():
        print(f"  {dup_type}: {count} documents")
    
    # Run exact deduplication
    print("\n--- WITH EXACT DEDUPLICATION ---")
    start_time = time.time()
    
    workflow_exact = MenoWorkflow()
    workflow_exact.load_data(data=data, text_column="text", deduplicate=True)
    workflow_exact.preprocess_documents()
    exact_results = workflow_exact.discover_topics(method="bertopic", num_topics=5)
    
    exact_time = time.time() - start_time
    
    # Count topics
    exact_topic_counts = exact_results["topic"].value_counts()
    
    # Run fuzzy deduplication
    print("\n--- WITH FUZZY DEDUPLICATION ---")
    
    # First deduplicate
    deduped_data, duplicate_map, fuzzy_groups = fuzzy_deduplicate(
        data, "text", threshold=0.85
    )
    
    start_time = time.time()
    
    workflow_fuzzy = MenoWorkflow()
    workflow_fuzzy.load_data(data=deduped_data, text_column="text")
    workflow_fuzzy.preprocess_documents()
    fuzzy_results = workflow_fuzzy.discover_topics(method="bertopic", num_topics=5)
    
    fuzzy_time = time.time() - start_time
    
    # Count topics
    fuzzy_topic_counts = fuzzy_results["topic"].value_counts()
    
    # Map topics to all documents
    index_to_topic = dict(zip(fuzzy_results['original_index'], fuzzy_results['topic']))
    
    topics_for_all = []
    for idx in data['original_index']:
        if idx in index_to_topic:
            topics_for_all.append(index_to_topic[idx])
        else:
            rep_idx = duplicate_map[idx]
            topics_for_all.append(index_to_topic[rep_idx])
    
    fuzzy_full_results = data.copy()
    fuzzy_full_results['topic_fuzzy'] = topics_for_all
    
    # Compare results
    print("\n=== RESULTS COMPARISON ===")
    print(f"Exact deduplication removed: {len(data) - len(exact_results)} documents")
    print(f"Fuzzy deduplication removed: {len(data) - len(deduped_data)} documents")
    print()
    print(f"Time with exact deduplication: {exact_time:.2f} seconds")
    print(f"Time with fuzzy deduplication: {fuzzy_time:.2f} seconds")
    
    # Compare topic distributions
    print("\nTopic distribution comparison:")
    comparison_df = pd.DataFrame({
        "Exact Deduplication": exact_topic_counts,
        "Fuzzy Deduplication": fuzzy_topic_counts
    })
    print(comparison_df)
    
    return {
        "exact_results": exact_results,
        "fuzzy_results": fuzzy_full_results,
        "exact_time": exact_time,
        "fuzzy_time": fuzzy_time,
        "fuzzy_groups": fuzzy_groups,
        "deduped_data": deduped_data,
        "original_data": data
    }


def visualize_duplicate_groups(data, fuzzy_groups, output_dir=None):
    """Visualize the fuzzy duplicate groups and their topic assignments."""
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Distribution of group sizes
    group_sizes = [len(group) for group in fuzzy_groups]
    axes[0].hist(group_sizes, bins=range(1, max(group_sizes) + 2), 
                alpha=0.7, color='royalblue', edgecolor='black')
    axes[0].set_xlabel('Number of Documents in Group')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Fuzzy Duplicate Group Sizes')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Topic consistency within groups
    consistency_by_size = {}
    for group in fuzzy_groups:
        size = len(group)
        topics = data.loc[group.index, 'assigned_topic'].values
        
        # Check consistency
        unique_topics = len(set(topics))
        consistency = 1.0 if unique_topics == 1 else 1.0 / unique_topics
        
        if size not in consistency_by_size:
            consistency_by_size[size] = []
        consistency_by_size[size].append(consistency)
    
    # Average consistency by group size
    sizes = []
    avg_consistency = []
    for size, consistencies in consistency_by_size.items():
        sizes.append(size)
        avg_consistency.append(np.mean(consistencies))
    
    axes[1].bar(sizes, avg_consistency, color='seagreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Group Size')
    axes[1].set_ylabel('Average Topic Consistency')
    axes[1].set_title('Topic Consistency by Duplicate Group Size')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fuzzy_duplicate_analysis.png'), 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main():
    """Main function demonstrating fuzzy deduplication."""
    print("==== MENO FUZZY DEDUPLICATION EXAMPLE ====\n")
    print("This example demonstrates how to use fuzzy matching for deduplication")
    print("to handle near-duplicate documents in topic modeling.\n")
    
    # Create output directory
    output_dir = Path("./fuzzy_deduplication_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run fuzzy deduplication example
    results = compare_fuzzy_vs_exact()
    
    # Map topics back to all documents for analysis
    data = results["original_data"]
    exact_topics = results["exact_results"]["topic"].values
    data['topic_exact'] = exact_topics
    
    # Visualize results
    visualize_duplicate_groups(
        results["fuzzy_results"], 
        results["fuzzy_groups"],
        output_dir
    )
    
    # Save results
    results["fuzzy_results"].to_csv(output_dir / "fuzzy_deduplication_results.csv", index=False)
    results["exact_results"].to_csv(output_dir / "exact_deduplication_results.csv", index=False)
    
    # Calculate agreement between exact and fuzzy deduplication
    topic_agreement = (data['topic_exact'] == data['topic_fuzzy']).mean()
    
    # Plot topic comparison
    plt.figure(figsize=(10, 6))
    
    # Count topics by method
    exact_counts = data['topic_exact'].value_counts().sort_index()
    fuzzy_counts = data['topic_fuzzy'].value_counts().sort_index()
    
    # Get all topics
    all_topics = sorted(set(exact_counts.index) | set(fuzzy_counts.index))
    
    # Create comparison dataframe
    comparison = pd.DataFrame(index=all_topics)
    comparison['Exact Deduplication'] = exact_counts
    comparison['Fuzzy Deduplication'] = fuzzy_counts
    comparison = comparison.fillna(0)
    
    # Plot
    comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Topic Distribution: Exact vs. Fuzzy Deduplication')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Documents')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_dir / "topic_comparison.png", dpi=300, bbox_inches='tight')
    
    print("\nResults Summary:")
    print(f"Topic agreement between methods: {topic_agreement:.2f}")
    print(f"Exact deduplication time: {results['exact_time']:.2f} seconds")
    print(f"Fuzzy deduplication time: {results['fuzzy_time']:.2f} seconds")
    print(f"\nResults saved to {output_dir.absolute()}")
    
    print("\nTo use fuzzy deduplication in your own projects:")
    print("1. Use the fuzzy_deduplicate() function to identify and remove near-duplicates")
    print("2. Run topic modeling on the deduplicated dataset")
    print("3. Map the topics back to all documents using the duplicate mapping")
    print("\nThis approach is especially useful for:")
    print("- Datasets with many similar but not identical documents")
    print("- Text with minor variations (different spacing, punctuation, etc.)")
    print("- Records with small edits or additions")


if __name__ == "__main__":
    main()
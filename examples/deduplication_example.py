"""
Deduplication Example for Meno

This example demonstrates how to use deduplication functionality in Meno,
including both exact and fuzzy deduplication methods, and how to process
the deduplicated texts with an external LLM.
"""

import pandas as pd
import numpy as np
from meno import MenoWorkflow
from meno.preprocessing.deduplication import TextDeduplicator, deduplicate_text


def create_sample_dataset(n_unique=100, n_duplicates=30, n_fuzzy=20):
    """Create a sample dataset with exact and fuzzy duplicates."""
    np.random.seed(42)
    
    # Create unique documents
    topics = ["technology", "health", "finance", "sports", "entertainment"]
    texts = []
    
    for i in range(n_unique):
        topic = topics[i % len(topics)]
        text = f"This is a document about {topic}. "
        text += f"It contains specific information about {topic} sector trends. "
        text += f"Document ID: {i}"
        texts.append(text)
    
    # Create DataFrame
    data = []
    
    # Add originals
    for i, text in enumerate(texts):
        data.append({
            "text": text,
            "id": f"doc_{i}",
            "is_duplicate": False,
            "duplicate_type": "original",
            "original_id": f"doc_{i}"
        })
    
    # Add exact duplicates
    for i in range(n_duplicates):
        idx = np.random.randint(0, len(texts))
        data.append({
            "text": texts[idx],
            "id": f"exact_dup_{i}",
            "is_duplicate": True,
            "duplicate_type": "exact",
            "original_id": f"doc_{idx}"
        })
    
    # Add fuzzy duplicates
    for i in range(n_fuzzy):
        idx = np.random.randint(0, len(texts))
        text = texts[idx]
        # Add some random noise
        words = text.split()
        # Add, modify, or remove 1-3 words
        ops = np.random.randint(1, 4)
        for _ in range(ops):
            op = np.random.choice(["add", "modify", "remove"])
            if op == "add" or len(words) < 5:
                pos = np.random.randint(0, len(words) + 1)
                words.insert(pos, f"additional_{np.random.randint(0, 100)}")
            elif op == "modify" and words:
                pos = np.random.randint(0, len(words))
                words[pos] = f"modified_{np.random.randint(0, 100)}"
            elif op == "remove" and len(words) > 5:
                pos = np.random.randint(0, len(words))
                words.pop(pos)
        
        fuzzy_text = " ".join(words)
        data.append({
            "text": fuzzy_text,
            "id": f"fuzzy_dup_{i}",
            "is_duplicate": True,
            "duplicate_type": "fuzzy",
            "original_id": f"doc_{idx}"
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def run_exact_deduplication_workflow():
    """Run a workflow with exact deduplication."""
    print("\n=== EXACT DEDUPLICATION WITH MENO WORKFLOW ===\n")
    
    # Create sample dataset
    data = create_sample_dataset()
    print(f"Original dataset size: {len(data)} documents")
    
    # Create workflow with exact deduplication
    workflow = MenoWorkflow()
    workflow.load_data(
        data=data,
        text_column="text",
        deduplicate=True  # Enable exact deduplication
    )
    
    print(f"After exact deduplication: {len(workflow.documents)} documents")
    print(f"Removed {len(data) - len(workflow.documents)} exact duplicates")
    
    # Continue normal workflow
    workflow.preprocess_documents()
    results = workflow.discover_topics(method="bertopic", num_topics=5)
    
    # Results already include topics mapped back to all documents
    print(f"\nAssigned {len(pd.unique(results['topic']))} topics to all {len(results)} documents (including duplicates)")
    
    # Count documents per topic
    topic_counts = results['topic'].value_counts().to_dict()
    print("\nDocuments per topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  Topic {topic}: {count} documents")
    
    return results


def run_fuzzy_deduplication():
    """Run a workflow with fuzzy deduplication."""
    print("\n=== FUZZY DEDUPLICATION WITH TEXT DEDUPLICATOR ===\n")
    
    # Create sample dataset
    data = create_sample_dataset()
    print(f"Original dataset size: {len(data)} documents")
    
    # Create deduplicator
    deduplicator = TextDeduplicator(similarity_threshold=0.8)
    
    # Run fuzzy deduplication
    deduplicated_data, duplicate_map, fuzzy_groups = deduplicator.deduplicate(
        data, 
        text_column="text",
        method="fuzzy",
        threshold=0.85
    )
    
    print(f"After fuzzy deduplication: {len(deduplicated_data)} documents")
    print(f"Removed {len(data) - len(deduplicated_data)} duplicates and near-duplicates")
    print(f"Found {len(fuzzy_groups)} groups of similar documents")
    
    # Display a sample fuzzy group
    if fuzzy_groups:
        print("\nSample fuzzy duplicate group:")
        sample_group = fuzzy_groups[0]
        for i, (idx, row) in enumerate(sample_group.iterrows()):
            print(f"  Doc {i+1}: {row['text'][:50]}...")
    
    # Run topic modeling on deduplicated data
    workflow = MenoWorkflow()
    workflow.load_data(
        data=deduplicated_data,
        text_column="text"
    )
    workflow.preprocess_documents()
    results = workflow.discover_topics(method="bertopic", num_topics=5)
    
    # Map topics back to all documents
    index_to_topic = dict(zip(results['original_index'], results['topic']))
    
    # Apply to original dataset
    topics_for_all = []
    for idx in data.index:
        if idx in index_to_topic:
            # This document was kept in the deduplicated set
            topics_for_all.append(index_to_topic[idx])
        else:
            # This document was removed as a duplicate - get topic from its representative
            rep_idx = duplicate_map[idx]
            topics_for_all.append(index_to_topic[rep_idx])
    
    data['topic'] = topics_for_all
    
    # Count documents per topic
    topic_counts = data['topic'].value_counts().to_dict()
    print("\nDocuments per topic after mapping back:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  Topic {topic}: {count} documents")
    
    return data


def simulate_llm_processing(texts, prompt="Assign a category to this text: "):
    """Simulate processing texts with an LLM."""
    # This function simulates what you would do with an actual LLM API
    # In a real scenario, you would send these texts to an API like OpenAI, Anthropic, etc.
    
    categories = ["Technology", "Healthcare", "Finance", "Sports", "Entertainment"]
    results = []
    
    for text in texts:
        # Simulate LLM category assignment
        word_counts = {}
        for category in categories:
            word_counts[category] = text.lower().count(category.lower())
        
        # Pick the category with the most mentions, or random if none
        max_count = max(word_counts.values())
        if max_count > 0:
            category = [c for c, count in word_counts.items() if count == max_count][0]
        else:
            category = np.random.choice(categories)
        
        # Create a summary (in reality, this would come from the LLM)
        summary = f"This text is about {category.lower()}."
        
        results.append({
            "category": category,
            "summary": summary,
            "confidence": np.random.uniform(0.7, 0.95)
        })
    
    return pd.DataFrame(results)


def run_deduplication_for_llm():
    """Run deduplication and process with simulated LLM."""
    print("\n=== DEDUPLICATION FOR EXTERNAL LLM PROCESSING ===\n")
    
    # Create sample dataset
    data = create_sample_dataset()
    print(f"Original dataset size: {len(data)} documents")
    
    # Create deduplicator
    deduplicator = TextDeduplicator(similarity_threshold=0.85)
    
    # Run deduplication
    deduplicated_data, duplicate_map, _ = deduplicator.deduplicate(
        data, 
        text_column="text",
        method="fuzzy",  # Could also use "exact"
        threshold=0.85
    )
    
    print(f"After deduplication: {len(deduplicated_data)} documents")
    print(f"Removed {len(data) - len(deduplicated_data)} duplicates and near-duplicates")
    
    # Simulate LLM processing on deduplicated data
    print(f"\nProcessing {len(deduplicated_data)} unique documents with simulated LLM...")
    llm_results = simulate_llm_processing(deduplicated_data['text'])
    
    # Add results to deduplicated data
    for col in llm_results.columns:
        deduplicated_data[col] = llm_results[col].values
    
    # Map LLM results back to all documents
    print("\nMapping LLM results back to all documents...")
    full_results = deduplicator.map_results_to_full_dataset(
        data,
        deduplicated_data,
        duplicate_map,
        ['category', 'summary', 'confidence']
    )
    
    # Verify results
    print(f"\nAll {len(full_results)} documents have LLM results:")
    print(f"  - Categories assigned: {sorted(full_results['category'].unique())}")
    print(f"  - Avg. confidence: {full_results['confidence'].mean():.2f}")
    
    # Show some stats
    category_counts = full_results['category'].value_counts()
    print("\nCategory distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} documents ({count/len(full_results)*100:.1f}%)")
    
    return full_results


def main():
    """Run all deduplication examples."""
    print("=== MENO DEDUPLICATION EXAMPLES ===")
    print("This script demonstrates how to use deduplication in Meno for:")
    print("1. Exact deduplication with MenoWorkflow")
    print("2. Fuzzy deduplication with TextDeduplicator")
    print("3. Processing deduplicated texts with an external LLM")
    
    # Run examples
    run_exact_deduplication_workflow()
    run_fuzzy_deduplication()
    run_deduplication_for_llm()
    
    print("\n=== COMPLETED DEDUPLICATION EXAMPLES ===")
    print("""
Key takeaways:
1. Use exact deduplication (deduplicate=True in workflow.load_data()) for identical documents
2. Use fuzzy deduplication from meno.preprocessing.deduplication for similar but not identical texts
3. Deduplication can significantly reduce processing time for LLMs
4. Always map results back to all documents to maintain the original dataset structure
    """)


if __name__ == "__main__":
    main()
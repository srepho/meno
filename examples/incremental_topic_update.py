"""
Incremental Topic Modeling Example

This example demonstrates incremental learning capabilities for topic modeling,
allowing a model to be updated with new documents without requiring full retraining.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time

from meno import MenoWorkflow
from meno.modeling.incremental import TopicUpdater
from meno.modeling.bertopic_model import BERTopicModel


def create_synthetic_dataset(num_documents=500, num_topics=5, seed=42):
    """Create a synthetic dataset with distinct topics."""
    np.random.seed(seed)
    
    # Define topics with characteristic words
    topics = {
        "Technology": ["computer", "software", "hardware", "system", "data", "technology", 
                      "digital", "network", "internet", "device"],
        "Health": ["medical", "health", "disease", "patient", "treatment", "doctor", 
                  "hospital", "medicine", "diagnosis", "therapy"],
        "Finance": ["money", "financial", "bank", "investment", "market", "stock", 
                   "economy", "fund", "investor", "profit"],
        "Education": ["school", "student", "learning", "teacher", "education", "university", 
                     "academic", "class", "college", "knowledge"],
        "Sports": ["team", "player", "game", "sports", "competition", "coach", 
                  "athlete", "league", "tournament", "championship"]
    }
    
    # Create documents
    documents = []
    topic_labels = []
    
    topic_names = list(topics.keys())
    for i in range(num_documents):
        # Choose a topic
        topic_idx = i % len(topic_names)  # Even distribution across topics
        topic = topic_names[topic_idx]
        topic_labels.append(topic)
        
        # Create a document with topic-specific words
        topic_words = topics[topic]
        
        # Base sentence with topic words
        words = np.random.choice(topic_words, size=np.random.randint(3, 7), replace=False)
        doc = " ".join(words)
        
        # Add some more sentences
        for _ in range(np.random.randint(1, 4)):
            words = np.random.choice(topic_words, size=np.random.randint(4, 8), replace=True)
            doc += ". " + " ".join(words)
            
        # Add some generic words
        generic_words = ["the", "and", "of", "in", "to", "a", "with", "for", "on", "is"]
        for _ in range(np.random.randint(5, 10)):
            position = np.random.randint(0, len(doc.split()))
            doc_parts = doc.split()
            doc_parts.insert(position, np.random.choice(generic_words))
            doc = " ".join(doc_parts)
            
        documents.append(doc)
    
    # Create a DataFrame
    df = pd.DataFrame({
        "text": documents,
        "topic": topic_labels,
        "id": [f"doc_{i}" for i in range(num_documents)]
    })
    
    return df


def plot_update_history(update_history, output_dir=None):
    """Plot metrics from update history."""
    if not update_history.updates:
        print("No updates to plot")
        return
    
    # Extract data
    timestamps = [u["timestamp"].split("T")[0] for u in update_history.updates]
    stability = [u["topic_stability"] for u in update_history.updates]
    execution_times = [u["execution_time"] for u in update_history.updates]
    doc_counts = [u["num_documents"] for u in update_history.updates]
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot topic stability
    axes[0].plot(range(len(stability)), stability, marker='o', linestyle='-', color='blue')
    axes[0].set_ylabel('Topic Stability (0-1)')
    axes[0].set_title('Topic Stability Across Updates')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot execution time
    axes[1].plot(range(len(execution_times)), execution_times, marker='s', linestyle='-', color='green')
    axes[1].set_ylabel('Execution Time (s)')
    axes[1].set_title('Update Execution Time')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot document counts
    axes[2].bar(range(len(doc_counts)), doc_counts, color='orange')
    axes[2].set_ylabel('Documents Added')
    axes[2].set_title('Number of Documents Per Update')
    axes[2].set_xlabel('Update Number')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Add x-tick labels
    plt.xticks(range(len(timestamps)), [f"Update {i+1}" for i in range(len(timestamps))])
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "update_history.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main():
    # Create output directory
    output_dir = Path("./incremental_output")
    output_dir.mkdir(exist_ok=True)
    
    print("Step 1: Creating initial dataset...")
    df_initial = create_synthetic_dataset(num_documents=300, num_topics=5)
    
    print(f"Initial dataset: {len(df_initial)} documents")
    
    # Save a reference copy
    df_initial.to_csv(output_dir / "initial_dataset.csv", index=False)
    
    print("\nStep 2: Training initial topic model...")
    workflow = MenoWorkflow(
        config_overrides={
            "modeling": {
                "embeddings": {
                    "model_name": "all-MiniLM-L6-v2",
                    "device": "cpu"
                },
                "clustering": {
                    "algorithm": "hdbscan",
                    "min_cluster_size": 5,
                    "min_samples": 5
                }
            }
        }
    )
    
    # Set up model
    start_time = time.time()
    workflow.load_data(data=df_initial, text_column="text", id_column="id")
    workflow.preprocess_documents()
    workflow.discover_topics(method="bertopic", num_topics="auto")
    initial_training_time = time.time() - start_time
    
    print(f"Initial model trained in {initial_training_time:.2f} seconds")
    
    # Get the underlying BERTopic model
    model = workflow.modeler.model
    
    # Check if the model is suitable for incremental learning
    if not isinstance(model, BERTopicModel):
        raise ValueError("Incremental updates require a BERTopic model")
    
    # Create topic updater
    topic_updater = TopicUpdater(model=model)
    
    # Get initial topics
    initial_topics = model.get_topic_info()
    print(f"\nInitial topics discovered: {len(initial_topics[initial_topics.Topic != -1])}")
    
    # Get document-topic assignments and save
    doc_topics = workflow.get_topic_assignments()
    doc_topics.to_csv(output_dir / "initial_document_topics.csv", index=False)
    
    # Create new batches for incremental updates
    print("\nStep 3: Creating new data batches for incremental updates...")
    df_update1 = create_synthetic_dataset(num_documents=100, num_topics=5, seed=43)
    df_update2 = create_synthetic_dataset(num_documents=150, num_topics=5, seed=44)
    df_update3 = create_synthetic_dataset(num_documents=200, num_topics=5, seed=45)
    
    # Save update datasets
    df_update1.to_csv(output_dir / "update1_dataset.csv", index=False)
    df_update2.to_csv(output_dir / "update2_dataset.csv", index=False)
    df_update3.to_csv(output_dir / "update3_dataset.csv", index=False)
    
    # Perform incremental update with first batch
    print("\nStep 4: Performing first incremental update (pure incremental)...")
    update1_docs = df_update1["text"].tolist()
    update1_ids = df_update1["id"].tolist()
    
    # Get the embeddings for the new documents
    embeddings1 = model.embedding_model.embed_documents(update1_docs)
    
    # Update model incrementally
    model, stats1 = topic_updater.update_model(
        documents=update1_docs,
        document_ids=update1_ids,
        embeddings=embeddings1,
        update_mode="incremental",
        verbose=True
    )
    
    # Get updated document-topic assignments
    doc_topics_after_update1 = workflow.get_topic_assignments()
    doc_topics_after_update1.to_csv(output_dir / "after_update1_document_topics.csv", index=False)
    
    topics_after_update1 = model.get_topic_info()
    print(f"Topics after update 1: {len(topics_after_update1[topics_after_update1.Topic != -1])}")
    print(f"Topic stability: {stats1['topic_stability']:.2f}")
    
    # Perform second update with sample of original documents
    print("\nStep 5: Performing second update (partial retrain)...")
    update2_docs = df_update2["text"].tolist()
    update2_ids = df_update2["id"].tolist()
    
    # Get embeddings for second batch
    embeddings2 = model.embedding_model.embed_documents(update2_docs)
    
    # Update with partial retraining
    model, stats2 = topic_updater.update_model(
        documents=update2_docs,
        document_ids=update2_ids,
        embeddings=embeddings2,
        update_mode="partial_retrain",
        sample_original_docs=True,
        sample_ratio=0.2,  # Use 20% of original docs for stability
        verbose=True
    )
    
    # Get updated document-topic assignments
    doc_topics_after_update2 = workflow.get_topic_assignments()
    doc_topics_after_update2.to_csv(output_dir / "after_update2_document_topics.csv", index=False)
    
    topics_after_update2 = model.get_topic_info()
    print(f"Topics after update 2: {len(topics_after_update2[topics_after_update2.Topic != -1])}")
    print(f"Topic stability: {stats2['topic_stability']:.2f}")
    
    # Third update - full retrain
    print("\nStep 6: Performing third update (full retrain)...")
    update3_docs = df_update3["text"].tolist()
    update3_ids = df_update3["id"].tolist()
    
    # Get embeddings for third batch
    embeddings3 = model.embedding_model.embed_documents(update3_docs)
    
    # Update with full retraining
    model, stats3 = topic_updater.update_model(
        documents=update3_docs,
        document_ids=update3_ids,
        embeddings=embeddings3,
        update_mode="full_retrain",
        merge_similar_topics=True,
        similarity_threshold=0.7,
        verbose=True
    )
    
    # Get final document-topic assignments
    doc_topics_final = workflow.get_topic_assignments()
    doc_topics_final.to_csv(output_dir / "final_document_topics.csv", index=False)
    
    topics_final = model.get_topic_info()
    print(f"Topics after update 3: {len(topics_final[topics_final.Topic != -1])}")
    print(f"Topic stability: {stats3['topic_stability']:.2f}")
    
    # Save the update history visualization
    print("\nStep 7: Generating update history visualization...")
    plot_update_history(topic_updater.update_history, output_dir=output_dir)
    
    # Save the updater state
    topic_updater.save(output_dir / "topic_updater.pkl")
    
    # Save the model
    model.save(output_dir / "final_model")
    
    # Print final summary
    print("\nIncremental Update Summary:")
    print("--------------------------")
    print(f"Initial training time: {initial_training_time:.2f} seconds")
    print(f"Update 1 time (incremental): {stats1['execution_time']:.2f} seconds")
    print(f"Update 2 time (partial retrain): {stats2['execution_time']:.2f} seconds")
    print(f"Update 3 time (full retrain): {stats3['execution_time']:.2f} seconds")
    print()
    print(f"Total documents processed: {topic_updater.update_history.total_documents_processed}")
    print(f"Final number of topics: {len(topics_final[topics_final.Topic != -1])}")
    print()
    print(f"All outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
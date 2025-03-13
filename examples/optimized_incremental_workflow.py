"""
Optimized Incremental Workflow Example

This example demonstrates how to combine memory optimization, document deduplication,
and incremental learning in a single integrated workflow for maximum efficiency.
It also includes visualization of topic drift over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from sklearn.metrics.pairwise import cosine_similarity

from meno import MenoWorkflow
from meno.modeling.incremental import TopicUpdater
from meno.modeling.bertopic_model import BERTopicModel
from meno.visualization.enhanced_viz.interactive_viz import create_topic_similarity_heatmap


def create_time_series_dataset(num_unique_initial=500, num_unique_updates=200, 
                              num_updates=3, duplicate_ratio=0.3, noise_level=0.1, seed=42):
    """Create a synthetic time series dataset with evolving topics and duplicates."""
    np.random.seed(seed)
    
    # Define topics with characteristic words
    base_topics = {
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
    
    # Words to add in each update to simulate topic drift
    topic_evolution = {
        "Technology": [
            ["cloud", "ai", "algorithm", "platform", "solution"],
            ["machine", "learning", "neural", "network", "compute"],
            ["blockchain", "crypto", "virtual", "reality", "metaverse"]
        ],
        "Health": [
            ["vaccine", "covid", "pandemic", "immunity", "virus"],
            ["mental", "wellness", "anxiety", "stress", "telemedicine"],
            ["genomics", "personalized", "wearable", "fitness", "tracking"]
        ],
        "Finance": [
            ["crypto", "bitcoin", "blockchain", "token", "wallet"],
            ["fintech", "payment", "digital", "banking", "transaction"],
            ["inflation", "recession", "interest", "rates", "federal"]
        ],
        "Education": [
            ["online", "remote", "virtual", "zoom", "digital"],
            ["equity", "accessibility", "inclusion", "diversity", "resources"],
            ["skills", "workforce", "career", "development", "preparation"]
        ],
        "Sports": [
            ["protocols", "bubble", "testing", "safety", "regulations"],
            ["analytics", "metrics", "performance", "data", "tracking"],
            ["streaming", "betting", "fantasy", "engagement", "monetization"]
        ]
    }
    
    # Function to create documents for a specific time period
    def create_period_docs(period, num_unique, base_topics, additional_words=None):
        docs = []
        topic_names = list(base_topics.keys())
        
        # Create unique documents
        for i in range(num_unique):
            # Choose a topic
            topic_idx = i % len(topic_names)
            topic = topic_names[topic_idx]
            
            # Get base topic words
            topic_words = base_topics[topic].copy()
            
            # Add evolution words if available
            if additional_words and topic in additional_words:
                topic_words.extend(additional_words[topic])
            
            # Create document with more structured content
            doc_words = np.random.choice(topic_words, size=np.random.randint(10, 20), replace=True)
            doc = " ".join(doc_words)
            
            # Add some generic words
            generic_words = ["the", "and", "of", "in", "to", "a", "with", "for", "on", "is"]
            for _ in range(np.random.randint(5, 10)):
                position = np.random.randint(0, len(doc.split()))
                doc_parts = doc.split()
                doc_parts.insert(position, np.random.choice(generic_words))
                doc = " ".join(doc_parts)
            
            # Create metadata
            doc_data = {
                "text": doc,
                "id": f"doc_{period}_{i}",
                "topic": topic,
                "period": period,
                "date": (datetime.now() + timedelta(days=period*30)).strftime("%Y-%m-%d"),
                "duplicate_type": "original"
            }
            
            docs.append(doc_data)
        
        return docs
    
    # Generate initial data
    all_docs = []
    
    # Create initial documents
    initial_docs = create_period_docs(0, num_unique_initial, base_topics)
    all_docs.extend(initial_docs)
    
    # Create update batches with evolving topics
    for update_idx in range(1, num_updates + 1):
        # Add new words to each topic to simulate drift
        update_words = {}
        for topic, evolution_list in topic_evolution.items():
            if update_idx <= len(evolution_list):
                update_words[topic] = evolution_list[update_idx-1]
        
        # Create documents for this period
        update_docs = create_period_docs(
            update_idx, 
            num_unique_updates, 
            base_topics, 
            update_words
        )
        
        # Add documents
        all_docs.extend(update_docs)
    
    # Add duplicates based on duplicate_ratio
    docs_with_duplicates = all_docs.copy()
    
    # Number of duplicates to add
    num_duplicates = int(len(all_docs) * duplicate_ratio)
    
    for i in range(num_duplicates):
        # Pick a random document to duplicate
        original_idx = np.random.randint(0, len(all_docs))
        original_doc = all_docs[original_idx].copy()
        
        # Decide duplicate type (exact or with noise)
        duplicate_type = np.random.choice(["exact", "noisy"], p=[0.4, 0.6])
        
        if duplicate_type == "exact":
            # Exact duplicate
            duplicate = original_doc.copy()
            duplicate["id"] = f"{original_doc['id']}_dup_exact_{i}"
            duplicate["duplicate_type"] = "exact_duplicate"
        else:
            # Add noise to create a near-duplicate
            words = original_doc["text"].split()
            
            # Number of words to modify
            num_changes = max(1, int(len(words) * noise_level))
            
            # Modify random words
            for _ in range(num_changes):
                change_type = np.random.choice(["replace", "delete", "insert"])
                
                if change_type == "replace" and words:
                    idx = np.random.randint(0, len(words))
                    words[idx] = f"altered_{words[idx]}"
                elif change_type == "delete" and len(words) > 5:
                    idx = np.random.randint(0, len(words))
                    words.pop(idx)
                elif change_type == "insert":
                    idx = np.random.randint(0, len(words) + 1)
                    words.insert(idx, f"added_word_{i}")
            
            duplicate = original_doc.copy()
            duplicate["text"] = " ".join(words)
            duplicate["id"] = f"{original_doc['id']}_dup_noisy_{i}"
            duplicate["duplicate_type"] = "near_duplicate"
        
        docs_with_duplicates.append(duplicate)
    
    # Shuffle the documents
    np.random.shuffle(docs_with_duplicates)
    
    # Convert to DataFrame
    df = pd.DataFrame(docs_with_duplicates)
    
    return df


def visualize_topic_drift(topic_updater, model, output_dir=None):
    """Visualize topic drift and evolution over time."""
    update_history = topic_updater.update_history
    
    if not update_history.updates:
        print("No update history to visualize")
        return
    
    # 1. Plot topic stability over time
    plt.figure(figsize=(10, 6))
    
    # Extract data
    timestamps = [i for i in range(1, len(update_history.updates) + 1)]
    stability = [u["topic_stability"] for u in update_history.updates]
    
    plt.plot(timestamps, stability, marker='o', linestyle='-', 
             color='royalblue', linewidth=2, markersize=8)
    plt.xlabel('Update Number')
    plt.ylabel('Topic Stability (0-1)')
    plt.title('Topic Stability Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(timestamps)
    plt.ylim(0, 1.05)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'topic_stability_over_time.png'), 
                   dpi=300, bbox_inches='tight')
    
    # 2. Create topic similarity heatmap if the model supports it
    if hasattr(model, 'topic_representations_'):
        topics = model.get_topic_info()
        relevant_topics = topics[topics.Topic != -1].Topic.tolist()
        
        if relevant_topics:
            # Use Meno's built-in similarity visualization
            fig = create_topic_similarity_heatmap(
                model, 
                min_topic=-1,  # Include noise topic
                return_figure=True
            )
            
            if output_dir:
                fig.write_html(os.path.join(output_dir, 'topic_similarity_heatmap.html'))
    
    # 3. Plot topic size evolution if available
    if hasattr(model, 'topic_sizes_'):
        plt.figure(figsize=(12, 6))
        
        # Get top N topics by size (exclude -1 noise topic)
        topic_sizes = {k: v for k, v in model.topic_sizes_.items() if k != -1}
        top_topics = sorted(topic_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_topic_ids = [t[0] for t in top_topics]
        
        # Plot sizes
        sizes = [model.topic_sizes_[t] for t in top_topic_ids]
        plt.bar(range(len(top_topic_ids)), sizes, alpha=0.7)
        plt.xticks(range(len(top_topic_ids)), [str(t) for t in top_topic_ids])
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.title('Top 10 Topics by Size')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'topic_sizes.png'), dpi=300, bbox_inches='tight')
    
    # 4. Plot topic word evolution
    if hasattr(model, 'topic_representations_'):
        # Get top 5 topics by size (exclude -1 noise topic)
        if hasattr(model, 'topic_sizes_'):
            topic_sizes = {k: v for k, v in model.topic_sizes_.items() if k != -1}
            top_topics = sorted(topic_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            top_topic_ids = [t[0] for t in top_topics]
        else:
            # If sizes not available, use first 5 non-noise topics
            top_topic_ids = [t for t in model.topic_representations_.keys() if t != -1][:5]
        
        # Plot top words for each topic
        plt.figure(figsize=(15, 10))
        
        for i, topic_id in enumerate(top_topic_ids):
            if topic_id in model.topic_representations_:
                words = model.topic_representations_[topic_id]
                if words:
                    plt.subplot(len(top_topic_ids), 1, i+1)
                    word_scores = [1.0] * len(words)  # Placeholder if scores not available
                    plt.barh(range(len(words)), word_scores, alpha=0.6, color=f'C{i}')
                    plt.yticks(range(len(words)), words)
                    plt.title(f'Topic {topic_id}')
                    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'topic_word_distribution.png'), 
                       dpi=300, bbox_inches='tight')


def run_optimized_incremental_workflow():
    """Run a workflow that combines memory optimization, deduplication, and incremental learning."""
    output_dir = Path("./optimized_incremental_output")
    output_dir.mkdir(exist_ok=True)
    
    print("==== MENO OPTIMIZED INCREMENTAL WORKFLOW EXAMPLE ====\n")
    print("This example demonstrates how to combine:")
    print("1. Memory optimization (8-bit quantization)")
    print("2. Document deduplication")
    print("3. Incremental learning")
    print("4. Topic drift visualization\n")
    
    # 1. Create synthetic time series dataset
    print("Creating synthetic time series dataset...")
    full_dataset = create_time_series_dataset(
        num_unique_initial=300,
        num_unique_updates=150,
        num_updates=3,
        duplicate_ratio=0.4
    )
    
    # Split into initial data and update batches
    initial_data = full_dataset[full_dataset['period'] == 0]
    updates = []
    for period in range(1, 4):
        updates.append(full_dataset[full_dataset['period'] == period])
    
    # Save dataset samples for reference
    full_dataset.to_csv(output_dir / "full_dataset.csv", index=False)
    initial_data.to_csv(output_dir / "initial_data.csv", index=False)
    
    # Print dataset stats
    print(f"Total dataset size: {len(full_dataset)} documents")
    print(f"Initial batch: {len(initial_data)} documents")
    for i, update in enumerate(updates):
        print(f"Update {i+1}: {len(update)} documents")
    
    # 2. Create optimized workflow
    print("\nInitializing optimized workflow...")
    workflow = MenoWorkflow(
        config_overrides={
            "modeling": {
                "embeddings": {
                    "model_name": "all-MiniLM-L6-v2",
                    "precision": "int8",   # 8-bit quantization
                    "use_mmap": True,      # Memory mapping
                    "quantize": True       # Quantize model weights
                },
                "clustering": {
                    "algorithm": "hdbscan",
                    "min_cluster_size": 10,
                    "min_samples": 5
                }
            }
        }
    )
    
    # 3. Process initial data with deduplication
    print("\nProcessing initial data with deduplication...")
    start_time = time.time()
    
    # Enable deduplication
    workflow.load_data(
        data=initial_data, 
        text_column="text",
        deduplicate=True  # Enable built-in deduplication
    )
    
    # Track removed duplicates
    initial_duplicates = len(initial_data) - len(workflow.documents)
    print(f"Removed {initial_duplicates} duplicates from initial data")
    
    # Continue processing
    workflow.preprocess_documents()
    initial_results = workflow.discover_topics(method="bertopic", num_topics="auto")
    
    initial_time = time.time() - start_time
    print(f"Initial model trained in {initial_time:.2f} seconds")
    
    # Get model and create topic updater
    model = workflow.modeler.model
    topic_updater = TopicUpdater(model=model)
    
    # Get initial topics
    initial_topics = model.get_topic_info()
    print(f"Initial topics discovered: {len(initial_topics[initial_topics.Topic != -1])}")
    
    # Save initial results
    initial_results.to_csv(output_dir / "initial_results.csv", index=False)
    
    # 4. Process incremental updates
    all_update_stats = []
    
    for i, update_batch in enumerate(updates):
        print(f"\nProcessing update batch {i+1}...")
        
        # First deduplicate the update batch
        update_batch['original_index'] = range(len(update_batch))
        duplicated = update_batch.duplicated(subset=["text"]).sum()
        update_batch_deduped = update_batch.drop_duplicates(subset=["text"]).copy()
        
        print(f"Removed {len(update_batch) - len(update_batch_deduped)} exact duplicates from update batch")
        
        update_texts = update_batch_deduped["text"].tolist()
        update_ids = update_batch_deduped["id"].tolist()
        
        # Choose update mode based on batch
        if i == 0:
            # First update: pure incremental
            update_mode = "incremental"
        elif i == 1:
            # Second update: partial retrain
            update_mode = "partial_retrain"
        else:
            # Last update: full retrain
            update_mode = "full_retrain"
        
        print(f"Using {update_mode} update mode...")
        
        # Get embeddings for the update batch
        start_time = time.time()
        embeddings = model.embedding_model.embed_documents(update_texts)
        
        # Update model
        model, stats = topic_updater.update_model(
            documents=update_texts,
            document_ids=update_ids,
            embeddings=embeddings,
            update_mode=update_mode,
            preserve_topic_ids=True,
            merge_similar_topics=(i == len(updates) - 1),  # Merge similar topics on final update
            verbose=True
        )
        
        # Add batch number to stats
        stats["batch"] = i + 1
        stats["update_mode"] = update_mode
        all_update_stats.append(stats)
        
        update_time = time.time() - start_time
        print(f"Update completed in {update_time:.2f} seconds")
        
        # Get updated topics
        updated_topics = model.get_topic_info()
        print(f"Topics after update {i+1}: {len(updated_topics[updated_topics.Topic != -1])}")
        print(f"Topic stability: {stats['topic_stability']:.2f}")
    
    # 5. Visualize topic drift
    print("\nGenerating topic drift visualizations...")
    visualize_topic_drift(topic_updater, model, output_dir)
    
    # 6. Save final model and results
    model.save(output_dir / "final_model")
    
    # Create update summary
    update_summary = pd.DataFrame(all_update_stats)
    update_summary.to_csv(output_dir / "update_stats.csv", index=False)
    
    # Plot update statistics
    plt.figure(figsize=(12, 8))
    
    # Plot execution times
    plt.subplot(2, 2, 1)
    plt.bar(update_summary['batch'], update_summary['execution_time'], color='royalblue')
    plt.xlabel('Update Batch')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Update Execution Time by Batch')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot topic stability
    plt.subplot(2, 2, 2)
    plt.bar(update_summary['batch'], update_summary['topic_stability'], color='seagreen')
    plt.xlabel('Update Batch')
    plt.ylabel('Topic Stability (0-1)')
    plt.title('Topic Stability by Batch')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot update modes
    plt.subplot(2, 2, 3)
    update_modes = update_summary['update_mode'].tolist()
    colors = ['skyblue' if mode == 'incremental' else 
              'lightgreen' if mode == 'partial_retrain' else 
              'salmon' for mode in update_modes]
    plt.bar(update_summary['batch'], [1] * len(update_summary), color=colors)
    plt.xlabel('Update Batch')
    plt.yticks([])
    plt.title('Update Mode by Batch')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Incremental'),
        Patch(facecolor='lightgreen', label='Partial Retrain'),
        Patch(facecolor='salmon', label='Full Retrain')
    ]
    plt.legend(handles=legend_elements, loc='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "update_performance.png", dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\nWorkflow complete!")
    print(f"All results saved to: {output_dir.absolute()}")
    print("\nOptimization Performance:")
    print(f"- Memory: Used 8-bit quantization and memory mapping")
    print(f"- Deduplication: Removed {initial_duplicates} duplicates from initial data")
    print(f"- Incremental learning: Processed {len(updates)} update batches using different strategies")
    print("\nFinal model statistics:")
    topics = model.get_topic_info()
    print(f"- Total topics: {len(topics[topics.Topic != -1])}")
    print(f"- Documents processed: {topic_updater.update_history.total_documents_processed}")


if __name__ == "__main__":
    run_optimized_incremental_workflow()
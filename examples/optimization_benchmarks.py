"""
Optimization Benchmarks

This script provides benchmarking utilities to evaluate the performance improvements from
memory optimization, document deduplication, and incremental learning.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import psutil
from pathlib import Path
import gc
import argparse

from meno import MenoWorkflow
from meno.modeling.incremental import TopicUpdater
from meno.modeling.embeddings import DocumentEmbedding


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def create_benchmark_dataset(num_documents=5000, num_duplicates=2000, 
                            words_per_doc=50, seed=42):
    """Create synthetic dataset for benchmarking."""
    np.random.seed(seed)
    
    # Create vocabulary
    topics = ["technology", "health", "finance", "sports", "entertainment"]
    topic_words = {
        "technology": ["computer", "software", "hardware", "system", "data", "network"],
        "health": ["medical", "health", "disease", "treatment", "doctor", "patient"],
        "finance": ["money", "financial", "bank", "investment", "market", "economy"],
        "sports": ["team", "player", "game", "sports", "coach", "athlete"],
        "entertainment": ["movie", "music", "actor", "film", "show", "celebrity"]
    }
    
    # Generate original documents
    documents = []
    
    for i in range(num_documents):
        # Assign a topic
        topic = topics[i % len(topics)]
        
        # Generate text using topic words
        words = []
        for _ in range(words_per_doc):
            if np.random.random() < 0.6:  # 60% topic-specific words
                words.append(np.random.choice(topic_words[topic]))
            else:  # 40% common words
                words.append(np.random.choice(["the", "and", "of", "in", "to", "a", "with", "for"]))
        
        doc_text = " ".join(words)
        documents.append({
            "text": doc_text,
            "id": f"doc_{i}",
            "topic": topic,
            "is_duplicate": False
        })
    
    # Add duplicates
    for i in range(num_duplicates):
        # Select a random document to duplicate
        orig_idx = np.random.randint(0, num_documents)
        duplicate = documents[orig_idx].copy()
        duplicate["id"] = f"dup_{i}"
        duplicate["is_duplicate"] = True
        documents.append(duplicate)
    
    # Shuffle documents
    np.random.shuffle(documents)
    
    # Convert to DataFrame
    df = pd.DataFrame(documents)
    
    return df


def benchmark_memory_optimizations(dataset, batch_size=1000):
    """Benchmark different memory optimization configurations."""
    print("\n=== MEMORY OPTIMIZATION BENCHMARK ===")
    
    # Define configurations to test
    configs = [
        {
            "name": "Baseline (float32)",
            "precision": "float32",
            "use_mmap": False,
            "quantize": False
        },
        {
            "name": "Half Precision (float16)",
            "precision": "float16", 
            "use_mmap": False,
            "quantize": False
        },
        {
            "name": "Memory Mapped (float32)",
            "precision": "float32",
            "use_mmap": True,
            "quantize": False
        },
        {
            "name": "Combined Optimizations",
            "precision": "float16",
            "use_mmap": True,
            "quantize": False
        },
        {
            "name": "8-bit Quantization",
            "precision": "int8",
            "use_mmap": True,
            "quantize": True
        }
    ]
    
    results = []
    
    # Run each configuration
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Clear previous data and run GC
        gc.collect()
        
        # Record starting memory
        start_memory = get_memory_usage()
        
        # Start timing
        start_time = time.time()
        
        # Initialize workflow with this configuration
        workflow = MenoWorkflow(
            config_overrides={
                "modeling": {
                    "embeddings": {
                        "model_name": "all-MiniLM-L6-v2",
                        "precision": config["precision"],
                        "use_mmap": config["use_mmap"],
                        "quantize": config["quantize"],
                        "batch_size": batch_size
                    }
                }
            }
        )
        
        # Run workflow
        workflow.load_data(data=dataset, text_column="text")
        workflow.preprocess_documents()
        
        # Skip topic discovery to focus on embeddings
        embeddings = workflow.modeler.embedding_model.embed_documents(
            dataset["text"].tolist()
        )
        
        # Record metrics
        end_time = time.time()
        peak_memory = get_memory_usage()
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_usage = peak_memory - start_memory
        embedding_size = embeddings.nbytes / (1024 * 1024)  # MB
        
        print(f"  Time: {processing_time:.2f}s")
        print(f"  Memory: {memory_usage:.2f} MB")
        print(f"  Embedding size: {embedding_size:.2f} MB")
        
        # Store results
        results.append({
            "configuration": config["name"],
            "time_seconds": processing_time,
            "memory_mb": memory_usage,
            "embedding_size_mb": embedding_size,
            "precision": config["precision"],
            "memory_mapped": config["use_mmap"],
            "quantized": config["quantize"]
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def benchmark_deduplication(dataset):
    """Benchmark performance impact of deduplication."""
    print("\n=== DEDUPLICATION BENCHMARK ===")
    
    # Count duplicates in the dataset
    exact_duplicates = dataset.duplicated(subset=["text"]).sum()
    total_docs = len(dataset)
    duplicate_pct = (exact_duplicates / total_docs) * 100
    
    print(f"Dataset: {total_docs} documents, {exact_duplicates} duplicates ({duplicate_pct:.1f}%)")
    
    results = []
    
    # Test without deduplication
    print("\nTesting WITHOUT deduplication...")
    
    # Clear memory
    gc.collect()
    
    # Start timing
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # Run workflow
    workflow_no_dedup = MenoWorkflow()
    workflow_no_dedup.load_data(data=dataset, text_column="text", deduplicate=False)
    workflow_no_dedup.preprocess_documents()
    workflow_no_dedup.discover_topics(method="bertopic", num_topics=10)
    
    # Record metrics
    end_time = time.time()
    peak_memory = get_memory_usage()
    
    # Calculate metrics
    processing_time_no_dedup = end_time - start_time
    memory_usage_no_dedup = peak_memory - start_memory
    
    print(f"  Time: {processing_time_no_dedup:.2f}s")
    print(f"  Memory: {memory_usage_no_dedup:.2f} MB")
    
    # Store results
    results.append({
        "deduplication": "Disabled",
        "time_seconds": processing_time_no_dedup,
        "memory_mb": memory_usage_no_dedup,
        "documents_processed": len(dataset)
    })
    
    # Test with deduplication
    print("\nTesting WITH deduplication...")
    
    # Clear memory
    gc.collect()
    
    # Start timing
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # Run workflow
    workflow_dedup = MenoWorkflow()
    workflow_dedup.load_data(data=dataset, text_column="text", deduplicate=True)
    workflow_dedup.preprocess_documents()
    workflow_dedup.discover_topics(method="bertopic", num_topics=10)
    
    # Record metrics
    end_time = time.time()
    peak_memory = get_memory_usage()
    
    # Calculate metrics
    processing_time_dedup = end_time - start_time
    memory_usage_dedup = peak_memory - start_memory
    docs_after_dedup = len(workflow_dedup.documents)
    
    print(f"  Time: {processing_time_dedup:.2f}s")
    print(f"  Memory: {memory_usage_dedup:.2f} MB")
    print(f"  Documents after deduplication: {docs_after_dedup}")
    
    # Store results
    results.append({
        "deduplication": "Enabled",
        "time_seconds": processing_time_dedup,
        "memory_mb": memory_usage_dedup,
        "documents_processed": docs_after_dedup
    })
    
    # Calculate improvement
    time_improvement = (processing_time_no_dedup - processing_time_dedup) / processing_time_no_dedup * 100
    memory_improvement = (memory_usage_no_dedup - memory_usage_dedup) / memory_usage_no_dedup * 100
    
    print(f"\nImprovement with deduplication:")
    print(f"  Time: {time_improvement:.1f}% faster")
    print(f"  Memory: {memory_improvement:.1f}% less memory")
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def benchmark_incremental_updates(initial_dataset, update_datasets):
    """Benchmark different incremental update strategies."""
    print("\n=== INCREMENTAL LEARNING BENCHMARK ===")
    
    # Define update modes to test
    update_modes = ["incremental", "partial_retrain", "full_retrain"]
    
    results = []
    
    # Initial training (same for all modes)
    print("\nTraining initial model...")
    
    # Clear memory
    gc.collect()
    
    # Train initial model
    workflow = MenoWorkflow()
    workflow.load_data(data=initial_dataset, text_column="text")
    workflow.preprocess_documents()
    workflow.discover_topics(method="bertopic", num_topics=10)
    
    # Get model
    model = workflow.modeler.model
    
    # Test each update mode with the same updates
    for mode in update_modes:
        print(f"\nTesting update mode: {mode}")
        
        # Create a copy of the model for this test
        import copy
        test_model = copy.deepcopy(model)
        
        # Create topic updater
        topic_updater = TopicUpdater(model=test_model)
        
        # Process all updates
        cumulative_time = 0
        update_times = []
        stability_scores = []
        
        for i, update_data in enumerate(update_datasets):
            # Start timing
            start_time = time.time()
            
            # Get update documents
            update_docs = update_data["text"].tolist()
            update_ids = update_data["id"].tolist()
            
            # Update model
            test_model, stats = topic_updater.update_model(
                documents=update_docs,
                document_ids=update_ids,
                update_mode=mode,
                verbose=False
            )
            
            # Record metrics
            end_time = time.time()
            update_time = end_time - start_time
            cumulative_time += update_time
            update_times.append(update_time)
            stability_scores.append(stats["topic_stability"])
            
            print(f"  Update {i+1}: {update_time:.2f}s, " +
                  f"stability: {stats['topic_stability']:.2f}")
        
        # Calculate average metrics
        avg_update_time = np.mean(update_times)
        avg_stability = np.mean(stability_scores)
        
        print(f"  Average update time: {avg_update_time:.2f}s")
        print(f"  Average stability: {avg_stability:.2f}")
        print(f"  Total update time: {cumulative_time:.2f}s")
        
        # Store results
        results.append({
            "update_mode": mode,
            "avg_update_time": avg_update_time,
            "avg_stability": avg_stability,
            "total_update_time": cumulative_time,
            "stability_scores": stability_scores,
            "update_times": update_times
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def plot_benchmark_results(memory_results, dedup_results, incremental_results, output_dir):
    """Plot benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Memory optimization plots
    plt.figure(figsize=(12, 5))
    
    # Memory usage
    plt.subplot(1, 2, 1)
    plt.bar(memory_results["configuration"], memory_results["memory_mb"], color='royalblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage by Configuration')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Processing time
    plt.subplot(1, 2, 2)
    plt.bar(memory_results["configuration"], memory_results["time_seconds"], color='seagreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time by Configuration')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_optimization_benchmark.png'), 
                dpi=300, bbox_inches='tight')
    
    # 2. Deduplication plots
    plt.figure(figsize=(12, 5))
    
    # Processing time
    plt.subplot(1, 2, 1)
    plt.bar(dedup_results["deduplication"], dedup_results["time_seconds"], color='royalblue')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time: With vs. Without Deduplication')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Memory usage
    plt.subplot(1, 2, 2)
    plt.bar(dedup_results["deduplication"], dedup_results["memory_mb"], color='seagreen')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage: With vs. Without Deduplication')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deduplication_benchmark.png'), 
                dpi=300, bbox_inches='tight')
    
    # 3. Incremental learning plots
    plt.figure(figsize=(12, 10))
    
    # Average update time
    plt.subplot(2, 2, 1)
    plt.bar(incremental_results["update_mode"], incremental_results["avg_update_time"], 
           color='royalblue')
    plt.ylabel('Time (seconds)')
    plt.title('Average Update Time by Mode')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Average stability
    plt.subplot(2, 2, 2)
    plt.bar(incremental_results["update_mode"], incremental_results["avg_stability"], 
           color='seagreen')
    plt.ylabel('Stability (0-1)')
    plt.title('Average Topic Stability by Mode')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Total update time
    plt.subplot(2, 2, 3)
    plt.bar(incremental_results["update_mode"], incremental_results["total_update_time"], 
           color='salmon')
    plt.ylabel('Time (seconds)')
    plt.title('Total Update Time by Mode')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Stability over updates
    plt.subplot(2, 2, 4)
    
    # Plot stability for each mode
    for idx, row in incremental_results.iterrows():
        mode = row["update_mode"]
        stability = row["stability_scores"]
        plt.plot(range(1, len(stability) + 1), stability, marker='o', label=mode)
    
    plt.xlabel('Update Number')
    plt.ylabel('Topic Stability (0-1)')
    plt.title('Topic Stability Over Updates')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'incremental_learning_benchmark.png'), 
                dpi=300, bbox_inches='tight')
    
    # 4. Combined optimization benchmark summary
    plt.figure(figsize=(12, 6))
    
    # Set up data
    categories = ['Memory Usage', 'Processing Time', 'Model Update Time']
    
    # Calculate relative values (lower is better)
    memory_reduction = memory_results["memory_mb"].values[0] / memory_results["memory_mb"].values[-1]
    time_reduction = dedup_results["time_seconds"].values[0] / dedup_results["time_seconds"].values[-1]
    
    # Get incremental vs full retrain speedup
    incremental_time = incremental_results[incremental_results["update_mode"] == "incremental"]["avg_update_time"].values[0]
    full_time = incremental_results[incremental_results["update_mode"] == "full_retrain"]["avg_update_time"].values[0]
    update_speedup = full_time / incremental_time
    
    improvements = [memory_reduction, time_reduction, update_speedup]
    
    # Plot
    plt.bar(categories, improvements, color=['royalblue', 'seagreen', 'salmon'])
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    plt.ylabel('Improvement Factor (higher is better)')
    plt.title('Combined Optimization Performance Improvement')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add text labels
    for i, v in enumerate(improvements):
        plt.text(i, v + 0.1, f"{v:.1f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_optimization_benchmark.png'), 
                dpi=300, bbox_inches='tight')
    
    return True


def main():
    """Run all benchmarks."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Meno optimization benchmarks')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                        help='Output directory for benchmark results')
    parser.add_argument('--docs', type=int, default=2000,
                        help='Number of documents to use for benchmarking')
    parser.add_argument('--duplicates', type=int, default=1000,
                        help='Number of duplicates to add to the dataset')
    parser.add_argument('--updates', type=int, default=3,
                        help='Number of update batches for incremental learning benchmark')
    parser.add_argument('--update-size', type=int, default=500,
                        help='Number of documents per update batch')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"=== MENO OPTIMIZATION BENCHMARKS ===")
    print(f"Dataset: {args.docs} documents + {args.duplicates} duplicates")
    print(f"Updates: {args.updates} batches of {args.update_size} documents each")
    print(f"Output directory: {args.output}")
    
    # Create dataset
    print("\nGenerating benchmark dataset...")
    dataset = create_benchmark_dataset(
        num_documents=args.docs,
        num_duplicates=args.duplicates,
        words_per_doc=50
    )
    
    # Create updates for incremental learning
    update_datasets = []
    for i in range(args.updates):
        update = create_benchmark_dataset(
            num_documents=args.update_size,
            num_duplicates=int(args.update_size * 0.2),  # 20% duplicates
            words_per_doc=50,
            seed=42 + i  # Different seed for each update
        )
        update_datasets.append(update)
    
    # Run memory optimization benchmark
    memory_results = benchmark_memory_optimizations(dataset.head(min(1000, len(dataset))))
    memory_results.to_csv(os.path.join(args.output, 'memory_optimization_results.csv'), index=False)
    
    # Run deduplication benchmark
    dedup_results = benchmark_deduplication(dataset)
    dedup_results.to_csv(os.path.join(args.output, 'deduplication_results.csv'), index=False)
    
    # Run incremental learning benchmark
    incremental_results = benchmark_incremental_updates(
        dataset.head(min(1000, len(dataset))),
        [update.head(min(args.update_size, len(update))) for update in update_datasets]
    )
    incremental_results.to_csv(os.path.join(args.output, 'incremental_learning_results.csv'), index=False)
    
    # Plot results
    plot_benchmark_results(
        memory_results, 
        dedup_results, 
        incremental_results, 
        args.output
    )
    
    print(f"\nAll benchmark results saved to: {os.path.abspath(args.output)}")
    
    # Print summary
    print("\n=== OPTIMIZATION SUMMARY ===")
    
    # Memory optimization
    best_memory_config = memory_results.loc[memory_results["memory_mb"].idxmin()]
    print(f"Best memory configuration: {best_memory_config['configuration']}")
    memory_reduction = memory_results["memory_mb"].iloc[0] / best_memory_config["memory_mb"]
    print(f"Memory reduction: {memory_reduction:.1f}x")
    
    # Deduplication
    time_improvement = (dedup_results["time_seconds"].iloc[0] - dedup_results["time_seconds"].iloc[1]) / dedup_results["time_seconds"].iloc[0] * 100
    memory_improvement = (dedup_results["memory_mb"].iloc[0] - dedup_results["memory_mb"].iloc[1]) / dedup_results["memory_mb"].iloc[0] * 100
    print(f"Deduplication improvement: {time_improvement:.1f}% faster, {memory_improvement:.1f}% less memory")
    
    # Incremental learning
    fastest_mode = incremental_results.loc[incremental_results["avg_update_time"].idxmin()]
    most_stable_mode = incremental_results.loc[incremental_results["avg_stability"].idxmax()]
    
    print(f"Fastest update mode: {fastest_mode['update_mode']}")
    print(f"Most stable update mode: {most_stable_mode['update_mode']}")
    
    # Compare incremental to full retrain
    incremental_time = incremental_results[incremental_results["update_mode"] == "incremental"]["avg_update_time"].values[0]
    full_time = incremental_results[incremental_results["update_mode"] == "full_retrain"]["avg_update_time"].values[0]
    update_speedup = full_time / incremental_time
    
    print(f"Incremental vs. full retrain speedup: {update_speedup:.1f}x")
    
    # Provide recommendation
    print("\nRecommended configuration for optimal performance:")
    print(f"1. Memory: {best_memory_config['precision']} precision, " +
          f"{'with' if best_memory_config['memory_mapped'] else 'without'} memory mapping, " +
          f"{'with' if best_memory_config['quantized'] else 'without'} quantization")
    print(f"2. Deduplication: Enabled (for {time_improvement:.1f}% faster processing)")
    print(f"3. Incremental updates: Use '{fastest_mode['update_mode']}' mode for speed, " +
          f"'{most_stable_mode['update_mode']}' mode for stability")


if __name__ == "__main__":
    main()
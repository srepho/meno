"""
Memory Optimization Example

This example demonstrates various memory optimization techniques available in Meno,
including 8-bit quantization, half-precision, and memory mapping.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import os
from pathlib import Path
import gc

from meno import MenoWorkflow
from meno.modeling.embeddings import DocumentEmbedding


def create_synthetic_documents(num_documents=10000, words_per_doc=100, seed=42):
    """Create synthetic documents for benchmarking."""
    np.random.seed(seed)
    
    # Create a vocabulary of words
    vocab = [f"word_{i}" for i in range(1000)]
    
    # Generate documents
    documents = []
    for i in range(num_documents):
        # Sample words from vocabulary with replacement
        words = np.random.choice(vocab, size=words_per_doc)
        # Create document
        doc = " ".join(words)
        documents.append(doc)
    
    return pd.DataFrame({
        "text": documents,
        "id": [f"doc_{i}" for i in range(num_documents)]
    })


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def benchmark_embedding_configurations(dataset, configs, document_counts=None):
    """Benchmark different embedding configurations."""
    if document_counts is None:
        document_counts = [100, 1000, 5000, 10000]
        
    # Keep only up to the maximum number of documents
    max_docs = max(document_counts)
    if len(dataset) > max_docs:
        dataset = dataset.iloc[:max_docs].copy()
    
    results = []
    
    for config in configs:
        config_name = config.get("name", "Unnamed Config")
        print(f"\nTesting configuration: {config_name}")
        
        # Extract embedding configuration
        embedding_config = config.get("embedding", {})
        
        for num_docs in document_counts:
            print(f"  Processing {num_docs} documents...")
            
            # Clear any previous cached data and run garbage collection
            gc.collect()
            
            # Use subset of data
            subset = dataset.iloc[:num_docs].copy()
            
            # Record starting memory
            start_memory = get_memory_usage()
            
            # Create embedding model
            embedder = DocumentEmbedding(**embedding_config)
            
            # Start timing
            start_time = time.time()
            
            # Generate embeddings
            texts = subset["text"].tolist()
            embeddings = embedder.embed_documents(texts)
            
            # End timing
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Record peak memory
            peak_memory = get_memory_usage()
            memory_increase = peak_memory - start_memory
            
            # Record memory usage of embeddings
            embedding_memory = embeddings.nbytes / 1024 / 1024  # MB
            
            # Record stats
            results.append({
                "configuration": config_name,
                "num_documents": num_docs,
                "time_seconds": elapsed,
                "docs_per_second": num_docs / elapsed,
                "memory_increase_mb": memory_increase,
                "embedding_memory_mb": embedding_memory,
                "bytes_per_doc": embedding_memory * 1024 * 1024 / num_docs,
                "embedding_shape": str(embeddings.shape),
                "embedding_dtype": str(embeddings.dtype)
            })
            
            print(f"    Completed in {elapsed:.2f} seconds, memory usage: {memory_increase:.2f} MB")
    
    return pd.DataFrame(results)


def run_topic_modeling_with_optimizations(dataset, configs):
    """Run full topic modeling with different optimization configurations."""
    results = []
    
    for config in configs:
        config_name = config.get("name", "Unnamed Config")
        print(f"\nRunning topic modeling with: {config_name}")
        
        # Clear previous data
        gc.collect()
        
        # Record starting memory
        start_memory = get_memory_usage()
        
        # Start timing
        start_time = time.time()
        
        # Configure workflow
        workflow = MenoWorkflow(config_overrides=config.get("workflow_config", {}))
        
        # Run workflow
        workflow.load_data(data=dataset, text_column="text", id_column="id")
        workflow.preprocess_documents()
        workflow.discover_topics(method="bertopic", num_topics=10)
        
        # End timing
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Get peak memory
        peak_memory = get_memory_usage()
        memory_increase = peak_memory - start_memory
        
        # Get topics
        topics = workflow.get_topics()
        
        # Record stats
        results.append({
            "configuration": config_name,
            "time_seconds": elapsed,
            "num_topics": len(topics),
            "memory_increase_mb": memory_increase
        })
        
        print(f"  Completed in {elapsed:.2f} seconds, memory usage: {memory_increase:.2f} MB")
        print(f"  Found {len(topics)} topics")
    
    return pd.DataFrame(results)


def plot_results(embedding_results, topic_results=None, output_dir=None):
    """Plot benchmark results."""
    # Create directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot embedding results
    plt.figure(figsize=(12, 10))
    
    # Memory usage by document count
    plt.subplot(2, 2, 1)
    for config in embedding_results['configuration'].unique():
        subset = embedding_results[embedding_results['configuration'] == config]
        plt.plot(subset['num_documents'], subset['memory_increase_mb'], marker='o', label=config)
    plt.xlabel('Number of Documents')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage by Configuration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Processing time by document count
    plt.subplot(2, 2, 2)
    for config in embedding_results['configuration'].unique():
        subset = embedding_results[embedding_results['configuration'] == config]
        plt.plot(subset['num_documents'], subset['time_seconds'], marker='o', label=config)
    plt.xlabel('Number of Documents')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time by Configuration')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Documents per second
    plt.subplot(2, 2, 3)
    for config in embedding_results['configuration'].unique():
        subset = embedding_results[embedding_results['configuration'] == config]
        plt.plot(subset['num_documents'], subset['docs_per_second'], marker='o', label=config)
    plt.xlabel('Number of Documents')
    plt.ylabel('Documents per Second')
    plt.title('Processing Throughput')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Memory per document
    plt.subplot(2, 2, 4)
    for config in embedding_results['configuration'].unique():
        subset = embedding_results[embedding_results['configuration'] == config]
        plt.plot(subset['num_documents'], subset['bytes_per_doc'] / 1024, marker='o', label=config)
    plt.xlabel('Number of Documents')
    plt.ylabel('KB per Document')
    plt.title('Memory Efficiency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'embedding_benchmark.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot topic modeling results if available
    if topic_results is not None:
        plt.figure(figsize=(12, 5))
        
        # Processing time
        plt.subplot(1, 2, 1)
        plt.bar(topic_results['configuration'], topic_results['time_seconds'])
        plt.xlabel('Configuration')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Topic Modeling Processing Time')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Memory usage
        plt.subplot(1, 2, 2)
        plt.bar(topic_results['configuration'], topic_results['memory_increase_mb'])
        plt.xlabel('Configuration')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Topic Modeling Memory Usage')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'topic_modeling_benchmark.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()


def main():
    """Main function to run memory optimization examples."""
    # Create output directory
    output_dir = Path("./memory_optimization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create dataset
    print("Generating synthetic dataset...")
    dataset = create_synthetic_documents(num_documents=15000, words_per_doc=50)
    dataset.to_csv(output_dir / "synthetic_dataset.csv", index=False)
    
    # Define configurations to test
    embedding_configs = [
        {
            "name": "Default (float32)",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "precision": "float32",
                "use_mmap": False,
                "batch_size": 32
            }
        },
        {
            "name": "Half Precision (float16)",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "precision": "float16",
                "use_mmap": False,
                "batch_size": 32
            }
        },
        {
            "name": "Memory-Mapped (float32)",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "precision": "float32",
                "use_mmap": True,
                "batch_size": 32
            }
        },
        {
            "name": "Memory-Mapped (float16)",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "precision": "float16",
                "use_mmap": True,
                "batch_size": 32
            }
        },
        {
            "name": "8-bit (int8)",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "precision": "int8",
                "quantize": False,  # Just embeddings, not model
                "use_mmap": True,
                "batch_size": 32
            }
        },
        # Try to include quantized model if bitsandbytes is available
        {
            "name": "Fully Quantized (int8)",
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "precision": "int8",
                "quantize": True,  # Quantize model weights too
                "use_mmap": True,
                "batch_size": 32
            }
        }
    ]
    
    # Run embedding benchmarks
    print("\nRunning embedding benchmarks...")
    embedding_results = benchmark_embedding_configurations(
        dataset,
        embedding_configs,
        document_counts=[100, 1000, 5000, 10000]
    )
    
    # Save results
    embedding_results.to_csv(output_dir / "embedding_benchmark_results.csv", index=False)
    
    # Define topic modeling configurations
    topic_configs = [
        {
            "name": "Default",
            "workflow_config": {
                "modeling": {
                    "embeddings": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "precision": "float32",
                        "use_mmap": False
                    }
                }
            }
        },
        {
            "name": "Memory-Optimized",
            "workflow_config": {
                "modeling": {
                    "embeddings": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "precision": "float16",
                        "use_mmap": True
                    }
                }
            }
        },
        {
            "name": "8-bit Quantized",
            "workflow_config": {
                "modeling": {
                    "embeddings": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "precision": "int8",
                        "use_mmap": True,
                        "quantize": True
                    }
                }
            }
        }
    ]
    
    # Run topic modeling with small dataset for speed
    small_dataset = dataset.iloc[:2000].copy()
    
    # Run topic modeling benchmarks
    print("\nRunning topic modeling benchmarks...")
    try:
        topic_results = run_topic_modeling_with_optimizations(small_dataset, topic_configs)
        # Save results
        topic_results.to_csv(output_dir / "topic_modeling_benchmark_results.csv", index=False)
    except Exception as e:
        print(f"Error running topic modeling benchmarks: {e}")
        topic_results = None
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(embedding_results, topic_results, output_dir)
    
    print(f"\nAll output saved to: {output_dir.absolute()}")
    
    # Print summary
    print("\nSummary of Memory Optimization Results:")
    print("---------------------------------------")
    
    # Print embedding summary
    print("\nEmbedding Memory Usage (10,000 documents):")
    summary = embedding_results[embedding_results['num_documents'] == 10000][
        ['configuration', 'memory_increase_mb', 'time_seconds']
    ].copy()
    
    # Calculate % reduction compared to baseline
    baseline = summary[summary['configuration'] == 'Default (float32)']['memory_increase_mb'].values[0]
    summary['memory_reduction_pct'] = 100 * (1 - (summary['memory_increase_mb'] / baseline))
    
    # Sort by memory usage
    summary = summary.sort_values('memory_increase_mb')
    
    # Print formatted summary
    for _, row in summary.iterrows():
        print(f"  {row['configuration']}: {row['memory_increase_mb']:.2f} MB " +
              f"({row['memory_reduction_pct']:.1f}% reduction, {row['time_seconds']:.2f} seconds)")
    
    # Print topic modeling summary if available
    if topic_results is not None:
        print("\nTopic Modeling Performance:")
        for _, row in topic_results.iterrows():
            print(f"  {row['configuration']}: {row['memory_increase_mb']:.2f} MB, " +
                  f"{row['time_seconds']:.2f} seconds, {row['num_topics']} topics")


if __name__ == "__main__":
    main()
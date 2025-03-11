"""Performance optimization techniques for preprocessing large datasets."""

import pandas as pd
import numpy as np
import os
import json
import time
import multiprocessing
from pathlib import Path
from functools import partial
from contextlib import contextmanager

from meno.preprocessing.acronyms import AcronymExpander
from meno.preprocessing.spelling import SpellingCorrector

# Create output directory
output_dir = Path("./output/performance")
os.makedirs(output_dir, exist_ok=True)

# Helper timing context manager
@contextmanager
def timer(task_name):
    """Context manager for timing code blocks."""
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{task_name}: {end_time - start_time:.4f} seconds")


# Generate sample data of varying sizes
def generate_sample_data(size, misspelling_rate=0.1, acronym_rate=0.05):
    """Generate sample data with controlled error rates."""
    # Common words for building sentences
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", 
        "for", "on", "with", "as", "at", "this", "by", "from", "they", 
        "we", "say", "her", "she", "or", "an", "will", "my", "all", "would",
        "customer", "product", "service", "company", "market", "business",
        "data", "system", "user", "report", "time", "information", "process"
    ]
    
    # Common misspellings
    misspellings = {
        "the": "teh",
        "with": "wiht",
        "from": "fromt",
        "they": "tehey",
        "would": "woudl",
        "customer": "custoemr",
        "product": "prodcut",
        "service": "servcie",
        "company": "compnay",
        "information": "informaiton",
        "process": "procses",
        "business": "busines",
        "system": "systme",
        "market": "markeet",
        "report": "reprot",
    }
    
    # Common acronyms
    acronyms = ["CEO", "CFO", "ROI", "KPI", "API", "CRM", "ERP", "HR", "IT", "AI"]
    
    # Generate random documents
    data = []
    for i in range(size):
        # Generate a random document
        doc_length = np.random.randint(10, 50)  # Random length
        words = np.random.choice(common_words, doc_length)
        
        # Introduce misspellings
        for j in range(len(words)):
            if words[j] in misspellings and np.random.random() < misspelling_rate:
                words[j] = misspellings[words[j]]
        
        # Add acronyms
        for j in range(len(words)):
            if np.random.random() < acronym_rate:
                words[j] = np.random.choice(acronyms)
        
        # Build document
        doc = " ".join(words)
        data.append(doc)
    
    return pd.DataFrame({"text": data})


# Basic sequential processing
def process_sequential(df, text_column="text"):
    """Process documents sequentially."""
    acronym_expander = AcronymExpander()
    spelling_corrector = SpellingCorrector()
    
    results = []
    for text in df[text_column]:
        # Correct spelling first
        corrected = spelling_corrector.correct_text(text)
        
        # Then expand acronyms
        expanded = acronym_expander.expand_acronyms(corrected)
        
        results.append(expanded)
    
    return results


# Batch processing
def process_batch(df, text_column="text", batch_size=100):
    """Process documents in batches."""
    acronym_expander = AcronymExpander()
    spelling_corrector = SpellingCorrector()
    
    results = []
    for i in range(0, len(df), batch_size):
        # Get batch
        batch = df[text_column].iloc[i:i+batch_size].tolist()
        
        # Process batch
        corrected_batch = spelling_corrector.correct_texts(batch)
        expanded_batch = acronym_expander.expand_acronyms_batch(corrected_batch)
        
        results.extend(expanded_batch)
    
    return results


# Parallel processing with multiprocessing
def process_chunk(chunk, text_column="text"):
    """Process a chunk of documents in parallel."""
    acronym_expander = AcronymExpander()
    spelling_corrector = SpellingCorrector()
    
    results = []
    for text in chunk[text_column]:
        corrected = spelling_corrector.correct_text(text)
        expanded = acronym_expander.expand_acronyms(corrected)
        results.append(expanded)
    
    return results


def process_parallel(df, text_column="text", n_jobs=None):
    """Process documents in parallel using multiprocessing."""
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Split data into chunks
    chunk_size = len(df) // n_jobs
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks in parallel
    with multiprocessing.Pool(n_jobs) as pool:
        results = pool.map(partial(process_chunk, text_column=text_column), chunks)
    
    # Flatten results
    flattened = [item for sublist in results for item in sublist]
    return flattened


# Cached processing
class CachedProcessor:
    """Processor with caching for repeated documents."""
    
    def __init__(self):
        """Initialize the cached processor."""
        self.acronym_expander = AcronymExpander()
        self.spelling_corrector = SpellingCorrector()
        self.cache = {}
    
    def process(self, df, text_column="text"):
        """Process documents with caching."""
        results = []
        cache_hits = 0
        cache_misses = 0
        
        for text in df[text_column]:
            # Check cache
            if text in self.cache:
                results.append(self.cache[text])
                cache_hits += 1
            else:
                # Process and cache
                corrected = self.spelling_corrector.correct_text(text)
                expanded = self.acronym_expander.expand_acronyms(corrected)
                
                self.cache[text] = expanded
                results.append(expanded)
                cache_misses += 1
        
        print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        return results


# Optimized processor with all techniques
class OptimizedProcessor:
    """Processor with multiple optimizations."""
    
    def __init__(self, cache_size=10000, batch_size=100, n_jobs=None):
        """Initialize the optimized processor."""
        self.acronym_expander = AcronymExpander()
        self.spelling_corrector = SpellingCorrector()
        self.cache = {}
        self.cache_size = cache_size
        self.batch_size = batch_size
        
        if n_jobs is None:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = n_jobs
    
    def _process_batch_no_cache(self, batch):
        """Process a batch without caching."""
        corrected_batch = self.spelling_corrector.correct_texts(batch)
        expanded_batch = self.acronym_expander.expand_acronyms_batch(corrected_batch)
        return expanded_batch
    
    def process(self, df, text_column="text"):
        """Process documents with optimizations."""
        results = [None] * len(df)
        
        # Identify unique documents
        texts = df[text_column].tolist()
        unique_texts = {}
        
        for i, text in enumerate(texts):
            if text not in unique_texts:
                unique_texts[text] = []
            unique_texts[text].append(i)
        
        # Process unique texts
        text_list = list(unique_texts.keys())
        
        # Check cache first
        cache_hits = 0
        to_process = []
        to_process_indices = []
        
        for i, text in enumerate(text_list):
            if text in self.cache:
                # Cache hit
                for idx in unique_texts[text]:
                    results[idx] = self.cache[text]
                cache_hits += 1
            else:
                # Cache miss
                to_process.append(text)
                to_process_indices.append(i)
        
        # Process in batches
        processed_results = []
        for i in range(0, len(to_process), self.batch_size):
            batch = to_process[i:i+self.batch_size]
            processed_batch = self._process_batch_no_cache(batch)
            processed_results.extend(processed_batch)
        
        # Update cache and results
        for i, text in enumerate(to_process):
            processed = processed_results[i]
            
            # Update cache if not too big
            if len(self.cache) < self.cache_size:
                self.cache[text] = processed
            
            # Update results
            for idx in unique_texts[text_list[to_process_indices[i]]]:
                results[idx] = processed
        
        print(f"Cache hits: {cache_hits}, Processed: {len(to_process)}")
        
        return results


# Run benchmarks
def run_benchmarks():
    """Run preprocessing benchmarks with various optimizations."""
    print("Generating sample data...")
    small_df = generate_sample_data(1000)
    medium_df = generate_sample_data(5000)
    large_df = generate_sample_data(10000)
    
    # Save sample data
    small_df.to_csv(output_dir / "small_sample.csv", index=False)
    medium_df.to_csv(output_dir / "medium_sample.csv", index=False)
    
    # Basic sequential processing
    print("\n--- Sequential Processing ---")
    
    with timer("Sequential (Small)"):
        small_sequential = process_sequential(small_df)
    
    with timer("Sequential (Medium)"):
        medium_sequential = process_sequential(medium_df)
    
    # Batch processing
    print("\n--- Batch Processing ---")
    
    with timer("Batch (Small)"):
        small_batch = process_batch(small_df, batch_size=100)
    
    with timer("Batch (Medium)"):
        medium_batch = process_batch(medium_df, batch_size=500)
    
    # Parallel processing
    print("\n--- Parallel Processing ---")
    
    with timer("Parallel (Small)"):
        small_parallel = process_parallel(small_df)
    
    with timer("Parallel (Medium)"):
        medium_parallel = process_parallel(medium_df)
    
    # Cached processing
    print("\n--- Cached Processing ---")
    
    # Create data with duplicates
    duplicate_df = pd.concat([small_df] * 3)
    print(f"Created duplicate dataset with {len(duplicate_df)} rows")
    
    cached_processor = CachedProcessor()
    
    with timer("Cached (With Duplicates)"):
        cached_results = cached_processor.process(duplicate_df)
    
    # Optimized processing
    print("\n--- Optimized Processing ---")
    
    optimized_processor = OptimizedProcessor(cache_size=1000, batch_size=200)
    
    with timer("Optimized (With Duplicates)"):
        optimized_results = optimized_processor.process(duplicate_df)
    
    with timer("Optimized (Medium)"):
        optimized_medium = optimized_processor.process(medium_df)
    
    # Large dataset benchmark
    print("\n--- Large Dataset Benchmark ---")
    
    print(f"Processing large dataset ({len(large_df)} documents)...")
    
    with timer("Sequential (Large)"):
        large_sequential_sample = process_sequential(large_df.head(500))
    
    with timer("Batch (Large)"):
        large_batch = process_batch(large_df, batch_size=1000)
    
    with timer("Parallel (Large)"):
        large_parallel = process_parallel(large_df)
    
    with timer("Optimized (Large)"):
        large_optimized = optimized_processor.process(large_df)
    
    # Save benchmark results
    benchmark_results = {
        "sequential": {
            "small": len(small_sequential),
            "medium": len(medium_sequential),
            "large_sample": len(large_sequential_sample),
        },
        "batch": {
            "small": len(small_batch),
            "medium": len(medium_batch),
            "large": len(large_batch),
        },
        "parallel": {
            "small": len(small_parallel),
            "medium": len(medium_parallel),
            "large": len(large_parallel),
        },
        "cached": {
            "duplicate": len(cached_results),
        },
        "optimized": {
            "duplicate": len(optimized_results),
            "medium": len(optimized_medium),
            "large": len(large_optimized),
        }
    }
    
    # Save results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Save processed examples
    examples_df = pd.DataFrame({
        "original": small_df["text"].head(5),
        "sequential": small_sequential[:5],
        "batch": small_batch[:5],
        "parallel": small_parallel[:5],
        "optimized": optimized_processor.process(small_df.head(5))
    })
    
    examples_df.to_csv(output_dir / "processed_examples.csv", index=False)
    
    print("\nBenchmark complete! Results saved to output directory.")


# Streaming processing example
def streaming_example():
    """Example of streaming processing for very large datasets."""
    print("\n--- Streaming Processing Example ---")
    
    # Generate a large dataset for streaming
    print("Generating streaming dataset...")
    chunk_size = 1000
    num_chunks = 5
    
    # Create a streaming processor
    acronym_expander = AcronymExpander()
    spelling_corrector = SpellingCorrector()
    
    # Process in chunks, simulating loading from disk
    total_processed = 0
    total_time = 0
    
    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks}...")
        
        # Generate chunk
        chunk = generate_sample_data(chunk_size)
        
        # Measure processing time
        start_time = time.time()
        
        # Process chunk
        corrected = spelling_corrector.correct_texts(chunk["text"])
        expanded = acronym_expander.expand_acronyms_batch(corrected)
        
        end_time = time.time()
        chunk_time = end_time - start_time
        total_time += chunk_time
        
        # Update counts
        total_processed += len(expanded)
        
        print(f"Chunk {i+1} processed in {chunk_time:.4f} seconds")
    
    print(f"Streamed processing complete: {total_processed} documents in {total_time:.4f} seconds")
    print(f"Average processing rate: {total_processed/total_time:.2f} documents/second")
    
    # Save streaming results
    streaming_results = {
        "total_documents": total_processed,
        "total_time": total_time,
        "documents_per_second": total_processed/total_time,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks
    }
    
    with open(output_dir / "streaming_results.json", "w") as f:
        json.dump(streaming_results, f, indent=2)


# Memory profiling example
def memory_profile_example():
    """Example of memory profiling during processing."""
    print("\n--- Memory Profiling Example ---")
    
    try:
        import psutil
        memory_available = True
    except ImportError:
        print("psutil not available. Install with: pip install psutil")
        memory_available = False
    
    if not memory_available:
        return
    
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Generate test data
    print("Generating test data...")
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    test_df = generate_sample_data(10000)
    
    after_gen_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after data generation: {after_gen_memory:.2f} MB")
    print(f"Delta: {after_gen_memory - initial_memory:.2f} MB")
    
    # Memory usage during different processing techniques
    memory_usage = {
        "initial": initial_memory,
        "after_data_generation": after_gen_memory,
        "techniques": {}
    }
    
    # 1. Sequential with and without caching
    print("\nSequential processing (no caching)...")
    before_memory = process.memory_info().rss / 1024 / 1024
    
    with timer("Sequential"):
        sequential_results = process_sequential(test_df.head(1000))
    
    after_memory = process.memory_info().rss / 1024 / 1024
    memory_usage["techniques"]["sequential"] = {
        "before": before_memory,
        "after": after_memory,
        "delta": after_memory - before_memory
    }
    print(f"Memory usage delta: {after_memory - before_memory:.2f} MB")
    
    # 2. With caching
    print("\nCached processing...")
    before_memory = process.memory_info().rss / 1024 / 1024
    
    cached_processor = CachedProcessor()
    with timer("Cached"):
        cached_results = cached_processor.process(test_df.head(1000))
    
    after_memory = process.memory_info().rss / 1024 / 1024
    memory_usage["techniques"]["cached"] = {
        "before": before_memory,
        "after": after_memory,
        "delta": after_memory - before_memory
    }
    print(f"Memory usage delta: {after_memory - before_memory:.2f} MB")
    
    # 3. Optimized
    print("\nOptimized processing...")
    before_memory = process.memory_info().rss / 1024 / 1024
    
    optimized_processor = OptimizedProcessor(cache_size=500)
    with timer("Optimized"):
        optimized_results = optimized_processor.process(test_df.head(1000))
    
    after_memory = process.memory_info().rss / 1024 / 1024
    memory_usage["techniques"]["optimized"] = {
        "before": before_memory,
        "after": after_memory,
        "delta": after_memory - before_memory
    }
    print(f"Memory usage delta: {after_memory - before_memory:.2f} MB")
    
    # Save memory profile results
    with open(output_dir / "memory_profile.json", "w") as f:
        json.dump(memory_usage, f, indent=2)
    
    print("\nMemory profiling complete! Results saved to output directory.")


if __name__ == "__main__":
    print("=== Meno Performance Optimization Examples ===")
    
    # Run benchmarks
    run_benchmarks()
    
    # Streaming example
    streaming_example()
    
    # Memory profiling
    memory_profile_example()
    
    print("\nAll examples completed successfully!")
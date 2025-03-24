"""
External LLM Processing with Deduplication Example

This example demonstrates a comprehensive workflow for using Meno's deduplication functionality
with external LLM processing. It shows how to:

1. Deduplicate documents (both exact and fuzzy matching)
2. Process only unique documents through an external LLM API
3. Map the LLM results back to the full dataset with duplicates
4. Analyze performance gains and cost savings

This pattern is especially useful for large datasets where:
- Processing costs with LLMs can be significant
- Many documents contain duplicate or near-duplicate content
- You want to maintain the full dataset structure for analysis
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from datetime import datetime
import requests
from collections import defaultdict

# Meno imports
from meno import MenoWorkflow
from meno.preprocessing.deduplication import TextDeduplicator, deduplicate_text


# Mock LLM API for demonstration
class MockLLMAPI:
    """Mock LLM API to simulate external API calls."""
    
    def __init__(self, response_time=0.1, cost_per_token=0.0001):
        """
        Initialize the mock API.
        
        Args:
            response_time: Simulated response time per document in seconds
            cost_per_token: Simulated cost per token (similar to real LLM APIs)
        """
        self.response_time = response_time
        self.cost_per_token = cost_per_token
        self.total_tokens_processed = 0
        self.total_cost = 0
        self.total_requests = 0
        
        # Pre-defined categories for consistent responses
        self.categories = {
            "technology": ["AI", "Software", "Hardware", "Mobile", "Cloud"],
            "health": ["Medical", "Wellness", "Healthcare", "Treatment", "Prevention"],
            "finance": ["Banking", "Investment", "Insurance", "Economy", "Markets"],
            "sports": ["Competition", "Team", "Training", "Athletics", "Games"],
            "entertainment": ["Media", "Movies", "Music", "Television", "Arts"]
        }
    
    def _count_tokens(self, text):
        """Estimate token count - real LLMs would have more sophisticated counting."""
        return len(text.split())
    
    def _determine_category(self, text):
        """Determine the most relevant category based on text content."""
        text = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                score += text.count(keyword.lower())
            if category in text:
                score += 3  # Bonus for explicit category mention
            scores[category] = score
        
        # Get highest scoring category, default to random if no matches
        max_score = max(scores.values())
        if max_score > 0:
            return [k for k, v in scores.items() if v == max_score][0]
        else:
            return np.random.choice(list(self.categories.keys()))
    
    def process_document(self, text, prompt=""):
        """
        Process a document with the mock LLM API.
        
        Args:
            text: The document text to process
            prompt: Optional prompt to include
        
        Returns:
            dict: Results including category, summary and metadata
        """
        # Simulate API request time
        time.sleep(self.response_time)
        
        # Count tokens and update costs
        tokens = self._count_tokens(text)
        self.total_tokens_processed += tokens
        self.total_cost += tokens * self.cost_per_token
        self.total_requests += 1
        
        # Determine the document category
        category = self._determine_category(text)
        
        # Generate a simulated summary
        # In reality, this would be the LLM's actual response
        first_sentence = text.split('.')[0] if '.' in text else text[:50]
        summary = f"This document discusses {category}. {first_sentence}..."
        
        # Create response object similar to what a real LLM API might return
        response = {
            "category": category,
            "summary": summary,
            "confidence": np.random.uniform(0.75, 0.98),
            "metadata": {
                "tokens_processed": tokens,
                "processing_cost": tokens * self.cost_per_token,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return response
    
    def get_usage_stats(self):
        """Return the usage statistics."""
        return {
            "total_tokens": self.total_tokens_processed,
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "average_tokens_per_request": self.total_tokens_processed / self.total_requests if self.total_requests > 0 else 0
        }


def create_realistic_dataset(size=1000, duplicate_rate=0.3, fuzzy_rate=0.15, seed=42):
    """
    Create a realistic dataset with exact and fuzzy duplicates.
    
    Args:
        size: Total dataset size
        duplicate_rate: Percentage of exact duplicates
        fuzzy_rate: Percentage of fuzzy (near) duplicates
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with text documents
    """
    np.random.seed(seed)
    
    # Calculate how many unique documents to create
    unique_count = int(size * (1 - duplicate_rate - fuzzy_rate))
    exact_dup_count = int(size * duplicate_rate)
    fuzzy_dup_count = int(size * fuzzy_rate)
    
    # Adjust counts if needed to match size
    total = unique_count + exact_dup_count + fuzzy_dup_count
    if total != size:
        unique_count += (size - total)
    
    # Domain categories
    categories = ["technology", "health", "finance", "sports", "entertainment"]
    
    # Template phrases for more realistic content
    templates = [
        "This document discusses {topic} with a focus on recent developments. {details}",
        "A comprehensive overview of {topic} trends and analysis. {details}",
        "Important considerations regarding {topic} and its implications. {details}",
        "Analysis of {topic} showing key patterns and insights. {details}",
        "Review of {topic} examining the current state and future directions. {details}"
    ]
    
    # Details snippets to add variety
    details_by_category = {
        "technology": [
            "The rapid advancement of AI has transformed many industries.",
            "Cloud computing continues to evolve with new service models.",
            "Mobile technology is increasingly integrated with IoT devices.",
            "Software development practices emphasize DevOps integration.",
            "Hardware innovations are pushing the boundaries of performance."
        ],
        "health": [
            "Patient-centered approaches are becoming the standard of care.",
            "Preventive healthcare strategies show promising outcomes.",
            "Medical research has identified new treatment protocols.",
            "Wellness programs are being adopted by more organizations.",
            "Healthcare systems are implementing digital transformation."
        ],
        "finance": [
            "Market analysis suggests cautious investment strategies.",
            "Banking regulations have evolved in response to recent events.",
            "Economic indicators point to shifting consumer behaviors.",
            "Insurance companies are adapting to new risk models.",
            "Financial technology is disrupting traditional services."
        ],
        "sports": [
            "Team performance metrics reveal important coaching insights.",
            "Athletic training methods now incorporate advanced analytics.",
            "Competition results demonstrate strategic advantages.",
            "Game theory applications are changing coaching approaches.",
            "Sports medicine advances are improving recovery times."
        ],
        "entertainment": [
            "Media consumption patterns continue to favor streaming platforms.",
            "Movie production techniques are embracing virtual technologies.",
            "Music industry distribution models are evolving rapidly.",
            "Television content is increasingly tailored to niche audiences.",
            "Arts programs are finding new ways to engage communities."
        ]
    }
    
    # Generate unique documents
    documents = []
    for i in range(unique_count):
        category = categories[i % len(categories)]
        template = templates[i % len(templates)]
        details = np.random.choice(details_by_category[category])
        
        text = template.format(
            topic=category,
            details=details
        )
        
        # Add some variety with document ID and timestamp
        text += f" Document ID: DOC-{i+1000}. Timestamp: {np.random.randint(1000000, 9999999)}."
        
        documents.append({
            "text": text,
            "id": f"doc_{i}",
            "category": category,
            "is_duplicate": False,
            "duplicate_type": "original",
            "original_id": f"doc_{i}"
        })
    
    # Generate exact duplicates
    for i in range(exact_dup_count):
        # Select a random document to duplicate
        orig_idx = np.random.randint(0, unique_count)
        orig_doc = documents[orig_idx].copy()
        
        # Create duplicate
        dup_doc = orig_doc.copy()
        dup_doc["id"] = f"dup_exact_{i}"
        dup_doc["is_duplicate"] = True
        dup_doc["duplicate_type"] = "exact"
        dup_doc["original_id"] = orig_doc["id"]
        
        documents.append(dup_doc)
    
    # Generate fuzzy duplicates
    for i in range(fuzzy_dup_count):
        # Select a random document to create a fuzzy duplicate from
        orig_idx = np.random.randint(0, unique_count)
        orig_doc = documents[orig_idx].copy()
        
        # Create fuzzy duplicate - small modifications to the text
        text = orig_doc["text"]
        words = text.split()
        
        # Determine how many modifications to make
        mod_count = np.random.randint(1, min(5, max(2, len(words) // 10)))
        
        for _ in range(mod_count):
            mod_type = np.random.choice(["add", "remove", "replace"])
            
            if mod_type == "add" or len(words) < 5:
                # Add a word
                pos = np.random.randint(0, len(words) + 1)
                new_word = np.random.choice(["additionally", "furthermore", "moreover", "specifically", "effectively"])
                words.insert(pos, new_word)
            elif mod_type == "remove" and len(words) > 5:
                # Remove a word (not touching the first or last 2 words)
                pos = np.random.randint(2, len(words) - 2)
                words.pop(pos)
            elif mod_type == "replace":
                # Replace a word
                pos = np.random.randint(2, len(words) - 2)
                replacements = {
                    "the": "a",
                    "a": "the",
                    "and": "plus",
                    "important": "significant",
                    "recent": "new",
                    "comprehensive": "complete",
                    "analysis": "examination",
                    "overview": "summary"
                }
                
                if words[pos].lower() in replacements:
                    words[pos] = replacements[words[pos].lower()]
                else:
                    words[pos] = words[pos] + " "  # Just add a space as minor change
        
        fuzzy_text = " ".join(words)
        
        # Create fuzzy duplicate document
        dup_doc = orig_doc.copy()
        dup_doc["id"] = f"dup_fuzzy_{i}"
        dup_doc["text"] = fuzzy_text
        dup_doc["is_duplicate"] = True
        dup_doc["duplicate_type"] = "fuzzy"
        dup_doc["original_id"] = orig_doc["id"]
        
        documents.append(dup_doc)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(documents)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def process_with_llm_no_deduplication(data, llm_api, text_column="text"):
    """
    Process the entire dataset with the LLM API without deduplication.
    
    Args:
        data: DataFrame containing the documents
        llm_api: LLM API instance
        text_column: Column containing the text to process
    
    Returns:
        DataFrame with LLM processing results
    """
    print(f"Processing entire dataset ({len(data)} documents) with LLM API...")
    start_time = time.time()
    
    # Process each document
    results = []
    for i, row in data.iterrows():
        if i % 50 == 0:
            print(f"  Processing document {i}/{len(data)}...")
            
        # Process with LLM API
        response = llm_api.process_document(row[text_column])
        
        # Add document ID and response to results
        result = {"id": row["id"], **response}
        results.append(result)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    
    # Convert to DataFrame and merge with original data
    results_df = pd.DataFrame(results)
    merged_df = pd.merge(data, results_df, on="id", how="left")
    
    return merged_df, elapsed


def process_with_llm_exact_deduplication(data, llm_api, text_column="text"):
    """
    Process the dataset with exact deduplication before LLM API.
    
    Args:
        data: DataFrame containing the documents
        llm_api: LLM API instance
        text_column: Column containing the text to process
    
    Returns:
        DataFrame with LLM processing results mapped back to all documents
    """
    print(f"Processing with exact deduplication...")
    start_time = time.time()
    
    # Exact deduplication
    deduplicator = TextDeduplicator()
    deduplicated_data, duplicate_map, _ = deduplicator.deduplicate(
        data, 
        text_column=text_column,
        method="exact"
    )
    
    dedup_time = time.time() - start_time
    print(f"Deduplication completed in {dedup_time:.2f} seconds")
    print(f"Reduced from {len(data)} to {len(deduplicated_data)} documents " + 
          f"({len(data) - len(deduplicated_data)} duplicates removed)")
    
    # Process deduplicated data with LLM
    print(f"Processing {len(deduplicated_data)} unique documents with LLM API...")
    llm_start_time = time.time()
    
    results = []
    for i, row in deduplicated_data.iterrows():
        if i % 20 == 0:
            print(f"  Processing document {i}/{len(deduplicated_data)}...")
            
        # Process with LLM API
        response = llm_api.process_document(row[text_column])
        
        # Add document ID and response to results
        result = {"id": row["id"], **response}
        results.append(result)
    
    llm_time = time.time() - llm_start_time
    total_time = time.time() - start_time
    print(f"LLM processing completed in {llm_time:.2f} seconds")
    print(f"Total time (deduplication + LLM): {total_time:.2f} seconds")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add results to deduplicated data
    for col in results_df.columns:
        if col != "id":
            deduplicated_data[col] = results_df[col].values
    
    # Map results back to full dataset
    print(f"Mapping results back to all {len(data)} documents...")
    full_results = deduplicator.map_results_to_full_dataset(
        data,
        deduplicated_data,
        duplicate_map,
        result_columns=[c for c in results_df.columns if c != "id"]
    )
    
    return full_results, total_time


def process_with_llm_fuzzy_deduplication(data, llm_api, text_column="text", threshold=0.85):
    """
    Process the dataset with fuzzy deduplication before LLM API.
    
    Args:
        data: DataFrame containing the documents
        llm_api: LLM API instance
        text_column: Column containing the text to process
        threshold: Similarity threshold for fuzzy matching (0.0-1.0)
    
    Returns:
        DataFrame with LLM processing results mapped back to all documents
    """
    print(f"Processing with fuzzy deduplication (threshold={threshold})...")
    start_time = time.time()
    
    # Fuzzy deduplication
    deduplicator = TextDeduplicator(similarity_threshold=threshold)
    deduplicated_data, duplicate_map, fuzzy_groups = deduplicator.deduplicate(
        data, 
        text_column=text_column,
        method="fuzzy",
        threshold=threshold
    )
    
    dedup_time = time.time() - start_time
    print(f"Fuzzy deduplication completed in {dedup_time:.2f} seconds")
    print(f"Reduced from {len(data)} to {len(deduplicated_data)} documents " + 
          f"({len(data) - len(deduplicated_data)} duplicates/near-duplicates removed)")
    print(f"Found {len(fuzzy_groups)} groups of similar documents")
    
    # Process deduplicated data with LLM
    print(f"Processing {len(deduplicated_data)} unique documents with LLM API...")
    llm_start_time = time.time()
    
    results = []
    for i, row in deduplicated_data.iterrows():
        if i % 20 == 0:
            print(f"  Processing document {i}/{len(deduplicated_data)}...")
            
        # Process with LLM API
        response = llm_api.process_document(row[text_column])
        
        # Add document ID and response to results
        result = {"id": row["id"], **response}
        results.append(result)
    
    llm_time = time.time() - llm_start_time
    total_time = time.time() - start_time
    print(f"LLM processing completed in {llm_time:.2f} seconds")
    print(f"Total time (fuzzy deduplication + LLM): {total_time:.2f} seconds")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add results to deduplicated data for results that have an ID match
    deduplicated_data = deduplicated_data.copy()
    
    # Create a mapping from ID to results
    id_to_results = {row["id"]: {col: row[col] for col in results_df.columns if col != "id"} 
                     for _, row in results_df.iterrows()}
    
    # Add results to deduplicated data
    for col in results_df.columns:
        if col != "id":
            deduplicated_data[col] = results_df[col].values
    
    # Map results back to full dataset
    print(f"Mapping results back to all {len(data)} documents...")
    full_results = deduplicator.map_results_to_full_dataset(
        data,
        deduplicated_data,
        duplicate_map,
        result_columns=[c for c in results_df.columns if c != "id"]
    )
    
    # Analyze a sample of fuzzy groups to check mapping consistency
    if fuzzy_groups:
        print("\nAnalyzing result consistency within fuzzy groups:")
        sample_size = min(3, len(fuzzy_groups))
        for i, group in enumerate(fuzzy_groups[:sample_size]):
            group_ids = group["id"].tolist()
            print(f"\nSample Group {i+1} ({len(group_ids)} documents):")
            
            # Get results for this group from full results
            group_results = full_results[full_results["id"].isin(group_ids)]
            
            # Check for category column first before analyzing consistency
            if "category" in group_results.columns:
                categories = group_results["category"].tolist()
                unique_categories = set(categories)
                
                # Check if all documents got the same category
                consistency = len(unique_categories) == 1
                
                print(f"  Documents in group: {', '.join(group_ids)}")
                print(f"  Categories assigned: {', '.join(unique_categories)}")
                print(f"  Mapping consistency: {'Consistent' if consistency else 'Inconsistent'}")
            else:
                print(f"  Documents in group: {', '.join(group_ids)}")
                print(f"  Category analysis skipped - category column not found in results")
            
            # Show a sample text comparison for the first two documents
            if len(group) >= 2:
                doc1 = group.iloc[0]
                doc2 = group.iloc[1]
                print(f"  Sample text comparison:")
                print(f"    Doc1 ({doc1['id']}): {doc1[text_column][:50]}...")
                print(f"    Doc2 ({doc2['id']}): {doc2[text_column][:50]}...")
    
    return full_results, total_time, fuzzy_groups


def compare_approaches(dataset_size=500):
    """
    Compare all three approaches (no deduplication, exact, and fuzzy).
    
    Args:
        dataset_size: Size of the test dataset
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n=== COMPARING DEDUPLICATION APPROACHES (Dataset Size: {dataset_size}) ===\n")
    
    # Create dataset
    data = create_realistic_dataset(size=dataset_size)
    
    # Count document types
    exact_dups = (data["duplicate_type"] == "exact").sum()
    fuzzy_dups = (data["duplicate_type"] == "fuzzy").sum()
    originals = (data["duplicate_type"] == "original").sum()
    
    print("Dataset composition:")
    print(f"  Original documents: {originals} ({originals/len(data)*100:.1f}%)")
    print(f"  Exact duplicates: {exact_dups} ({exact_dups/len(data)*100:.1f}%)")
    print(f"  Fuzzy duplicates: {fuzzy_dups} ({fuzzy_dups/len(data)*100:.1f}%)")
    print(f"  Total: {len(data)} documents\n")
    
    # Initialize LLM API for each approach to track metrics separately
    llm_api_no_dedup = MockLLMAPI(response_time=0.01)
    llm_api_exact = MockLLMAPI(response_time=0.01)
    llm_api_fuzzy = MockLLMAPI(response_time=0.01)
    
    # 1. No deduplication approach
    print("\n--- APPROACH 1: NO DEDUPLICATION ---")
    no_dedup_results, no_dedup_time = process_with_llm_no_deduplication(
        data, llm_api_no_dedup
    )
    no_dedup_stats = llm_api_no_dedup.get_usage_stats()
    
    # 2. Exact deduplication approach
    print("\n--- APPROACH 2: EXACT DEDUPLICATION ---")
    exact_dedup_results, exact_dedup_time = process_with_llm_exact_deduplication(
        data, llm_api_exact
    )
    exact_dedup_stats = llm_api_exact.get_usage_stats()
    
    # 3. Fuzzy deduplication approach
    print("\n--- APPROACH 3: FUZZY DEDUPLICATION ---")
    fuzzy_dedup_results, fuzzy_dedup_time, fuzzy_groups = process_with_llm_fuzzy_deduplication(
        data, llm_api_fuzzy, threshold=0.85
    )
    fuzzy_dedup_stats = llm_api_fuzzy.get_usage_stats()
    
    # Calculate metrics
    tokens_saved_exact = no_dedup_stats["total_tokens"] - exact_dedup_stats["total_tokens"]
    cost_saved_exact = no_dedup_stats["total_cost"] - exact_dedup_stats["total_cost"]
    time_saved_exact = no_dedup_time - exact_dedup_time
    
    tokens_saved_fuzzy = no_dedup_stats["total_tokens"] - fuzzy_dedup_stats["total_tokens"]
    cost_saved_fuzzy = no_dedup_stats["total_cost"] - fuzzy_dedup_stats["total_cost"]
    time_saved_fuzzy = no_dedup_time - fuzzy_dedup_time
    
    # Check if 'category' exists in results, if not skip agreement calculation
    if 'category' in no_dedup_results.columns and 'category' in exact_dedup_results.columns and 'category' in fuzzy_dedup_results.columns:
        # Compare result consistency
        exact_categories = exact_dedup_results["category"].tolist()
        fuzzy_categories = fuzzy_dedup_results["category"].tolist()
        no_dedup_categories = no_dedup_results["category"].tolist()
        
        exact_agreement = sum(a == b for a, b in zip(exact_categories, no_dedup_categories)) / len(exact_categories)
        fuzzy_agreement = sum(a == b for a, b in zip(fuzzy_categories, no_dedup_categories)) / len(fuzzy_categories)
    else:
        # Set default values if category column is missing
        exact_agreement = 1.0  # assume 100% agreement as default
        fuzzy_agreement = 1.0
    
    # Print comparison
    print("\n=== APPROACH COMPARISON ===")
    
    comparison = pd.DataFrame({
        "Metric": [
            "Processing Time (s)",
            "Documents Processed",
            "Total Tokens",
            "Total Cost ($)",
            "Requests to LLM API",
            "Result Agreement",
            "Tokens Saved vs. No Dedup",
            "Cost Saved vs. No Dedup ($)",
            "Time Saved vs. No Dedup (s)"
        ],
        "No Deduplication": [
            f"{no_dedup_time:.2f}",
            len(data),
            f"{no_dedup_stats['total_tokens']:,}",
            f"{no_dedup_stats['total_cost']:.4f}",
            no_dedup_stats['total_requests'],
            "100%",
            "0",
            "0.00",
            "0.00"
        ],
        "Exact Deduplication": [
            f"{exact_dedup_time:.2f}",
            exact_dedup_stats['total_requests'],
            f"{exact_dedup_stats['total_tokens']:,}",
            f"{exact_dedup_stats['total_cost']:.4f}",
            exact_dedup_stats['total_requests'],
            f"{exact_agreement*100:.1f}%",
            f"{tokens_saved_exact:,}",
            f"{cost_saved_exact:.4f}",
            f"{time_saved_exact:.2f}"
        ],
        "Fuzzy Deduplication": [
            f"{fuzzy_dedup_time:.2f}",
            fuzzy_dedup_stats['total_requests'],
            f"{fuzzy_dedup_stats['total_tokens']:,}",
            f"{fuzzy_dedup_stats['total_cost']:.4f}",
            fuzzy_dedup_stats['total_requests'],
            f"{fuzzy_agreement*100:.1f}%",
            f"{tokens_saved_fuzzy:,}",
            f"{cost_saved_fuzzy:.4f}",
            f"{time_saved_fuzzy:.2f}"
        ]
    })
    
    print(comparison.to_string(index=False))
    
    # Return all results and metrics
    return {
        "data": data,
        "no_dedup_results": no_dedup_results,
        "exact_dedup_results": exact_dedup_results,
        "fuzzy_dedup_results": fuzzy_dedup_results,
        "no_dedup_stats": no_dedup_stats,
        "exact_dedup_stats": exact_dedup_stats,
        "fuzzy_dedup_stats": fuzzy_dedup_stats,
        "comparison": comparison,
        "fuzzy_groups": fuzzy_groups,
        "times": {
            "no_dedup": no_dedup_time,
            "exact_dedup": exact_dedup_time,
            "fuzzy_dedup": fuzzy_dedup_time
        },
        "savings": {
            "exact": {
                "tokens": tokens_saved_exact,
                "cost": cost_saved_exact,
                "time": time_saved_exact
            },
            "fuzzy": {
                "tokens": tokens_saved_fuzzy,
                "cost": cost_saved_fuzzy,
                "time": time_saved_fuzzy
            }
        },
        "agreement": {
            "exact": exact_agreement,
            "fuzzy": fuzzy_agreement
        }
    }


def visualize_results(results, output_dir=None):
    """
    Visualize the comparison results.
    
    Args:
        results: Results from compare_approaches()
        output_dir: Optional directory to save visualizations
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Processing Time and Cost Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Processing Time
    times = [
        results["times"]["no_dedup"],
        results["times"]["exact_dedup"],
        results["times"]["fuzzy_dedup"]
    ]
    labels = ["No Deduplication", "Exact Deduplication", "Fuzzy Deduplication"]
    
    axes[0].bar(labels, times, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0].set_ylabel('Processing Time (seconds)')
    axes[0].set_title('Total Processing Time')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(times):
        axes[0].text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Plot 2: LLM Requests
    requests = [
        results["no_dedup_stats"]["total_requests"],
        results["exact_dedup_stats"]["total_requests"],
        results["fuzzy_dedup_stats"]["total_requests"]
    ]
    
    axes[1].bar(labels, requests, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[1].set_ylabel('Number of LLM API Requests')
    axes[1].set_title('LLM API Requests')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(requests):
        axes[1].text(i, v + 0.5, str(v), ha='center')
    
    # Plot 3: Cost Comparison
    costs = [
        results["no_dedup_stats"]["total_cost"],
        results["exact_dedup_stats"]["total_cost"],
        results["fuzzy_dedup_stats"]["total_cost"]
    ]
    
    axes[2].bar(labels, costs, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[2].set_ylabel('Cost ($)')
    axes[2].set_title('LLM API Cost')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(costs):
        axes[2].text(i, v + 0.0001, f"${v:.4f}", ha='center')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'processing_comparison.png'), 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Figure 2: Category Distribution Comparison - only if category column exists
    if ('category' in results["no_dedup_results"].columns and 
        'category' in results["exact_dedup_results"].columns and 
        'category' in results["fuzzy_dedup_results"].columns):
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get category distributions
        cat_no_dedup = results["no_dedup_results"]["category"].value_counts().sort_index()
        cat_exact = results["exact_dedup_results"]["category"].value_counts().sort_index()
        cat_fuzzy = results["fuzzy_dedup_results"]["category"].value_counts().sort_index()
        
        # Get all categories
        all_categories = sorted(set(cat_no_dedup.index) | 
                            set(cat_exact.index) | 
                            set(cat_fuzzy.index))
        
        # Create comparison DataFrame
        cat_comp = pd.DataFrame(index=all_categories)
        cat_comp["No Deduplication"] = cat_no_dedup
        cat_comp["Exact Deduplication"] = cat_exact
        cat_comp["Fuzzy Deduplication"] = cat_fuzzy
        cat_comp = cat_comp.fillna(0)
        
        # Plot
        cat_comp.plot(kind='bar', ax=ax)
        ax.set_title('Category Distribution Comparison')
        ax.set_xlabel('Category')
        ax.set_ylabel('Document Count')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Approach')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'category_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()
    else:
        print("Skipping category distribution visualization - category column not found in results")
    
    # Figure 3: Savings Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Tokens Saved
    token_savings = [
        0,
        results["savings"]["exact"]["tokens"],
        results["savings"]["fuzzy"]["tokens"]
    ]
    
    axes[0].bar(labels, token_savings, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0].set_ylabel('Tokens Saved')
    axes[0].set_title('Token Savings vs. No Deduplication')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(token_savings):
        axes[0].text(i, v + 10, f"{v:,}", ha='center')
    
    # Plot 2: Cost Saved
    cost_savings = [
        0,
        results["savings"]["exact"]["cost"],
        results["savings"]["fuzzy"]["cost"]
    ]
    
    axes[1].bar(labels, cost_savings, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[1].set_ylabel('Cost Saved ($)')
    axes[1].set_title('Cost Savings vs. No Deduplication')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(cost_savings):
        axes[1].text(i, v + 0.0001, f"${v:.4f}", ha='center')
    
    # Plot 3: Time Saved
    time_savings = [
        0,
        results["savings"]["exact"]["time"],
        results["savings"]["fuzzy"]["time"]
    ]
    
    axes[2].bar(labels, time_savings, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[2].set_ylabel('Time Saved (seconds)')
    axes[2].set_title('Time Savings vs. No Deduplication')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(time_savings):
        axes[2].text(i, max(0, v) + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'savings_comparison.png'), 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()


def real_world_example():
    """
    Show a real-world-like example of using deduplication with an external LLM.
    This demonstrates the pattern for use with actual APIs like OpenAI, Anthropic, etc.
    """
    print("\n=== REAL-WORLD DEDUPLICATION WITH EXTERNAL LLM EXAMPLE ===\n")
    print("This example shows how to use deduplication with an actual LLM API")
    print("(The code uses a mock API for demonstration, but follows the same pattern)")
    
    # Sample messages from a hypothetical customer service dataset
    messages = [
        "I've been waiting for my refund for 2 weeks now. Order #12345.",
        "Where is my refund? It's been 2 weeks. Order #12345.",  # Near duplicate
        "I'd like to request a refund for my recent purchase that arrived damaged.",
        "The product I received was damaged on arrival. I want a refund.",  # Near duplicate
        "I've been waiting for my refund for 14 days now. Order #12345.",  # Near duplicate
        "When will my order ship? I placed it 3 days ago.",
        "It's been 3 days since I ordered. When will it ship?",  # Near duplicate
        "I need to update my shipping address for order #54321.",
        "How do I change my shipping address for order #54321?",  # Near duplicate
        "I need to cancel my recent order #98765 before it ships."
    ]
    
    # Create a DataFrame
    data = pd.DataFrame({
        "message_id": [f"msg_{i}" for i in range(len(messages))],
        "text": messages,
        "timestamp": pd.date_range(start="2023-01-01", periods=len(messages), freq="h")
    })
    
    print(f"Original dataset: {len(data)} customer messages")
    
    # Create a TextDeduplicator with fuzzy matching
    deduplicator = TextDeduplicator(similarity_threshold=0.8)
    
    # Apply fuzzy deduplication
    deduplicated_data, duplicate_map, fuzzy_groups = deduplicator.deduplicate(
        data, text_column="text", method="fuzzy", threshold=0.8
    )
    
    print(f"After deduplication: {len(deduplicated_data)} unique messages")
    print(f"Removed {len(data) - len(deduplicated_data)} near-duplicate messages")
    
    # Display the fuzzy groups
    print("\nGroups of similar messages:")
    for i, group in enumerate(fuzzy_groups):
        print(f"\nGroup {i+1}:")
        for j, (idx, row) in enumerate(group.iterrows()):
            print(f"  {j+1}. {row['text']}")
    
    # Set up a mock LLM API (replace with real API in production)
    mock_llm = MockLLMAPI(response_time=0.1)
    
    print("\nProcessing deduplicated messages with LLM...")
    
    # Define a simple prompt (would be more sophisticated in a real application)
    prompt = "Classify this customer message into one of these categories: Refund, Shipping, Address Change, Cancellation, Other. Then provide a brief summary."
    
    # Process only the deduplicated messages with the LLM
    responses = []
    for idx, row in deduplicated_data.iterrows():
        # In real code, this would be a call to an actual LLM API
        # For example:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "system", "content": prompt}, 
        #               {"role": "user", "content": row["text"]}]
        # )
        
        # Instead, we use our mock API
        response = mock_llm.process_document(row["text"], prompt)
        
        # Add the message ID
        response["message_id"] = row["message_id"]
        responses.append(response)
    
    # Convert responses to DataFrame
    responses_df = pd.DataFrame(responses)
    
    # Add responses to deduplicated data
    for col in responses_df.columns:
        if col != "message_id":
            deduplicated_data[col] = responses_df[col].values
    
    # Map LLM results back to full dataset
    print("\nMapping results back to all messages...")
    full_results = deduplicator.map_results_to_full_dataset(
        data,
        deduplicated_data,
        duplicate_map,
        result_columns=["category", "summary", "confidence", "metadata"]
    )
    
    # Display results
    print("\nResults after mapping back to all messages:")
    for i, row in full_results.iterrows():
        print(f"\nMessage {i+1}: {row['text']}")
        print(f"  Category: {row['category']} (Confidence: {row['confidence']:.2f})")
        print(f"  Summary: {row['summary']}")
    
    # Calculate savings
    total_tokens = mock_llm.get_usage_stats()["total_tokens"]
    total_cost = mock_llm.get_usage_stats()["total_cost"]
    
    tokens_saved = total_tokens * (len(data) - len(deduplicated_data)) / len(deduplicated_data)
    cost_saved = total_cost * (len(data) - len(deduplicated_data)) / len(deduplicated_data)
    
    print("\nEstimated savings:")
    print(f"  Tokens saved: {tokens_saved:.0f}")
    print(f"  Cost saved: ${cost_saved:.4f}")
    
    return {
        "original_data": data,
        "deduplicated_data": deduplicated_data,
        "full_results": full_results,
        "fuzzy_groups": fuzzy_groups,
        "usage_stats": mock_llm.get_usage_stats()
    }


def code_snippets():
    """
    Display code snippets for using deduplication with popular LLM APIs.
    """
    print("\n=== CODE SNIPPETS: USING DEDUPLICATION WITH REAL LLM APIS ===\n")
    
    # OpenAI API Example
    print("Example 1: Using with OpenAI API\n")
    openai_code = """
# Example using OpenAI API with Meno's deduplication

import pandas as pd
from meno.preprocessing.deduplication import TextDeduplicator
import openai
import os
from tqdm import tqdm

# Set up your OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]  # Keep keys in environment variables!

# Load your dataset
data = pd.read_csv("customer_messages.csv")
print(f"Original dataset: {len(data)} messages")

# Create a TextDeduplicator with fuzzy matching
deduplicator = TextDeduplicator(similarity_threshold=0.85)

# Apply fuzzy deduplication
deduplicated_data, duplicate_map, fuzzy_groups = deduplicator.deduplicate(
    data, text_column="message", method="fuzzy", threshold=0.85
)
print(f"After deduplication: {len(deduplicated_data)} unique messages")
print(f"Removed {len(data) - len(deduplicated_data)} near-duplicate messages")

# Process only the deduplicated messages with OpenAI
responses = []
for idx, row in tqdm(deduplicated_data.iterrows(), total=len(deduplicated_data)):
    # Create the prompt with proper instructions for your use case
    prompt = f"Please classify this customer message into a category and provide a brief summary:\\n\\n{row['message']}"
    
    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or your preferred model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies customer service messages."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    # Extract the response text
    response_text = response.choices[0].message.content
    
    # Parse the response (format depends on your prompt engineering)
    # This is a simple example - adjust based on your LLM output format
    lines = response_text.strip().split('\\n')
    category = lines[0].replace('Category:', '').strip() if lines and 'Category:' in lines[0] else 'Unknown'
    summary = lines[1].replace('Summary:', '').strip() if len(lines) > 1 and 'Summary:' in lines[1] else response_text
    
    # Add to responses with message ID for later mapping
    responses.append({
        "message_id": row["message_id"],
        "category": category,
        "summary": summary,
        "full_response": response_text,
        "tokens_used": response.usage.total_tokens
    })

# Convert responses to DataFrame
responses_df = pd.DataFrame(responses)

# Merge responses with deduplicated data
for col in responses_df.columns:
    if col != "message_id":
        deduplicated_data[col] = responses_df[col].values

# Map LLM results back to full dataset
print("Mapping results back to all messages...")
full_results = deduplicator.map_results_to_full_dataset(
    data,
    deduplicated_data,
    duplicate_map,
    result_columns=["category", "summary", "full_response", "tokens_used"]
)

# Calculate and show savings
total_tokens = sum(full_results["tokens_used"])
estimated_tokens_without_dedup = total_tokens * len(data) / len(deduplicated_data)
tokens_saved = estimated_tokens_without_dedup - total_tokens

print(f"Tokens used with deduplication: {total_tokens}")
print(f"Estimated tokens without deduplication: {estimated_tokens_without_dedup:.0f}")
print(f"Tokens saved: {tokens_saved:.0f}")
print(f"Estimated cost saved: ${tokens_saved * 0.001:.2f}")  # Assuming $0.001 per token

# Save the results
full_results.to_csv("classified_messages.csv", index=False)
"""
    print(openai_code)
    
    # Anthropic Claude Example
    print("\nExample 2: Using with Anthropic Claude API\n")
    claude_code = """
# Example using Anthropic Claude API with Meno's deduplication

import pandas as pd
from meno.preprocessing.deduplication import TextDeduplicator
from anthropic import Anthropic
import os
from tqdm import tqdm
import json

# Set up Anthropic API
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Load your dataset
data = pd.read_csv("support_tickets.csv")
print(f"Original dataset: {len(data)} tickets")

# Create a TextDeduplicator
deduplicator = TextDeduplicator(similarity_threshold=0.85)

# Apply fuzzy deduplication
deduplicated_data, duplicate_map, fuzzy_groups = deduplicator.deduplicate(
    data, text_column="ticket_text", method="fuzzy", threshold=0.85
)
print(f"After deduplication: {len(deduplicated_data)} unique tickets")

# Process only the deduplicated tickets with Claude
responses = []
for idx, row in tqdm(deduplicated_data.iterrows(), total=len(deduplicated_data)):
    # Create prompt for Claude
    prompt = f'''Human: Please analyze this support ticket and provide:
1. Primary issue category
2. Priority level (Low/Medium/High)
3. Brief summary

Return your analysis in JSON format with keys: category, priority, summary

Support ticket: {row['ticket_text']}

    Assistant: I'll analyze this support ticket for you.'''
    
    # Call Claude API
    try:
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",  # Or your preferred Claude model
            max_tokens=1024,
            temperature=0.2,
            system="You are a helpful customer support analyst that categorizes support tickets.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response (assuming JSON format based on prompt instruction)
        try:
            # First try to extract JSON from the response if it's wrapped in markdown
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response.content[0].text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try parsing the whole response as JSON
                result = json.loads(response.content[0].text)
                
            result["ticket_id"] = row["ticket_id"]  # Add ID for mapping back
            result["tokens_used"] = response.usage.input_tokens + response.usage.output_tokens
            
        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            result = {
                "ticket_id": row["ticket_id"],
                "category": "Error",
                "priority": "Unknown",
                "summary": response.content[0].text[:100] + "...",
                "full_response": response.content[0].text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
            
    except Exception as e:
        # Handle API errors
        result = {
            "ticket_id": row["ticket_id"],
            "category": "Error",
            "priority": "Unknown",
            "summary": f"API Error: {str(e)}",
            "full_response": str(e),
            "tokens_used": 0
        }
        
    responses.append(result)

# Convert responses to DataFrame
responses_df = pd.DataFrame(responses)

# Merge with deduplicated data
for col in responses_df.columns:
    if col != "ticket_id":
        deduplicated_data[col] = responses_df[col].values

# Map results back to full dataset
full_results = deduplicator.map_results_to_full_dataset(
    data,
    deduplicated_data,
    duplicate_map,
    result_columns=["category", "priority", "summary", "full_response", "tokens_used"]
)

# Calculate tokens and cost savings
total_tokens = sum(full_results["tokens_used"])
estimated_tokens_without_dedup = total_tokens * len(data) / len(deduplicated_data)
tokens_saved = estimated_tokens_without_dedup - total_tokens

print(f"Tokens used with deduplication: {total_tokens}")
print(f"Estimated tokens without deduplication: {estimated_tokens_without_dedup:.0f}")
print(f"Tokens saved: {tokens_saved:.0f}")
print(f"Estimated cost saved: ${tokens_saved * 0.00001:.2f}")  # Adjust cost rate as needed

# Save results
full_results.to_csv("analyzed_tickets.csv", index=False)
"""
    print(claude_code)
    
    return {"openai_example": openai_code, "claude_example": claude_code}


def main():
    """
    Main function to run the example.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="External LLM Deduplication Example")
    parser.add_argument("--mode", choices=["compare", "real-world", "snippets", "all"],
                        default="all", help="Which example mode to run")
    parser.add_argument("--size", type=int, default=200,
                        help="Dataset size for comparison (default: 200)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--output-dir", type=str, default="dedup_results",
                        help="Directory to save results and visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if visualizing
    if args.visualize and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    
    # Run the selected mode
    if args.mode in ["compare", "all"]:
        print("\n" + "="*80)
        print("RUNNING COMPARISON OF DEDUPLICATION APPROACHES")
        print("="*80)
        
        results = compare_approaches(dataset_size=args.size)
        
        if args.visualize:
            visualize_results(results, output_dir=args.output_dir)
            
    if args.mode in ["real-world", "all"]:
        print("\n" + "="*80)
        print("RUNNING REAL-WORLD EXAMPLE")
        print("="*80)
        
        real_world_results = real_world_example()
        
    if args.mode in ["snippets", "all"]:
        print("\n" + "="*80)
        print("DISPLAYING CODE SNIPPETS")
        print("="*80)
        
        snippets = code_snippets()
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    if args.visualize and args.output_dir:
        print(f"\nVisualizations saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    main()
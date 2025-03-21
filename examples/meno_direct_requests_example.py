"""
Example demonstrating direct API requests with full Meno functionality.

This example shows how to:
1. Use direct API requests with the requests library
2. Integrate with Meno's caching and deduplication features 
3. Process batches of texts efficiently with deduplication and caching
4. Configure API endpoint, model, and other parameters simply

The goal is to provide a simple entry point for users who need to use direct 
requests while benefiting from Meno's optimization features.
"""

import requests
import time
import json
import os
import hashlib
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union

# Configuration - Change these to your values
API_KEY = "your_api_key_here"  # Your API key for the LLM service
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"  # API endpoint
MODEL = "gpt-3.5-turbo"  # Model to use
API_VERSION = None  # Only needed for some APIs

# Cache settings
ENABLE_CACHE = True  # Whether to use response caching (saves API costs)
CACHE_TTL = 24 * 60 * 60  # Cache time-to-live in seconds (24 hours)
CACHE_DIR = Path.home() / ".meno" / "llm_cache"  # Where to store cache files

# Memory cache for faster repeat access
_MEMORY_CACHE = {}

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(text: str, model: str, system_prompt: str) -> str:
    """Generate a unique cache key from the request parameters."""
    # Create a string that combines all parameters that make a unique request
    combined = f"{text}|{model}|{system_prompt}"
    # Hash it to create a fixed-length key that's safe for filenames
    return hashlib.md5(combined.encode()).hexdigest()

def get_from_cache(cache_key: str) -> Optional[str]:
    """Get a cached response if it exists and is still valid."""
    # Check memory cache first
    if cache_key in _MEMORY_CACHE:
        timestamp, value = _MEMORY_CACHE[cache_key]
        if time.time() < timestamp:  # Still valid
            return value
        else:  # Expired
            del _MEMORY_CACHE[cache_key]
    
    # Check disk cache
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check expiration
            if cached_data.get('expires', 0) > time.time():
                # Add to memory cache for faster access next time
                _MEMORY_CACHE[cache_key] = (cached_data['expires'], cached_data['value'])
                return cached_data['value']
            else:
                # Expired - delete the file
                cache_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, KeyError, IOError):
            # Invalid cache file, ignore it
            cache_file.unlink(missing_ok=True)
    
    return None

def add_to_cache(cache_key: str, value: str, ttl: int = CACHE_TTL) -> None:
    """Add a response to both memory and disk cache."""
    expires = time.time() + ttl
    
    # Add to memory cache
    _MEMORY_CACHE[cache_key] = (expires, value)
    
    # Add to disk cache for persistence
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'expires': expires,
                'value': value,
                'created': time.time()
            }, f)
    except IOError:
        # If we can't write to disk, just keep the memory cache
        print(f"Warning: Could not write to cache file: {cache_file}")

def generate_with_llm(
    text: str, 
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 60, 
    enable_cache: bool = ENABLE_CACHE,
    temperature: float = 0.7,
    max_tokens: int = 150
) -> str:
    """
    Make a direct API call to generate a response from the given text.
    Includes caching for efficiency.
    
    Args:
        text: The user input text to process
        system_prompt: The system prompt to use
        timeout: Request timeout in seconds
        enable_cache: Whether to use response caching
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        
    Returns:
        The generated response text or an error message
    """
    # Generate a cache key from all relevant parameters
    cache_key = get_cache_key(text, MODEL, system_prompt)
    
    # Check cache first if enabled
    if enable_cache:
        cached_result = get_from_cache(cache_key)
        if cached_result is not None:
            print(f"Cache hit for prompt: {text[:30]}...")
            return cached_result
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add API version if provided (needed for some APIs)
    if API_VERSION:
        if "?" in API_ENDPOINT:
            full_endpoint = f"{API_ENDPOINT}&api-version={API_VERSION}"
        else:
            full_endpoint = f"{API_ENDPOINT}?api-version={API_VERSION}"
    else:
        full_endpoint = API_ENDPOINT
    
    try:
        response = requests.post(
            full_endpoint, 
            headers=headers, 
            json=payload,
            timeout=timeout
        )
        
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        response_data = response.json()
        
        if not response_data.get('choices') or len(response_data['choices']) == 0:
            result = "[No response generated.]"
        else:
            result = response_data['choices'][0]['message']['content'].strip()
        
        # Cache the result if caching is enabled
        if enable_cache:
            add_to_cache(cache_key, result)
            
        return result
        
    except requests.exceptions.Timeout:
        return "[Error: Request timed out]"
    except requests.exceptions.RequestException as e:
        return f"[Error: {e}]"
    except ValueError as e:  # JSON parsing error
        return f"[Error: Invalid response format - {e}]"
    except Exception as e:
        return f"[Error: Unexpected error - {e}]"

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity ratio between two strings."""
    # Early termination for very different length texts
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Quick length-based filter to avoid unnecessary computation
    # If lengths differ by more than 30%, they're likely different
    if abs(len1 - len2) / max(len1, len2) > 0.3:
        return 0.0
    
    # Quick check for exact match
    if text1 == text2:
        return 1.0
        
    # Quick check for very different texts by comparing the first N chars
    preview_len = min(50, min(len1, len2))
    if text1[:preview_len].lower() != text2[:preview_len].lower() and \
       SequenceMatcher(None, text1[:preview_len], text2[:preview_len]).ratio() < 0.5:
        return 0.0
        
    # For texts that pass the quick checks, perform full comparison
    return SequenceMatcher(None, text1, text2).ratio()

def identify_duplicates(
    texts: List[str],
    threshold: float = 0.92,
    max_comparisons: Optional[int] = None,
    preprocess: bool = True
) -> Dict[int, int]:
    """
    Identify fuzzy duplicates in a list of texts.
    
    This utility function finds texts that are similar to each other based on a 
    similarity threshold. Use this for deduplication before sending texts to LLMs
    to save on API costs.
    
    Args:
        texts: List of text strings to check for duplicates
        threshold: Similarity threshold (0.0-1.0), by default 0.92
            Higher values are more strict (require more similarity to consider duplicates)
        max_comparisons: Maximum number of comparisons to perform (for very large datasets)
            If None, all pairs will be compared
        preprocess: Whether to preprocess texts for faster comparison
            
    Returns:
        Dictionary mapping duplicate indices to their representative index
    """
    if not texts:
        return {}
        
    n_texts = len(texts)
    
    # Calculate total possible comparisons
    total_comparisons = (n_texts * (n_texts - 1)) // 2
    
    # If max_comparisons is set and less than total, sample proportionally
    sampling_enabled = max_comparisons is not None and max_comparisons < total_comparisons
    
    # Preprocess texts for faster comparison if requested
    if preprocess:
        compare_texts = []
        for text in texts:
            # Simple preprocessing - lowercase and trim spaces
            simplified = text.lower().strip()
            # Remove common punctuation
            for char in ',.:;?!':
                simplified = simplified.replace(char, '')
            compare_texts.append(simplified)
    else:
        compare_texts = texts
    
    # Create length-based groups for optimization
    length_groups = defaultdict(list)
    for i, text in enumerate(compare_texts):
        # Group texts by length ranges (each range is ~10% of the text length)
        length_range = len(text) // 10
        length_groups[length_range].append(i)
    
    duplicate_map = {}
    processed = set()
    
    # Process each length group
    for length_range, indices in length_groups.items():
        # Skip if no texts in this range
        if not indices:
            continue
            
        # Check adjacent length ranges too (handle boundary cases)
        adjacent_indices = []
        for adj_range in [length_range-1, length_range, length_range+1]:
            if adj_range in length_groups:
                adjacent_indices.extend(length_groups[adj_range])
        
        # Compare texts within this length group and adjacent groups
        for idx, i in enumerate(indices):
            if i in processed:
                continue
                
            processed.add(i)
            text1 = compare_texts[i]
            
            # Determine comparison targets
            comparison_targets = [j for j in adjacent_indices if j > i and j not in processed]
            
            # Sample if needed
            if sampling_enabled and len(comparison_targets) > max_comparisons:
                import random
                comparison_targets = random.sample(comparison_targets, max_comparisons)
            
            for j in comparison_targets:
                text2 = compare_texts[j]
                
                # Calculate similarity
                similarity = calculate_text_similarity(text1, text2)
                
                # If similar enough, mark as duplicate
                if similarity >= threshold:
                    duplicate_map[j] = i
                    processed.add(j)
    
    return duplicate_map

def process_texts_with_llm(
    texts: List[str], 
    system_prompt: str = "You are a helpful assistant.",
    max_workers: Optional[int] = None, 
    timeout: int = 60,
    deduplicate: bool = True,
    deduplication_threshold: float = 0.92,
    enable_cache: bool = ENABLE_CACHE,
    show_progress: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 150
) -> List[Dict[str, Any]]:
    """
    Process multiple texts with deduplication, parallel execution, and caching.
    
    Combines all of Meno's optimization features (fuzzy deduplication, parallel processing,
    caching) using direct API calls.
    
    Args:
        texts: List of text prompts to process
        system_prompt: The system prompt to use
        max_workers: Maximum number of worker threads (None = auto-determined)
        timeout: Request timeout in seconds
        deduplicate: Whether to perform deduplication
        deduplication_threshold: Similarity threshold for deduplication
        enable_cache: Whether to use response caching
        show_progress: Whether to print progress information
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        
    Returns:
        List of dictionaries containing the input text, response, and metadata
    """
    if not texts:
        return []
    
    start_time = time.time()
    
    # Deduplication to reduce API calls    
    if deduplicate:
        # Identify duplicates with optimized algorithm
        duplicate_map = identify_duplicates(
            texts, 
            threshold=deduplication_threshold,
            preprocess=True
        )
        
        # Create a set of unique text indices
        unique_indices = set(range(len(texts)))
        for dup_idx in duplicate_map:
            unique_indices.discard(dup_idx)
            
        # Create list of unique texts for processing
        unique_texts = [texts[idx] for idx in sorted(unique_indices)]
        
        if show_progress:
            dedup_time = time.time() - start_time
            print(f"Deduplication reduced {len(texts)} texts to {len(unique_texts)} unique texts (took {dedup_time:.2f}s)")
    else:
        unique_texts = texts
        unique_indices = set(range(len(texts)))
        duplicate_map = {}
    
    # Process unique texts with ThreadPoolExecutor
    results = []
    
    # Create a lock for thread-safe progress updates
    progress_lock = threading.Lock()
    completed_count = 0
    cache_count = 0
    
    def process_single_text(text: str, index: int) -> Dict[str, Any]:
        nonlocal completed_count, cache_count
        
        # Check if we already have this in cache
        cache_key = get_cache_key(text, MODEL, system_prompt)
        from_cache = False
        
        if enable_cache:
            cached_result = get_from_cache(cache_key)
            if cached_result is not None:
                from_cache = True
                # Thread-safe increment of cache count
                with progress_lock:
                    cache_count += 1
        
        start_time = time.time()
        
        # Only make API call if not in cache
        if from_cache:
            response = cached_result
        else:
            response = generate_with_llm(
                text=text,
                system_prompt=system_prompt,
                timeout=timeout,
                enable_cache=enable_cache,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        end_time = time.time()
        
        result = {
            "index": index,
            "input": text,
            "response": response,
            "time_taken": end_time - start_time,
            "success": not response.startswith("[Error:"),
            "from_cache": from_cache
        }
        
        # Update progress with thread safety
        if show_progress:
            with progress_lock:
                completed_count += 1
                status = "✓" if result["success"] else "✗"
                cache_status = "(cached)" if from_cache else ""
                print(f"Completed {completed_count}/{len(unique_texts)}: {status} {cache_status}")
        
        return result
    
    # Determine optimal number of workers if not specified
    if max_workers is None:
        # Use min(32, os.cpu_count() + 4) as recommended for IO-bound tasks
        import os
        default_workers = min(32, (os.cpu_count() or 4) + 4)
        # But also consider the number of texts (no need for more workers than texts)
        max_workers = min(default_workers, len(unique_texts))
    
    # Using ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = [
            executor.submit(process_single_text, text, i) 
            for i, text in enumerate(unique_texts)
        ]
        
        # Collect results as they complete
        unique_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                unique_results.append(result)
            except Exception as e:
                # Find the index of the failed future
                for i, f in enumerate(futures):
                    if f == future:
                        idx = i
                        break
                else:
                    idx = len(unique_results)
                
                # Add error result
                unique_results.append({
                    "index": idx,
                    "input": unique_texts[idx] if idx < len(unique_texts) else "Unknown",
                    "response": f"[Error: Thread execution failed - {e}]",
                    "time_taken": 0,
                    "success": False,
                    "from_cache": False
                })
    
    # If no deduplication was done, return results directly
    if not deduplicate:
        return sorted(unique_results, key=lambda x: x["index"])
    
    # Map results back to include duplicates
    final_results = []
    
    # Create a map from unique indices to results
    orig_indices = sorted(unique_indices)
    idx_to_orig_idx = {i: orig_idx for i, orig_idx in enumerate(orig_indices)}
    index_to_result = {r["index"]: r for r in unique_results}
    
    # Fill in results for all original texts
    for i in range(len(texts)):
        if i in unique_indices:
            # Find the corresponding result in unique_results
            idx_in_unique = orig_indices.index(i)
            result = index_to_result[idx_in_unique].copy()
            result["index"] = i  # Restore original index
            final_results.append(result)
        else:
            # This is a duplicate, copy from its source with adjusted metadata
            source_idx = duplicate_map[i]
            idx_in_unique = orig_indices.index(source_idx)
            source_result = index_to_result[idx_in_unique].copy()
            
            # Update the metadata for this duplicate
            source_result["index"] = i
            source_result["input"] = texts[i]
            source_result["is_duplicate"] = True
            source_result["duplicate_of"] = source_idx
            source_result["time_taken"] = 0  # No API call was made
            
            final_results.append(source_result)
    
    # Sort results by original index to maintain order
    final_results = sorted(final_results, key=lambda x: x["index"])
    
    # Show summary if requested
    if show_progress:
        total_time = time.time() - start_time
        duplicate_count = len(duplicate_map)
        api_call_count = len(unique_texts) - cache_count
        
        print(f"\nPerformance Summary:")
        print(f"- Total texts processed: {len(texts)}")
        print(f"- Duplicates detected: {duplicate_count} ({duplicate_count/len(texts)*100:.1f}%)")
        print(f"- Cache hits: {cache_count} ({cache_count/len(unique_texts)*100:.1f}% of unique texts)")
        print(f"- API calls made: {api_call_count}")
        print(f"- Total processing time: {total_time:.2f}s")
        
        # Calculate the estimated API cost savings
        if api_call_count > 0:
            saved_calls = duplicate_count + cache_count
            saved_percentage = saved_calls / len(texts) * 100
            print(f"- Estimated API call savings: {saved_calls} calls ({saved_percentage:.1f}%)")
    
    return final_results

def categorize_texts(texts: List[str], categories: List[str]) -> List[str]:
    """
    Categorize a list of texts into predefined categories using the LLM.
    Uses deduplication and caching for efficiency.
    
    Args:
        texts: List of texts to categorize
        categories: List of possible categories
        
    Returns:
        List of assigned categories
    """
    categories_str = ", ".join(categories)
    
    system_prompt = f"""You are a text categorization assistant. 
    Categorize each text into exactly one of these categories: {categories_str}.
    Respond with ONLY the category name, nothing else."""
    
    prompts = []
    for text in texts:
        prompts.append(f"Categorize this text into one of these categories ({categories_str}):\n\n{text}")
    
    print(f"Categorizing {len(texts)} texts...")
    results = process_texts_with_llm(
        texts=prompts,
        system_prompt=system_prompt,
        deduplicate=True,
        deduplication_threshold=0.92,
        enable_cache=True,
        show_progress=True,
        temperature=0.3  # Lower temperature for more consistent categorization
    )
    
    # Extract the assigned categories
    assigned_categories = []
    for i, result in enumerate(results):
        response = result["response"]
        
        # Check if the response matches a category exactly
        if response in categories:
            assigned_categories.append(response)
        else:
            # Try to find a matching category
            matched = False
            for category in categories:
                if category.lower() in response.lower():
                    assigned_categories.append(category)
                    matched = True
                    break
            
            if not matched:
                print(f"Warning: Could not match response '{response}' to a category for text {i+1}")
                assigned_categories.append("uncategorized")
    
    return assigned_categories

def generate_topic_labels(keyword_groups: List[List[str]]) -> List[str]:
    """
    Generate topic labels for groups of keywords.
    Uses deduplication and caching for efficiency.
    
    Args:
        keyword_groups: List of keyword groups to generate labels for
        
    Returns:
        List of topic labels
    """
    system_prompt = """You are a topic labeling expert. 
    Create a concise, descriptive label (1-3 words) for each set of keywords.
    The label should capture the core theme represented by the keywords."""
    
    prompts = []
    for keywords in keyword_groups:
        keywords_str = ", ".join(keywords)
        prompts.append(f"Create a concise topic label (1-3 words) for these keywords: {keywords_str}")
    
    print(f"Generating labels for {len(keyword_groups)} keyword groups...")
    results = process_texts_with_llm(
        texts=prompts,
        system_prompt=system_prompt,
        deduplicate=True,
        deduplication_threshold=0.85,  # Allow for more fuzzy matching of similar keyword groups
        enable_cache=True,
        show_progress=True
    )
    
    # Extract the labels
    labels = [result["response"] for result in results]
    return labels

# Example usage
if __name__ == "__main__":
    print("MENO DIRECT REQUESTS INTEGRATION EXAMPLE")
    print("Note: This example requires a valid API key.")
    
    if API_KEY == "your_api_key_here":
        print("Please update the API_KEY variable at the top of this file to run the example.")
        exit()
    
    # Example 1: Simple text generation
    print("\n=== Simple Text Generation Example ===")
    text = "What are three interesting facts about machine learning?"
    
    print(f"Generating response for: {text}")
    response = generate_with_llm(text)
    print(f"\nResponse: {response}")
    
    # Example 2: Text categorization with deduplication and caching
    print("\n=== Text Categorization Example ===")
    categories = ["business", "technology", "health", "entertainment", "sports"]
    
    texts = [
        # Technology group (similar)
        "Apple announced its newest iPhone with improved camera and battery life.",
        "Apple just revealed their latest iPhone model with better photos and longer battery.",
        "The tech giant unveiled a new smartphone with enhanced capabilities.",
        
        # Sports (distinct)
        "The basketball team won the championship after an intense overtime period.",
        
        # Health (distinct)
        "Researchers discovered a new treatment that could help patients with diabetes.",
        
        # Entertainment (similar)
        "The streaming service released a new comedy series that critics are praising.",
        "A popular streaming platform launched a comedy show that reviewers love."
    ]
    
    categories_result = categorize_texts(texts, categories)
    
    print("\nCategorization Results:")
    for i, (text, category) in enumerate(zip(texts, categories_result)):
        print(f"{i+1}. Text: {text[:50]}...")
        print(f"   Category: {category}")
    
    # Example 3: Topic labeling with deduplication and caching
    print("\n=== Topic Labeling Example ===")
    keyword_groups = [
        # Finance group (similar)
        ["finance", "banking", "investment", "stocks", "money"],
        ["investment", "finance", "stocks", "banking", "wealth"],  # Same keywords, different order
        
        # AI group (distinct)
        ["machine learning", "artificial intelligence", "neural networks", "deep learning"],
        
        # Environment group (similar)
        ["climate", "environment", "sustainability", "renewable", "green"],
        ["environment", "sustainability", "ecology", "green energy"],  # Similar theme
        
        # Technology group (distinct)
        ["smartphone", "mobile", "app", "android", "ios"]
    ]
    
    labels = generate_topic_labels(keyword_groups)
    
    print("\nTopic Labeling Results:")
    for i, (keywords, label) in enumerate(zip(keyword_groups, labels)):
        print(f"{i+1}. Keywords: {', '.join(keywords)}")
        print(f"   Label: {label}")
    
    print("\nAll examples completed!")
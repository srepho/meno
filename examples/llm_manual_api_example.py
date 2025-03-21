"""
Example demonstrating direct LLM API usage with Meno's utility functions.

This example shows how to:
1. Make direct API calls to OpenAI without using the full LLMTopicLabeler class
2. Process multiple texts concurrently with ThreadPoolExecutor
3. Use fuzzy deduplication to save on API costs by only processing unique content
4. Use caching to avoid redundant API calls and reduce costs
5. Use the optimized deduplication algorithm for large datasets
6. Use the generate_text_with_llm function which provides a consistent interface for both
   standard OpenAI and Azure OpenAI APIs
"""

from meno.modeling.llm_topic_labeling import (
    generate_call_from_text,
    process_texts_with_threadpool,
    generate_text_with_llm,
    identify_fuzzy_duplicates,
    process_texts_with_deduplication
)
import time
import os
from pathlib import Path

# If you want to try this example, replace with your actual API key
API_KEY = "your_openai_api_key_here"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"  # Default model, can be changed

# Cache directory for demonstration (uncomment to use custom path)
# os.environ["MENO_CACHE_DIR"] = str(Path.home() / ".meno" / "custom_cache")


def demo_direct_api_call():
    """Demonstrate a single direct API call with caching."""
    print("\n=== Single Direct API Call with Caching Example ===")

    # Simple topic request
    text = "Please provide a concise name for this topic based on these keywords: healthcare, doctor, nurse, hospital, patient"
    system_prompt = "You are a topic labeling expert that can identify concise topic names from keywords."
    
    print("First call (will use API):")
    start_time = time.time()
    result1 = generate_call_from_text(
        text=text,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model=MODEL,  # Using a lower-cost model for this example
        system_prompt=system_prompt,
        enable_cache=True,      # Enable caching (default)
        cache_ttl=3600          # Cache for 1 hour (default is 24 hours)
    )
    time1 = time.time() - start_time
    
    print(f"Input: {text}")
    print(f"Topic Name: {result1}")
    print(f"Processing time: {time1:.2f} seconds")
    
    # Make the same call again to demonstrate caching
    print("\nSecond call with identical parameters (should use cache):")
    start_time = time.time()
    result2 = generate_call_from_text(
        text=text,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model=MODEL,
        system_prompt=system_prompt,
        enable_cache=True
    )
    time2 = time.time() - start_time
    
    print(f"Topic Name: {result2}")
    print(f"Processing time: {time2:.2f} seconds")
    print(f"Speed improvement: {time1/time2:.1f}x faster with caching")


def demo_concurrent_processing():
    """Demonstrate concurrent processing of multiple texts with caching."""
    print("\n=== Concurrent Text Processing with Caching Example ===")
    
    # Sample topic keyword sets to process
    texts = [
        "Keywords: finance, bank, investment, money, budget, savings",
        "Keywords: travel, vacation, tourism, hotel, flight, destination",
        "Keywords: technology, computer, software, hardware, programming, algorithm",
        "Keywords: food, cooking, recipe, restaurant, chef, ingredient"
    ]
    
    system_prompt = "You are a topic labeling expert. Generate a concise 1-3 word topic name."
    
    print(f"First batch - Processing {len(texts)} texts concurrently...")
    start_time = time.time()
    results1 = process_texts_with_threadpool(
        texts=texts,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model=MODEL,
        system_prompt=system_prompt,
        max_workers=4,       # Process all texts simultaneously
        enable_cache=True,   # Enable caching
        show_progress=True   # Show progress during processing
    )
    total_time1 = time.time() - start_time
    
    # Process the same texts again to demonstrate caching
    print(f"\nSecond batch - Processing same {len(texts)} texts again (should use cache)...")
    start_time = time.time()
    results2 = process_texts_with_threadpool(
        texts=texts,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model=MODEL,
        system_prompt=system_prompt,
        max_workers=4,
        enable_cache=True,
        show_progress=True
    )
    total_time2 = time.time() - start_time
    
    # Report on the difference
    print(f"\nCaching Performance:")
    print(f"- First run (API calls): {total_time1:.2f} seconds")
    print(f"- Second run (cached): {total_time2:.2f} seconds")
    print(f"- Speed improvement: {total_time1/total_time2:.1f}x faster with caching")
    
    # Check cache status in the results
    cached_count = sum(1 for r in results2 if r.get("from_cache", False))
    print(f"- Cache hits: {cached_count}/{len(texts)} ({cached_count/len(texts)*100:.0f}%)")


def demo_optimized_deduplication():
    """Demonstrate optimized fuzzy deduplication algorithm."""
    print("\n=== Optimized Fuzzy Deduplication Example ===")
    
    # Create a larger list with similar texts to demonstrate optimization
    base_texts = [
        "Keywords: finance, bank, investment, money, budget, savings",
        "Keywords: travel, vacation, tourism, hotel, flight, destination", 
        "Keywords: technology, computers, software, programming, development",
        "Keywords: healthcare, doctor, medicine, hospital, patient"
    ]
    
    # Generate variations to create a larger dataset with duplicates
    texts = []
    for base in base_texts:
        texts.append(base)  # Original
        
        # Add variations (similar to original)
        words = base.replace("Keywords: ", "").split(", ")
        texts.append(f"Keywords: {', '.join(words[::-1])}")  # Reversed order
        texts.append(f"Keywords: {', '.join(sorted(words))}")  # Alphabetical order
        texts.append(f"Topic words: {', '.join(words)}")  # Different prefix
        
    # Add some completely different texts
    texts.extend([
        "Keywords: education, school, learning, student, teacher, classroom",
        "Keywords: entertainment, movie, film, actor, cinema, television",
        "Keywords: politics, government, election, policy, democracy"
    ])
    
    # Benchmark standard deduplication
    start_time = time.time()
    print(f"Running standard deduplication on {len(texts)} texts...")
    duplicates1 = identify_fuzzy_duplicates(texts, threshold=0.8)
    time1 = time.time() - start_time
    
    # Benchmark optimized deduplication
    start_time = time.time()
    print(f"Running optimized deduplication on {len(texts)} texts...")
    
    # Create simplified texts for faster comparison
    simplified_texts = []
    for text in texts:
        # Simple preprocessing - lowercase and remove punctuation
        simplified = text.lower().strip()
        for char in ',.;:?!':
            simplified = simplified.replace(char, '')
        simplified_texts.append(simplified)
    
    duplicates2 = identify_fuzzy_duplicates(
        texts, 
        threshold=0.8,
        simplified_texts=simplified_texts  # Pass preprocessed texts
    )
    time2 = time.time() - start_time
    
    # Print results
    print(f"\nStandard deduplication:")
    print(f"- Found {len(duplicates1)} duplicates in {time1:.4f} seconds")
    
    print(f"\nOptimized deduplication:")
    print(f"- Found {len(duplicates2)} duplicates in {time2:.4f} seconds")
    print(f"- Speed improvement: {time1/time2:.1f}x faster")
    
    # Show some example duplicates
    print("\nExample duplicates found:")
    count = 0
    for dup_idx, source_idx in duplicates2.items():
        print(f"Text {dup_idx + 1} is similar to Text {source_idx + 1}")
        print(f"  Original: {texts[source_idx]}")
        print(f"  Duplicate: {texts[dup_idx]}")
        print()
        count += 1
        if count >= 3:  # Show only first 3 examples
            break


def demo_integrated_deduplication_and_caching():
    """Demonstrate combining deduplication and caching for maximum efficiency."""
    print("\n=== Integrated Deduplication and Caching Example ===")
    
    # Create a test dataset with duplicates and repetitions
    texts = [
        # Finance group (similar texts)
        "Keywords: finance, bank, investment, money, budget, savings",
        "Keywords: banking, investment, finance, budget, money, savings",  # Similar to first
        "Keywords: financial, banking, investing, budgeting, monetary",    # Similar to first
        
        # Travel group (similar texts)
        "Keywords: travel, vacation, tourism, hotel, flight, destination",
        "Keywords: vacation, tourism, travel, hotels, flights, destinations",  # Similar
        "Keywords: tourism, travel, vacations, hotels, flights",  # Similar
        
        # Technology group (similar texts)
        "Keywords: technology, computer, software, hardware, programming, algorithm",
        "Keywords: computers, hardware, software, coding, programming",  # Similar
        
        # Unique texts
        "Keywords: food, cooking, recipe, restaurant, chef, ingredient",
        "Keywords: sports, athletics, competition, exercise, fitness"
    ]
    
    # Process with full optimization
    print(f"Processing {len(texts)} texts with deduplication and caching...")
    system_prompt = "You are a topic labeling expert. Generate a concise 1-3 word topic name."
    
    start_time = time.time()
    results = process_texts_with_deduplication(
        texts=texts,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model=MODEL,
        system_prompt=system_prompt,
        deduplicate=True,                  # Enable deduplication
        deduplication_threshold=0.8,       # Similarity threshold (lower = more aggressive)
        enable_cache=True,                 # Enable response caching
        preprocess_for_deduplication=True, # Enable text preprocessing for faster comparison
        show_progress=True                 # Show detailed progress and statistics
    )
    
    # Run it again to demonstrate full caching
    print(f"\nProcessing same {len(texts)} texts again (should be fully cached)...")
    results2 = process_texts_with_deduplication(
        texts=texts,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model=MODEL,
        system_prompt=system_prompt,
        deduplicate=True,
        deduplication_threshold=0.8,
        enable_cache=True,
        show_progress=True
    )


def demo_structured_api_interface():
    """Demonstrate the more structured generate_text_with_llm function."""
    print("\n=== Unified API Interface Example ===")
    
    # Example 1: Standard OpenAI API
    text = "Analyze these keywords and provide a topic name: environment, climate, sustainability, recycling, green"
    
    print("Using standard OpenAI API:")
    response = generate_text_with_llm(
        text=text,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model_name=MODEL,
        use_azure=False,
        system_prompt="You are a topic labeling assistant specialized in generating concise topic names.",
        temperature=0.3,  # More deterministic output
        max_tokens=50     # We only need a short response
    )
    
    print(f"Input: {text}")
    print(f"Response: {response}")
    
    # Example 2: Azure OpenAI API (commented out - would need Azure credentials)
    """
    print("\nUsing Azure OpenAI API:")
    azure_response = generate_text_with_llm(
        text=text,
        api_key="your-azure-api-key",
        api_endpoint="https://your-resource.openai.azure.com",
        deployment_id="your-deployment-name",
        use_azure=True,
        system_prompt="You are a topic labeling assistant specialized in generating concise topic names."
    )
    
    print(f"Input: {text}")
    print(f"Response: {azure_response}")
    """


if __name__ == "__main__":
    print("MENO DIRECT LLM API USAGE EXAMPLES")
    print("Note: This example requires a valid OpenAI API key.")
    print("Update the API_KEY variable at the top of this file to run the example.")
    
    # Skip running the examples if using the placeholder API key
    if API_KEY == "your_openai_api_key_here":
        print("\nExample is using a placeholder API key. Please update with your actual key to run the examples.")
        
        # We can still run the deduplication demos since they don't require an API key
        print("\nDemonstrating optimized deduplication (no API key needed):")
        demo_optimized_deduplication()
    else:
        # Run all demos with an actual API key
        demo_direct_api_call()
        demo_concurrent_processing()
        demo_optimized_deduplication()
        demo_integrated_deduplication_and_caching()
        demo_structured_api_interface()
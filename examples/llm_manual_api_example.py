"""
Example demonstrating direct LLM API usage with Meno's utility functions.

This example shows how to:
1. Make direct API calls to OpenAI without using the full LLMTopicLabeler class
2. Process multiple texts concurrently with ThreadPoolExecutor
3. Use fuzzy deduplication to save on API costs by only processing unique content
4. Use the generate_text_with_llm function which provides a consistent interface for both
   standard OpenAI and Azure OpenAI APIs
"""

from meno.modeling.llm_topic_labeling import (
    generate_call_from_text,
    process_texts_with_threadpool,
    generate_text_with_llm,
    identify_fuzzy_duplicates,
    process_texts_with_deduplication
)

# If you want to try this example, replace with your actual API key
API_KEY = "your_openai_api_key_here"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def demo_direct_api_call():
    """Demonstrate a single direct API call."""
    print("\n=== Single Direct API Call Example ===")

    # Simple topic request
    text = "Please provide a concise name for this topic based on these keywords: healthcare, doctor, nurse, hospital, patient"
    system_prompt = "You are a topic labeling expert that can identify concise topic names from keywords."
    
    result = generate_call_from_text(
        text=text,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model="gpt-3.5-turbo",  # Using a lower-cost model for this example
        system_prompt=system_prompt
    )
    
    print(f"Input: {text}")
    print(f"Topic Name: {result}")


def demo_concurrent_processing():
    """Demonstrate concurrent processing of multiple texts."""
    print("\n=== Concurrent Text Processing Example ===")
    
    # Sample topic keyword sets to process
    texts = [
        "Keywords: finance, bank, investment, money, budget, savings",
        "Keywords: travel, vacation, tourism, hotel, flight, destination",
        "Keywords: technology, computer, software, hardware, programming, algorithm",
        "Keywords: food, cooking, recipe, restaurant, chef, ingredient"
    ]
    
    system_prompt = "You are a topic labeling expert. Generate a concise 1-3 word topic name."
    
    print(f"Processing {len(texts)} texts concurrently...")
    results = process_texts_with_threadpool(
        texts=texts,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model="gpt-3.5-turbo",
        system_prompt=system_prompt,
        max_workers=4  # Process all texts simultaneously
    )
    
    # Print results
    for r in results:
        print(f"\nKeywords: {r['input'].replace('Keywords: ', '')}")
        print(f"Topic: {r['response']}")
        print(f"Processing time: {r['time_taken']:.2f} seconds")


def demo_deduplication():
    """Demonstrate fuzzy deduplication to save on API costs."""
    print("\n=== Fuzzy Deduplication Example ===")
    
    # Create a list with some similar texts
    texts = [
        "Keywords: finance, bank, investment, money, budget, savings",
        "Keywords: banking, investment, finance, budget, money, savings",  # Similar to first text
        "Keywords: financial services, banking, investments, monetary policy",  # Similar to first text 
        "Keywords: travel, vacation, tourism, hotel, flight, destination",
        "Keywords: travel, tourism, hotels, vacations, holiday destinations",  # Similar to fourth text
        "Keywords: technology, computer, software, hardware, programming, algorithm",
        "Keywords: computers, hardware, software development, programming languages",  # Similar to sixth text
        "Keywords: food, cooking, recipe, restaurant, chef, ingredient",
        "Keywords: completely different topic about sports and athletics"
    ]
    
    # First, identify duplicate texts
    print("Identifying duplicates with fuzzy matching...")
    duplicates = identify_fuzzy_duplicates(texts, threshold=0.8)  # Lower threshold for demo
    
    print(f"Found {len(duplicates)} potential duplicates:")
    for dup_idx, source_idx in duplicates.items():
        print(f"Text {dup_idx + 1} is similar to Text {source_idx + 1}")
        print(f"  Original: {texts[source_idx]}")
        print(f"  Duplicate: {texts[dup_idx]}")
        print()
    
    # Process texts with deduplication
    print("\nProcessing texts with automatic deduplication...")
    system_prompt = "You are a topic labeling expert. Generate a concise 1-3 word topic name."
    
    # Note: This function handles deduplication internally
    results = process_texts_with_deduplication(
        texts=texts,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model="gpt-3.5-turbo",
        system_prompt=system_prompt,
        deduplicate=True,
        deduplication_threshold=0.8  # Lower threshold for demo
    )
    
    # Print the results, highlighting duplicates
    print("\nResults with deduplication:")
    for r in results:
        is_duplicate = r.get("is_duplicate", False)
        dup_marker = "[DUPLICATE] " if is_duplicate else ""
        source_info = f" (of #{r.get('duplicate_of', '') + 1})" if is_duplicate else ""
        
        print(f"\n{dup_marker}Text #{r['index'] + 1}{source_info}: {r['input']}")
        print(f"Topic: {r['response']}")
        
        if is_duplicate:
            print("Note: This result was copied from the similar text, saving an API call")


def demo_structured_api_interface():
    """Demonstrate the more structured generate_text_with_llm function."""
    print("\n=== Structured API Interface Example ===")
    
    # Example 1: Standard OpenAI API
    text = "Analyze these keywords and provide a topic name: environment, climate, sustainability, recycling, green"
    
    response = generate_text_with_llm(
        text=text,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model_name="gpt-3.5-turbo",
        use_azure=False,
        system_prompt="You are a topic labeling assistant specialized in generating concise topic names.",
        temperature=0.3,  # More deterministic output
        max_tokens=50     # We only need a short response
    )
    
    print("Standard OpenAI API Example:")
    print(f"Input: {text}")
    print(f"Response: {response}")
    
    # Example 2: Azure OpenAI API (commented out - would need Azure credentials)
    """
    azure_response = generate_text_with_llm(
        text=text,
        api_key="your-azure-api-key",
        api_endpoint="https://your-resource.openai.azure.com",
        deployment_id="your-deployment-name",
        use_azure=True,
        system_prompt="You are a topic labeling assistant specialized in generating concise topic names."
    )
    
    print("\nAzure OpenAI API Example:")
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
        
        # We can still run the deduplication detection part since it doesn't require an API key
        print("\nDemonstrating text similarity detection (no API key needed):")
        
        # Sample texts with similarities
        texts = [
            "Keywords: finance, bank, investment, money, budget, savings",
            "Keywords: banking, investment, finance, budget, money, savings",  # Similar to first text
            "Keywords: financial services, banking, investments, monetary policy",  # Similar to first text 
            "Keywords: completely different topic about sports and athletics"
        ]
        
        # Identify duplicates without making API calls
        duplicates = identify_fuzzy_duplicates(texts, threshold=0.8)
        
        print(f"Found {len(duplicates)} potential duplicates:")
        for dup_idx, source_idx in duplicates.items():
            print(f"Text {dup_idx + 1} is similar to Text {source_idx + 1}")
            print(f"  Original: {texts[source_idx]}")
            print(f"  Duplicate: {texts[dup_idx]}")
            print()
    else:
        demo_direct_api_call()
        demo_concurrent_processing()
        demo_deduplication()  # Added this new demo
        demo_structured_api_interface()
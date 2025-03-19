from typing import List, Dict, Any, Optional, Union
from meno.modeling.llm_topic_labeling import (
    generate_call_from_text,
    process_texts_with_threadpool,
    format_chat_completion,
    generate_text_with_llm,
    identify_fuzzy_duplicates,
    process_texts_with_deduplication
)


# Example usage:
if __name__ == "__main__":
    # Configuration
    api_key = "your_api_key_here"
    api_endpoint = "https://api.openai.com/v1/chat/completions"
    
    # Single text example
    text = "Hello!"
    result = generate_call_from_text(text, api_key, api_endpoint)
    print(f"Single result: {result}\n")
    
    # Multiple texts example
    texts = [
        "What's the weather like today?",
        "Tell me a joke",
        "What is the capital of France?",
        "How do I make pancakes?"
    ]
    
    results = process_texts_with_threadpool(texts, api_key, api_endpoint)
    
    # Print results
    print("\nAll results:")
    for result in results:
        print(f"\nPrompt {result['index']+1}: {result['input']}")
        print(f"Response: {result['response']}")
        print(f"Time: {result['time_taken']:.2f} seconds")
    
    # Example with deduplication
    print("\n\nDeduplication example:")
    # Create texts with some similar content
    texts_with_duplicates = [
        "What are some good books about machine learning?",
        "Can you recommend books on machine learning?",  # Similar to the first
        "What are the best movies from the 1990s?",
        "Tell me about artificial intelligence applications",
        "What are practical applications of AI?",  # Similar to the previous
        "How do I learn to play the guitar?"
    ]
    
    # Just demonstrate the deduplication without making API calls
    print("Identifying similar texts...")
    duplicates = identify_fuzzy_duplicates(
        texts_with_duplicates, 
        threshold=0.7  # Lower threshold to catch more similarities for demo
    )
    
    if duplicates:
        print(f"Found {len(duplicates)} similar texts:")
        for dup_idx, source_idx in duplicates.items():
            print(f"Text {dup_idx+1} is similar to text {source_idx+1}")
            print(f"  Original: {texts_with_duplicates[source_idx]}")
            print(f"  Similar:  {texts_with_duplicates[dup_idx]}")
    else:
        print("No similar texts found.")
    
    # Now do the actual processing with deduplication
    print("\nProcessing with deduplication (skipping actual API calls with dummy key)...")
    if api_key != "your_api_key_here":
        # Only run if a real API key is provided
        dedup_results = process_texts_with_deduplication(
            texts=texts_with_duplicates,
            api_key=api_key,
            api_endpoint=api_endpoint,
            model="gpt-3.5-turbo",
            deduplicate=True,
            deduplication_threshold=0.7
        )
        
        # Print results
        print("\nResults with deduplication:")
        for r in dedup_results:
            is_duplicate = r.get("is_duplicate", False)
            dup_str = " (DUPLICATE)" if is_duplicate else ""
            print(f"\nText{dup_str}: {r['input']}")
            print(f"Response: {r['response']}")
    
    # Example using generate_text_with_llm (more structured interface)
    print("\n\nUsing generate_text_with_llm function:")
    
    # For standard OpenAI
    text = "Explain how topic modeling works"
    if api_key != "your_api_key_here":
        response = generate_text_with_llm(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            model_name="gpt-3.5-turbo",
            use_azure=False,
            system_prompt="You are a helpful expert in machine learning and NLP.",
            max_tokens=500
        )
        print(f"\nResponse: {response}")
    else:
        print("Skipping API call with dummy key.")
    
    # For Azure OpenAI (commented out as it requires Azure credentials)
    """
    azure_response = generate_text_with_llm(
        text=text,
        api_key="your-azure-api-key",
        api_endpoint="https://your-resource.openai.azure.com",
        deployment_id="your-deployment-name",
        use_azure=True,
        system_prompt="You are a helpful expert in machine learning and NLP."
    )
    print(f"\nAzure Response: {azure_response}")
    """
"""
Example demonstrating the enhanced generate_text_with_llm function with requests library support.

This example shows how to:
1. Use the enhanced generate_text_with_llm function with the 'requests' library option
2. Use the 'openai' SDK option for comparison
3. Use with both standard OpenAI and Azure OpenAI APIs

The function has been enhanced to support:
- Multiple library backends (OpenAI SDK or direct requests)
- Caching for the requests implementation
- Proper error handling and timeout configuration
"""

import os
import time
from meno.modeling.llm_topic_labeling import generate_text_with_llm

# Configuration - replace with your actual values
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"  # or gpt-4, etc.

# For Azure OpenAI (if you're using it)
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "your_azure_api_key")
AZURE_ENDPOINT = "https://your-resource.openai.azure.com" 
AZURE_DEPLOYMENT = "your-deployment-name"
AZURE_API_VERSION = "2023-05-15"

def demo_basic_usage():
    """Demonstrate basic usage of the enhanced generate_text_with_llm function."""
    print("\n=== Basic Usage Examples ===")
    
    # Simple prompt for all examples
    prompt = "What are three interesting facts about machine learning?"
    
    # Example 1: Using the requests library with OpenAI
    print("\n1. Using requests library with OpenAI:")
    print(f"Prompt: {prompt}")
    
    try:
        start_time = time.time()
        response = generate_text_with_llm(
            text=prompt,
            api_key=OPENAI_API_KEY,
            api_endpoint=OPENAI_API_ENDPOINT,
            model_name=OPENAI_MODEL,
            use_azure=False,
            library="requests",  # Use requests library
            enable_cache=True,   # Enable caching
            timeout=30           # 30 second timeout
        )
        end_time = time.time()
        
        print(f"\nResponse (requests): {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Using the OpenAI SDK (default)
    print("\n2. Using OpenAI SDK:")
    print(f"Prompt: {prompt}")
    
    try:
        start_time = time.time()
        response = generate_text_with_llm(
            text=prompt,
            api_key=OPENAI_API_KEY,
            api_endpoint=None,  # Not needed for standard OpenAI with SDK
            model_name=OPENAI_MODEL,
            use_azure=False,
            library="openai"  # Use OpenAI SDK
        )
        end_time = time.time()
        
        print(f"\nResponse (SDK): {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

def demo_caching():
    """Demonstrate the caching functionality of the requests implementation."""
    print("\n=== Caching Example ===")
    
    prompt = "Explain the concept of neural networks in simple terms."
    print(f"Prompt: {prompt}")
    
    # First call - should hit the API
    print("\nFirst call (should use API):")
    try:
        start_time = time.time()
        response1 = generate_text_with_llm(
            text=prompt,
            api_key=OPENAI_API_KEY,
            api_endpoint=OPENAI_API_ENDPOINT,
            model_name=OPENAI_MODEL,
            use_azure=False,
            library="requests",
            enable_cache=True
        )
        time1 = time.time() - start_time
        
        print(f"Response: {response1[:100]}...")  # Show just the beginning
        print(f"Time taken: {time1:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
    
    # Second call with identical parameters - should use cache
    print("\nSecond call (should use cache):")
    try:
        start_time = time.time()
        response2 = generate_text_with_llm(
            text=prompt,
            api_key=OPENAI_API_KEY,
            api_endpoint=OPENAI_API_ENDPOINT,
            model_name=OPENAI_MODEL,
            use_azure=False,
            library="requests",
            enable_cache=True
        )
        time2 = time.time() - start_time
        
        print(f"Response: {response2[:100]}...")  # Show just the beginning
        print(f"Time taken: {time2:.2f} seconds")
        
        if time1 > 0 and time2 > 0:
            print(f"Speed improvement: {time1/time2:.1f}x faster with caching")
    except Exception as e:
        print(f"Error: {e}")

def demo_azure_openai():
    """Demonstrate usage with Azure OpenAI."""
    # Skip if no Azure credentials
    if AZURE_API_KEY == "your_azure_api_key" or AZURE_DEPLOYMENT == "your-deployment-name":
        print("\n=== Azure OpenAI Example (Skipped - No credentials) ===")
        print("To run this example, provide your Azure OpenAI credentials.")
        return
    
    print("\n=== Azure OpenAI Example ===")
    
    prompt = "What are the main differences between Azure OpenAI and standard OpenAI?"
    print(f"Prompt: {prompt}")
    
    # Using requests with Azure
    print("\n1. Using requests library with Azure OpenAI:")
    try:
        full_endpoint = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
        response = generate_text_with_llm(
            text=prompt,
            api_key=AZURE_API_KEY,
            api_endpoint=full_endpoint,
            deployment_id=AZURE_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
            use_azure=True,
            library="requests",
            enable_cache=True
        )
        
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Using SDK with Azure
    print("\n2. Using OpenAI SDK with Azure OpenAI:")
    try:
        response = generate_text_with_llm(
            text=prompt,
            api_key=AZURE_API_KEY,
            api_endpoint=AZURE_ENDPOINT,
            deployment_id=AZURE_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
            use_azure=True,
            library="openai"
        )
        
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

def demo_error_handling():
    """Demonstrate error handling."""
    print("\n=== Error Handling Example ===")
    
    # Example with invalid API key
    print("\n1. Invalid API Key:")
    try:
        response = generate_text_with_llm(
            text="This should fail due to invalid API key",
            api_key="invalid_key_123",
            api_endpoint=OPENAI_API_ENDPOINT,
            model_name=OPENAI_MODEL,
            use_azure=False,
            library="requests"
        )
        
        print(f"Response: {response}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Example with invalid model
    print("\n2. Invalid Model Name:")
    try:
        response = generate_text_with_llm(
            text="This should fail due to invalid model name",
            api_key=OPENAI_API_KEY,
            api_endpoint=OPENAI_API_ENDPOINT,
            model_name="non-existent-model",
            use_azure=False,
            library="requests"
        )
        
        print(f"Response: {response}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Example with invalid library
    print("\n3. Invalid Library:")
    try:
        response = generate_text_with_llm(
            text="This should fail due to invalid library",
            api_key=OPENAI_API_KEY,
            api_endpoint=OPENAI_API_ENDPOINT,
            model_name=OPENAI_MODEL,
            use_azure=False,
            library="invalid_library"
        )
        
        print(f"Response: {response}")
    except Exception as e:
        print(f"Exception: {e}")

# Example usage
if __name__ == "__main__":
    print("ENHANCED GENERATE_TEXT_WITH_LLM EXAMPLES")
    print("This example demonstrates using the enhanced generate_text_with_llm function.")
    print("It now supports both the OpenAI SDK and direct requests via the 'library' parameter.")
    
    # Check if we have API credentials
    if OPENAI_API_KEY == "your_openai_api_key":
        print("\nNo API key provided. Please set your OpenAI API key to run the examples.")
        print("You can do this by setting the OPENAI_API_KEY environment variable.")
        exit()
    
    # Run the examples
    demo_basic_usage()
    demo_caching()
    demo_azure_openai()
    demo_error_handling()
    
    print("\nAll examples completed!")
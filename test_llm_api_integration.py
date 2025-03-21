"""
Test the LLM API integration features with real examples.

This script demonstrates the various ways to use the enhanced generate_text_with_llm function 
with both the OpenAI SDK and direct requests approaches, to make it easier for users 
to understand how to use the API in their own code.

This is a manual test script that should be run with real API credentials.
"""

import os
import time
import hashlib
import json
from pathlib import Path

# Import Meno's LLM functions
from meno.modeling.llm_topic_labeling import generate_text_with_llm

# API keys - replace with your own or set environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")

# Azure OpenAI configuration
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "")
AZURE_API_VERSION = "2023-05-15"

# Cache directory for demonstration
CACHE_DIR = Path.home() / ".meno" / "test_cache"

def setup():
    """Check and setup test environment."""
    if not OPENAI_API_KEY:
        print("⚠️ No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
        print("  Standard OpenAI tests will be skipped.")
        
    if not AZURE_API_KEY or not AZURE_ENDPOINT or not AZURE_DEPLOYMENT:
        print("⚠️ Azure OpenAI configuration incomplete.")
        print("  Set AZURE_API_KEY, AZURE_ENDPOINT, and AZURE_DEPLOYMENT environment variables.")
        print("  Azure tests will be skipped.")
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")

def test_standard_openai_sdk():
    """Test using the OpenAI SDK with standard OpenAI API."""
    if not OPENAI_API_KEY:
        print("Skipping standard OpenAI SDK test (no API key)")
        return
    
    print("\n=== Testing Standard OpenAI API with SDK ===")
    prompt = "What are three interesting facts about neural networks?"
    
    # Start timer
    start_time = time.time()
    
    # Call the function with the SDK library option
    response = generate_text_with_llm(
        text=prompt,
        api_key=OPENAI_API_KEY,
        api_endpoint=None,  # Default endpoint for OpenAI
        model_name="gpt-3.5-turbo",  # Using a smaller model to save costs
        use_azure=False,
        library="openai",  # Use OpenAI SDK
        temperature=0.7
    )
    
    # Calculate time
    elapsed = time.time() - start_time
    
    # Print results
    print(f"Prompt: {prompt}")
    print(f"Response (SDK): {response}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    return response

def test_standard_openai_requests():
    """Test using the requests library with standard OpenAI API."""
    if not OPENAI_API_KEY:
        print("Skipping standard OpenAI requests test (no API key)")
        return
    
    print("\n=== Testing Standard OpenAI API with Requests ===")
    prompt = "What are three interesting facts about neural networks?"
    
    # Start timer
    start_time = time.time()
    
    # Call the function with the requests library option
    response = generate_text_with_llm(
        text=prompt,
        api_key=OPENAI_API_KEY,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",  # Using a smaller model to save costs
        use_azure=False,
        library="requests",  # Use direct requests
        temperature=0.7,
        timeout=30,  # 30 second timeout
        enable_cache=True,  # Enable caching
        cache_dir=str(CACHE_DIR)
    )
    
    # Calculate time
    elapsed = time.time() - start_time
    
    # Print results
    print(f"Prompt: {prompt}")
    print(f"Response (Requests): {response}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    return response

def test_azure_openai_sdk():
    """Test using the OpenAI SDK with Azure OpenAI API."""
    if not AZURE_API_KEY or not AZURE_ENDPOINT or not AZURE_DEPLOYMENT:
        print("Skipping Azure OpenAI SDK test (incomplete configuration)")
        return
    
    print("\n=== Testing Azure OpenAI API with SDK ===")
    prompt = "What are three interesting facts about neural networks?"
    
    # Start timer
    start_time = time.time()
    
    # Call the function with the SDK library option for Azure
    response = generate_text_with_llm(
        text=prompt,
        api_key=AZURE_API_KEY,
        api_endpoint=AZURE_ENDPOINT,
        deployment_id=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        use_azure=True,
        library="openai",  # Use OpenAI SDK
        temperature=0.7
    )
    
    # Calculate time
    elapsed = time.time() - start_time
    
    # Print results
    print(f"Prompt: {prompt}")
    print(f"Response (Azure SDK): {response}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    return response

def test_azure_openai_requests():
    """Test using the requests library with Azure OpenAI API."""
    if not AZURE_API_KEY or not AZURE_ENDPOINT or not AZURE_DEPLOYMENT:
        print("Skipping Azure OpenAI requests test (incomplete configuration)")
        return
    
    print("\n=== Testing Azure OpenAI API with Requests ===")
    prompt = "What are three interesting facts about neural networks?"
    
    # Construct Azure endpoint for direct API call
    api_endpoint = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    
    # Start timer
    start_time = time.time()
    
    # Call the function with the requests library option for Azure
    response = generate_text_with_llm(
        text=prompt,
        api_key=AZURE_API_KEY,
        api_endpoint=api_endpoint,
        deployment_id=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        use_azure=True,
        library="requests",  # Use direct requests
        temperature=0.7,
        timeout=30,  # 30 second timeout
        enable_cache=True,  # Enable caching
        cache_dir=str(CACHE_DIR)
    )
    
    # Calculate time
    elapsed = time.time() - start_time
    
    # Print results
    print(f"Prompt: {prompt}")
    print(f"Response (Azure Requests): {response}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    return response

def test_caching():
    """Test the caching functionality of the requests implementation."""
    if not OPENAI_API_KEY:
        print("Skipping caching test (no API key)")
        return
    
    print("\n=== Testing Caching Functionality ===")
    prompt = "What is the capital of France?"
    
    # Clear any existing cache for this prompt
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_path = CACHE_DIR / f"llm_cache_{prompt_hash}.json"
    if cache_path.exists():
        cache_path.unlink()
        print(f"Cleared existing cache for this prompt: {cache_path}")
    
    print("\nFirst call (should use API):")
    start_time = time.time()
    response1 = generate_text_with_llm(
        text=prompt,
        api_key=OPENAI_API_KEY,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",
        use_azure=False,
        library="requests",
        enable_cache=True,
        cache_dir=str(CACHE_DIR)
    )
    time1 = time.time() - start_time
    
    print(f"Response: {response1}")
    print(f"Time taken: {time1:.2f} seconds")
    
    # Verify cache file was created
    if cache_path.exists():
        print(f"Cache file created: {cache_path}")
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            print(f"Cache data: {cache_data['content'][:50]}... (timestamp: {cache_data.get('timestamp')})")
    
    # Second call with identical parameters
    print("\nSecond call (should use cache):")
    start_time = time.time()
    response2 = generate_text_with_llm(
        text=prompt,
        api_key=OPENAI_API_KEY,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",
        use_azure=False,
        library="requests",
        enable_cache=True,
        cache_dir=str(CACHE_DIR)
    )
    time2 = time.time() - start_time
    
    print(f"Response: {response2}")
    print(f"Time taken: {time2:.2f} seconds")
    
    if time1 > 0 and time2 > 0:
        print(f"Speed improvement: {time1/time2:.1f}x faster with caching")
    
    # Consistency check
    if response1 == response2:
        print("✅ Responses are identical - caching works correctly")
    else:
        print("⚠️ Responses differ - possible caching issue")
    
    # Now test with caching disabled
    print("\nThird call (with caching disabled):")
    start_time = time.time()
    response3 = generate_text_with_llm(
        text=prompt,
        api_key=OPENAI_API_KEY,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",
        use_azure=False,
        library="requests",
        enable_cache=False  # Disable caching
    )
    time3 = time.time() - start_time
    
    print(f"Response: {response3[:50]}...")
    print(f"Time taken: {time3:.2f} seconds")
    
    if time2 > 0 and time3 > 0:
        print(f"Comparison: No-cache call is {time3/time2:.1f}x slower than cached call")
    
    return response1, response2, response3

def test_error_handling():
    """Test error handling functionality."""
    print("\n=== Testing Error Handling ===")
    
    # Test with invalid API key
    print("\n1. Invalid API Key:")
    response = generate_text_with_llm(
        text="This should fail",
        api_key="invalid_key_123",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",
        library="requests"
    )
    print(f"Response: {response}")
    
    # Test with invalid model name
    print("\n2. Invalid Model Name:")
    response = generate_text_with_llm(
        text="This should fail",
        api_key=OPENAI_API_KEY if OPENAI_API_KEY else "fake_key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="non-existent-model",
        library="requests"
    )
    print(f"Response: {response}")
    
    # Test with invalid library
    print("\n3. Invalid Library:")
    response = generate_text_with_llm(
        text="This should fail",
        api_key=OPENAI_API_KEY if OPENAI_API_KEY else "fake_key",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",
        library="invalid_library"
    )
    print(f"Response: {response}")
    
    # Test with invalid endpoint
    print("\n4. Invalid Endpoint:")
    response = generate_text_with_llm(
        text="This should fail",
        api_key=OPENAI_API_KEY if OPENAI_API_KEY else "fake_key",
        api_endpoint="https://invalid-endpoint.example.com",
        model_name="gpt-3.5-turbo",
        library="requests",
        timeout=5  # Short timeout to avoid long wait
    )
    print(f"Response: {response}")

def run_all_tests():
    """Run all tests sequentially."""
    # Setup test environment
    setup()
    
    # Run tests
    tests = [
        test_standard_openai_sdk,
        test_standard_openai_requests,
        test_azure_openai_sdk,
        test_azure_openai_requests,
        test_caching,
        test_error_handling
    ]
    
    results = {}
    for test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_func.__name__}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_func.__name__] = "PASS" if result is not None else "SKIP"
        except Exception as e:
            print(f"Error: {e}")
            results[test_func.__name__] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result == "PASS" else "⚠️ SKIP" if result == "SKIP" else f"❌ {result}"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    run_all_tests()
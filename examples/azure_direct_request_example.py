"""
Example demonstrating direct API calls to Azure OpenAI using requests library.

This example shows how to:
1. Make direct API calls to Azure OpenAI endpoint using the requests library
2. Process multiple texts concurrently with ThreadPoolExecutor
3. Use proper Azure API parameters (api-version, deployment_id instead of model)
4. Handle responses correctly

Usage:
1. Set your Azure OpenAI API key and endpoint
2. Set your deployment name (model deployment in Azure)
3. Run the script
"""

import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union

# Configuration - Replace with your Azure OpenAI settings
API_KEY = "your_azure_api_key_here"
RESOURCE_NAME = "your-resource-name"  # This is part of your endpoint URL
DEPLOYMENT_ID = "your-deployment-name"  # This is the deployment name in Azure
API_VERSION = "2023-12-01-preview"  # Update as needed

# Construct the API endpoint URL
API_ENDPOINT = f"https://{RESOURCE_NAME}.openai.azure.com/openai/deployments/{DEPLOYMENT_ID}/chat/completions?api-version={API_VERSION}"


def generate_azure_completion(
    text: str, 
    api_key: str,
    endpoint: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 150,
    timeout: int = 60
) -> str:
    """
    Make a direct API call to Azure OpenAI using the requests library.
    
    Args:
        text: The user input text to process
        api_key: Your Azure OpenAI API key
        endpoint: The full Azure OpenAI endpoint URL (including deployment and api-version)
        system_prompt: The system prompt to use
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in the response
        timeout: Request timeout in seconds
        
    Returns:
        The generated response text or an error message
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key  # Note: Azure uses 'api-key' instead of Authorization header
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        response_data = response.json()
        
        if not response_data.get('choices') or len(response_data['choices']) == 0:
            return "[No response generated.]"
            
        return response_data['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.Timeout:
        return "[Error: Request timed out]"
    except requests.exceptions.RequestException as e:
        return f"[Error: {e}]"
    except ValueError as e:  # JSON parsing error
        return f"[Error: Invalid response format - {e}]"
    except Exception as e:
        return f"[Error: Unexpected error - {e}]"


def process_texts_with_threadpool(
    texts: List[str],
    api_key: str,
    endpoint: str,
    system_prompt: str = "You are a helpful assistant.",
    max_workers: Optional[int] = None,
    timeout: int = 60,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple texts concurrently using a ThreadPoolExecutor.
    
    Args:
        texts: List of text prompts to process
        api_key: Your Azure API key
        endpoint: The Azure endpoint URL
        system_prompt: The system prompt to use
        max_workers: Maximum number of worker threads (None = auto-determined)
        timeout: Request timeout in seconds
        show_progress: Whether to print progress information
        
    Returns:
        List of dictionaries containing the input text, response, and timing information
    """
    results = []
    
    # Create a lock for thread-safe progress updates
    progress_lock = threading.Lock()
    completed_count = 0
    
    def process_single_text(text: str, index: int) -> Dict[str, Any]:
        nonlocal completed_count
        
        start_time = time.time()
        response = generate_azure_completion(
            text=text,
            api_key=api_key,
            endpoint=endpoint,
            system_prompt=system_prompt,
            timeout=timeout
        )
        end_time = time.time()
        
        result = {
            "index": index,
            "input": text,
            "response": response,
            "time_taken": end_time - start_time,
            "success": not response.startswith("[Error:")
        }
        
        # Update progress with thread safety
        if show_progress:
            with progress_lock:
                completed_count += 1
                print(f"Completed {completed_count}/{len(texts)}: {'✓' if result['success'] else '✗'}")
        
        return result
    
    # Determine optimal number of workers if not specified
    if max_workers is None:
        # Use min(32, os.cpu_count() + 4) as recommended for IO-bound tasks
        import os
        default_workers = min(32, (os.cpu_count() or 4) + 4)
        # But also consider the number of texts (no need for more workers than texts)
        max_workers = min(default_workers, len(texts))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = [
            executor.submit(process_single_text, text, i) 
            for i, text in enumerate(texts)
        ]
        
        # Collect results as they complete
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Find the index of the failed future
                for i, f in enumerate(futures):
                    if f == future:
                        idx = i
                        break
                else:
                    idx = len(results)
                
                # Add error result
                results.append({
                    "index": idx,
                    "input": texts[idx] if idx < len(texts) else "Unknown",
                    "response": f"[Error: Thread execution failed - {e}]",
                    "time_taken": 0,
                    "success": False
                })
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x["index"])
    
    # Show summary if requested
    if show_progress and results:
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"\nSummary: {success_count}/{len(results)} successful")
    
    return results


# Example usage
if __name__ == "__main__":
    print("AZURE OPENAI DIRECT API USAGE EXAMPLE")
    print("Note: This example requires a valid Azure OpenAI API key and deployment.")
    print(f"Configured endpoint: {API_ENDPOINT}")
    
    # Skip running the examples if using the placeholder API key
    if API_KEY == "your_azure_api_key_here":
        print("\nExample is using a placeholder API key. Please update with your actual key and configuration to run the examples.")
        exit()
    
    # Basic request test
    text = "What's your favorite part about being an AI assistant?"
    print("\n=== Basic Request Test ===")
    print(f"Input: {text}")
    
    start_time = time.time()
    result = generate_azure_completion(
        text=text,
        api_key=API_KEY,
        endpoint=API_ENDPOINT
    )
    time_taken = time.time() - start_time
    
    print(f"Response: {result}")
    print(f"Processing time: {time_taken:.2f} seconds")
    
    # Batch processing test
    print("\n=== Batch Processing Test ===")
    texts = [
        "What's the weather like today?",
        "Tell me a joke",
        "What is the capital of France?",
        "How do I make pancakes?"
    ]
    
    batch_results = process_texts_with_threadpool(
        texts=texts,
        api_key=API_KEY,
        endpoint=API_ENDPOINT,
        max_workers=4,
        show_progress=True
    )
    
    print("\nBatch Results:")
    for result in batch_results:
        print(f"\nPrompt {result['index']+1}: {result['input']}")
        print(f"Response: {result['response']}")
        print(f"Time: {result['time_taken']:.2f} seconds")
    
    print("\nAll examples completed!")
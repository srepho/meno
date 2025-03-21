import requests
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union

def generate_call_from_text(text: str, api_key: str, api_endpoint: str, 
                           model: str = "gpt-4o", system_prompt: str = "You are a helpful assistant.",
                           timeout: int = 60) -> str:
    """
    Make a single API call to generate a response from the given text.
    
    Args:
        text: The user input text to process
        api_key: Your API key for authentication
        api_endpoint: The API endpoint URL
        model: The model to use for generation
        system_prompt: The system prompt to use
        timeout: Request timeout in seconds
        
    Returns:
        The generated response text or an error message
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }
    
    try:
        response = requests.post(
            api_endpoint, 
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


def process_texts_with_threadpool(texts: List[str], api_key: str, api_endpoint: str,
                                 model: str = "gpt-4o", system_prompt: str = "You are a helpful assistant.",
                                 max_workers: Optional[int] = None, timeout: int = 60) -> List[Dict[str, Any]]:
    """
    Process multiple texts concurrently using a ThreadPoolExecutor.
    
    Args:
        texts: List of text prompts to process
        api_key: Your API key for authentication
        api_endpoint: The API endpoint URL
        model: The model to use for generation
        system_prompt: The system prompt to use
        max_workers: Maximum number of worker threads (None = auto-determined)
        timeout: Request timeout in seconds
        
    Returns:
        List of dictionaries containing the input text, response, and timing information
    """
    results = []
    
    def process_single_text(text: str, index: int) -> Dict[str, Any]:
        start_time = time.time()
        response = generate_call_from_text(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            model=model,
            system_prompt=system_prompt,
            timeout=timeout
        )
        end_time = time.time()
        
        return {
            "index": index,
            "input": text,
            "response": response,
            "time_taken": end_time - start_time,
            "success": not response.startswith("[Error:")
        }
    
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
                # Optional: Print progress
                print(f"Completed {result['index']+1}/{len(texts)}: {'✓' if result['success'] else '✗'}")
            except Exception as e:
                results.append({
                    "index": len(results),
                    "input": texts[len(results)] if len(results) < len(texts) else "Unknown",
                    "response": f"[Error: Thread execution failed - {e}]",
                    "time_taken": 0,
                    "success": False
                })
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x["index"])
    return results


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
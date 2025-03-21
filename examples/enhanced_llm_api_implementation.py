"""
Enhanced LLM API implementation supporting multiple library options.

This example shows how to extend Meno's generate_text_with_llm function
to support multiple implementation libraries (OpenAI SDK, requests, etc.)
while maintaining consistent behavior.
"""

import requests
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_text_with_llm(
    text: str,
    api_key: str,
    api_endpoint: str,
    deployment_id: str = None, 
    model_name: str = "gpt-4o",
    api_version: str = "2023-05-15",
    use_azure: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    user_prompt_prefix: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    library: str = "openai",  # New parameter for selecting implementation library
    timeout: int = 60,        # New parameter for requests timeout
    enable_cache: bool = True # New parameter for caching (requests implementation)
) -> str:
    """Generate text using LLM APIs with support for multiple implementation libraries.
    
    This utility function makes it easy to generate text using either OpenAI's SDK
    or direct HTTP requests, with proper parameter configurations for each service.
    
    Parameters
    ----------
    text : str
        The input text/prompt to send to the model
    api_key : str
        The API key for the LLM service
    api_endpoint : str
        For Azure: The azure_endpoint (e.g., "https://your-resource.openai.azure.com")
        For OpenAI: Base URL, usually "https://api.openai.com/v1/chat/completions"
    deployment_id : str, optional
        Azure deployment name, required when use_azure=True
    model_name : str, optional
        For OpenAI: Model name like "gpt-4o" or "gpt-3.5-turbo", by default "gpt-4o"
        Ignored when use_azure=True (deployment_id is used instead)
    api_version : str, optional
        API version, by default "2023-05-15" - mainly used for Azure
    use_azure : bool, optional
        Whether to use Azure OpenAI, by default False
    system_prompt : str, optional
        System prompt for the model, by default "You are a helpful assistant."
    user_prompt_prefix : str, optional
        Prefix to add before the input text
    temperature : float, optional
        Temperature for response generation, by default 0.7
    max_tokens : int, optional
        Maximum tokens in the response, by default 1000
    library : str, optional
        Which library to use for implementation: "openai" or "requests", by default "openai"
    timeout : int, optional
        Timeout in seconds for requests implementation, by default 60
    enable_cache : bool, optional
        Whether to enable caching for requests implementation, by default True
        
    Returns
    -------
    str
        Generated text response from the model
        
    Examples
    --------
    # Using the OpenAI SDK:
    >>> response = generate_text_with_llm(
    ...     text="Explain the benefits of Python 3.10",
    ...     api_key="your-openai-api-key",
    ...     api_endpoint="https://api.openai.com/v1/chat/completions",
    ...     model_name="gpt-4o",
    ...     use_azure=False,
    ...     library="openai"
    ... )
    
    # Using direct requests:
    >>> response = generate_text_with_llm(
    ...     text="Explain the benefits of Python 3.10",
    ...     api_key="your-openai-api-key",
    ...     api_endpoint="https://api.openai.com/v1/chat/completions",
    ...     model_name="gpt-4o",
    ...     use_azure=False,
    ...     library="requests"
    ... )
    
    # Using direct requests with Azure:
    >>> response = generate_text_with_llm(
    ...     text="Tell me a joke about Azure cloud services",
    ...     api_key="your-azure-api-key",
    ...     api_endpoint="https://your-resource.openai.azure.com/openai/deployments/your-deployment-name/chat/completions",
    ...     deployment_id="your-deployment-name",
    ...     use_azure=True,
    ...     api_version="2023-05-15",
    ...     library="requests"
    ... )
    """
    # Prepare the full user prompt
    user_prompt = f"{user_prompt_prefix}{text}"
    
    if library.lower() == "openai":
        return _generate_with_openai_sdk(
            text=user_prompt,
            api_key=api_key,
            api_endpoint=api_endpoint,
            deployment_id=deployment_id,
            model_name=model_name,
            api_version=api_version,
            use_azure=use_azure,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif library.lower() == "requests":
        return _generate_with_requests(
            text=user_prompt,
            api_key=api_key,
            api_endpoint=api_endpoint,
            deployment_id=deployment_id,
            model_name=model_name,
            api_version=api_version,
            use_azure=use_azure,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            enable_cache=enable_cache
        )
    else:
        raise ValueError(f"Unsupported library: {library}. Supported options are 'openai' and 'requests'.")

def _generate_with_openai_sdk(
    text: str,
    api_key: str,
    api_endpoint: str,
    deployment_id: str = None, 
    model_name: str = "gpt-4o",
    api_version: str = "2023-05-15",
    use_azure: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> str:
    """Implementation using the OpenAI SDK."""
    try:
        # Import the necessary client
        if use_azure:
            from openai import AzureOpenAI as ClientClass
            if not deployment_id:
                raise ValueError("deployment_id is required when using Azure OpenAI")
                
            # Create Azure OpenAI client
            client = ClientClass(
                api_key=api_key,
                azure_endpoint=api_endpoint,
                api_version=api_version
            )
            
            # Make the API call with deployment_id
            response = client.chat.completions.create(
                deployment_id=deployment_id,  # Azure uses deployment_id
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            from openai import OpenAI as ClientClass
            
            # Create standard OpenAI client
            client_kwargs = {"api_key": api_key}
            if api_endpoint:
                client_kwargs["base_url"] = api_endpoint
                
            client = ClientClass(**client_kwargs)
            
            # Make the API call with model
            response = client.chat.completions.create(
                model=model_name,  # Standard OpenAI uses model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Extract and return the response content
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return "[No response generated]"
            
    except Exception as e:
        logger.error(f"Error generating text with OpenAI SDK: {e}")
        return f"[Error: {str(e)}]"

def _generate_with_requests(
    text: str,
    api_key: str,
    api_endpoint: str,
    deployment_id: str = None, 
    model_name: str = "gpt-4o",
    api_version: str = "2023-05-15",
    use_azure: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: int = 60,
    enable_cache: bool = True
) -> str:
    """Implementation using the requests library."""
    # Implement caching if enabled
    if enable_cache:
        import hashlib
        import time
        import os
        from pathlib import Path
        
        # Generate a cache key
        cache_key = f"{text}|{model_name if not use_azure else deployment_id}|{system_prompt}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Set up cache directory
        cache_dir = Path.home() / ".meno" / "llm_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = cache_dir / f"{cache_hash}.json"
        
        # Check if we have a cached result
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if the cache is still valid (default: 24 hours)
                if cached_data.get('expires', 0) > time.time():
                    logger.debug(f"Using cached result for: {text[:30]}...")
                    return cached_data['value']
            except (json.JSONDecodeError, KeyError):
                # Invalid cache file, ignore it
                pass
    
    # Prepare headers based on whether we're using Azure or standard OpenAI
    if use_azure:
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    # Prepare the messages payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    # Prepare the request payload
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add model for standard OpenAI (not Azure)
    if not use_azure:
        payload["model"] = model_name
    
    # Prepare the endpoint URL
    if use_azure and "api-version" not in api_endpoint and "api_version" not in api_endpoint:
        # Add api-version if using Azure and it's not already in the URL
        if "?" in api_endpoint:
            api_endpoint = f"{api_endpoint}&api-version={api_version}"
        else:
            api_endpoint = f"{api_endpoint}?api-version={api_version}"
    
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
            result = "[No response generated]"
        else:
            result = response_data['choices'][0]['message']['content'].strip()
        
        # Cache the result if caching is enabled
        if enable_cache:
            # Cache for 24 hours by default
            expires = time.time() + (24 * 60 * 60)
            cache_data = {
                'value': result,
                'expires': expires,
                'created': time.time()
            }
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"Failed to write to cache file: {e}")
        
        return result
    
    except requests.exceptions.Timeout:
        return "[Error: Request timed out]"
    except requests.exceptions.RequestException as e:
        return f"[Error: {e}]"
    except json.JSONDecodeError:
        return "[Error: Invalid JSON response]"
    except Exception as e:
        logger.error(f"Error generating text with requests: {e}")
        return f"[Error: {str(e)}]"

# Example usage
if __name__ == "__main__":
    # Example configuration
    API_KEY = "your_api_key_here"
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-3.5-turbo"
    
    print("ENHANCED LLM API IMPLEMENTATION EXAMPLE")
    print("Note: This example requires a valid API key.")
    
    if API_KEY == "your_api_key_here":
        print("Please update the API_KEY variable to run the examples.")
        exit()
    
    # Example 1: Using OpenAI SDK
    print("\n=== Example using OpenAI SDK ===")
    prompt = "What are three interesting facts about machine learning?"
    
    print(f"Generating response for: {prompt}")
    response1 = generate_text_with_llm(
        text=prompt,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model_name=MODEL,
        library="openai"
    )
    print(f"\nResponse (OpenAI SDK): {response1}")
    
    # Example 2: Using requests library
    print("\n=== Example using requests library ===")
    
    print(f"Generating response for: {prompt}")
    response2 = generate_text_with_llm(
        text=prompt,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model_name=MODEL,
        library="requests",
        enable_cache=True
    )
    print(f"\nResponse (requests): {response2}")
    
    # Example 3: Cache test with requests
    print("\n=== Cache test with requests ===")
    
    print(f"Generating response again (should use cache):")
    start_time = time.time()
    response3 = generate_text_with_llm(
        text=prompt,
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        model_name=MODEL,
        library="requests",
        enable_cache=True
    )
    end_time = time.time()
    
    print(f"\nResponse (cached): {response3}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    print("\nAll examples completed!")
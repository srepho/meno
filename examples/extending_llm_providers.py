"""
Example demonstrating how to extend the generate_text_with_llm function to support other LLM providers.

This example shows how to add support for:
1. Google's Gemini AI
2. Anthropic's Claude
3. Cohere
4. Any other LLM provider with an API

The approach uses the same pattern as the OpenAI SDK vs requests implementation,
allowing you to choose your implementation library while maintaining a consistent interface.
"""

import os
import time
import json
import requests
import hashlib
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the base function for reference
try:
    from meno.modeling.llm_topic_labeling import generate_text_with_llm as base_generate_text_with_llm
except ImportError:
    # If Meno is not installed, define a placeholder
    def base_generate_text_with_llm(*args, **kwargs):
        return "Meno not installed - this is just a reference implementation"

# Check for available libraries
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# API keys - replace with your own or set environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

# Cache directory
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".meno", "llm_cache")


def generate_text_with_llm_extended(
    text: str,
    api_key: str,
    api_endpoint: Optional[str] = None,
    model_name: str = "gpt-4o",
    system_prompt: str = "You are a helpful assistant.",
    user_prompt_prefix: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    library: str = "openai",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    # Additional parameters for specific providers
    provider: str = "openai",
    version: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Enhanced version of generate_text_with_llm that supports multiple LLM providers.
    
    Parameters
    ----------
    text : str
        The text prompt to send to the LLM
    api_key : str
        API key for the LLM provider
    api_endpoint : Optional[str], optional
        API endpoint URL, by default None
    model_name : str, optional
        Name of the model to use, by default "gpt-4o"
    system_prompt : str, optional
        System prompt to use, by default "You are a helpful assistant."
    user_prompt_prefix : str, optional
        Prefix to add before the user's text, by default ""
    temperature : float, optional
        Temperature setting for generation, by default 0.7
    max_tokens : int, optional
        Maximum tokens to generate, by default 1000
    library : str, optional
        Library implementation to use ("sdk", "requests"), by default "openai"
    timeout : int, optional
        Timeout in seconds for requests, by default 60
    enable_cache : bool, optional
        Whether to enable response caching, by default True
    cache_dir : Optional[str], optional
        Directory to store cache files, by default None
    provider : str, optional
        LLM provider to use ("openai", "google", "anthropic", "cohere"), by default "openai"
    version : Optional[str], optional
        API version to use with the provider, by default None
    additional_params : Optional[Dict[str, Any]], optional
        Additional parameters to pass to the provider's API, by default None
    
    Returns
    -------
    str
        The generated text or error message
    """
    # Combine user text with prefix if provided
    full_text = f"{user_prompt_prefix} {text}" if user_prompt_prefix else text
    
    # Handle different providers
    if provider == "openai":
        # Use the base implementation for OpenAI and Azure
        return base_generate_text_with_llm(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt_prefix=user_prompt_prefix,
            temperature=temperature,
            max_tokens=max_tokens,
            library=library,
            timeout=timeout,
            enable_cache=enable_cache,
            cache_dir=cache_dir
        )
    
    elif provider == "google":
        # Google Gemini implementation
        try:
            if library == "sdk":
                if not GOOGLE_AVAILABLE:
                    return f"[Error: Google Generative AI SDK not installed. Install with 'pip install google-generativeai']"
                
                return _generate_with_google_sdk(
                    text=full_text,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    additional_params=additional_params
                )
            
            elif library == "requests":
                return _generate_with_google_requests(
                    text=full_text,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    timeout=timeout,
                    enable_cache=enable_cache,
                    cache_dir=cache_dir,
                    additional_params=additional_params
                )
            
            else:
                return f"[Error: Unsupported library '{library}' for Google. Use 'sdk' or 'requests']"
        
        except Exception as e:
            return f"[Error: Google AI generation failed: {str(e)}]"
    
    elif provider == "anthropic":
        # Anthropic Claude implementation
        try:
            if library == "sdk":
                if not ANTHROPIC_AVAILABLE:
                    return f"[Error: Anthropic SDK not installed. Install with 'pip install anthropic']"
                
                return _generate_with_anthropic_sdk(
                    text=full_text,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    additional_params=additional_params
                )
            
            elif library == "requests":
                return _generate_with_anthropic_requests(
                    text=full_text,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    timeout=timeout,
                    enable_cache=enable_cache,
                    cache_dir=cache_dir,
                    version=version,
                    additional_params=additional_params
                )
            
            else:
                return f"[Error: Unsupported library '{library}' for Anthropic. Use 'sdk' or 'requests']"
        
        except Exception as e:
            return f"[Error: Anthropic generation failed: {str(e)}]"
    
    elif provider == "cohere":
        # Cohere implementation
        try:
            if library == "sdk":
                if not COHERE_AVAILABLE:
                    return f"[Error: Cohere SDK not installed. Install with 'pip install cohere']"
                
                return _generate_with_cohere_sdk(
                    text=full_text,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    additional_params=additional_params
                )
            
            elif library == "requests":
                return _generate_with_cohere_requests(
                    text=full_text,
                    api_key=api_key,
                    api_endpoint=api_endpoint,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    enable_cache=enable_cache,
                    cache_dir=cache_dir,
                    additional_params=additional_params
                )
            
            else:
                return f"[Error: Unsupported library '{library}' for Cohere. Use 'sdk' or 'requests']"
        
        except Exception as e:
            return f"[Error: Cohere generation failed: {str(e)}]"
    
    else:
        return f"[Error: Unsupported provider '{provider}'. Supported providers: 'openai', 'google', 'anthropic', 'cohere']"


def _generate_cache_key(text: str, model: str, params: Dict[str, Any]) -> str:
    """Generate a cache key based on request parameters."""
    # Create a string combining all relevant parameters
    param_str = f"{text}|{model}|{json.dumps(params, sort_keys=True)}"
    # Generate a hash for the cache key
    return hashlib.md5(param_str.encode()).hexdigest()


def _get_from_cache(cache_key: str, cache_dir: str, ttl: int = 86400) -> Optional[str]:
    """Try to get a response from cache."""
    cache_file = os.path.join(cache_dir, f"llm_cache_{cache_key}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            cached_time = cache_data.get('timestamp', 0)
            current_time = time.time()
            
            if current_time - cached_time <= ttl:
                return cache_data.get('content')
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
    
    return None


def _save_to_cache(response: str, cache_key: str, cache_dir: str) -> None:
    """Save a response to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"llm_cache_{cache_key}.json")
    
    try:
        cache_data = {
            'content': response,
            'timestamp': time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Error writing to cache: {e}")


# Google Gemini implementations

def _generate_with_google_sdk(
    text: str,
    api_key: str,
    model_name: str = "gemini-pro",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using Google's Generative AI SDK."""
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Get the model
    model = genai.GenerativeModel(model_name)
    
    # Prepare generation parameters
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    # Add any additional parameters
    if additional_params:
        generation_config.update(additional_params)
    
    # Create a chat session with the system prompt
    chat = model.start_chat(history=[
        {"role": "user", "parts": [system_prompt]},
        {"role": "model", "parts": ["I'll help you as requested."]}
    ])
    
    # Send the user's message
    response = chat.send_message(text, generation_config=generation_config)
    
    # Return the response
    return response.text


def _generate_with_google_requests(
    text: str,
    api_key: str,
    model_name: str = "gemini-pro",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using Google's Generative AI API via direct requests."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Prepare request parameters
    params = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topP": 0.95,
        "topK": 40,
    }
    
    # Add any additional parameters
    if additional_params:
        params.update(additional_params)
    
    # Generate cache key if caching is enabled
    if enable_cache:
        cache_key = _generate_cache_key(
            text=f"{system_prompt}\n{text}", 
            model=model_name, 
            params=params
        )
        
        # Try to get from cache
        cached_response = _get_from_cache(cache_key, cache_dir)
        if cached_response:
            return cached_response
    
    # Prepare the API request
    api_url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
    
    # Create a messages array with system prompt and user content
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": system_prompt}]
            },
            {
                "role": "model",
                "parts": [{"text": "I'll help you as requested."}]
            },
            {
                "role": "user",
                "parts": [{"text": text}]
            }
        ],
        "generationConfig": params
    }
    
    # Send the request
    response = requests.post(
        api_url,
        params={"key": api_key},
        json=payload,
        timeout=timeout
    )
    
    # Process the response
    response.raise_for_status()
    result = response.json()
    
    # Extract the generated text
    generated_text = ""
    if "candidates" in result and result["candidates"]:
        for part in result["candidates"][0]["content"]["parts"]:
            if "text" in part:
                generated_text += part["text"]
    
    # Save to cache if enabled
    if enable_cache and generated_text:
        _save_to_cache(generated_text, cache_key, cache_dir)
    
    return generated_text


# Anthropic Claude implementations

def _generate_with_anthropic_sdk(
    text: str,
    api_key: str,
    model_name: str = "claude-3-opus-20240229",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using Anthropic's Claude SDK."""
    # Initialize the client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare request parameters
    params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": text}
        ]
    }
    
    # Add any additional parameters
    if additional_params:
        params.update(additional_params)
    
    # Make the API call
    response = client.messages.create(**params)
    
    # Return the response
    return response.content[0].text


def _generate_with_anthropic_requests(
    text: str,
    api_key: str,
    model_name: str = "claude-3-opus-20240229",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    version: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using Anthropic's Claude API via direct requests."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    api_version = version or "2023-06-01"
    
    # Prepare request parameters
    params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": text}
        ]
    }
    
    # Add any additional parameters
    if additional_params:
        params.update(additional_params)
    
    # Generate cache key if caching is enabled
    if enable_cache:
        cache_key = _generate_cache_key(
            text=f"{system_prompt}\n{text}", 
            model=model_name, 
            params=params
        )
        
        # Try to get from cache
        cached_response = _get_from_cache(cache_key, cache_dir)
        if cached_response:
            return cached_response
    
    # Prepare the API request
    api_url = "https://api.anthropic.com/v1/messages"
    
    # Set up headers
    headers = {
        "x-api-key": api_key,
        "anthropic-version": api_version,
        "content-type": "application/json"
    }
    
    # Send the request
    response = requests.post(
        api_url,
        headers=headers,
        json=params,
        timeout=timeout
    )
    
    # Process the response
    response.raise_for_status()
    result = response.json()
    
    # Extract the generated text
    generated_text = result.get("content", [{}])[0].get("text", "")
    
    # Save to cache if enabled
    if enable_cache and generated_text:
        _save_to_cache(generated_text, cache_key, cache_dir)
    
    return generated_text


# Cohere implementations

def _generate_with_cohere_sdk(
    text: str,
    api_key: str,
    model_name: str = "command",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using Cohere's SDK."""
    # Initialize the client
    client = cohere.Client(api_key)
    
    # Prepare request parameters
    params = {
        "model": model_name,
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    # Add any additional parameters
    if additional_params:
        params.update(additional_params)
    
    # Make the API call
    response = client.generate(**params)
    
    # Return the response
    return response.generations[0].text


def _generate_with_cohere_requests(
    text: str,
    api_key: str,
    api_endpoint: Optional[str] = None,
    model_name: str = "command",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate text using Cohere's API via direct requests."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    api_url = api_endpoint or "https://api.cohere.ai/v1/generate"
    
    # Prepare request parameters
    params = {
        "model": model_name,
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    # Add any additional parameters
    if additional_params:
        params.update(additional_params)
    
    # Generate cache key if caching is enabled
    if enable_cache:
        cache_key = _generate_cache_key(
            text=text, 
            model=model_name, 
            params=params
        )
        
        # Try to get from cache
        cached_response = _get_from_cache(cache_key, cache_dir)
        if cached_response:
            return cached_response
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Send the request
    response = requests.post(
        api_url,
        headers=headers,
        json=params,
        timeout=timeout
    )
    
    # Process the response
    response.raise_for_status()
    result = response.json()
    
    # Extract the generated text
    generated_text = result.get("generations", [{}])[0].get("text", "")
    
    # Save to cache if enabled
    if enable_cache and generated_text:
        _save_to_cache(generated_text, cache_key, cache_dir)
    
    return generated_text


# Example usage

def demo_google_gemini():
    """Demonstrate using Google's Gemini model."""
    if not GOOGLE_API_KEY:
        print("⚠️ No Google API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    print("\n=== Testing Google Gemini Integration ===")
    prompt = "What are the key differences between transformers and RNNs in deep learning?"
    
    # Test with SDK if available
    if GOOGLE_AVAILABLE:
        print("\n1. Using Google SDK:")
        try:
            start_time = time.time()
            response = generate_text_with_llm_extended(
                text=prompt,
                api_key=GOOGLE_API_KEY,
                provider="google",
                library="sdk",
                model_name="gemini-pro",
                temperature=0.7,
                max_tokens=800
            )
            elapsed = time.time() - start_time
            
            print(f"Prompt: {prompt}")
            print(f"Response (SDK): {response[:200]}...")
            print(f"Time taken: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error with Google SDK: {e}")
    else:
        print("Google Generative AI SDK not installed. Skipping SDK test.")
    
    # Test with direct requests
    print("\n2. Using direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_extended(
            text=prompt,
            api_key=GOOGLE_API_KEY,
            provider="google",
            library="requests",
            model_name="gemini-pro",
            temperature=0.7,
            max_tokens=800,
            enable_cache=True
        )
        elapsed = time.time() - start_time
        
        print(f"Prompt: {prompt}")
        print(f"Response (Requests): {response[:200]}...")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error with direct requests: {e}")


def demo_anthropic_claude():
    """Demonstrate using Anthropic's Claude model."""
    if not ANTHROPIC_API_KEY:
        print("⚠️ No Anthropic API key found. Set the ANTHROPIC_API_KEY environment variable.")
        return
    
    print("\n=== Testing Anthropic Claude Integration ===")
    prompt = "Compare and contrast different approaches to prompt engineering for large language models."
    
    # Test with SDK if available
    if ANTHROPIC_AVAILABLE:
        print("\n1. Using Anthropic SDK:")
        try:
            start_time = time.time()
            response = generate_text_with_llm_extended(
                text=prompt,
                api_key=ANTHROPIC_API_KEY,
                provider="anthropic",
                library="sdk",
                model_name="claude-3-sonnet-20240229",  # Use a less expensive model
                temperature=0.7,
                max_tokens=800
            )
            elapsed = time.time() - start_time
            
            print(f"Prompt: {prompt}")
            print(f"Response (SDK): {response[:200]}...")
            print(f"Time taken: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error with Anthropic SDK: {e}")
    else:
        print("Anthropic SDK not installed. Skipping SDK test.")
    
    # Test with direct requests
    print("\n2. Using direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_extended(
            text=prompt,
            api_key=ANTHROPIC_API_KEY,
            provider="anthropic",
            library="requests",
            model_name="claude-3-haiku-20240307",  # Use the smallest model to save costs
            temperature=0.7,
            max_tokens=800,
            enable_cache=True,
            version="2023-06-01"  # Specify API version
        )
        elapsed = time.time() - start_time
        
        print(f"Prompt: {prompt}")
        print(f"Response (Requests): {response[:200]}...")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error with direct requests: {e}")


def demo_cohere():
    """Demonstrate using Cohere's models."""
    if not COHERE_API_KEY:
        print("⚠️ No Cohere API key found. Set the COHERE_API_KEY environment variable.")
        return
    
    print("\n=== Testing Cohere Integration ===")
    prompt = "What are the most important considerations for deploying machine learning models in production?"
    
    # Test with SDK if available
    if COHERE_AVAILABLE:
        print("\n1. Using Cohere SDK:")
        try:
            start_time = time.time()
            response = generate_text_with_llm_extended(
                text=prompt,
                api_key=COHERE_API_KEY,
                provider="cohere",
                library="sdk",
                model_name="command",
                temperature=0.7,
                max_tokens=800
            )
            elapsed = time.time() - start_time
            
            print(f"Prompt: {prompt}")
            print(f"Response (SDK): {response[:200]}...")
            print(f"Time taken: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Error with Cohere SDK: {e}")
    else:
        print("Cohere SDK not installed. Skipping SDK test.")
    
    # Test with direct requests
    print("\n2. Using direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_extended(
            text=prompt,
            api_key=COHERE_API_KEY,
            provider="cohere",
            library="requests",
            model_name="command",
            temperature=0.7,
            max_tokens=800,
            enable_cache=True
        )
        elapsed = time.time() - start_time
        
        print(f"Prompt: {prompt}")
        print(f"Response (Requests): {response[:200]}...")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error with direct requests: {e}")


def demo_openai_integration():
    """Demonstrate that the original OpenAI integration still works."""
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("⚠️ No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
        return
    
    print("\n=== Testing Standard OpenAI Integration ===")
    prompt = "What are the best practices for fine-tuning language models for specific tasks?"
    
    # Test with our extended function but OpenAI provider
    print("\n1. Using OpenAI with direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_extended(
            text=prompt,
            api_key=openai_api_key,
            provider="openai",  # Specify OpenAI provider
            library="requests",
            model_name="gpt-3.5-turbo",  # Use a less expensive model
            temperature=0.7,
            max_tokens=800,
            enable_cache=True
        )
        elapsed = time.time() - start_time
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response[:200]}...")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error with OpenAI: {e}")


def run_demos():
    """Run all the demo functions."""
    print("DEMONSTRATING MULTI-PROVIDER LLM INTEGRATION")
    print("This example shows how to extend the generate_text_with_llm function to support multiple LLM providers.")
    print("It demonstrates integration with OpenAI, Google Gemini, Anthropic Claude, and Cohere.")
    print("Note: You need to set API keys as environment variables to run the examples.")
    
    # Run the demos
    demo_openai_integration()
    demo_google_gemini()
    demo_anthropic_claude()
    demo_cohere()


if __name__ == "__main__":
    run_demos()
"""
LLM Provider implementations for various services.

This module provides integrations with multiple LLM providers:
- Google Gemini
- Anthropic Claude
- Hugging Face
- AWS Bedrock
- OpenAI (via the base implementation)

Each provider implementation supports both SDK and direct requests approaches.
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import requests
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".meno", "llm_cache")

# Check for available libraries
GOOGLE_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False
BEDROCK_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    pass

try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    pass


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

def generate_with_gemini_sdk(
    text: str,
    api_key: str,
    model_name: str = "gemini-pro",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using Google's Generative AI SDK."""
    if not GOOGLE_AVAILABLE:
        return f"[Error: Google Generative AI SDK not installed. Install with 'pip install google-generativeai']"
    
    try:
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
    except Exception as e:
        return f"[Error: Google AI generation failed: {str(e)}]"


def generate_with_gemini_requests(
    text: str,
    api_key: str,
    model_name: str = "gemini-pro",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using Google's Generative AI API via direct requests."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    try:
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
    except Exception as e:
        return f"[Error: Google AI generation failed: {str(e)}]"


# Anthropic Claude implementations

def generate_with_anthropic_sdk(
    text: str,
    api_key: str,
    model_name: str = "claude-3-sonnet-20240229",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using Anthropic's Claude SDK."""
    if not ANTHROPIC_AVAILABLE:
        return f"[Error: Anthropic SDK not installed. Install with 'pip install anthropic']"
    
    try:
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
    except Exception as e:
        return f"[Error: Anthropic generation failed: {str(e)}]"


def generate_with_anthropic_requests(
    text: str,
    api_key: str,
    model_name: str = "claude-3-sonnet-20240229",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    api_version: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using Anthropic's Claude API via direct requests."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    api_version = api_version or "2023-06-01"
    
    try:
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
        generated_text = ""
        if "content" in result and result["content"]:
            for content_item in result["content"]:
                if "text" in content_item:
                    generated_text += content_item["text"]
        
        # Save to cache if enabled
        if enable_cache and generated_text:
            _save_to_cache(generated_text, cache_key, cache_dir)
        
        return generated_text
    except Exception as e:
        return f"[Error: Anthropic generation failed: {str(e)}]"


# HuggingFace Inference API implementations

def generate_with_huggingface_sdk(
    text: str,
    api_key: str,
    model_name: str = "meta-llama/Llama-2-70b-chat-hf",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using HuggingFace Inference API with SDK."""
    if not HUGGINGFACE_AVAILABLE:
        return f"[Error: HuggingFace Hub SDK not installed. Install with 'pip install huggingface_hub']"
    
    try:
        # Initialize the client with the API token
        client = InferenceClient(api_key=api_key)
        
        # Prepare the prompt with system instruction
        prompt = f"{system_prompt}\n\nUser: {text}\n\nAssistant:"
        
        # Prepare parameters
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": temperature > 0
        }
        
        # Add any additional parameters
        if additional_params:
            params.update(additional_params)
        
        # Make the API call
        response = client.text_generation(
            prompt=prompt,
            model=model_name,
            **params
        )
        
        return response
    except Exception as e:
        return f"[Error: HuggingFace generation failed: {str(e)}]"


def generate_with_huggingface_requests(
    text: str,
    api_key: str,
    model_name: str = "meta-llama/Llama-2-70b-chat-hf",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using HuggingFace Inference API via direct requests."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    try:
        # Prepare the prompt with system instruction
        prompt = f"{system_prompt}\n\nUser: {text}\n\nAssistant:"
        
        # Prepare parameters for the API
        params = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": temperature > 0
            },
            "options": {
                "wait_for_model": True
            }
        }
        
        # Add any additional parameters
        if additional_params and "parameters" in additional_params:
            params["parameters"].update(additional_params["parameters"])
        
        # Generate cache key if caching is enabled
        if enable_cache:
            cache_key = _generate_cache_key(
                text=prompt, 
                model=model_name, 
                params=params
            )
            
            # Try to get from cache
            cached_response = _get_from_cache(cache_key, cache_dir)
            if cached_response:
                return cached_response
        
        # Prepare the API request
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
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
        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text", "")
            # Remove the prompt from the beginning if it's included
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
        else:
            generated_text = result.get("generated_text", "")
        
        # Save to cache if enabled
        if enable_cache and generated_text:
            _save_to_cache(generated_text, cache_key, cache_dir)
        
        return generated_text
    except Exception as e:
        return f"[Error: HuggingFace generation failed: {str(e)}]"


# AWS Bedrock implementations

@lru_cache(maxsize=1)
def _get_bedrock_runtime_client(region_name: str = "us-east-1"):
    """Get a cached Bedrock runtime client."""
    return boto3.client("bedrock-runtime", region_name=region_name)


def generate_with_bedrock_sdk(
    text: str,
    api_key: str,  # AWS access key ID (can be ignored if using instance profile)
    api_secret: Optional[str] = None,  # AWS secret key (can be ignored if using instance profile)
    model_name: str = "anthropic.claude-3-sonnet-20240229",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    region_name: str = "us-east-1",
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Generate text using AWS Bedrock SDK."""
    if not BEDROCK_AVAILABLE:
        return f"[Error: AWS Boto3 SDK not installed. Install with 'pip install boto3']"
    
    try:
        # Set up AWS credentials if provided
        if api_key and api_secret:
            os.environ["AWS_ACCESS_KEY_ID"] = api_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = api_secret
        
        # Get Bedrock runtime client
        bedrock_runtime = _get_bedrock_runtime_client(region_name)
        
        # Prepare model-specific parameters based on the provider
        provider = model_name.split('.')[0].lower()
        
        if provider == "anthropic":
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": text}
                ]
            }
        elif provider == "amazon":
            body = {
                "inputText": text,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "stopSequences": []
                }
            }
        elif provider == "cohere":
            body = {
                "prompt": text,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif provider == "ai21":
            body = {
                "prompt": text,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        elif provider == "meta":
            body = {
                "prompt": f"{system_prompt}\n\nUser: {text}\n\nAssistant:",
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        else:
            return f"[Error: Unsupported model provider: {provider}]"
        
        # Add any additional parameters
        if additional_params:
            body.update(additional_params)
        
        # Make the API call
        response = bedrock_runtime.invoke_model(
            modelId=model_name,
            body=json.dumps(body)
        )
        
        # Parse the response based on the provider
        response_body = json.loads(response["body"].read().decode("utf-8"))
        
        if provider == "anthropic":
            return response_body.get("content", [{}])[0].get("text", "")
        elif provider == "amazon":
            return response_body.get("results", [{}])[0].get("outputText", "")
        elif provider == "cohere":
            return response_body.get("generations", [{}])[0].get("text", "")
        elif provider == "ai21":
            return response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
        elif provider == "meta":
            return response_body.get("generation", "")
        else:
            return "[Error: Unable to parse response]"
    except Exception as e:
        return f"[Error: AWS Bedrock generation failed: {str(e)}]"


def generate_with_bedrock_requests(
    text: str,
    api_key: str,  # AWS access key ID
    api_secret: str,  # AWS secret key
    model_name: str = "anthropic.claude-3-sonnet-20240229",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: str = "You are a helpful assistant.",
    region_name: str = "us-east-1",
    timeout: int = 60,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Generate text using AWS Bedrock via direct requests with AWS Signature V4.
    
    Note: This is a complex implementation requiring AWS SigV4 signing.
    The SDK approach is strongly recommended for Bedrock.
    
    The implementation here uses boto3 for signing, which is simpler and more reliable
    than implementing SigV4 from scratch.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    try:
        # Import boto3 here to avoid dependency errors if not installed
        if not BEDROCK_AVAILABLE:
            return f"[Error: AWS Boto3 SDK not installed. Install with 'pip install boto3']"
        
        # Prepare model-specific parameters based on the provider
        provider = model_name.split('.')[0].lower()
        
        if provider == "anthropic":
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": text}
                ]
            }
        elif provider == "amazon":
            body = {
                "inputText": text,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "stopSequences": []
                }
            }
        elif provider == "cohere":
            body = {
                "prompt": text,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif provider == "ai21":
            body = {
                "prompt": text,
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        elif provider == "meta":
            body = {
                "prompt": f"{system_prompt}\n\nUser: {text}\n\nAssistant:",
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        else:
            return f"[Error: Unsupported model provider: {provider}]"
        
        # Add any additional parameters
        if additional_params:
            body.update(additional_params)
        
        # Generate cache key if caching is enabled
        if enable_cache:
            cache_key = _generate_cache_key(
                text=f"{system_prompt}\n{text}", 
                model=model_name, 
                params=body
            )
            
            # Try to get from cache
            cached_response = _get_from_cache(cache_key, cache_dir)
            if cached_response:
                return cached_response
        
        # Set up AWS credentials
        os.environ["AWS_ACCESS_KEY_ID"] = api_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = api_secret
        
        # Use boto3 for the request, which handles SigV4 signing
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)
        
        # Make the API call
        response = bedrock_runtime.invoke_model(
            modelId=model_name,
            body=json.dumps(body)
        )
        
        # Parse the response based on the provider
        response_body = json.loads(response["body"].read().decode("utf-8"))
        
        if provider == "anthropic":
            generated_text = response_body.get("content", [{}])[0].get("text", "")
        elif provider == "amazon":
            generated_text = response_body.get("results", [{}])[0].get("outputText", "")
        elif provider == "cohere":
            generated_text = response_body.get("generations", [{}])[0].get("text", "")
        elif provider == "ai21":
            generated_text = response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
        elif provider == "meta":
            generated_text = response_body.get("generation", "")
        else:
            return "[Error: Unable to parse response]"
        
        # Save to cache if enabled
        if enable_cache and generated_text:
            _save_to_cache(generated_text, cache_key, cache_dir)
        
        return generated_text
    except Exception as e:
        return f"[Error: AWS Bedrock generation failed: {str(e)}]"


# Provider registry for extensibility
PROVIDER_REGISTRY = {
    "google": {
        "sdk": generate_with_gemini_sdk,
        "requests": generate_with_gemini_requests
    },
    "anthropic": {
        "sdk": generate_with_anthropic_sdk,
        "requests": generate_with_anthropic_requests
    },
    "huggingface": {
        "sdk": generate_with_huggingface_sdk,
        "requests": generate_with_huggingface_requests
    },
    "bedrock": {
        "sdk": generate_with_bedrock_sdk,
        "requests": generate_with_bedrock_requests
    }
}
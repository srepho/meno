"""Extended LLM API integration for meno.

This module provides an enhanced version of the generate_text_with_llm function
that supports multiple LLM providers:
- OpenAI (original implementation)
- Google Gemini 
- Anthropic Claude
- Hugging Face
- AWS Bedrock

Each provider implementation supports both SDK and direct requests approaches.
"""

from typing import Dict, Any, Optional, Union, List
import os
import time
import json
import hashlib
import logging
from pathlib import Path
import importlib.util
import requests
from functools import lru_cache

# Import the base implementation
from meno.modeling.llm_topic_labeling import generate_text_with_llm as base_generate_text_with_llm

# Import provider implementations
from meno.utils.llm_providers import (
    generate_with_gemini_sdk,
    generate_with_gemini_requests,
    generate_with_anthropic_sdk,
    generate_with_anthropic_requests,
    generate_with_huggingface_sdk,
    generate_with_huggingface_requests,
    generate_with_bedrock_sdk,
    generate_with_bedrock_requests,
    PROVIDER_REGISTRY,
    DEFAULT_CACHE_DIR
)

# Configure logging
logger = logging.getLogger(__name__)

# Check for available libraries
GOOGLE_AVAILABLE = importlib.util.find_spec("google.generativeai") is not None
ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
HUGGINGFACE_AVAILABLE = importlib.util.find_spec("huggingface_hub") is not None
BEDROCK_AVAILABLE = importlib.util.find_spec("boto3") is not None
OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None


def generate_text_with_llm_multi(
    text: str,
    api_key: str,
    api_endpoint: Optional[str] = None,
    deployment_id: Optional[str] = None,
    model_name: str = "gpt-4o",
    api_version: str = "2023-05-15",
    use_azure: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    user_prompt_prefix: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    library: str = "openai",  # Key parameter for selecting implementation
    timeout: int = 60,        # For requests timeout
    enable_cache: bool = True,  # For caching (requests implementation)
    cache_dir: Optional[str] = None,  # Cache directory
    provider: str = "openai",  # LLM provider to use: openai, google, anthropic, huggingface, bedrock
    api_secret: Optional[str] = None,  # Secondary API credential (e.g., AWS secret key for Bedrock)
    region_name: str = "us-east-1",  # Region for AWS Bedrock
    additional_params: Optional[Dict[str, Any]] = None  # Additional provider-specific parameters
) -> str:
    """Generate text using multiple LLM providers with a unified interface.
    
    This enhanced function extends the original generate_text_with_llm to support
    multiple LLM providers including Google Gemini, Anthropic Claude, Hugging Face,
    and AWS Bedrock, along with the original OpenAI support.
    
    Parameters
    ----------
    text : str
        The input text/prompt to send to the model
    api_key : str
        The API key for the LLM service
    api_endpoint : Optional[str], optional
        API endpoint URL or base URL, by default None
    deployment_id : Optional[str], optional
        Azure deployment name, required when use_azure=True, by default None
    model_name : str, optional
        Model name/ID to use, by default "gpt-4o"
        For different providers, use appropriate model names:
        - OpenAI: "gpt-4o", "gpt-3.5-turbo"
        - Google: "gemini-pro", "gemini-pro-vision"
        - Anthropic: "claude-3-opus-20240229", "claude-3-sonnet-20240229"
        - Hugging Face: "meta-llama/Llama-2-70b-chat-hf"
        - AWS Bedrock: "anthropic.claude-3-sonnet-20240229", "amazon.titan-text-express-v1"
    api_version : str, optional
        API version, by default "2023-05-15"
    use_azure : bool, optional
        Whether to use Azure OpenAI, by default False
    system_prompt : str, optional
        System prompt for the model, by default "You are a helpful assistant."
    user_prompt_prefix : str, optional
        Prefix to add before the input text, by default ""
    temperature : float, optional
        Temperature for response generation, by default 0.7
    max_tokens : int, optional
        Maximum tokens in the response, by default 1000
    library : str, optional
        Library to use for implementation: "sdk" or "requests", by default "openai"
    timeout : int, optional
        Timeout in seconds for requests implementation, by default 60
    enable_cache : bool, optional
        Whether to enable caching for requests implementation, by default True
    cache_dir : Optional[str], optional
        Directory to store cache files, by default None (uses ~/.meno/llm_cache)
    provider : str, optional
        LLM provider to use: "openai", "google", "anthropic", "huggingface", "bedrock", by default "openai"
    api_secret : Optional[str], optional
        Secondary API credential (e.g., AWS secret key for Bedrock), by default None
    region_name : str, optional
        Region for AWS Bedrock, by default "us-east-1"
    additional_params : Optional[Dict[str, Any]], optional
        Additional provider-specific parameters, by default None
    
    Returns
    -------
    str
        Generated text response from the model
    
    Notes
    -----
    For backward compatibility, if the provider is "openai", this function
    will use the original implementation which handles both OpenAI SDK and requests.
    """
    # Prepare full text with prefix if provided
    full_text = f"{user_prompt_prefix} {text}" if user_prompt_prefix else text
    
    # Set default cache directory if not provided
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Handle compatibility for OpenAI-specific implementation
    if provider.lower() == "openai":
        # Map "sdk" or "requests" to "openai" for backward compatibility
        openai_library = library
        if library == "sdk":
            openai_library = "openai"
            
        # Use the original implementation for OpenAI and Azure
        return base_generate_text_with_llm(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            deployment_id=deployment_id,
            model_name=model_name,
            api_version=api_version,
            use_azure=use_azure,
            system_prompt=system_prompt,
            user_prompt_prefix=user_prompt_prefix,
            temperature=temperature,
            max_tokens=max_tokens,
            library=openai_library,
            timeout=timeout,
            enable_cache=enable_cache,
            cache_dir=cache_dir
        )
    
    # For other providers, use the new implementation
    try:
        # Normalize provider name and library
        provider_lower = provider.lower()
        library_lower = library.lower()
        
        # Standardize library name
        if library_lower == "openai":
            library_lower = "sdk"
        
        # Check if provider is supported
        if provider_lower not in PROVIDER_REGISTRY:
            return f"[Error: Unsupported provider '{provider}'. Supported providers: 'openai', 'google', 'anthropic', 'huggingface', 'bedrock']"
        
        # Check if library is supported for this provider
        if library_lower not in PROVIDER_REGISTRY[provider_lower]:
            return f"[Error: Unsupported library '{library}' for {provider}. Use 'sdk' or 'requests']"
        
        # Get the appropriate implementation function
        generator_func = PROVIDER_REGISTRY[provider_lower][library_lower]
        
        # Prepare common parameters
        params = {
            "text": full_text,
            "api_key": api_key,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
        
        # Add library-specific parameters
        if library_lower == "requests":
            params.update({
                "timeout": timeout,
                "enable_cache": enable_cache,
                "cache_dir": cache_dir,
            })
        
        # Add provider-specific parameters
        if provider_lower == "google":
            # Google Gemini specific parameters
            if api_endpoint:
                params["api_endpoint"] = api_endpoint
            if additional_params:
                params["additional_params"] = additional_params
                
        elif provider_lower == "anthropic":
            # Anthropic Claude specific parameters
            if api_endpoint:
                params["api_endpoint"] = api_endpoint
            if api_version:
                params["api_version"] = api_version
            if additional_params:
                params["additional_params"] = additional_params
                
        elif provider_lower == "huggingface":
            # Hugging Face specific parameters
            if api_endpoint:
                params["api_endpoint"] = api_endpoint
            if additional_params:
                params["additional_params"] = additional_params
                
        elif provider_lower == "bedrock":
            # AWS Bedrock specific parameters
            if api_secret:
                params["api_secret"] = api_secret
            if region_name:
                params["region_name"] = region_name
            if additional_params:
                params["additional_params"] = additional_params
        
        # Call the implementation function
        return generator_func(**params)
    
    except Exception as e:
        return f"[Error: {provider} generation failed: {str(e)}]"


# For backward compatibility and easy migration
generate_text_with_llm_extended = generate_text_with_llm_multi
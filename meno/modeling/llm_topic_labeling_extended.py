"""Extended LLM API integration for meno.

This module provides an enhanced version of the generate_text_with_llm function
that supports multiple LLM providers:
- OpenAI (original implementation)
- Azure OpenAI (dedicated provider)
- Google Gemini 
- Anthropic Claude
- Hugging Face
- AWS Bedrock

Each provider implementation supports both SDK and direct requests approaches.
"""

from typing import Dict, Any, Optional, Union, List
import os
import logging

# Import the base implementation
from meno.modeling.llm_topic_labeling import generate_text_with_llm as base_generate_text_with_llm

# Import provider implementations
try:
    from meno.utils.llm_providers import PROVIDER_REGISTRY, DEFAULT_CACHE_DIR
except ImportError:
    # Define placeholders if module not available
    PROVIDER_REGISTRY = {}
    DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".meno", "llm_cache")

# Configure logging
logger = logging.getLogger(__name__)


def generate_text_with_llm_multi(
    text: str,
    api_key: Optional[str] = None,
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
    # Set default cache directory if not provided
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Handle compatibility for OpenAI-specific implementation
    if provider.lower() == "openai":
        # Map "sdk" or "requests" to "openai" for backward compatibility
        openai_library = library
        if library == "sdk":
            openai_library = "openai"
            
        # Use the original implementation for OpenAI
        return base_generate_text_with_llm(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            deployment_id=deployment_id,
            model_name=model_name,
            api_version=api_version,
            use_azure=use_azure,  # This can still be True for backward compatibility
            system_prompt=system_prompt,
            user_prompt_prefix=user_prompt_prefix,
            temperature=temperature,
            max_tokens=max_tokens,
            library=openai_library,
            timeout=timeout,
            enable_cache=enable_cache,
            cache_dir=cache_dir
        )
    
    # Dedicated Azure OpenAI provider
    elif provider.lower() == "azure":
        # Force use_azure to True for the Azure provider
        azure_use_azure = True
        
        # Ensure we have the required parameters
        if not deployment_id:
            logger.warning("deployment_id is required for Azure OpenAI but was not provided")
            
        if not api_endpoint:
            logger.warning("api_endpoint is required for Azure OpenAI but was not provided")
        
        # Map "sdk" or "requests" to appropriate library for Azure
        azure_library = library
        if library == "sdk":
            azure_library = "openai"
        
        # Check if we have provider-specific implementation
        if azure_library in PROVIDER_REGISTRY.get("azure", {}):
            try:
                # Try to use provider-specific implementation if available
                provider_func = PROVIDER_REGISTRY["azure"][azure_library]
                return provider_func(
                    text=text,
                    api_key=api_key,
                    api_endpoint=api_endpoint,
                    deployment_id=deployment_id,
                    model_name=model_name,  
                    api_version=api_version,
                    system_prompt=system_prompt,
                    user_prompt_prefix=user_prompt_prefix,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    enable_cache=enable_cache,
                    cache_dir=cache_dir,
                    **additional_params or {}
                )
            except Exception as e:
                logger.warning(f"Error using Azure provider-specific implementation: {e}. Falling back to base implementation.")
                # Fall back to base implementation
                pass
                
        # Use the original implementation with Azure settings
        return base_generate_text_with_llm(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            deployment_id=deployment_id,
            model_name=model_name,  # Note: For Azure, model_name is ignored, deployment_id is used
            api_version=api_version,
            use_azure=azure_use_azure,  # Always True for the Azure provider
            system_prompt=system_prompt,
            user_prompt_prefix=user_prompt_prefix,
            temperature=temperature,
            max_tokens=max_tokens,
            library=azure_library,
            timeout=timeout,
            enable_cache=enable_cache,
            cache_dir=cache_dir
        )
    
    # For other providers (placeholder implementation)
    return f"[Placeholder] {provider.title()} implementation using {library} with model {model_name}"


# For backward compatibility and easy migration
generate_text_with_llm_extended = generate_text_with_llm_multi
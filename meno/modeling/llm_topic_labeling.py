"""LLM-based topic labeling for topic models.

This module provides a way to generate human-readable topic names using Language Models
(LLMs). It supports both local models via HuggingFace and remote models via OpenAI.
It also supports batch processing of texts for classification with efficient token usage.
"""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar, Callable, Generator, Set, Type
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import re
import os
import warnings
from tqdm import tqdm
import importlib.util
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib
import functools
import pickle
from collections import deque, defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import requests

logger = logging.getLogger(__name__)

# Check for available LLM backends
OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None

if TRANSFORMERS_AVAILABLE:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from transformers.utils import is_torch_available
        TORCH_AVAILABLE = is_torch_available()
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = False


class RateLimiter:
    """Rate limiter to control the rate of API requests.
    
    This class implements a token bucket algorithm to rate limit requests.
    
    Parameters
    ----------
    requests_per_minute : int, optional
        Maximum number of requests per minute, by default 60
    burst_limit : int, optional
        Maximum number of requests that can be made in a burst, by default None
        If None, burst_limit equals requests_per_minute
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: Optional[int] = None):
        """Initialize the rate limiter."""
        self.requests_per_minute = max(1, requests_per_minute)
        self.burst_limit = burst_limit if burst_limit is not None else requests_per_minute
        self.token_bucket = min(self.requests_per_minute, self.burst_limit)
        self.last_refill_time = datetime.now()
        self.refill_rate = self.requests_per_minute / 60.0  # tokens per second
        self.lock = threading.Lock()
    
    def _refill_bucket(self) -> None:
        """Refill the token bucket based on elapsed time."""
        now = datetime.now()
        time_passed = (now - self.last_refill_time).total_seconds()
        self.last_refill_time = now
        
        # Calculate tokens to add based on time passed
        new_tokens = time_passed * self.refill_rate
        
        # Add tokens to bucket, up to burst limit
        self.token_bucket = min(self.token_bucket + new_tokens, self.burst_limit)
    
    def acquire(self) -> bool:
        """Acquire a token from the bucket, waiting if necessary.
        
        Returns
        -------
        bool
            True if token acquired, False if it would exceed the rate limit
        """
        with self.lock:
            # Refill the bucket first
            self._refill_bucket()
            
            # Check if we can take a token
            if self.token_bucket >= 1:
                self.token_bucket -= 1
                return True
            else:
                # Calculate wait time to get a token
                wait_time = (1 - self.token_bucket) / self.refill_rate
                
                # If wait time is reasonable (less than 5 seconds), wait and retry
                if wait_time <= 5.0:
                    time.sleep(wait_time)
                    self._refill_bucket()
                    self.token_bucket -= 1
                    return True
                
                return False
    
    def wait_for_token(self) -> None:
        """Wait until a token is available and acquire it."""
        while not self.acquire():
            time.sleep(0.1)  # Small sleep to avoid busy waiting


class LLMTopicLabeler:
    """LLM-based topic labeler to generate human-readable topic names and classify texts.
    
    This class provides methods to generate descriptive topic names for topic models
    using Language Models (LLMs). It supports both local HuggingFace models and
    OpenAI API if available. It also supports batch processing of texts for classification
    with efficient token usage.
    
    Parameters
    ----------
    model_type : str, optional
        Type of model to use, by default "openai"
        Options: "local", "openai", "auto"
        If "auto", will use OpenAI if available, otherwise fall back to local model
    model_name : str, optional
        Name of the model to use
        For OpenAI standard API, this is the model name (e.g., "gpt-3.5-turbo", "gpt-4o")
        For Azure OpenAI, this is the deployment name
        For local models, default is "google/flan-t5-small"
    max_new_tokens : int, optional
        Maximum number of tokens to generate, by default 50
    temperature : float, optional
        Temperature for generation, by default 0.7
    enable_fallback : bool, optional
        Whether to enable fallback to rule-based labeling if LLM fails, by default True
    device : str, optional
        Device to use for local models, by default "auto"
        Options: "auto", "cpu", "cuda", "mps"
    verbose : bool, optional
        Whether to show verbose output, by default False
    api_key : Optional[str], optional
        API key for OpenAI or Azure OpenAI, by default None
        If None and model_type="openai", will try to use the OPENAI_API_KEY environment variable
    api_endpoint : Optional[str], optional
        API endpoint URL for OpenAI or Azure OpenAI, by default None
        For Azure, this is the azure_endpoint (e.g., "https://your-resource.openai.azure.com")
        For OpenAI standard API, this can be used for custom base URLs
    api_version : str, optional
        API version to use with OpenAI API, by default "2023-05-15"
        Required for Azure OpenAI, optional for standard OpenAI
    use_azure : bool, optional
        Whether to use Azure OpenAI client, by default True
    max_token_limit : Optional[int], optional
        Maximum allowed token length for prompts, by default None
        If exceeded, will fail with a warning unless enable_fallback is True
    max_parallel_requests : int, optional
        Maximum number of parallel requests to make when batching, by default 4
    requests_per_minute : Optional[int], optional
        Maximum number of requests per minute for rate limiting, by default None
        If None, no rate limiting is applied
    burst_limit : Optional[int], optional
        Maximum number of requests that can be made in a burst for rate limiting, by default None
        If None, burst_limit equals requests_per_minute
    system_prompt_template : Optional[str], optional
        Template for system prompt in OpenAI messages, by default None
        Uses a default template if None
    user_prompt_template : Optional[str], optional
        Template for user prompt in OpenAI messages, by default None
        Uses a default template if None
    batch_size : Optional[int], optional
        Maximum number of texts to process in a single API call, by default 20
        Actual batch size may be smaller depending on token limits
    deduplicate : bool, optional
        Whether to deduplicate similar texts before processing, by default False
    deduplication_threshold : float, optional
        Similarity threshold for deduplication (0.0-1.0), by default 0.92
        Higher values are more strict (require more similarity to consider as duplicates)
    enable_cache : bool, optional
        Whether to cache classification results to disk, by default True
    cache_dir : Optional[str], optional
        Directory to store cache files, by default "./.meno_cache"
    cache_ttl : int, optional
        Time-to-live for cache entries in seconds, by default 86400 (1 day)
    
    Attributes
    ----------
    model_type : str
        Type of model being used
    model_name : str
        Name of the model being used
    model : Any
        The loaded model (if using local model)
    tokenizer : Any
        The tokenizer for the model (if using local model)
    rate_limiter : Optional[RateLimiter]
        Rate limiter for controlling API request rate
    confidence_scores : Dict[str, float]
        Dictionary of confidence scores for the last batch of classifications
        
    Example
    -------
    >>> from meno.modeling.llm_topic_labeling import LLMTopicLabeler
    >>> labeler = LLMTopicLabeler(model_type="openai", model_name="gpt-3.5-turbo")
    >>> texts = ["This is about technology and AI", "The stock market fell by 2% today"]
    >>> results = labeler.classify_texts(texts)
    >>> confidences = labeler.confidence_scores
    """
    
    def __init__(
        self,
        model_type: str = "openai",
        model_name: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        enable_fallback: bool = True,
        device: str = "auto",
        verbose: bool = False,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None, 
        api_version: str = "2023-05-15",
        use_azure: Optional[bool] = True,
        max_token_limit: Optional[int] = None,
        max_parallel_requests: int = 4,
        requests_per_minute: Optional[int] = None,
        burst_limit: Optional[int] = None,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        batch_size: int = 20,
        deduplicate: bool = False,
        deduplication_threshold: float = 0.92,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 86400,
    ):
        """Initialize the LLM topic labeler."""
        self.verbose = verbose
        self.enable_fallback = enable_fallback
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_token_limit = max_token_limit
        self.max_parallel_requests = max_parallel_requests
        self.batch_size = batch_size
        self.deduplicate = deduplicate
        self.deduplication_threshold = deduplication_threshold
        
        # Caching settings
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".meno_cache")
        self.cache_ttl = cache_ttl
        
        # Initialize confidence scores
        self.confidence_scores = {}
        
        # Set default prompt templates
        self.system_prompt_template = system_prompt_template or "You are a helpful assistant that classifies text into topics."
        self.user_prompt_template = user_prompt_template or "Classify the following text into the most appropriate topic: {{text}}"
        
        # Initialize rate limiter if requests_per_minute is specified
        self.rate_limiter = None
        if requests_per_minute is not None:
            self.rate_limiter = RateLimiter(
                requests_per_minute=requests_per_minute,
                burst_limit=burst_limit
            )
            if self.verbose:
                logger.info(f"Rate limiter initialized with {requests_per_minute} requests per minute")
                
        # Initialize cache
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            if self.verbose:
                logger.info(f"Cache enabled, using directory: {self.cache_dir}")
        
        # Determine model type
        if model_type == "auto":
            if OPENAI_AVAILABLE:
                self.model_type = "openai"
            elif TRANSFORMERS_AVAILABLE:
                self.model_type = "local"
            else:
                raise ImportError(
                    "No LLM backend is available. Please install either 'openai' or 'transformers' package."
                )
        else:
            self.model_type = model_type
            
        # Set default model names
        if model_name is None:
            if self.model_type == "openai":
                self.model_name = "gpt-3.5-turbo"
            elif self.model_type == "local":
                self.model_name = "google/flan-t5-small"
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            self.model_name = model_name
            
        # Initialize the model based on type
        if self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI package is not installed. Please install it with 'pip install openai'."
                )
            
            import openai
            
            # Set up client configuration
            client_kwargs = {}
            
            # Configure API key - rename from parameter name to avoid confusion
            openai_api_key = api_key
            
            if not openai_api_key:
                # Try to get from environment
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning("No API key provided. Please provide an API key using api_key parameter or set OPENAI_API_KEY environment variable.")
            
            # Initialize appropriate client based on use_azure flag
            if use_azure:
                # Use AzureOpenAI client
                if not api_endpoint:
                    raise ValueError("api_endpoint is required when using Azure OpenAI")
                
                logger.info(f"Using Azure OpenAI with deployment: {self.model_name}")
                logger.info(f"Using API version: {api_version}")
                
                # Initialize Azure OpenAI client
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    api_key=openai_api_key,
                    azure_endpoint=api_endpoint,
                    api_version=api_version
                )
                
                logger.info(f"Using Azure OpenAI endpoint: {api_endpoint}")
                self.is_azure = True
            else:
                # Use standard OpenAI client
                from openai import OpenAI
                
                client_kwargs = {"api_key": openai_api_key}
                
                # Configure standard OpenAI client parameters
                if api_endpoint:
                    client_kwargs["base_url"] = api_endpoint
                
                # Initialize standard OpenAI client
                self.client = OpenAI(**client_kwargs)
                logger.info(f"Using OpenAI model: {self.model_name}")
                
                # Log custom API configuration if used
                if api_endpoint:
                    logger.info(f"Using custom API endpoint: {api_endpoint}")
                
                self.is_azure = False
                
        elif self.model_type == "local":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers package is not installed. Please install it with 'pip install transformers'."
                )
                
            # Determine device
            if device == "auto":
                if TORCH_AVAILABLE:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else \
                                 "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else \
                                 "cpu"
                else:
                    self.device = "cpu"
            else:
                self.device = device
                
            logger.info(f"Loading local model: {self.model_name} on {self.device}")
            
            # Set model path
            model_path = self.model_name
            
            # Check if it's a local path
            if Path(self.model_name).exists() and Path(self.model_name).is_dir():
                logger.info(f"Using local model files from: {self.model_name}")
                model_path = str(Path(self.model_name).absolute())
            
            # Load tokenizer and model
            try:
                # Try to load the tokenizer
                tokenizer_kwargs = {}
                if self.device != "cpu":
                    # Use disk offloading for large models
                    tokenizer_kwargs["use_fast"] = True
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    **tokenizer_kwargs
                )
                
                # Configure model loading parameters
                model_kwargs = {}
                
                # Add device parameters for GPU acceleration or offloading
                if self.device != "cpu":
                    # Check if model is a 8bit-compatible size
                    small_model_prefixes = ["google/flan-t5", "facebook/opt-125m", "EleutherAI/pythia", 
                                           "bigscience/bloom-560m", "facebook/opt-350m", 
                                           "microsoft/phi-1", "stabilityai/stablelm-base-alpha-3b"]
                    
                    is_small_model = any(self.model_name.startswith(prefix) for prefix in small_model_prefixes) or \
                                    "1b" in self.model_name.lower() or \
                                    "2b" in self.model_name.lower() or \
                                    "3b" in self.model_name.lower() or \
                                    "small" in self.model_name.lower()
                    
                    # Appropriate device mapping based on model size and available hardware
                    if is_small_model or "7b" in self.model_name.lower() or "8b" in self.model_name.lower():
                        if self.device == "cuda":
                            model_kwargs["device_map"] = "auto"
                            
                            # Use quantization if available for larger models
                            if not is_small_model and importlib.util.find_spec("bitsandbytes") is not None:
                                model_kwargs["load_in_8bit"] = True
                                logger.info("Using 8-bit quantization for memory efficiency")
                    else:
                        # For larger models or CPU, use best defaults
                        model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
                        
                        # Try 8-bit quantization for large models if bitsandbytes is available
                        if self.device == "cuda" and importlib.util.find_spec("bitsandbytes") is not None:
                            model_kwargs["load_in_8bit"] = True
                            logger.info("Using 8-bit quantization for memory efficiency")
                
                # Try to load the model directly first
                try:
                    logger.info(f"Loading model with parameters: {model_kwargs}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        **model_kwargs
                    )
                    
                    # Configure pipeline
                    pipeline_kwargs = {
                        "model": self.model,
                        "tokenizer": self.tokenizer,
                        "max_length": self.max_new_tokens + 512,  # Add buffer for prompt
                        "temperature": self.temperature
                    }
                    
                    # Set device for pipeline
                    if self.device == "cuda":
                        pipeline_kwargs["device"] = 0
                    elif self.device == "mps":
                        pipeline_kwargs["device"] = -1  # MPS support varies by transformers version
                    else:
                        pipeline_kwargs["device"] = -1
                    
                    self.pipeline = pipeline(
                        "text-generation",
                        **pipeline_kwargs
                    )
                    
                except Exception as e:
                    # Fallback to using pipeline directly
                    logger.warning(f"Failed to load model directly, falling back to pipeline: {e}")
                    self.model = None
                    
                    self.pipeline = pipeline(
                        "text-generation",
                        model=model_path,
                        device=0 if self.device == "cuda" else -1,
                    )
            except Exception as e:
                raise ImportError(f"Failed to load the model: {e}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        logger.info("LLM Topic Labeler initialized successfully")
        
    def generate_topic_name(
        self,
        keywords: List[str],
        example_docs: Optional[List[str]] = None,
        detailed: bool = False,
    ) -> str:
        """Generate a human-readable topic name based on keywords and example documents.
        
        Parameters
        ----------
        keywords : List[str]
            List of keywords representing the topic
        example_docs : Optional[List[str]], optional
            List of example documents for the topic, by default None
        detailed : bool, optional
            Whether to generate a more detailed topic description, by default False
            
        Returns
        -------
        str
            Generated topic name
        """
        # Build the prompt
        prompt = self._build_prompt(keywords, example_docs, detailed)
        
        # Try to generate a name with the LLM
        try:
            if self.model_type == "openai":
                return self._generate_openai(prompt)
            elif self.model_type == "local":
                return self._generate_local(prompt)
        except Exception as e:
            logger.warning(f"Failed to generate topic name with LLM: {e}")
            if not self.enable_fallback:
                raise
            
        # Fallback to rule-based labeling if LLM fails
        return self._fallback_labeling(keywords)
        
    def _build_prompt(
        self,
        keywords: List[str],
        example_docs: Optional[List[str]] = None,
        detailed: bool = False,
    ) -> str:
        """Build a prompt for the LLM based on keywords and example documents.
        
        Parameters
        ----------
        keywords : List[str]
            List of keywords representing the topic
        example_docs : Optional[List[str]], optional
            List of example documents for the topic, by default None
        detailed : bool, optional
            Whether to generate a more detailed topic description, by default False
            
        Returns
        -------
        str
            Generated prompt
        """
        # Format keywords
        keyword_str = ", ".join(keywords[:20])  # Limit to top 20 keywords
        
        # Basic prompt
        if detailed:
            base_prompt = (
                f"You are a topic modeling assistant. Given the following keywords and example documents, "
                f"generate a descriptive and specific topic name that captures the essence of this topic. "
                f"The name should be a concise phrase (4-8 words) that describes the topic clearly.\n\n"
                f"Keywords: {keyword_str}\n"
            )
        else:
            base_prompt = (
                f"You are a topic modeling assistant. Given the following keywords, "
                f"generate a concise topic name (2-5 words) that captures the main theme.\n\n"
                f"Keywords: {keyword_str}\n"
            )
        
        # Add example documents if available
        if example_docs and len(example_docs) > 0:
            # Select up to 3 example documents, and truncate them to 100 words each
            sample_docs = example_docs[:3]
            truncated_docs = [" ".join(doc.split()[:100]) + ("..." if len(doc.split()) > 100 else "") 
                             for doc in sample_docs]
            docs_str = "\n".join([f"- {doc}" for doc in truncated_docs])
            base_prompt += f"\nExample documents:\n{docs_str}\n"
        
        # Finish the prompt
        if detailed:
            base_prompt += "\nGenerate a descriptive topic name that captures the specific subject matter:"
        else:
            base_prompt += "\nTopic name:"
        
        # Check token limit if specified
        if self.max_token_limit is not None:
            token_count = self._count_tokens(base_prompt)
            if token_count > self.max_token_limit:
                if self.enable_fallback:
                    logger.warning(
                        f"Prompt token count ({token_count}) exceeds limit ({self.max_token_limit}). "
                        "Using fallback labeling instead."
                    )
                    raise ValueError(f"Token limit exceeded: {token_count} > {self.max_token_limit}")
                else:
                    raise ValueError(
                        f"Prompt token count ({token_count}) exceeds limit ({self.max_token_limit}) "
                        "and fallback is disabled."
                    )
                
        return base_prompt
        
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Parameters
        ----------
        text : str
            The text to count tokens for
            
        Returns
        -------
        int
            The number of tokens in the text
        """
        if self.model_type == "openai":
            # Use tiktoken if available, otherwise estimate
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model(self.model_name)
                return len(encoding.encode(text))
            except ImportError:
                # Rough estimate based on GPT tokenization pattern (approx 4 chars per token)
                return len(text) // 4
        elif self.model_type == "local" and hasattr(self, "tokenizer"):
            # Use the model's tokenizer if available
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate based on average token length
            return len(text.split())
    
    def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a topic name using OpenAI API.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the API
        system_prompt : Optional[str], optional
            System prompt to use, by default None
            If None, uses a default system prompt
            
        Returns
        -------
        str
            Generated topic name
        """
        try:
            # Apply rate limiting if enabled
            if self.rate_limiter is not None:
                self.rate_limiter.wait_for_token()
                if self.verbose:
                    logger.debug("Rate limiter token acquired for OpenAI request")
            
            # Use provided system prompt or default
            if system_prompt is None:
                system_prompt = "You are a topic modeling assistant that generates concise, descriptive topic names."
            
            # Create messages array which is required for both client types
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use the appropriate parameter name based on whether we're using Azure or standard OpenAI
            if self.is_azure:
                # For Azure OpenAI, use deployment_id parameter
                response = self.client.chat.completions.create(
                    deployment_id=self.model_name,  # Use deployment name for Azure
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
            else:
                # For standard OpenAI, use model parameter
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
            
            # Extract the generated text
            result = response.choices[0].message.content.strip()
            
            # Clean up the result (remove quotes, normalize whitespace)
            result = re.sub(r'^["\']|["\']$', '', result)
            result = re.sub(r'\s+', ' ', result).strip()
            
            if self.verbose:
                logger.info(f"Generated topic name: {result}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise
            
    def _batch_generate_openai(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None
    ) -> Tuple[List[str], Dict[int, float]]:
        """Generate responses for multiple prompts using a single OpenAI API call.
        
        Parameters
        ----------
        prompts : List[str]
            List of prompts to send to the API
        system_prompt : Optional[str], optional
            System prompt to use, by default None
            If None, uses the class's system_prompt_template
            
        Returns
        -------
        Tuple[List[str], Dict[int, float]]
            Tuple containing:
            - List of generated responses
            - Dictionary mapping prompt indices to confidence scores
        """
        # Check cache first for each prompt
        cached_results = []
        cached_indices = []
        confidence_scores = {}
        
        if self.enable_cache:
            for i, prompt in enumerate(prompts):
                cache_key = self._get_cache_key(prompt, system_prompt or self.system_prompt_template, self.user_prompt_template)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    result, confidence = cached_result
                    cached_results.append((i, result))
                    cached_indices.append(i)
                    confidence_scores[i] = confidence
                    
                    if self.verbose:
                        logger.debug(f"Cache hit for prompt {i}")
        
        # Filter out prompts that were found in cache
        if cached_indices:
            filtered_prompts = [p for i, p in enumerate(prompts) if i not in cached_indices]
        else:
            filtered_prompts = prompts
            
        # If all results were in cache, return early
        if not filtered_prompts:
            # Reconstruct the full result list
            all_results = [""] * len(prompts)
            for i, result in cached_results:
                all_results[i] = result
                
            return all_results, confidence_scores
                    
        try:
            # Apply rate limiting if enabled
            if self.rate_limiter is not None:
                self.rate_limiter.wait_for_token()
                if self.verbose:
                    logger.debug("Rate limiter token acquired for batch OpenAI request")
            
            # Use provided system prompt or default
            if system_prompt is None:
                system_prompt = self.system_prompt_template
            
            # Create a combined message with all prompts
            combined_prompt = "Process the following texts and provide a classification for each:\n\n"
            for i, prompt in enumerate(filtered_prompts):
                combined_prompt += f"TEXT {i+1}:\n{prompt}\n\n"
            
            combined_prompt += "FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n"
            combined_prompt += "TEXT 1: [classification] (confidence: HIGH/MEDIUM/LOW)\nTEXT 2: [classification] (confidence: HIGH/MEDIUM/LOW)\n..."
                
            # Create messages array which is required for both client types
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_prompt}
            ]
            
            # Use the appropriate parameter name based on whether we're using Azure or standard OpenAI
            if self.is_azure:
                # For Azure OpenAI, use deployment_id parameter
                response = self.client.chat.completions.create(
                    deployment_id=self.model_name,  # Use deployment name for Azure
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
            else:
                # For standard OpenAI, use model parameter
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
            
            # Extract the generated text
            result = response.choices[0].message.content.strip()
            
            # Parse the result to extract individual responses and confidence
            parsed_results = []
            result_lines = result.split('\n')
            
            # Map from index in filtered_prompts to index in original prompts
            if cached_indices:
                idx_map = {new_idx: old_idx for new_idx, old_idx in 
                          enumerate([i for i in range(len(prompts)) if i not in cached_indices])}
            else:
                idx_map = {i: i for i in range(len(filtered_prompts))}
                
            for i, prompt in enumerate(filtered_prompts):
                result_prefix = f"TEXT {i+1}: "
                
                for line in result_lines:
                    if line.startswith(result_prefix):
                        # Try to extract classification and confidence
                        confidence_value = 0.7  # Default medium confidence
                        
                        # Try to parse confidence if included
                        confidence_match = re.search(r'\(confidence:\s*(HIGH|MEDIUM|LOW)\)', line, re.IGNORECASE)
                        if confidence_match:
                            confidence_level = confidence_match.group(1).upper()
                            if confidence_level == "HIGH":
                                confidence_value = 0.9
                            elif confidence_level == "MEDIUM":
                                confidence_value = 0.7
                            elif confidence_level == "LOW":
                                confidence_value = 0.5
                                
                            # Remove the confidence part for the final classification
                            classification_text = line[:confidence_match.start()].strip()
                            classification_text = classification_text[len(result_prefix):].strip()
                        else:
                            # Just extract the classification without confidence
                            classification_text = line[len(result_prefix):].strip()
                        
                        # Clean up the result
                        classification = re.sub(r'^["\']|["\']$', '', classification_text)
                        classification = re.sub(r'\s+', ' ', classification).strip()
                        
                        # Map to the original index
                        orig_idx = idx_map[i]
                        parsed_results.append((orig_idx, classification))
                        confidence_scores[orig_idx] = confidence_value
                        
                        # Cache the result
                        if self.enable_cache:
                            cache_key = self._get_cache_key(prompt, system_prompt, self.user_prompt_template)
                            self._cache_result(cache_key, classification, confidence_value)
                            
                        break
                else:
                    # If no match found, add a placeholder
                    orig_idx = idx_map[i]
                    parsed_results.append((orig_idx, "Unclassified"))
                    confidence_scores[orig_idx] = 0.3  # Low confidence for unclassified
            
            # Combine cached and new results
            all_results = [""] * len(prompts)
            
            # Add cached results
            for i, result in cached_results:
                all_results[i] = result
                
            # Add new results
            for i, result in parsed_results:
                all_results[i] = result
                
            if self.verbose:
                logger.info(f"Generated {len(parsed_results)} classifications in batch mode")
                logger.info(f"Used {len(cached_results)} cached results")
                
            return all_results, confidence_scores
            
        except Exception as e:
            logger.error(f"Error in batch generation with OpenAI: {e}")
            raise
    
    def _generate_local(self, prompt: str) -> str:
        """Generate a topic name using a local HuggingFace model.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model
            
        Returns
        -------
        str
            Generated topic name
        """
        try:
            # Apply rate limiting if enabled
            if self.rate_limiter is not None:
                self.rate_limiter.wait_for_token()
                if self.verbose:
                    logger.debug("Rate limiter token acquired for local model request")
                    
            # Generate text
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_return_sequences=1,
                do_sample=True,
            )
            
            # Extract the generated text
            if isinstance(outputs, list):
                result = outputs[0]["generated_text"]
            else:
                result = outputs
                
            # Remove the prompt from the result (if it's included)
            if result.startswith(prompt):
                result = result[len(prompt):].strip()
                
            # Clean up the result (remove quotes, normalize whitespace)
            result = re.sub(r'^["\']|["\']$', '', result)
            result = re.sub(r'\s+', ' ', result).strip()
            
            if self.verbose:
                logger.info(f"Generated topic name: {result}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating with local model: {e}")
            raise
    
    def _fallback_labeling(self, keywords: List[str]) -> str:
        """Generate a topic name using a rule-based approach as fallback.
        
        Parameters
        ----------
        keywords : List[str]
            List of keywords representing the topic
            
        Returns
        -------
        str
            Generated topic name
        """
        if not keywords:
            return "Unknown Topic"
            
        # Use top keyword as main theme
        main_theme = keywords[0].title()
        
        # Add 2-3 supporting keywords if available
        if len(keywords) > 1:
            supporting = ", ".join(keywords[1:min(4, len(keywords))])
            return f"{main_theme}: {supporting}"
        else:
            return main_theme
    
    def _process_topic_batch(
        self,
        topic_batch: List[Tuple[int, List[str], Optional[List[str]], bool]],
    ) -> List[Tuple[int, str]]:
        """Process a batch of topics in parallel.
        
        Parameters
        ----------
        topic_batch : List[Tuple[int, List[str], Optional[List[str]], bool]]
            List of tuples containing (topic_id, keywords, example_docs, detailed)
            
        Returns
        -------
        List[Tuple[int, str]]
            List of tuples containing (topic_id, topic_name)
        """
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            # Submit all tasks
            future_to_topic = {
                executor.submit(
                    self.generate_topic_name, 
                    keywords, 
                    example_docs, 
                    detailed
                ): (topic_id, keywords)
                for topic_id, keywords, example_docs, detailed in topic_batch
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_topic):
                topic_id, keywords = future_to_topic[future]
                try:
                    topic_name = future.result()
                    results.append((topic_id, topic_name))
                except Exception as e:
                    logger.warning(f"Failed to generate name for topic {topic_id}: {e}")
                    # Fallback to simple naming
                    topic_names = self._fallback_labeling(keywords)
                    results.append((topic_id, topic_names))
                    
        return results
        
    def classify_texts(
        self,
        texts: List[str],
        categories: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        batch_size: Optional[int] = None,
        progress_bar: Union[bool, str, Dict[str, Any]] = True,
        deduplicate: Optional[bool] = None,
        deduplication_threshold: Optional[float] = None,
    ) -> List[str]:
        """Classify a list of texts using LLM.
        
        This method efficiently processes multiple texts for classification,
        using batching and deduplication to optimize API calls.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to classify
        categories : Optional[List[str]], optional
            List of predefined categories to choose from, by default None
            If provided, the model will classify into these categories
        system_prompt : Optional[str], optional
            Custom system prompt, by default None
            If None, uses the class's system_prompt_template
        user_prompt_template : Optional[str], optional
            Custom user prompt template, by default None
            If None, uses the class's user_prompt_template
            Should include {{text}} placeholder for text insertion
        batch_size : Optional[int], optional
            Maximum number of texts per batch, by default None
            If None, uses the class's batch_size
        progress_bar : bool, optional
            Whether to show a progress bar, by default True
        deduplicate : Optional[bool], optional
            Whether to deduplicate similar texts, by default None
            If None, uses the class's deduplicate setting
        deduplication_threshold : Optional[float], optional
            Similarity threshold for deduplication, by default None
            If None, uses the class's deduplication_threshold
            
        Returns
        -------
        List[str]
            List of classification results (one per input text)
        """
        if len(texts) == 0:
            return []
            
        # Set defaults from class attributes if not specified
        if batch_size is None:
            batch_size = self.batch_size
            
        if deduplicate is None:
            deduplicate = self.deduplicate
            
        if deduplication_threshold is None:
            deduplication_threshold = self.deduplication_threshold
            
        if user_prompt_template is None:
            user_prompt_template = self.user_prompt_template
            
        if system_prompt is None:
            system_prompt = self.system_prompt_template
            
        # Modify system prompt if categories are provided
        if categories and system_prompt == self.system_prompt_template:
            category_list = ", ".join(categories)
            system_prompt = f"You are a helpful assistant that classifies text into one of these categories: {category_list}."
            
        # Set up progress tracking
        total_texts = len(texts)
        use_simple_progress = False
        simple_progress_interval = 5  # Default interval
        
        # Check if progress_bar is a string or dictionary for configuration
        if isinstance(progress_bar, str) and progress_bar.lower() == "simple":
            use_simple_progress = True
        elif isinstance(progress_bar, dict) and progress_bar.get("type", "") == "simple":
            use_simple_progress = True
            # Allow customizing the interval
            if "interval" in progress_bar and isinstance(progress_bar["interval"], int) and progress_bar["interval"] > 0:
                simple_progress_interval = progress_bar["interval"]
        
        if progress_bar and not use_simple_progress:
            try:
                pbar = tqdm(total=total_texts, desc="Classifying texts")
            except Exception as e:
                logger.warning(f"Failed to create tqdm progress bar: {e}. Using simple progress.")
                use_simple_progress = True
                
        if use_simple_progress:
            print(f"Classifying {total_texts} texts... (progress updates every {simple_progress_interval} items)")
            
        # Check if we're in OpenAI mode (only OpenAI supports batch processing)
        if self.model_type != "openai":
            logger.warning("Batch processing is only available with OpenAI models. Using parallel processing instead.")
            
            # Define a function to process a single text
            def process_text(text):
                # Format the prompt
                prompt = user_prompt_template.replace("{{text}}", text)
                
                if self.model_type == "openai":
                    return self._generate_openai(prompt, system_prompt)
                else:
                    return self._generate_local(prompt)
                    
            # Process texts in parallel
            results = []
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    
                    # Submit all tasks
                    futures = [executor.submit(process_text, text) for text in batch]
                    
                    # Process completed tasks
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Failed to classify text: {e}")
                            results.append("Unclassified")
                            
                        if progress_bar:
                            if use_simple_progress:
                                if len(results) % simple_progress_interval == 0:  # Show progress at specified interval
                                    print(f"  Progress: {len(results)}/{total_texts} texts processed")
                            else:
                                pbar.update(1)
                            
            if progress_bar:
                if use_simple_progress:
                    print(f"  Completed: {total_texts}/{total_texts} texts processed")
                else:
                    pbar.close()
                
            return results
            
        # Deduplication (for OpenAI mode)
        if deduplicate:
            # Create duplicate map
            duplicate_map = self._identify_fuzzy_duplicates(texts, deduplication_threshold)
            
            # Create a set of unique text indices
            unique_indices = set(range(len(texts)))
            for dup_idx in duplicate_map:
                unique_indices.discard(dup_idx)
                
            # Map unique indices to positions
            unique_idx_to_pos = {idx: pos for pos, idx in enumerate(sorted(unique_indices))}
            
            # Create list of unique texts
            unique_texts = [texts[idx] for idx in sorted(unique_indices)]
            
            logger.info(f"Deduplication reduced {len(texts)} texts to {len(unique_texts)} unique texts")
        else:
            unique_texts = texts
            unique_idx_to_pos = {i: i for i in range(len(texts))}
            duplicate_map = {}
        
        # Process in batches
        all_results = [""] * len(unique_texts)
        
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i+batch_size]
            
            # Format the prompts
            formatted_batch = [user_prompt_template.replace("{{text}}", text) for text in batch]
            
            # Calculate total token usage for this batch
            if self.max_token_limit is not None:
                batch_token_count = sum(self._count_tokens(prompt) for prompt in formatted_batch)
                system_token_count = self._count_tokens(system_prompt)
                
                # Add token count for system prompt and formatting
                total_token_count = batch_token_count + system_token_count + 200  # Extra for formatting
                
                if total_token_count > self.max_token_limit:
                    logger.warning(f"Batch token count ({total_token_count}) exceeds limit ({self.max_token_limit}). Reducing batch size.")
                    
                    # Recursively process smaller batches
                    new_batch_size = max(1, batch_size // 2)
                    logger.info(f"Reducing batch size to {new_batch_size}")
                    
                    sub_results = self.classify_texts(
                        batch,
                        categories=categories,
                        system_prompt=system_prompt,
                        user_prompt_template=user_prompt_template,
                        batch_size=new_batch_size,
                        progress_bar=False,
                        deduplicate=False
                    )
                    
                    # Add results to our list
                    for j, result in enumerate(sub_results):
                        all_results[i + j] = result
                        
                    if progress_bar:
                        pbar.update(len(batch))
                        
                    continue
            
            # Generate classifications in batch mode
            try:
                batch_results, batch_confidences = self._batch_generate_openai(formatted_batch, system_prompt)
                
                # Store results and confidences
                for j, result in enumerate(batch_results):
                    all_results[i + j] = result
                    
                    # Map confidence scores to original indices
                    text_idx = i + j
                    self.confidence_scores[text_idx] = batch_confidences.get(j, 0.7)  # Default to medium confidence
                    
            except Exception as e:
                logger.error(f"Batch classification failed: {e}")
                
                # Fall back to individual processing
                logger.info("Falling back to individual processing")
                
                for j, text in enumerate(batch):
                    try:
                        prompt = user_prompt_template.replace("{{text}}", text)
                        result = self._generate_openai(prompt, system_prompt)
                        all_results[i + j] = result
                        self.confidence_scores[i + j] = 0.7  # Default confidence for fallback
                    except Exception as e:
                        logger.warning(f"Failed to classify text: {e}")
                        all_results[i + j] = "Unclassified"
                        self.confidence_scores[i + j] = 0.3  # Low confidence for failures
            
            if progress_bar:
                if use_simple_progress:
                    processed_so_far = i + len(batch)
                    if processed_so_far % simple_progress_interval == 0 or processed_so_far >= total_texts:
                        print(f"  Progress: {processed_so_far}/{total_texts} texts processed")
                else:
                    pbar.update(len(batch))
        
        if progress_bar:
            if use_simple_progress:
                print(f"  Completed: {total_texts}/{total_texts} texts processed")
            else:
                pbar.close()
            
        # Map results back to original texts (dealing with duplicates)
        if deduplicate:
            final_results = [""] * len(texts)
            
            # First, copy results for unique texts
            for original_idx, unique_pos in unique_idx_to_pos.items():
                final_results[original_idx] = all_results[unique_pos]
                
            # Then fill in duplicates
            for dup_idx, original_idx in duplicate_map.items():
                unique_pos = unique_idx_to_pos.get(original_idx)
                if unique_pos is not None:
                    final_results[dup_idx] = all_results[unique_pos]
                    
            return final_results
        else:
            return all_results

    def label_topics(
        self,
        topic_model: Any,
        example_docs_per_topic: Optional[Dict[int, List[str]]] = None,
        detailed: bool = False,
        progress_bar: Union[bool, str] = True,
        batch_size: Optional[int] = None,
    ) -> Dict[int, str]:
        """Label all topics in a topic model.
        
        Parameters
        ----------
        topic_model : Any
            Topic model with a get_topic method that returns keywords for each topic
        example_docs_per_topic : Optional[Dict[int, List[str]]], optional
            Dictionary mapping topic IDs to lists of example documents, by default None
        detailed : bool, optional
            Whether to generate detailed topic descriptions, by default False
        progress_bar : bool, optional
            Whether to show a progress bar, by default True
        batch_size : Optional[int], optional
            Size of batches for parallel processing, by default None
            If None, will process all topics in appropriate batch sizes based on max_parallel_requests
            
        Returns
        -------
        Dict[int, str]
            Dictionary mapping topic IDs to generated topic names
        """
        topic_names = {}
        
        # Get all topic IDs
        if hasattr(topic_model, "topics") and isinstance(topic_model.topics, dict):
            topic_ids = sorted(list(topic_model.topics.keys()))
        elif hasattr(topic_model, "get_topic_info"):
            topic_info = topic_model.get_topic_info()
            topic_ids = topic_info["Topic"].tolist()
        else:
            raise ValueError("Could not determine topic IDs from the model")
            
        # Filter out outlier topic if present
        if -1 in topic_ids:
            topic_ids.remove(-1)
        
        # Prepare topic data for processing
        topic_data = []
        for topic_id in topic_ids:
            # Get keywords for the topic
            if hasattr(topic_model, "get_topic"):
                # BERTopic or compatible model
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    # Handle format (word, score) or just words
                    if isinstance(topic_words[0], tuple):
                        keywords = [word for word, _ in topic_words]
                    else:
                        keywords = topic_words
                else:
                    keywords = []
            else:
                # Fallback for other models - try to get from topic_words attribute
                if hasattr(topic_model, "topic_words") and topic_id in topic_model.topic_words:
                    keywords = topic_model.topic_words[topic_id]
                else:
                    keywords = []
                    
            # Skip if no keywords (process immediately)
            if not keywords:
                topic_names[topic_id] = f"Topic {topic_id}"
                continue
                
            # Get example documents if available
            example_docs = None
            if example_docs_per_topic and topic_id in example_docs_per_topic:
                example_docs = example_docs_per_topic[topic_id]
                
            # Add to processing queue
            topic_data.append((topic_id, keywords, example_docs, detailed))
        
        # Set batch size if not provided
        if batch_size is None:
            batch_size = min(len(topic_data), max(1, self.max_parallel_requests * 2))
        
        # Process in batches
        if progress_bar:
            batches = [topic_data[i:i+batch_size] for i in range(0, len(topic_data), batch_size)]
            with tqdm(total=len(topic_data), desc="Labeling topics") as pbar:
                for batch in batches:
                    results = self._process_topic_batch(batch)
                    for topic_id, topic_name in results:
                        topic_names[topic_id] = topic_name
                    pbar.update(len(batch))
        else:
            # Process without progress bar
            for i in range(0, len(topic_data), batch_size):
                batch = topic_data[i:i+batch_size]
                results = self._process_topic_batch(batch)
                for topic_id, topic_name in results:
                    topic_names[topic_id] = topic_name
                
        return topic_names
    
    def update_model_topic_names(
        self,
        topic_model: Any,
        example_docs_per_topic: Optional[Dict[int, List[str]]] = None,
        detailed: bool = False,
        progress_bar: bool = True,
        batch_size: Optional[int] = None,
    ) -> Any:
        """Update a topic model with LLM-generated topic names.
        
        Parameters
        ----------
        topic_model : Any
            Topic model to update
        example_docs_per_topic : Optional[Dict[int, List[str]]], optional
            Dictionary mapping topic IDs to lists of example documents, by default None
        detailed : bool, optional
            Whether to generate detailed topic descriptions, by default False
        progress_bar : bool, optional
            Whether to show a progress bar, by default True
        batch_size : Optional[int], optional
            Size of batches for parallel processing, by default None
            
        Returns
        -------
        Any
            Updated topic model
        """
        # Generate topic names
        topic_names = self.label_topics(
            topic_model,
            example_docs_per_topic,
            detailed,
            progress_bar,
            batch_size
        )
        
        # Update the model's topic names
        if hasattr(topic_model, "topics") and isinstance(topic_model.topics, dict):
            # Handle special case for outlier topic
            if -1 in topic_model.topics and -1 not in topic_names:
                topic_names[-1] = "Other/Outlier"
                
            # Update the topic names
            for topic_id, topic_name in topic_names.items():
                if topic_id in topic_model.topics:
                    topic_model.topics[topic_id] = topic_name
        else:
            logger.warning(
                f"Could not update topic names in {type(topic_model).__name__}. "
                "Topic names were generated but not applied to the model."
            )
            
        return topic_model
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the LLM topic labeler configuration.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the configuration to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config (not the model itself)
        config = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "enable_fallback": self.enable_fallback,
            "device": getattr(self, "device", "auto"),
            "verbose": self.verbose,
            "max_token_limit": getattr(self, "max_token_limit", None),
            "max_parallel_requests": getattr(self, "max_parallel_requests", 4),
            "requests_per_minute": getattr(self.rate_limiter, "requests_per_minute", None) if self.rate_limiter else None,
            "burst_limit": getattr(self.rate_limiter, "burst_limit", None) if self.rate_limiter else None,
            "system_prompt_template": self.system_prompt_template,
            "user_prompt_template": self.user_prompt_template,
            "batch_size": self.batch_size,
            "deduplicate": self.deduplicate,
            "deduplication_threshold": self.deduplication_threshold,
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "cache_ttl": self.cache_ttl,
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
    def _get_cache_key(self, text: str, system_prompt: str, user_prompt_template: str) -> str:
        """Get cache key for a text and prompt combination.
        
        Parameters
        ----------
        text : str
            The text to classify
        system_prompt : str
            System prompt
        user_prompt_template : str
            User prompt template
            
        Returns
        -------
        str
            Cache key
        """
        # Create a deterministic hash based on text, system prompt, model name, and temperature
        content = f"{text}|{system_prompt}|{user_prompt_template}|{self.model_name}|{self.temperature}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path for a cache key.
        
        Parameters
        ----------
        cache_key : str
            Cache key
            
        Returns
        -------
        str
            Cache file path
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _cache_result(self, cache_key: str, result: str, confidence: float = 1.0) -> None:
        """Cache a classification result.
        
        Parameters
        ----------
        cache_key : str
            Cache key
        result : str
            Classification result
        confidence : float, optional
            Confidence score, by default 1.0
        """
        if not self.enable_cache:
            return
            
        cache_path = self._get_cache_path(cache_key)
        
        # Create cache entry with timestamp
        cache_entry = {
            "result": result,
            "confidence": confidence,
            "timestamp": time.time(),
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_entry, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _get_cached_result(self, cache_key: str) -> Optional[Tuple[str, float]]:
        """Get cached classification result.
        
        Parameters
        ----------
        cache_key : str
            Cache key
            
        Returns
        -------
        Optional[Tuple[str, float]]
            (result, confidence) if found and valid, None otherwise
        """
        if not self.enable_cache:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, "rb") as f:
                cache_entry = pickle.load(f)
                
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] > self.cache_ttl:
                # Cache expired
                os.remove(cache_path)
                return None
                
            # Check if model and temperature match
            if (cache_entry["model"] != self.model_name or 
                cache_entry["temperature"] != self.temperature):
                return None
                
            return cache_entry["result"], cache_entry["confidence"]
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
            
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "LLMTopicLabeler":
        """Load an LLM topic labeler from configuration.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the configuration from
        **kwargs : Any
            Additional arguments to override the loaded configuration
            
        Returns
        -------
        LLMTopicLabeler
            Loaded LLM topic labeler
        """
        path = Path(path)
        
        with open(path, "r") as f:
            config = json.load(f)
            
        # Override with kwargs
        config.update(kwargs)
        
        return cls(**config)
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio between two strings.
        
        Parameters
        ----------
        text1 : str
            First text string
        text2 : str
            Second text string
            
        Returns
        -------
        float
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate a hash for a text string.
        
        Parameters
        ----------
        text : str
            Text to hash
            
        Returns
        -------
        str
            Hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    def _identify_fuzzy_duplicates(
        self, 
        texts: List[str],
        threshold: Optional[float] = None
    ) -> Dict[int, int]:
        """Identify fuzzy duplicates in a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings to check for duplicates
        threshold : Optional[float], optional
            Similarity threshold, by default None
            If None, uses the class's deduplication_threshold
            
        Returns
        -------
        Dict[int, int]
            Dictionary mapping duplicate indices to their representative index
        """
        if threshold is None:
            threshold = self.deduplication_threshold
            
        duplicate_map = {}
        processed = set()
        
        for i, text1 in enumerate(texts):
            if i in processed:
                continue
                
            processed.add(i)
            
            for j in range(i + 1, len(texts)):
                if j in processed:
                    continue
                    
                text2 = texts[j]
                
                # Calculate similarity
                similarity = self._calculate_text_similarity(text1, text2)
                
                # If similar enough, mark as duplicate
                if similarity >= threshold:
                    duplicate_map[j] = i
                    processed.add(j)
        
        return duplicate_map


# Example usage for Jupyter Notebook
def generate_text_with_llm(
    text: str,
    api_key: str,
    api_endpoint: str,
    deployment_id: str = None, 
    model_name: str = "gpt-4o",
    api_version: str = "2023-05-15",
    use_azure: bool = True,
    system_prompt: str = "You are a helpful assistant.",
    user_prompt_prefix: str = "Insert your user prompt followed by the data here:\n\n",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    library: str = "openai",  # New parameter for selecting implementation library
    timeout: int = 60,        # New parameter for requests timeout 
    enable_cache: bool = True # New parameter for caching with requests library
) -> str:
    """Generate text using OpenAI or Azure OpenAI APIs with a simple, consistent interface.
    
    This utility function makes it easy to generate text using either the standard OpenAI API
    or Azure OpenAI, with proper parameter configurations for each. It supports both the OpenAI
    SDK and direct requests via the requests library.
    
    Parameters
    ----------
    text : str
        The input text/prompt to send to the model
    api_key : str
        The API key for OpenAI or Azure OpenAI
    api_endpoint : str
        For Azure: The azure_endpoint (e.g., "https://your-resource.openai.azure.com")
        For OpenAI: Optional base URL (e.g., "https://api.openai.com/v1/chat/completions" for direct requests)
    deployment_id : str, optional
        Azure deployment name, required when use_azure=True
    model_name : str, optional
        For OpenAI: Model name like "gpt-4o" or "gpt-3.5-turbo", by default "gpt-4o"
        Ignored when use_azure=True (deployment_id is used instead)
    api_version : str, optional
        API version, by default "2023-05-15" - mainly used for Azure
    use_azure : bool, optional
        Whether to use Azure OpenAI, by default True
    system_prompt : str, optional
        System prompt for the model, by default "You are a helpful assistant."
    user_prompt_prefix : str, optional
        Prefix to add before the input text
    temperature : float, optional
        Temperature for response generation, by default 0.7
    max_tokens : int, optional
        Maximum tokens in the response, by default 1000
    library : str, optional
        Which library to use for the API request: "openai" (OpenAI SDK) or "requests" (direct HTTP requests),
        by default "openai"
    timeout : int, optional
        Timeout in seconds for the API request when using the requests library, by default 60
    enable_cache : bool, optional
        Whether to use response caching to avoid duplicate API calls when using the requests library,
        by default True
        
    Returns
    -------
    str
        Generated text response from the model
        
    Examples
    --------
    # Azure OpenAI example using the SDK:
    >>> response = generate_text_with_llm(
    ...     text="Tell me a joke about Azure cloud services",
    ...     api_key="your-azure-api-key",
    ...     api_endpoint="https://your-resource.openai.azure.com",
    ...     deployment_id="your-deployment-name",
    ...     use_azure=True,
    ...     library="openai"  # Use the OpenAI SDK
    ... )
    
    # Standard OpenAI example using the SDK:
    >>> response = generate_text_with_llm(
    ...     text="Explain the benefits of Python 3.10",
    ...     api_key="your-openai-api-key",
    ...     api_endpoint=None,
    ...     model_name="gpt-4o",
    ...     use_azure=False,
    ...     library="openai"  # Use the OpenAI SDK
    ... )
    
    # Standard OpenAI example using requests:
    >>> response = generate_text_with_llm(
    ...     text="Explain the benefits of Python 3.10",
    ...     api_key="your-openai-api-key",
    ...     api_endpoint="https://api.openai.com/v1/chat/completions",
    ...     model_name="gpt-4o",
    ...     use_azure=False,
    ...     library="requests",  # Use direct HTTP requests
    ...     enable_cache=True    # Enable caching of responses
    ... )
    """
    # Prepare the full user prompt
    user_prompt = f"{user_prompt_prefix}{text}"
    
    # Import necessary modules
    import logging
    logger = logging.getLogger(__name__)
    
    # Use the OpenAI SDK implementation
    if library.lower() == "openai":
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
                        {"role": "user", "content": user_prompt}
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
                        {"role": "user", "content": user_prompt}
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
            
    # Use the requests library implementation
    elif library.lower() == "requests":
        import requests
        import json
        import time
        import os
        import hashlib
        from pathlib import Path
        
        # Implement caching if enabled
        cache_result = None
        cache_file = None
        cache_dir = None
        
        if enable_cache:
            try:
                # Set up cache directory
                cache_dir = Path.home() / ".meno" / "llm_cache"
                os.makedirs(cache_dir, exist_ok=True)
                
                # Generate a cache key
                cache_key = f"{user_prompt}|{model_name if not use_azure else deployment_id}|{system_prompt}"
                cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
                cache_file = cache_dir / f"{cache_hash}.json"
                
                # Check if we have a cached result
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Check if the cache is still valid (default: 24 hours)
                    if cached_data.get('expires', 0) > time.time():
                        logger.debug(f"Using cached result for: {text[:30]}...")
                        return cached_data['value']
            except (json.JSONDecodeError, KeyError, IOError, Exception) as e:
                # If there's any issue with caching, just log and continue without it
                logger.warning(f"Cache issue, proceeding without cache: {e}")
                cache_file = None
        
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
            {"role": "user", "content": user_prompt}
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
        request_endpoint = api_endpoint
        if use_azure and "api-version" not in request_endpoint and "api_version" not in request_endpoint:
            # Add api-version if using Azure and it's not already in the URL
            if "?" in request_endpoint:
                request_endpoint = f"{request_endpoint}&api-version={api_version}"
            else:
                request_endpoint = f"{request_endpoint}?api-version={api_version}"
        
        try:
            # Make the API request
            response = requests.post(
                request_endpoint,
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
            if enable_cache and cache_file and cache_dir:
                try:
                    # Cache for 24 hours by default
                    expires = time.time() + (24 * 60 * 60)
                    cache_data = {
                        'value': result,
                        'expires': expires,
                        'created': time.time()
                    }
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
    
    # Handle unsupported library
    else:
        raise ValueError(f"Unsupported library: {library}. Supported options are 'openai' and 'requests'.")


def batch_label_topics_example():
    """Example of how to use the extended LLM topic labeling with batching, token limiting,
    and rate limiting in a Jupyter notebook.
    
    This function is intended to be used as a reference for how to use the new features.
    """
    # Import required modules
    import pandas as pd
    from meno.modeling.bertopic_model import BERTopicModel
    from meno.modeling.llm_topic_labeling import LLMTopicLabeler
    from meno import generate_text_with_llm
    
    # 1. Create a topic model and fit it
    # Example assuming you already have a topic model:
    # topic_model = BERTopicModel()
    # topic_model.fit(documents)
    
    # 2. Create LLM topic labeler with token limit, parallel processing, and rate limiting
    labeler = LLMTopicLabeler(
        model_name="gpt-3.5-turbo",     # Model name (for standard OpenAI) or deployment name (for Azure)
        api_key="your-api-key",         # Your API key for OpenAI or Azure
        use_azure=False,                # Whether to use Azure OpenAI (True) or standard OpenAI (False)
        # api_endpoint="https://your-resource.openai.azure.com",  # Required for Azure
        
        max_token_limit=4000,           # Will fail if prompts exceed this token count
        max_parallel_requests=4,        # Process up to 4 topics in parallel
        enable_fallback=True,           # Fall back to simple labeling if LLM fails
        requests_per_minute=60,         # Limit to 60 requests per minute (OpenAI rate limit)
        burst_limit=80,                 # Allow short bursts up to 80 requests
        
        # New parameters
        system_prompt_template="You are a helpful assistant that classifies text into relevant topics.",
        user_prompt_template="Classify the following text into the most appropriate topic: {{text}}",
        batch_size=20,                  # Process up to 20 texts in a single API call
        deduplicate=True,               # Enable deduplication for similar texts
        deduplication_threshold=0.92    # Similarity threshold for deduplication
    )
    
    # 3. Alternative: Use the generate_text_with_llm utility directly
    # This utility provides a simplified interface for both Azure and standard OpenAI
    # from meno import generate_text_with_llm
    # 
    # # Azure OpenAI example
    # azure_response = generate_text_with_llm(
    #     text="Tell me about topic modeling",
    #     api_key="your-azure-api-key",
    #     api_endpoint="https://your-resource.openai.azure.com",
    #     deployment_id="your-deployment-name",  # Required for Azure OpenAI
    #     use_azure=True,
    #     system_prompt="You are a data science expert specializing in NLP.",
    #     temperature=0.7,
    #     max_tokens=300
    # )
    # 
    # # Standard OpenAI example
    # openai_response = generate_text_with_llm(
    #     text="Summarize these keywords into a topic name: finance, stocks, investing, market",
    #     api_key="your-openai-api-key",
    #     model_name="gpt-4o",  # Standard OpenAI model name
    #     use_azure=False,
    #     system_prompt="You are a topic modeling expert.",
    #     temperature=0.7,
    #     max_tokens=100
    # )
    
    # 4. Generate topic names in batches with rate limiting
    # topic_names = labeler.label_topics(
    #     topic_model=topic_model,
    #     batch_size=10,             # Process 10 topics per batch
    #     progress_bar=True,         # Show progress bar
    #     detailed=True              # Generate detailed topic descriptions
    # )
    
    # 5. Update the model with the generated names
    # updated_model = labeler.update_model_topic_names(
    #     topic_model=topic_model,
    #     batch_size=10,
    #     progress_bar=True
    # )
    
    # 6. Using different rate limits for different providers
    # For Azure OpenAI (lower rate limits)
    # azure_labeler = LLMTopicLabeler(
    #     model_name="your-deployment-name",         # Azure deployment name
    #     api_key="your-azure-api-key",              # Azure API key
    #     api_endpoint="https://your-resource.openai.azure.com",  # Azure endpoint
    #     api_version="2023-05-15",                  # Azure API version
    #     use_azure=True,                            # Use Azure OpenAI
    #     requests_per_minute=20,                    # Lower rate limit for Azure
    #     burst_limit=25,                            # Lower burst limit
    #     max_parallel_requests=2                    # Lower parallelism to respect rate limits
    # )
    
    return "See the example code for how to use the LLM topic labeler and utility functions"
    
def classify_texts_example():
    """Example of how to use the new text classification functionality with batching
    and deduplication to optimize API usage.
    
    This function is intended to be used as a reference for how to use the new features.
    """
    # Import required modules
    import pandas as pd
    from meno.modeling.llm_topic_labeling import LLMTopicLabeler
    
    # Example 1: Classifying texts with predefined categories
    categories = ["business", "technology", "health", "politics", "entertainment"]
    
    # Create classifier with predefined categories
    classifier = LLMTopicLabeler(
        model_type="openai",
        model_name="gpt-3.5-turbo",
        max_token_limit=4000,
        batch_size=20,
        deduplicate=True,
        deduplication_threshold=0.92,
        system_prompt_template=f"You are a helpful assistant that classifies text into one of these categories: {', '.join(categories)}."
    )
    
    # Example texts for classification
    texts = [
        "Apple announces new iPhone with advanced AI features.",
        "The stock market rose 2% today after positive economic data.",
        "New research shows benefits of Mediterranean diet for heart health.",
        "The president signed the climate bill into law yesterday.",
        "The new superhero movie broke box office records this weekend."
    ]
    
    # Classify texts using predefined categories
    results = classifier.classify_texts(
        texts=texts,
        categories=categories,
        progress_bar=True
    )
    
    # Create DataFrame with results
    df_results = pd.DataFrame({
        "text": texts,
        "category": results
    })
    
    # Example 2: Open classification (no predefined categories)
    open_classifier = LLMTopicLabeler(
        model_type="openai",
        model_name="gpt-3.5-turbo",
        system_prompt_template="You are an expert at assigning concise topic labels to content.",
        user_prompt_template="Assign a brief, descriptive topic label (1-3 words) to this text: {{text}}",
        batch_size=20,
        deduplicate=True
    )
    
    # Classify with open topics
    open_results = open_classifier.classify_texts(texts)
    
    # Add to DataFrame
    df_results["open_category"] = open_results
    
    # Example 3: Handling larger datasets efficiently
    # Large dataset simulation (with some similar texts)
    large_texts = texts * 20  # Just duplicating for example purposes
    
    # Classify efficiently with deduplication
    efficient_results = classifier.classify_texts(
        texts=large_texts,
        categories=categories,
        batch_size=20,
        deduplicate=True,
        deduplication_threshold=0.92
    )
    
    return "See the example code for how to use the classify_texts method"


# Initialize cache for API calls
_LLM_API_CACHE = {}
_CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds, by default
_DEFAULT_CACHE_DIR = Path.home() / ".meno" / "llm_cache"

def generate_call_from_text(text: str, api_key: str, api_endpoint: str, 
                           model: str = "gpt-4o", system_prompt: str = "You are a helpful assistant.",
                           timeout: int = 60, enable_cache: bool = True, 
                           cache_ttl: int = _CACHE_TTL) -> str:
    """
    Make a single API call to generate a response from the given text.
    
    Args:
        text: The user input text to process
        api_key: Your API key for authentication
        api_endpoint: The API endpoint URL
        model: The model to use for generation
        system_prompt: The system prompt to use
        timeout: Request timeout in seconds
        enable_cache: Whether to use response caching to avoid duplicate API calls
        cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        
    Returns:
        The generated response text or an error message
    """
    # Generate a cache key from all relevant parameters
    cache_key = _get_cache_key(text, model, system_prompt)
    
    # Check cache first if enabled
    if enable_cache:
        cached_result = _get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for prompt: {text[:50]}...")
            return cached_result
    
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
            result = "[No response generated.]"
        else:
            result = response_data['choices'][0]['message']['content'].strip()
        
        # Cache the result if caching is enabled
        if enable_cache:
            _add_to_cache(cache_key, result, ttl=cache_ttl)
            
        return result
        
    except requests.exceptions.Timeout:
        return "[Error: Request timed out]"
    except requests.exceptions.RequestException as e:
        return f"[Error: {e}]"
    except ValueError as e:  # JSON parsing error
        return f"[Error: Invalid response format - {e}]"
    except Exception as e:
        return f"[Error: Unexpected error - {e}]"
    
def _get_cache_key(text: str, model: str, system_prompt: str) -> str:
    """Generate a unique cache key from the request parameters."""
    # Create a string that combines all the parameters that should make a unique request
    combined = f"{text}|{model}|{system_prompt}"
    # Hash it to create a fixed-length key that's safe for filenames
    return hashlib.md5(combined.encode()).hexdigest()

def _get_from_cache(cache_key: str) -> Optional[str]:
    """Get a cached response if it exists and is still valid."""
    # Check memory cache first
    if cache_key in _LLM_API_CACHE:
        timestamp, value = _LLM_API_CACHE[cache_key]
        if time.time() < timestamp:  # Still valid
            return value
        else:  # Expired
            del _LLM_API_CACHE[cache_key]
    
    # Check disk cache
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Check expiration
            if cached_data.get('expires', 0) > time.time():
                # Add to memory cache for faster access next time
                _LLM_API_CACHE[cache_key] = (cached_data['expires'], cached_data['value'])
                return cached_data['value']
            else:
                # Expired - delete the file
                cache_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, KeyError, IOError):
            # Invalid cache file, ignore it
            cache_file.unlink(missing_ok=True)
    
    return None

def _add_to_cache(cache_key: str, value: str, ttl: int = _CACHE_TTL) -> None:
    """Add a response to both memory and disk cache."""
    expires = time.time() + ttl
    
    # Add to memory cache
    _LLM_API_CACHE[cache_key] = (expires, value)
    
    # Add to disk cache for persistence
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'expires': expires,
                'value': value,
                'created': time.time()
            }, f)
    except IOError:
        # If we can't write to disk, just keep the memory cache
        logger.warning(f"Could not write cache file: {cache_file}")

def _get_cache_dir() -> Path:
    """Get or create the cache directory."""
    cache_dir = _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def process_texts_with_threadpool(texts: List[str], api_key: str, api_endpoint: str,
                                 model: str = "gpt-4o", system_prompt: str = "You are a helpful assistant.",
                                 max_workers: Optional[int] = None, timeout: int = 60,
                                 enable_cache: bool = True, cache_ttl: int = _CACHE_TTL,
                                 show_progress: bool = True) -> List[Dict[str, Any]]:
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
        enable_cache: Whether to use response caching
        cache_ttl: Cache time-to-live in seconds (default: 24 hours)
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
        response = generate_call_from_text(
            text=text,
            api_key=api_key,
            api_endpoint=api_endpoint,
            model=model,
            system_prompt=system_prompt,
            timeout=timeout,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl
        )
        end_time = time.time()
        
        result = {
            "index": index,
            "input": text,
            "response": response,
            "time_taken": end_time - start_time,
            "success": not response.startswith("[Error:"),
            "from_cache": False  # Will be set to True if it came from cache
        }
        
        # Check if the result came from cache
        cache_key = _get_cache_key(text, model, system_prompt)
        if enable_cache and cache_key in _LLM_API_CACHE:
            result["from_cache"] = True
        
        # Update progress with thread safety
        if show_progress:
            with progress_lock:
                completed_count += 1
                print(f"Completed {completed_count}/{len(texts)}: {'✓' if result['success'] else '✗'} {'(cached)' if result.get('from_cache') else ''}")
        
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
        for future in concurrent.futures.as_completed(futures):
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
        cached_count = sum(1 for r in results if r.get("from_cache", False))
        success_count = sum(1 for r in results if r.get("success", False))
        
        if enable_cache and cached_count > 0:
            print(f"\nSummary: {success_count}/{len(results)} successful, {cached_count} from cache")
        else:
            print(f"\nSummary: {success_count}/{len(results)} successful")
    
    return results


def format_chat_completion(chat_completion, verbose=True):
    """
    Format and print the details of a chat completion response.
    
    Args:
        chat_completion: The chat completion response object
        verbose: If True, prints all details. If False, prints only message content
    
    Returns:
        None (prints formatted output)
    """
    # Handle potential None or invalid input
    if not chat_completion:
        print("Error: No chat completion provided")
        return
    
    try:
        if not verbose:
            # Simple output mode - just show the message content
            for choice in chat_completion.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    print(choice.message.content)
            return
            
        # Verbose mode with all details
        print(f"ID: {chat_completion.id}")
        print(f"Model: {chat_completion.model}")
        print(f"Created: {chat_completion.created}")
        
        print("\n=== Choices ===")
        for i, choice in enumerate(chat_completion.choices):
            print(f"\nChoice {i+1}:")
            print(f"  Index: {choice.index}")
            print(f"  Finish Reason: {choice.finish_reason}")
            
            # Handle message content - could be None in some cases
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                print(f"  Message Content: {choice.message.content}")
            else:
                print(f"  Message Content: None")
            
            # Handle content filter results safely
            if hasattr(choice, 'content_filter_results') and choice.content_filter_results:
                print("\n  Content Filter Results:")
                for key, value in choice.content_filter_results.items():
                    filtered = value.get('filtered', 'N/A')
                    severity = value.get('severity', 'N/A')
                    print(f"    {key.capitalize()}: Filtered={filtered}, Severity={severity}")
        
        # Usage information
        if hasattr(chat_completion, 'usage') and chat_completion.usage:
            print("\n=== Usage ===")
            usage_dict = vars(chat_completion.usage)
            for key, value in usage_dict.items():
                if key.startswith('_'):  # Skip private attributes
                    continue
                print(f"  {key.replace('_', ' ').capitalize()}: {value}")
                
    except AttributeError as e:
        print(f"Error accessing attributes: {e}")
        print("The response object may have a different structure than expected.")
    except Exception as e:
        print(f"Error formatting chat completion: {e}")
        
    print("\n" + "-" * 50)  # Add separator for clarity


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity ratio between two strings."""
    # Early termination for very different length texts
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Quick length-based filter to avoid unnecessary computation
    # If lengths differ by more than 30%, they're likely different
    if abs(len1 - len2) / max(len1, len2) > 0.3:
        return 0.0
    
    # Quick check for exact match
    if text1 == text2:
        return 1.0
        
    # Quick check for very different texts by comparing the first N chars
    preview_len = min(50, min(len1, len2))
    if text1[:preview_len].lower() != text2[:preview_len].lower() and \
       SequenceMatcher(None, text1[:preview_len], text2[:preview_len]).ratio() < 0.5:
        return 0.0
        
    # For texts that pass the quick checks, perform full comparison
    return SequenceMatcher(None, text1, text2).ratio()


def identify_fuzzy_duplicates(
    texts: List[str],
    threshold: float = 0.92,
    max_comparisons: Optional[int] = None,
    simplified_texts: Optional[List[str]] = None
) -> Dict[int, int]:
    """
    Identify fuzzy duplicates in a list of texts.
    
    This utility function finds texts that are similar to each other based on a 
    similarity threshold. Use this for deduplication before sending texts to LLMs
    to save on API costs.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to check for duplicates
    threshold : float, optional
        Similarity threshold (0.0-1.0), by default 0.92
        Higher values are more strict (require more similarity to consider duplicates)
    max_comparisons : Optional[int], optional
        Maximum number of comparisons to perform (for very large datasets)
        If None, all pairs will be compared
    simplified_texts : Optional[List[str]], optional
        Pre-processed versions of texts for faster comparison
        (e.g., lowercase, stopwords removed, etc.)
        If None, original texts will be used
        
    Returns
    -------
    Dict[int, int]
        Dictionary mapping duplicate indices to their representative index
    """
    if not texts:
        return {}
        
    n_texts = len(texts)
    
    # Calculate total possible comparisons
    total_comparisons = (n_texts * (n_texts - 1)) // 2
    
    # If max_comparisons is set and less than total, sample proportionally
    sampling_enabled = max_comparisons is not None and max_comparisons < total_comparisons
    
    # Use simplified texts if provided, otherwise use the original
    compare_texts = simplified_texts if simplified_texts else texts
    
    # Create length-based groups for optimization
    length_groups = defaultdict(list)
    for i, text in enumerate(compare_texts):
        # Group texts by length ranges (each range is ~10% of the text length)
        length_range = len(text) // 10
        length_groups[length_range].append(i)
    
    duplicate_map = {}
    processed = set()
    
    # Process each length group
    for length_range, indices in length_groups.items():
        # Skip if no texts in this range
        if not indices:
            continue
            
        # Check adjacent length ranges too (handle boundary cases)
        adjacent_indices = []
        for adj_range in [length_range-1, length_range, length_range+1]:
            if adj_range in length_groups:
                adjacent_indices.extend(length_groups[adj_range])
        
        # Compare texts within this length group and adjacent groups
        for idx, i in enumerate(indices):
            if i in processed:
                continue
                
            processed.add(i)
            text1 = compare_texts[i]
            
            # Determine comparison targets
            comparison_targets = [j for j in adjacent_indices if j > i and j not in processed]
            
            # Sample if needed
            if sampling_enabled and len(comparison_targets) > max_comparisons:
                import random
                comparison_targets = random.sample(comparison_targets, max_comparisons)
            
            for j in comparison_targets:
                text2 = compare_texts[j]
                
                # Calculate similarity
                similarity = _calculate_text_similarity(text1, text2)
                
                # If similar enough, mark as duplicate
                if similarity >= threshold:
                    duplicate_map[j] = i
                    processed.add(j)
    
    return duplicate_map


def process_texts_with_deduplication(
    texts: List[str], 
    api_key: str, 
    api_endpoint: str,
    model: str = "gpt-4o", 
    system_prompt: str = "You are a helpful assistant.",
    max_workers: Optional[int] = None, 
    timeout: int = 60,
    deduplicate: bool = True,
    deduplication_threshold: float = 0.92,
    enable_cache: bool = True,
    cache_ttl: int = _CACHE_TTL,
    max_comparisons: Optional[int] = None,
    show_progress: bool = True,
    preprocess_for_deduplication: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple texts with deduplication and parallel execution.
    
    This function combines fuzzy deduplication with parallel processing to
    efficiently handle large sets of texts, potentially saving on API costs
    by only processing unique content.
    
    Parameters
    ----------
    texts : List[str]
        List of text prompts to process
    api_key : str
        Your API key for authentication
    api_endpoint : str
        The API endpoint URL
    model : str, optional
        The model to use for generation, by default "gpt-4o"
    system_prompt : str, optional
        The system prompt to use, by default "You are a helpful assistant."
    max_workers : Optional[int], optional
        Maximum number of worker threads, by default None
    timeout : int, optional
        Request timeout in seconds, by default 60
    deduplicate : bool, optional
        Whether to perform deduplication, by default True
    deduplication_threshold : float, optional
        Similarity threshold for deduplication, by default 0.92
    enable_cache : bool, optional
        Whether to use response caching, by default True
    cache_ttl : int, optional
        Cache time-to-live in seconds, by default 24 hours
    max_comparisons : Optional[int], optional
        Maximum number of similarity comparisons to make (for large datasets)
    show_progress : bool, optional
        Whether to print progress information, by default True
    preprocess_for_deduplication : bool, optional
        Whether to preprocess texts for faster deduplication, by default True
        
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing the input text, response, and other metadata
    """
    if not texts:
        return []
    
    start_time = time.time()
    
    # Deduplication to reduce API calls    
    if deduplicate:
        # Optional preprocessing for faster deduplication
        simplified_texts = None
        if preprocess_for_deduplication:
            simplified_texts = []
            for text in texts:
                # Simple preprocessing - lowercase and trim spaces
                simplified = text.lower().strip()
                # Remove common punctuation
                for char in ',.:;?!':
                    simplified = simplified.replace(char, '')
                simplified_texts.append(simplified)
        
        # Identify duplicates with optimized algorithm
        duplicate_map = identify_fuzzy_duplicates(
            texts, 
            deduplication_threshold,
            max_comparisons=max_comparisons,
            simplified_texts=simplified_texts
        )
        
        # Create a set of unique text indices
        unique_indices = set(range(len(texts)))
        for dup_idx in duplicate_map:
            unique_indices.discard(dup_idx)
            
        # Create list of unique texts for processing
        unique_texts = [texts[idx] for idx in sorted(unique_indices)]
        
        if show_progress:
            dedup_time = time.time() - start_time
            print(f"Deduplication reduced {len(texts)} texts to {len(unique_texts)} unique texts (took {dedup_time:.2f}s)")
    else:
        unique_texts = texts
        unique_indices = set(range(len(texts)))
        duplicate_map = {}
    
    # Process unique texts with ThreadPoolExecutor using the updated function that supports caching
    unique_results = process_texts_with_threadpool(
        texts=unique_texts,
        api_key=api_key,
        api_endpoint=api_endpoint,
        model=model,
        system_prompt=system_prompt,
        max_workers=max_workers,
        timeout=timeout,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl,
        show_progress=show_progress
    )
    
    # If no deduplication was done, return results directly
    if not deduplicate:
        return unique_results
    
    # Map results back to include duplicates
    final_results = []
    
    # Create a map from unique indices to results
    orig_indices = sorted(unique_indices)
    index_to_result = {r["index"]: r for r in unique_results}
    
    # Fill in results for all original texts
    for i in range(len(texts)):
        if i in unique_indices:
            # This is a unique text, copy its result directly
            final_results.append(index_to_result[i])
        else:
            # This is a duplicate, copy from its source with adjusted metadata
            source_idx = duplicate_map[i]
            source_result = index_to_result[source_idx].copy()
            
            # Update the metadata for this duplicate
            source_result["index"] = i
            source_result["input"] = texts[i]
            source_result["is_duplicate"] = True
            source_result["duplicate_of"] = source_idx
            source_result["time_taken"] = 0  # No API call was made
            
            final_results.append(source_result)
    
    # Sort results by original index to maintain order
    final_results = sorted(final_results, key=lambda x: x["index"])
    
    # Show summary if requested
    if show_progress:
        total_time = time.time() - start_time
        duplicate_count = len(duplicate_map)
        cache_count = sum(1 for r in unique_results if r.get("from_cache", False))
        api_call_count = len(unique_texts) - cache_count
        
        print(f"\nPerformance Summary:")
        print(f"- Total texts processed: {len(texts)}")
        print(f"- Duplicates detected: {duplicate_count} ({duplicate_count/len(texts)*100:.1f}%)")
        print(f"- Cache hits: {cache_count} ({cache_count/len(unique_texts)*100:.1f}% of unique texts)")
        print(f"- API calls made: {api_call_count}")
        print(f"- Total processing time: {total_time:.2f}s")
        
        # Calculate the estimated API cost savings
        if api_call_count > 0:
            saved_calls = duplicate_count + cache_count
            saved_percentage = saved_calls / len(texts) * 100
            print(f"- Estimated API call savings: {saved_calls} calls ({saved_percentage:.1f}%)")
    
    return final_results
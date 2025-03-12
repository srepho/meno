"""LLM-based topic labeling for topic models.

This module provides a way to generate human-readable topic names using Language Models
(LLMs). It supports both local models via HuggingFace and remote models via OpenAI.
"""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar, Callable, Generator
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import re
import warnings
from tqdm import tqdm
import importlib.util
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections import deque
from datetime import datetime, timedelta

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
    """LLM-based topic labeler to generate human-readable topic names.
    
    This class provides methods to generate descriptive topic names for topic models
    using Language Models (LLMs). It supports both local HuggingFace models and
    OpenAI API if available.
    
    Parameters
    ----------
    model_type : str, optional
        Type of model to use, by default "local"
        Options: "local", "openai", "auto"
        If "auto", will use OpenAI if available, otherwise fall back to local model
    model_name : str, optional
        Name of the model to use, by default "google/flan-t5-small" for local models
        For OpenAI, default is "gpt-3.5-turbo"
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
    openai_api_key : Optional[str], optional
        OpenAI API key, by default None
        If None and model_type="openai", will try to use the OPENAI_API_KEY environment variable
    api_endpoint : Optional[str], optional
        Custom API endpoint URL for OpenAI API, by default None
        Can be used for self-hosted models, proxies, or Azure OpenAI endpoints
    api_version : Optional[str], optional
        API version to use with OpenAI API, by default None
        Useful when using Azure OpenAI or other custom OpenAI API deployments
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
    """
    
    def __init__(
        self,
        model_type: str = "local",
        model_name: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        enable_fallback: bool = True,
        device: str = "auto",
        verbose: bool = False,
        openai_api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None, 
        api_version: Optional[str] = None,
        max_token_limit: Optional[int] = None,
        max_parallel_requests: int = 4,
        requests_per_minute: Optional[int] = None,
        burst_limit: Optional[int] = None,
    ):
        """Initialize the LLM topic labeler."""
        self.verbose = verbose
        self.enable_fallback = enable_fallback
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_token_limit = max_token_limit
        self.max_parallel_requests = max_parallel_requests
        
        # Initialize rate limiter if requests_per_minute is specified
        self.rate_limiter = None
        if requests_per_minute is not None:
            self.rate_limiter = RateLimiter(
                requests_per_minute=requests_per_minute,
                burst_limit=burst_limit
            )
            if self.verbose:
                logger.info(f"Rate limiter initialized with {requests_per_minute} requests per minute")
        
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
            
            # Configure API key
            if openai_api_key:
                client_kwargs["api_key"] = openai_api_key
                
            # Configure custom API endpoint (base URL)
            if api_endpoint:
                client_kwargs["base_url"] = api_endpoint
                
            # Configure API version 
            if api_version:
                client_kwargs["api_version"] = api_version
                
            # Initialize OpenAI client with parameters
            self.client = openai.OpenAI(**client_kwargs)
                
            logger.info(f"Using OpenAI model: {self.model_name}")
            
            # Log custom API configuration if used
            if api_endpoint:
                logger.info(f"Using custom API endpoint: {api_endpoint}")
            if api_version:
                logger.info(f"Using API version: {api_version}")
                
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
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate a topic name using OpenAI API.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the API
            
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
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a topic modeling assistant that generates concise, descriptive topic names."},
                    {"role": "user", "content": prompt}
                ],
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

    def label_topics(
        self,
        topic_model: Any,
        example_docs_per_topic: Optional[Dict[int, List[str]]] = None,
        detailed: bool = False,
        progress_bar: bool = True,
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
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
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


# Example usage for Jupyter Notebook
def batch_label_topics_example():
    """Example of how to use the extended LLM topic labeling with batching, token limiting,
    and rate limiting in a Jupyter notebook.
    
    This function is intended to be used as a reference for how to use the new features.
    """
    # Import required modules
    import pandas as pd
    from meno.modeling.bertopic_model import BERTopicModel
    from meno.modeling.llm_topic_labeling import LLMTopicLabeler
    
    # 1. Create a topic model and fit it
    # Example assuming you already have a topic model:
    # topic_model = BERTopicModel()
    # topic_model.fit(documents)
    
    # 2. Create LLM topic labeler with token limit, parallel processing, and rate limiting
    labeler = LLMTopicLabeler(
        model_type="openai",         # Using OpenAI as an example for rate limiting
        model_name="gpt-3.5-turbo",
        max_token_limit=1000,        # Will fail if prompts exceed this token count
        max_parallel_requests=4,     # Process up to 4 topics in parallel
        enable_fallback=True,        # Fall back to simple labeling if LLM fails
        requests_per_minute=60,      # Limit to 60 requests per minute (OpenAI rate limit)
        burst_limit=80,              # Allow short bursts up to 80 requests
        # Optionally configure custom API settings
        # openai_api_key="your-api-key",  
        # api_endpoint="https://your-custom-endpoint.com",  # For custom endpoints
        # api_version="2023-07-01"  # For specific API versions
    )
    
    # 3. Generate topic names in batches with rate limiting
    # topic_names = labeler.label_topics(
    #     topic_model=topic_model,
    #     batch_size=10,             # Process 10 topics per batch
    #     progress_bar=True,         # Show progress bar
    #     detailed=True              # Generate detailed topic descriptions
    # )
    
    # 4. Update the model with the generated names
    # updated_model = labeler.update_model_topic_names(
    #     topic_model=topic_model,
    #     batch_size=10,
    #     progress_bar=True
    # )
    
    # 5. Using different rate limits for different providers
    # For Azure OpenAI (lower rate limits)
    # azure_labeler = LLMTopicLabeler(
    #     model_type="openai",
    #     model_name="gpt-4",
    #     requests_per_minute=20,                    # Lower rate limit for Azure
    #     burst_limit=25,                            # Lower burst limit
    #     openai_api_key="your_azure_key",          
    #     api_endpoint="https://your-resource.openai.azure.com",  # Azure endpoint
    #     api_version="2023-05-15",                 # Azure API version
    #     max_parallel_requests=2                   # Lower parallelism to respect rate limits
    # )
    
    # For local models (higher rate limits, limited by hardware)
    # local_labeler = LLMTopicLabeler(
    #     model_type="local",
    #     model_name="google/flan-t5-small",
    #     requests_per_minute=100,    # Higher limit since it's local
    #     max_parallel_requests=2     # Limited by GPU memory or CPU
    # )
    
    return "See the example code for how to use the LLM topic labeler with batching, token limiting, and rate limiting"
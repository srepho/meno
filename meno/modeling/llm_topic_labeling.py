"""LLM-based topic labeling for topic models.

This module provides a way to generate human-readable topic names using Language Models
(LLMs). It supports both local models via HuggingFace and remote models via OpenAI.
"""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar, Callable
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import re
import warnings
from tqdm import tqdm
import importlib.util

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
    ):
        """Initialize the LLM topic labeler."""
        self.verbose = verbose
        self.enable_fallback = enable_fallback
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
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
            if openai_api_key:
                openai.api_key = openai_api_key
                
            self.client = openai.OpenAI()
                
            logger.info(f"Using OpenAI model: {self.model_name}")
                
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
            
        return base_prompt
    
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
    
    def label_topics(
        self,
        topic_model: Any,
        example_docs_per_topic: Optional[Dict[int, List[str]]] = None,
        detailed: bool = False,
        progress_bar: bool = True,
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
            
        # Process each topic
        iterable = tqdm(topic_ids, desc="Labeling topics") if progress_bar else topic_ids
        for topic_id in iterable:
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
                    
            # Skip if no keywords
            if not keywords:
                topic_names[topic_id] = f"Topic {topic_id}"
                continue
                
            # Get example documents if available
            example_docs = None
            if example_docs_per_topic and topic_id in example_docs_per_topic:
                example_docs = example_docs_per_topic[topic_id]
                
            # Generate a name for this topic
            try:
                topic_name = self.generate_topic_name(keywords, example_docs, detailed)
                topic_names[topic_id] = topic_name
            except Exception as e:
                logger.warning(f"Failed to generate name for topic {topic_id}: {e}")
                # Fallback to simple naming
                topic_names[topic_id] = self._fallback_labeling(keywords)
                
        return topic_names
    
    def update_model_topic_names(
        self,
        topic_model: Any,
        example_docs_per_topic: Optional[Dict[int, List[str]]] = None,
        detailed: bool = False,
        progress_bar: bool = True,
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
            progress_bar
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
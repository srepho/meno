"""Text normalization utilities for preprocessing text data."""

from typing import List, Dict, Set, Optional, Union, Any
import re
import string
from collections import Counter
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from ..utils.config import StopwordsConfig


class TextNormalizer:
    """Class for normalizing and cleaning text data.
    
    This class handles text normalization operations including:
    - Lowercasing
    - Punctuation removal
    - Number removal
    - Lemmatization
    - Stopword removal
    
    Parameters
    ----------
    lowercase : bool, optional
        Whether to convert text to lowercase, by default True
    remove_punctuation : bool, optional
        Whether to remove punctuation, by default True
    remove_numbers : bool, optional
        Whether to remove numeric characters, by default False
    lemmatize : bool, optional
        Whether to lemmatize tokens, by default True
    language : str, optional
        spaCy language model to use, by default "en"
    stopwords_config : StopwordsConfig, optional
        Configuration for stopword handling, by default None
    
    Attributes
    ----------
    lowercase : bool
        Whether to convert text to lowercase
    remove_punctuation : bool
        Whether to remove punctuation
    remove_numbers : bool
        Whether to remove numeric characters
    lemmatize : bool
        Whether to lemmatize tokens
    language : str
        spaCy language model code
    nlp : spacy.language.Language
        Loaded spaCy language model
    stopwords : Set[str]
        Set of stopwords to remove
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        lemmatize: bool = True,
        language: str = "en",
        stopwords_config: Optional[StopwordsConfig] = None,
    ):
        """Initialize the text normalizer with specified options."""
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.language = language
        
        # Load spaCy model
        # Use the smallest possible model based on needs
        if lemmatize:
            self.nlp = self._load_spacy_model(language)
        else:
            # If not lemmatizing, just need tokenization
            self.nlp = spacy.blank(language)
        
        # Set up stopwords
        self.stopwords = self._setup_stopwords(stopwords_config)
    
    def _load_spacy_model(self, language: str) -> Language:
        """Load the appropriate spaCy language model.
        
        Parameters
        ----------
        language : str
            Language code (e.g., "en" for English)
        
        Returns
        -------
        Language
            Loaded spaCy language model
        
        Raises
        ------
        ValueError
            If the language is not supported or model can't be loaded
        """
        try:
            # Try to load the small model first (faster, less memory)
            model_name = f"{language}_core_web_sm"
            nlp = spacy.load(model_name, disable=["ner", "parser"])
            return nlp
        except OSError:
            try:
                # Fall back to md model if sm not available
                model_name = f"{language}_core_web_md"
                nlp = spacy.load(model_name, disable=["ner", "parser"])
                return nlp
            except OSError:
                # Try to download the model if not available
                try:
                    import subprocess
                    subprocess.run([
                        "python", "-m", "spacy", "download", f"{language}_core_web_sm"
                    ], check=True)
                    return spacy.load(f"{language}_core_web_sm", disable=["ner", "parser"])
                except Exception as e:
                    # If all else fails, fall back to blank model with warning
                    import warnings
                    warnings.warn(
                        f"Could not load spaCy model for {language}. "
                        f"Falling back to blank model. Lemmatization will be limited. "
                        f"Error: {str(e)}"
                    )
                    return spacy.blank(language)
    
    def _setup_stopwords(self, config: Optional[StopwordsConfig] = None) -> Set[str]:
        """Set up stopwords based on configuration.
        
        Parameters
        ----------
        config : Optional[StopwordsConfig], optional
            Stopwords configuration, by default None
        
        Returns
        -------
        Set[str]
            Set of stopwords to remove
        """
        if config is None:
            # Default config
            use_default = True
            custom_stopwords = set()
            keep_words = set()
        else:
            use_default = config.use_default
            custom_stopwords = set(config.custom)
            keep_words = set(config.keep)
        
        # Start with spaCy's default stopwords if requested
        if use_default and hasattr(self.nlp, "Defaults") and hasattr(self.nlp.Defaults, "stop_words"):
            stopwords = self.nlp.Defaults.stop_words.copy()
        else:
            stopwords = set()
        
        # Add custom stopwords
        stopwords.update(custom_stopwords)
        
        # Remove words that should be kept
        stopwords.difference_update(keep_words)
        
        return stopwords
    
    def normalize_text(self, text: str) -> str:
        """Normalize a single text string.
        
        Parameters
        ----------
        text : str
            Text to normalize
        
        Returns
        -------
        str
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Preprocess before spaCy
        if self.lowercase:
            text = text.lower()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract tokens based on settings
        tokens = []
        for token in doc:
            # Skip punctuation if requested
            if self.remove_punctuation and token.is_punct:
                continue
            
            # Skip numbers if requested
            if self.remove_numbers and token.like_num:
                continue
            
            # Skip stopwords
            if token.text.lower() in self.stopwords:
                continue
            
            # Use lemma if requested, otherwise use the original text
            if self.lemmatize and token.lemma_ != "-PRON-":
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        
        # Join tokens back into a string
        normalized_text = " ".join(tokens)
        
        return normalized_text
    
    def normalize_batch(
        self, 
        texts: Union[List[str], pd.Series],
        batch_size: int = 1000,
        n_process: int = 1,
    ) -> Union[List[str], pd.Series]:
        """Normalize a batch of texts.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Texts to normalize
        batch_size : int, optional
            Batch size for processing, by default 1000
        n_process : int, optional
            Number of processes for parallel processing, by default 1
            Set to -1 to use all available cores
        
        Returns
        -------
        Union[List[str], pd.Series]
            Normalized texts in the same format as input
        """
        # Convert pandas Series to list if needed
        is_series = isinstance(texts, pd.Series)
        if is_series:
            text_list = texts.tolist()
        else:
            text_list = texts
        
        # Process in batches to avoid memory issues with large datasets
        normalized_texts = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            
            # Process batch with SpaCy's pipe for efficiency
            docs = self.nlp.pipe(
                batch,
                batch_size=min(batch_size, 128),  # SpaCy's internal batch size
                n_process=n_process,
            )
            
            # Process each doc
            batch_results = []
            for doc in docs:
                # Extract tokens based on settings
                tokens = []
                for token in doc:
                    # Skip punctuation if requested
                    if self.remove_punctuation and token.is_punct:
                        continue
                    
                    # Skip numbers if requested
                    if self.remove_numbers and token.like_num:
                        continue
                    
                    # Skip stopwords
                    if token.text.lower() in self.stopwords:
                        continue
                    
                    # Use lemma if requested, otherwise use the original text
                    if self.lemmatize and token.lemma_ != "-PRON-":
                        tokens.append(token.lemma_)
                    else:
                        tokens.append(token.text)
                
                # Join tokens back into a string
                normalized_text = " ".join(tokens)
                batch_results.append(normalized_text)
            
            normalized_texts.extend(batch_results)
        
        # Return in the same format as input
        if is_series:
            return pd.Series(normalized_texts, index=texts.index)
        else:
            return normalized_texts


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    lemmatize: bool = True,
    language: str = "en",
    stopwords: Optional[List[str]] = None,
) -> str:
    """Function for one-off text normalization without creating a TextNormalizer instance.
    
    Parameters
    ----------
    text : str
        Text to normalize
    lowercase : bool, optional
        Whether to convert text to lowercase, by default True
    remove_punctuation : bool, optional
        Whether to remove punctuation, by default True
    remove_numbers : bool, optional
        Whether to remove numeric characters, by default False
    lemmatize : bool, optional
        Whether to lemmatize tokens, by default True
    language : str, optional
        spaCy language model to use, by default "en"
    stopwords : Optional[List[str]], optional
        Additional stopwords to remove, by default None
    
    Returns
    -------
    str
        Normalized text
    """
    # Create stopwords config
    config = StopwordsConfig(
        use_default=True,
        custom=stopwords or [],
        keep=[],
    )
    
    # Create normalizer
    normalizer = TextNormalizer(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        lemmatize=lemmatize,
        language=language,
        stopwords_config=config,
    )
    
    # Normalize text
    return normalizer.normalize_text(text)
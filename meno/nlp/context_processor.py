"""Context-aware text processing with language model integration."""

from typing import Dict, List, Optional, Union, Any, Tuple, Set
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
from collections import defaultdict

# Optional dependencies for advanced NLP features
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class ContextualProcessor:
    """Context-aware text processing using language models.
    
    This class provides enhanced text processing capabilities using
    language models to understand context and make smarter corrections
    and expansions. It can be used to augment the basic acronym expansion
    and spelling correction with contextual understanding.
    
    Parameters
    ----------
    use_spacy : bool, optional
        Whether to use SpaCy for NLP processing, by default True
    use_transformers : bool, optional
        Whether to use Hugging Face transformers for language modeling, by default False
    model_name : str, optional
        Name of the language model to use, by default "en_core_web_sm" for SpaCy
        or "distilbert-base-uncased" for transformers
    device : str, optional
        Device to run the model on ("cpu" or "cuda"), by default "cpu"
    max_length : int, optional
        Maximum length for transformer inputs, by default 128
    batch_size : int, optional
        Batch size for processing, by default 32
    
    Attributes
    ----------
    nlp : Any
        SpaCy NLP model or transformer pipeline
    use_spacy : bool
        Whether SpaCy is being used
    use_transformers : bool
        Whether transformers are being used
    model_name : str
        Name of the loaded model
    device : str
        Device being used for processing
    """
    
    def __init__(
        self,
        use_spacy: bool = True,
        use_transformers: bool = False,
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 128,
        batch_size: int = 32,
    ):
        """Initialize the contextual processor with specified options."""
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize language model
        if self.use_spacy:
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "SpaCy is required for context-aware processing. "
                    "Install with 'pip install spacy' and download a model with "
                    "'python -m spacy download en_core_web_sm'"
                )
            
            # Default to small English model if not specified
            self.model_name = model_name or "en_core_web_sm"
            
            try:
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Loaded SpaCy model: {self.model_name}")
            except OSError:
                logger.warning(
                    f"SpaCy model {self.model_name} not found. Attempting to download..."
                )
                try:
                    # Download model if not available
                    import subprocess
                    subprocess.check_call(
                        [
                            "python", "-m", "spacy", "download", 
                            self.model_name.replace("_", "-")
                        ]
                    )
                    self.nlp = spacy.load(self.model_name)
                    logger.info(f"Downloaded and loaded SpaCy model: {self.model_name}")
                except Exception as e:
                    raise ImportError(
                        f"Failed to download SpaCy model {self.model_name}: {e}. "
                        f"Please download it manually with "
                        f"'python -m spacy download {self.model_name.replace('_', '-')}'"
                    )
        
        elif self.use_transformers:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers is required for advanced language modeling. "
                    "Install with 'pip install transformers'"
                )
            
            # Default to DistilBERT if not specified
            self.model_name = model_name or "distilbert-base-uncased"
            
            try:
                # Create a fill-mask pipeline for contextual word prediction
                self.nlp = pipeline(
                    "fill-mask", 
                    model=self.model_name, 
                    device=0 if self.device == "cuda" else -1
                )
                logger.info(f"Loaded transformer model: {self.model_name}")
            except Exception as e:
                raise ImportError(
                    f"Failed to load transformer model {self.model_name}: {e}"
                )
        else:
            logger.warning(
                "No NLP model selected. Using basic processing only. "
                "Install SpaCy or transformers for enhanced features."
            )
            self.nlp = None
            self.model_name = None
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text.
        
        Parameters
        ----------
        text : str
            Text to extract entities from
            
        Returns
        -------
        List[Tuple[str, str]]
            List of (entity text, entity type) tuples
        """
        if not self.use_spacy or not self.nlp:
            logger.warning("SpaCy is required for entity extraction.")
            return []
        
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def suggest_corrections(
        self, 
        misspelled_word: str, 
        context_text: str,
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """Suggest corrections for a misspelled word based on context.
        
        Parameters
        ----------
        misspelled_word : str
            Misspelled word to correct
        context_text : str
            Context text containing the word
        top_n : int, optional
            Number of top suggestions to return, by default 3
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (correction, confidence) tuples
        """
        if not self.nlp:
            logger.warning("No NLP model available for contextual correction.")
            return []
        
        if self.use_spacy:
            return self._spacy_correct(misspelled_word, context_text, top_n)
        elif self.use_transformers:
            return self._transformer_correct(misspelled_word, context_text, top_n)
        else:
            return []
    
    def _spacy_correct(
        self, 
        misspelled_word: str, 
        context_text: str, 
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Use SpaCy for contextual word correction."""
        # Replace the misspelled word with a mask token
        # We'll use [MASK] as a placeholder
        word_pattern = r'\b' + re.escape(misspelled_word) + r'\b'
        mask_text = re.sub(word_pattern, "[MASK]", context_text)
        
        # Process the masked text
        doc = self.nlp(mask_text)
        
        # Find tokens near the mask
        mask_positions = [i for i, token in enumerate(doc) if token.text == "[MASK]"]
        if not mask_positions:
            return []
        
        # For each mask position, analyze surrounding context
        suggestions = []
        for pos in mask_positions:
            # Get context window
            start = max(0, pos - 5)
            end = min(len(doc), pos + 6)
            context_tokens = [token for i, token in enumerate(doc) if start <= i < end and i != pos]
            
            # Use word vectors to find similar words
            if hasattr(self.nlp, 'vocab') and hasattr(self.nlp.vocab, 'vectors') and self.nlp.vocab.vectors.size:
                context_vectors = [token.vector for token in context_tokens if token.has_vector]
                if context_vectors:
                    # Average the context vectors
                    avg_vector = np.mean(context_vectors, axis=0)
                    
                    # Find most similar words
                    most_similar = []
                    for word in self.nlp.vocab:
                        if word.is_alpha and len(word.text) > 1 and word.has_vector:
                            similarity = np.dot(avg_vector, word.vector) / (
                                np.linalg.norm(avg_vector) * np.linalg.norm(word.vector)
                            )
                            most_similar.append((word.text, similarity))
                    
                    # Sort and filter
                    most_similar.sort(key=lambda x: x[1], reverse=True)
                    suggestions.extend(most_similar[:top_n])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for word, score in suggestions:
            if word.lower() not in seen:
                seen.add(word.lower())
                unique_suggestions.append((word, float(score)))
        
        return unique_suggestions[:top_n]
    
    def _transformer_correct(
        self, 
        misspelled_word: str, 
        context_text: str, 
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Use transformer models for contextual word correction."""
        # Replace the misspelled word with the model's mask token
        mask_token = self.nlp.tokenizer.mask_token
        word_pattern = r'\b' + re.escape(misspelled_word) + r'\b'
        
        # Find all occurrences of the word and create a masked version for each
        all_matches = list(re.finditer(word_pattern, context_text))
        if not all_matches:
            return []
        
        suggestions = []
        for match in all_matches:
            start, end = match.span()
            masked_text = context_text[:start] + mask_token + context_text[end:]
            
            # Truncate if needed to fit model's max length
            if len(masked_text) > self.max_length:
                # Find the mask position
                mask_pos = masked_text.find(mask_token)
                
                # Create a context window around the mask
                half_window = (self.max_length - len(mask_token)) // 2
                start_pos = max(0, mask_pos - half_window)
                end_pos = min(len(masked_text), mask_pos + len(mask_token) + half_window)
                
                masked_text = masked_text[start_pos:end_pos]
            
            try:
                # Get predictions for the masked token
                predictions = self.nlp(masked_text)
                
                # Add to suggestions
                for pred in predictions[:top_n]:
                    token = pred["token_str"].strip()
                    score = pred["score"]
                    suggestions.append((token, score))
            except Exception as e:
                logger.warning(f"Error in transformer prediction: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for word, score in suggestions:
            if word.lower() not in seen and word.lower() != misspelled_word.lower():
                seen.add(word.lower())
                unique_suggestions.append((word, float(score)))
        
        return unique_suggestions[:top_n]
    
    def extract_potential_acronyms(self, text: str) -> Dict[str, List[str]]:
        """Extract potential acronyms and their expansions from text.
        
        Parameters
        ----------
        text : str
            Text to analyze
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping acronyms to potential expansions
        """
        if not self.use_spacy or not self.nlp:
            logger.warning("SpaCy is required for acronym extraction.")
            return {}
        
        # Process the text with SpaCy
        doc = self.nlp(text)
        
        # Find patterns like "Organization Name (ON)" or "ON (Organization Name)"
        acronyms = {}
        
        # Pattern 1: "Full Name (ACRONYM)"
        pattern1 = re.compile(r'([A-Za-z][A-Za-z\s]+)\s+\(([A-Z][A-Z0-9]{1,})\)')
        for match in pattern1.finditer(text):
            full_name, acronym = match.groups()
            if self._validate_acronym(full_name, acronym):
                if acronym not in acronyms:
                    acronyms[acronym] = []
                acronyms[acronym].append(full_name.strip())
        
        # Pattern 2: "ACRONYM (Full Name)"
        pattern2 = re.compile(r'([A-Z][A-Z0-9]{1,})\s+\(([A-Za-z][A-Za-z\s]+)\)')
        for match in pattern2.finditer(text):
            acronym, full_name = match.groups()
            if self._validate_acronym(full_name, acronym):
                if acronym not in acronyms:
                    acronyms[acronym] = []
                acronyms[acronym].append(full_name.strip())
        
        # Advanced NLP-based extraction
        if hasattr(doc, 'noun_chunks'):
            # Look for noun chunks that could be expansions
            for chunk in doc.noun_chunks:
                # Get first letters of each word in the chunk
                words = [token.text for token in chunk if token.is_alpha]
                if len(words) >= 2:  # Need at least 2 words for an acronym
                    first_letters = ''.join(word[0].upper() for word in words if word)
                    
                    # If we have a potential acronym elsewhere in the text
                    for token in doc:
                        if (token.text.isupper() and 
                            len(token.text) >= 2 and 
                            token.text == first_letters):
                            
                            if token.text not in acronyms:
                                acronyms[token.text] = []
                            acronyms[token.text].append(chunk.text)
        
        return acronyms
    
    def _validate_acronym(self, full_name: str, acronym: str) -> bool:
        """Validate if an acronym matches its potential expansion.
        
        Parameters
        ----------
        full_name : str
            Potential full expansion
        acronym : str
            Acronym to validate
            
        Returns
        -------
        bool
            True if the acronym is valid for the expansion
        """
        # Simple validation: Check if the acronym can be formed from the words
        words = full_name.split()
        
        # Case 1: First letter of each word
        first_letters = ''.join(word[0].upper() for word in words if word)
        if acronym in first_letters:
            return True
            
        # Case 2: First letters of significant words (skip common articles)
        skip_words = {'a', 'an', 'the', 'of', 'for', 'to', 'in', 'on', 'by', 'at', 'and'}
        sig_first_letters = ''.join(word[0].upper() for word in words 
                                   if word and word.lower() not in skip_words)
        if acronym in sig_first_letters:
            return True
            
        # Case 3: More flexible - check if acronym letters appear in order
        full_upper = full_name.upper()
        i, j = 0, 0
        while i < len(acronym) and j < len(full_upper):
            if acronym[i] == full_upper[j]:
                i += 1
            j += 1
        
        return i == len(acronym)
    
    def learn_from_corpus(
        self, 
        corpus: Union[List[str], pd.Series],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Learn domain-specific terminology and patterns from a corpus.
        
        Parameters
        ----------
        corpus : Union[List[str], pd.Series]
            Collection of texts to analyze
        output_path : Optional[str], optional
            Path to save learned information, by default None
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of learned information
        """
        if not self.use_spacy or not self.nlp:
            logger.warning("SpaCy is required for corpus learning.")
            return {}
        
        # Convert to list if needed
        if isinstance(corpus, pd.Series):
            text_list = corpus.tolist()
        else:
            text_list = corpus
        
        # Initialize containers for learned information
        learned_info = {
            "acronyms": {},            # Acronyms and their expansions
            "terminology": {},         # Domain-specific terminology with frequency
            "entity_types": {},        # Named entity types and examples
            "common_phrases": {},      # Common multi-word phrases
            "correction_patterns": {}, # Common correction patterns
        }
        
        # Frequency tracking
        term_freq = defaultdict(int)
        phrase_freq = defaultdict(int)
        entity_examples = defaultdict(list)
        
        # Process each text
        for text in text_list:
            # Skip empty texts
            if not text or not isinstance(text, str):
                continue
                
            # Process with SpaCy
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text
                
                if entity_type not in learned_info["entity_types"]:
                    learned_info["entity_types"][entity_type] = {
                        "count": 0,
                        "examples": []
                    }
                
                learned_info["entity_types"][entity_type]["count"] += 1
                
                # Add example if not too many already
                if len(entity_examples[entity_type]) < 10 and entity_text not in entity_examples[entity_type]:
                    entity_examples[entity_type].append(entity_text)
            
            # Extract potential acronyms
            acronyms = self.extract_potential_acronyms(text)
            for acronym, expansions in acronyms.items():
                if acronym not in learned_info["acronyms"]:
                    learned_info["acronyms"][acronym] = {}
                    
                for expansion in expansions:
                    if expansion not in learned_info["acronyms"][acronym]:
                        learned_info["acronyms"][acronym][expansion] = 0
                    learned_info["acronyms"][acronym][expansion] += 1
            
            # Track terminology and phrases
            for token in doc:
                # Skip non-alpha tokens and short words
                if not token.is_alpha or len(token.text) < 4:
                    continue
                
                # Check if likely domain-specific terminology
                if (not token.is_stop and 
                    token.pos_ in ("NOUN", "ADJ", "PROPN", "VERB") and
                    not token.like_num):
                    
                    term = token.lemma_
                    term_freq[term] += 1
            
            # Find noun chunks for phrases
            if hasattr(doc, 'noun_chunks'):
                for chunk in doc.noun_chunks:
                    if len(chunk) > 1 and len(chunk.text.split()) > 1:
                        phrase = chunk.text
                        phrase_freq[phrase] += 1
        
        # Filter to likely domain terminology based on frequency
        if term_freq:
            min_freq = max(3, len(text_list) // 10)  # Adjust threshold based on corpus size
            learned_info["terminology"] = {term: freq for term, freq in term_freq.items() 
                                        if freq >= min_freq}
        
        # Add common phrases
        if phrase_freq:
            phrase_threshold = max(2, len(text_list) // 20)
            learned_info["common_phrases"] = {phrase: freq for phrase, freq in phrase_freq.items() 
                                           if freq >= phrase_threshold}
        
        # Add entity examples
        for entity_type, examples in entity_examples.items():
            learned_info["entity_types"][entity_type]["examples"] = examples
        
        # Save to file if requested
        if output_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(learned_info, f, indent=2)
                logger.info(f"Saved learned corpus information to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save learned information: {e}")
        
        return learned_info


def process_context(
    text: str,
    use_spacy: bool = True,
    use_transformers: bool = False,
) -> Dict[str, Any]:
    """One-off context processing function.
    
    Parameters
    ----------
    text : str
        Text to process
    use_spacy : bool, optional
        Whether to use SpaCy, by default True
    use_transformers : bool, optional
        Whether to use transformers, by default False
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of processed information
    """
    processor = ContextualProcessor(
        use_spacy=use_spacy,
        use_transformers=use_transformers
    )
    
    result = {
        "entities": processor.extract_entities(text),
        "acronyms": processor.extract_potential_acronyms(text),
    }
    
    return result
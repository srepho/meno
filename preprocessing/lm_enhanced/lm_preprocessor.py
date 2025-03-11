"""Language model enhanced preprocessor.

This module provides language model enhanced versions of the acronym expansion
and spelling correction components, as well as a unified preprocessor that
combines them.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Set, Any, Callable
from pathlib import Path
import os
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError(
        "The sentence-transformers package is required for LMPreprocessor. "
        "Please install it with `pip install sentence-transformers`."
    )

# Import Meno components
from ..acronyms import AcronymExpander, expand_acronyms
from ..spelling import SpellingCorrector, correct_spelling
from ..normalization import TextNormalizer, normalize_text
from ...modeling.embeddings import DocumentEmbedding

# Set up logging
logger = logging.getLogger(__name__)


class LMAcronymExpander(AcronymExpander):
    """Language model enhanced acronym expander.
    
    This class extends the base AcronymExpander with language model capabilities
    to improve acronym detection and expansion using semantic context.
    
    Parameters
    ----------
    custom_mappings : Dict[str, str], optional
        Custom acronym to expansion mappings, by default None
    min_length : int, optional
        Minimum length for acronyms, by default 2
    domain : str, optional
        Domain-specific acronyms to include, by default None.
        Options: "healthcare", "tech", "finance", "legal"
    contextual_expansion : bool, optional
        Whether to extract and learn acronyms from context, by default True
    ignore_case : bool, optional
        Whether to ignore case when matching acronyms, by default False
    model : Union[str, SentenceTransformer], optional
        Language model to use, by default "sentence-transformers/all-MiniLM-L6-v2"
    context_window : int, optional
        Number of tokens to consider for context around an acronym, by default 50
    min_similarity : float, optional
        Minimum similarity score required for context matching, by default 0.6
    learning_rate : float, optional
        Rate at which to learn from new context, by default 0.7
    use_gpu : bool, optional
        Whether to use GPU for embeddings if available, by default False
    """
    
    def __init__(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
        min_length: int = 2,
        domain: Optional[str] = None,
        contextual_expansion: bool = True,
        ignore_case: bool = False,
        model: Union[str, SentenceTransformer] = "sentence-transformers/all-MiniLM-L6-v2",
        context_window: int = 50,
        min_similarity: float = 0.6,
        learning_rate: float = 0.7,
        use_gpu: bool = False,
    ):
        # Initialize the base class
        super().__init__(
            custom_mappings=custom_mappings,
            min_length=min_length,
            domain=domain,
            contextual_expansion=contextual_expansion,
            ignore_case=ignore_case
        )
        
        # Initialize language model
        self.context_window = context_window
        self.min_similarity = min_similarity
        self.learning_rate = learning_rate
        
        # Set up model
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
            if not use_gpu:
                self.model = self.model.to("cpu")
        else:
            self.model = model
        
        # Create embeddings for known acronyms and their contexts
        self.acronym_contexts = {}
        self.acronym_embeddings = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings for known acronyms and their definitions."""
        # Generate example contexts for known acronyms
        contexts = []
        acronyms = []
        
        for acronym, expansion in self.acronym_dict.items():
            # Create a simple example context
            context = f"The {acronym} ({expansion}) is commonly used in this field."
            contexts.append(context)
            acronyms.append(acronym)
            
            # Store the context
            self.acronym_contexts[acronym] = [expansion]
        
        # Generate embeddings in batch
        if contexts:
            try:
                embeddings = self.model.encode(contexts, convert_to_tensor=True)
                
                # Store embeddings
                for i, acronym in enumerate(acronyms):
                    self.acronym_embeddings[acronym] = embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")
    
    def expand_acronyms(self, text: str) -> str:
        """Expand acronyms in text with language model support.
        
        Parameters
        ----------
        text : str
            Text with potential acronyms
            
        Returns
        -------
        str
            Text with expanded acronyms
        """
        if not text:
            return text
        
        # First pass: extract and learn parenthetical acronyms
        if self.contextual_expansion:
            text = self._extract_and_learn_parenthetical_acronyms(text)
        
        # Extract all potential acronyms
        acronyms = self.extract_acronyms(text)
        
        # Sort acronyms by length (descending) to avoid replacing substrings
        acronyms = sorted(acronyms, key=len, reverse=True)
        
        # Context-aware acronym expansion
        result = text
        for acronym in acronyms:
            # Skip if already expanded in text (e.g., "CEO (Chief Executive Officer)")
            if re.search(rf"{re.escape(acronym)}\s+\([^)]+\)", result):
                continue
            
            # Skip if too short
            if len(acronym) < self.min_length:
                continue
            
            # Get expansion with context awareness
            expansion = self._get_best_expansion_for_context(acronym, text)
            
            if expansion:
                # Replace with expanded version
                result = re.sub(
                    rf"\b{re.escape(acronym)}\b",
                    f"{acronym} ({expansion})",
                    result
                )
        
        return result
    
    def _get_best_expansion_for_context(self, acronym: str, text: str) -> Optional[str]:
        """Find the best expansion for an acronym based on text context.
        
        Parameters
        ----------
        acronym : str
            The acronym to expand
        text : str
            The text containing the acronym
            
        Returns
        -------
        Optional[str]
            The best expansion or None if not found
        """
        # Direct lookup for known acronyms
        if acronym in self.acronym_dict:
            return self.acronym_dict[acronym]
        
        # Case-insensitive lookup if enabled
        if self.ignore_case and acronym.upper() in self.acronym_dict:
            return self.acronym_dict[acronym.upper()]
        
        # Find the context around the acronym
        context = self._extract_context(acronym, text)
        
        if not context:
            return None
        
        # Get context embedding
        try:
            context_embedding = self.model.encode(context, convert_to_tensor=True)
            
            # Compare with known acronym contexts
            best_score = -1
            best_expansion = None
            
            for acr, emb in self.acronym_embeddings.items():
                if len(acr) < 2:  # Skip very short acronyms
                    continue
                
                # Calculate similarity
                similarity = util.pytorch_cos_sim(context_embedding, emb).item()
                
                if similarity > best_score and similarity > self.min_similarity:
                    best_score = similarity
                    best_expansion = self.acronym_dict[acr]
            
            return best_expansion
        
        except Exception as e:
            logger.warning(f"Error computing embeddings: {e}")
            return None
    
    def _extract_context(self, acronym: str, text: str) -> str:
        """Extract the context around an acronym.
        
        Parameters
        ----------
        acronym : str
            The acronym to find
        text : str
            The text containing the acronym
            
        Returns
        -------
        str
            The extracted context
        """
        # Find all occurrences of the acronym
        pattern = re.compile(rf"\b{re.escape(acronym)}\b")
        matches = list(pattern.finditer(text))
        
        if not matches:
            return ""
        
        # Take the first occurrence for simplicity
        match = matches[0]
        
        # Extract context window
        start = max(0, match.start() - self.context_window)
        end = min(len(text), match.end() + self.context_window)
        
        # Get the context
        context = text[start:end]
        
        return context
    
    def learn_from_context(self, acronym: str, expansion: str, context: str):
        """Learn a new acronym-expansion pair with its context.
        
        Parameters
        ----------
        acronym : str
            The acronym to learn
        expansion : str
            The expansion of the acronym
        context : str
            The context in which the acronym appears
        """
        # Add to dictionary
        self.acronym_dict[acronym] = expansion
        
        # Add to context store
        if acronym not in self.acronym_contexts:
            self.acronym_contexts[acronym] = []
        
        self.acronym_contexts[acronym].append(context)
        
        # Update embeddings
        try:
            # Create new embedding
            new_embedding = self.model.encode(context, convert_to_tensor=True)
            
            if acronym in self.acronym_embeddings:
                # Blend with existing embedding
                existing = self.acronym_embeddings[acronym]
                blended = (
                    existing * (1 - self.learning_rate) + 
                    new_embedding * self.learning_rate
                )
                self.acronym_embeddings[acronym] = blended
            else:
                # Add new embedding
                self.acronym_embeddings[acronym] = new_embedding
        
        except Exception as e:
            logger.warning(f"Error updating embeddings: {e}")
    
    def suggest_expansions(self, acronym: str, context: Optional[str] = None, n: int = 5) -> List[Tuple[str, float]]:
        """Suggest possible expansions for an acronym, considering context if provided.
        
        Parameters
        ----------
        acronym : str
            The acronym to find expansions for
        context : Optional[str], optional
            Context in which the acronym appears, by default None
        n : int, optional
            Number of suggestions to return, by default 5
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (expansion, score) tuples
        """
        # Direct lookup for exact match
        if acronym in self.acronym_dict:
            return [(self.acronym_dict[acronym], 1.0)]
        
        # Case-insensitive lookup if enabled
        if self.ignore_case and acronym.upper() in self.acronym_dict:
            return [(self.acronym_dict[acronym.upper()], 1.0)]
        
        # Prepare suggestions
        suggestions = []
        
        # First, try string similarity
        for acr, expansion in self.acronym_dict.items():
            # Skip short acronyms
            if len(acr) < 2:
                continue
            
            # Calculate string similarity
            score = SequenceMatcher(None, acronym, acr).ratio()
            
            if score > 0.7:  # Threshold for string similarity
                suggestions.append((expansion, score))
        
        # If we have context, use embeddings
        if context and self.acronym_embeddings:
            try:
                # Embed the context
                context_embedding = self.model.encode(context, convert_to_tensor=True)
                
                # Calculate similarities
                for acr, emb in self.acronym_embeddings.items():
                    if len(acr) < 2:  # Skip very short acronyms
                        continue
                    
                    # Calculate similarity
                    similarity = util.pytorch_cos_sim(context_embedding, emb).item()
                    
                    if similarity > self.min_similarity:
                        expansion = self.acronym_dict[acr]
                        suggestions.append((expansion, similarity))
            
            except Exception as e:
                logger.warning(f"Error computing embeddings: {e}")
        
        # Sort by score and take top n
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        return suggestions[:n]
    
    def generate_acronym_expansion(self, acronym: str, context: str) -> Optional[str]:
        """Generate a potential expansion for an unknown acronym.
        
        Parameters
        ----------
        acronym : str
            The acronym to generate an expansion for
        context : str
            The context in which the acronym appears
            
        Returns
        -------
        Optional[str]
            A generated expansion or None if generation fails
        """
        # First, try to extract from context using pattern matching
        # Look for patterns like "X (Full Expansion)"
        pattern = re.compile(rf"{re.escape(acronym)}\s*\(([^)]+)\)")
        matches = pattern.findall(context)
        
        if matches:
            return matches[0]
        
        # Try to find words that match the acronym letters
        words = re.findall(r'\b\w+\b', context)
        
        # Convert acronym to uppercase for matching
        upper_acronym = acronym.upper()
        
        # Find combinations of words that could form the acronym
        candidates = []
        
        # Simple approach: find words starting with each letter of the acronym
        for i in range(len(words) - len(acronym) + 1):
            candidate = []
            match = True
            
            for j, letter in enumerate(upper_acronym):
                if i + j >= len(words):
                    match = False
                    break
                
                word = words[i + j]
                if not word or word[0].upper() != letter:
                    match = False
                    break
                
                candidate.append(word)
            
            if match:
                candidates.append(" ".join(candidate))
        
        if candidates:
            return candidates[0]
        
        # If all fails, try acronym suggestions
        suggestions = self.suggest_expansions(acronym, context)
        
        if suggestions:
            return suggestions[0][0]
        
        return None


class LMSpellingCorrector(SpellingCorrector):
    """Language model enhanced spelling corrector.
    
    This class extends the base SpellingCorrector with language model capabilities
    to improve spelling correction using semantic context.
    
    Parameters
    ----------
    dictionary : Dict[str, str], optional
        Custom spelling correction dictionary, by default None
    min_word_length : int, optional
        Minimum length for words to be considered, by default 3
    max_distance : int, optional
        Maximum Levenshtein distance for similarity, by default 2
    min_score : int, optional
        Minimum similarity score (0-100), by default 80
    ignore_case : bool, optional
        Whether to ignore case when matching words, by default True
    domain : str, optional
        Domain-specific dictionary to include, by default None.
        Options: "medical", "technical", "financial", "legal"
    use_keyboard_proximity : bool, optional
        Whether to consider keyboard proximity for typos, by default True
    learn_corrections : bool, optional
        Whether to learn from corrections, by default True
    model : Union[str, SentenceTransformer], optional
        Language model to use, by default "sentence-transformers/all-MiniLM-L6-v2"
    context_window : int, optional
        Number of tokens to consider for context, by default 5
    semantic_threshold : float, optional
        Minimum semantic similarity for replacement, by default 0.75
    use_gpu : bool, optional
        Whether to use GPU for embeddings if available, by default False
    """
    
    def __init__(
        self,
        dictionary: Optional[Dict[str, str]] = None,
        min_word_length: int = 3,
        max_distance: int = 2,
        min_score: int = 80,
        ignore_case: bool = True,
        domain: Optional[str] = None,
        use_keyboard_proximity: bool = True,
        learn_corrections: bool = True,
        model: Union[str, SentenceTransformer] = "sentence-transformers/all-MiniLM-L6-v2",
        context_window: int = 5,
        semantic_threshold: float = 0.75,
        use_gpu: bool = False,
    ):
        # Initialize the base class
        super().__init__(
            dictionary=dictionary,
            min_word_length=min_word_length,
            max_distance=max_distance,
            min_score=min_score,
            ignore_case=ignore_case,
            domain=domain,
            use_keyboard_proximity=use_keyboard_proximity,
            learn_corrections=learn_corrections
        )
        
        # Initialize language model
        self.context_window = context_window
        self.semantic_threshold = semantic_threshold
        
        # Set up model
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
            if not use_gpu:
                self.model = self.model.to("cpu")
        else:
            self.model = model
        
        # Word embeddings cache
        self.word_embeddings = {}
        
        # Build embeddings for common words
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings for common words."""
        # Get common corrections
        common_words = set()
        
        # Add correct forms from dictionary
        for _, correction in self.dictionary.items():
            common_words.add(correction)
        
        # Add some common English words (limited sample for efficiency)
        common_english = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
        ]
        
        common_words.update(common_english)
        
        # Generate embeddings for common words (limit batch size for memory efficiency)
        word_list = list(common_words)[:1000]  # Limit to 1000 words
        
        if word_list:
            try:
                # Create sentences with context
                sentences = [f"The {word} is common." for word in word_list]
                
                # Generate embeddings
                embeddings = self.model.encode(sentences, convert_to_tensor=True)
                
                # Store embeddings
                for i, word in enumerate(word_list):
                    self.word_embeddings[word] = embeddings[i]
            
            except Exception as e:
                logger.warning(f"Failed to initialize word embeddings: {e}")
    
    def correct_word(self, word: str, context: Optional[str] = None) -> str:
        """Correct a potentially misspelled word using semantic context if available.
        
        Parameters
        ----------
        word : str
            Word to potentially correct
        context : Optional[str], optional
            Context around the word, by default None
            
        Returns
        -------
        str
            Corrected word or original if no correction needed
        """
        # Skip correction for short words
        if len(word) < self.min_word_length:
            return word
        
        # Skip correction for words in ignore list
        if word in self.ignore_words:
            return word
        
        # Direct lookup in dictionary
        lookup_word = word if not self.ignore_case else word.lower()
        
        if lookup_word in self.dictionary:
            correction = self.dictionary[lookup_word]
            
            # Apply case preservation if original had specific case
            if word[0].isupper() and not self.ignore_case:
                correction = correction[0].upper() + correction[1:]
            
            return correction
        
        # Context-based correction if context is provided
        if context:
            context_correction = self._correct_with_context(word, context)
            if context_correction != word:
                # Add to learned corrections if enabled
                if self.learn_corrections:
                    self.learned_corrections[lookup_word] = context_correction
                return context_correction
        
        # Fall back to standard spelling correction
        return super().correct_word(word)
    
    def _correct_with_context(self, word: str, context: str) -> str:
        """Correct a word using its semantic context.
        
        Parameters
        ----------
        word : str
            Word to potentially correct
        context : str
            Context around the word
            
        Returns
        -------
        str
            Corrected word or original if no correction needed
        """
        # Get word in lowercase for comparison
        lookup_word = word.lower() if self.ignore_case else word
        
        # Skip if word is too short or in ignore list
        if len(word) < self.min_word_length or word in self.ignore_words:
            return word
        
        # If word is in our dictionary, use that correction
        if lookup_word in self.dictionary:
            correction = self.dictionary[lookup_word]
            return correction
        
        try:
            # Generate embeddings for word in context
            word_context = self._extract_context_window(word, context)
            word_context_embedding = self.model.encode(word_context, convert_to_tensor=True)
            
            # Find potential corrections based on string similarity
            candidates = self._get_correction_candidates(word)
            
            if not candidates:
                return word  # No correction candidates
            
            # Create context snippets with each candidate
            candidate_contexts = [
                word_context.replace(word, candidate) for candidate in candidates
            ]
            
            # Embed all candidate contexts
            candidate_embeddings = self.model.encode(candidate_contexts, convert_to_tensor=True)
            
            # Calculate semantic similarity
            similarities = util.pytorch_cos_sim(
                word_context_embedding, candidate_embeddings
            )[0]
            
            # Find best candidate
            best_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_idx].item()
            
            # Return best candidate if it's above the semantic threshold
            if best_similarity > self.semantic_threshold:
                best_correction = candidates[best_idx]
                
                # Apply case preservation if original had specific case
                if word[0].isupper() and not self.ignore_case:
                    best_correction = best_correction[0].upper() + best_correction[1:]
                
                # Add to learned corrections if enabled
                if self.learn_corrections:
                    self.learned_corrections[lookup_word] = best_correction
                
                return best_correction
            
            return word  # No good correction found
        
        except Exception as e:
            logger.warning(f"Error in context-based correction: {e}")
            return word
    
    def _extract_context_window(self, word: str, text: str) -> str:
        """Extract a window of text around a word.
        
        Parameters
        ----------
        word : str
            The word to find in the text
        text : str
            The text containing the word
            
        Returns
        -------
        str
            A window of text around the word
        """
        # Find the position of the word in the text
        pattern = re.compile(rf"\b{re.escape(word)}\b")
        match = pattern.search(text)
        
        if not match:
            return text  # Word not found, return full text
        
        # Split text into words
        words = text.split()
        
        # Find the index of the word
        word_index = None
        for i, w in enumerate(words):
            if w == word or w.strip(".,!?;:()[]{}\"'") == word:
                word_index = i
                break
        
        if word_index is None:
            return text  # Word not found, return full text
        
        # Extract context window
        start = max(0, word_index - self.context_window)
        end = min(len(words), word_index + self.context_window + 1)
        
        # Join the words in the window
        context = " ".join(words[start:end])
        
        return context
    
    def _get_correction_candidates(self, word: str, max_candidates: int = 5) -> List[str]:
        """Get potential correction candidates for a word.
        
        Parameters
        ----------
        word : str
            Word to find corrections for
        max_candidates : int, optional
            Maximum number of candidates to return, by default 5
            
        Returns
        -------
        List[str]
            List of potential corrections
        """
        candidates = []
        lookup_word = word.lower() if self.ignore_case else word
        
        # First, check dictionary corrections
        for misspelled, correction in self.dictionary.items():
            # Calculate similarity
            similarity = SequenceMatcher(None, lookup_word, misspelled).ratio()
            
            if similarity >= self.min_score / 100:
                candidates.append((correction, similarity))
        
        # Check keyboard proximity typos
        if self.use_keyboard_proximity:
            for valid_word in set(self.dictionary.values()):
                if self._is_keyboard_typo(lookup_word, valid_word):
                    similarity = 0.85  # Default score for keyboard typos
                    candidates.append((valid_word, similarity))
        
        # Sort by similarity and take top candidates
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        # Extract just the words
        candidate_words = [cand[0] for cand in candidates[:max_candidates]]
        
        # If we have too few candidates, add common words that are close
        if len(candidate_words) < 3:
            # Try to find similar words from common English
            for common_word in self.word_embeddings.keys():
                if len(common_word) < 3:  # Skip very short words
                    continue
                
                similarity = SequenceMatcher(None, lookup_word, common_word).ratio()
                
                if similarity >= self.min_score / 100 and common_word not in candidate_words:
                    candidates.append((common_word, similarity))
            
            # Sort again and update candidate words
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            candidate_words = [cand[0] for cand in candidates[:max_candidates]]
        
        return candidate_words
    
    def correct_text(self, text: str) -> str:
        """Correct spelling in text with semantic context awareness.
        
        Parameters
        ----------
        text : str
            Text to correct
            
        Returns
        -------
        str
            Corrected text
        """
        if not text:
            return text
        
        # Split text into words
        words = re.findall(r'\b\w+\b', text)
        
        # Detect potential misspellings
        for word in words:
            # Skip short words
            if len(word) < self.min_word_length:
                continue
            
            # Skip words that are likely correct
            if word in self.ignore_words:
                continue
            
            # Check if it's potentially misspelled
            lookup_word = word.lower() if self.ignore_case else word
            if lookup_word not in self.dictionary and lookup_word not in self.dictionary.values():
                # Extract context for this word
                context = self._extract_context_window(word, text)
                
                # Get correction with context
                correction = self._correct_with_context(word, context)
                
                # Replace in text if corrected
                if correction != word:
                    # Use word boundary in pattern to avoid replacing substrings
                    pattern = rf"\b{re.escape(word)}\b"
                    text = re.sub(pattern, correction, text)
        
        return text
    
    def find_likely_misspellings(self, text: str, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Find words in text that are likely misspelled based on semantic context.
        
        Parameters
        ----------
        text : str
            Text to analyze
        threshold : float, optional
            Confidence threshold for flagging words, by default 0.8
            
        Returns
        -------
        List[Tuple[str, str, float]]
            List of (word, suggested_correction, confidence) tuples
        """
        results = []
        
        if not text:
            return results
        
        # Split text into words
        words = re.findall(r'\b\w+\b', text)
        
        # Detect potential misspellings using semantic context
        for word in words:
            # Skip short words
            if len(word) < self.min_word_length:
                continue
            
            # Skip words that are likely correct
            if word in self.ignore_words:
                continue
            
            # Check if it's potentially misspelled
            lookup_word = word.lower() if self.ignore_case else word
            
            if lookup_word not in self.dictionary.values():
                # Get context for this word
                context = self._extract_context_window(word, text)
                
                try:
                    # Embed the context
                    context_embedding = self.model.encode(context, convert_to_tensor=True)
                    
                    # Get correction candidates
                    candidates = self._get_correction_candidates(word)
                    
                    if not candidates:
                        continue
                    
                    # Create context with each candidate
                    candidate_contexts = [
                        context.replace(word, candidate) for candidate in candidates
                    ]
                    
                    # Embed all candidate contexts
                    candidate_embeddings = self.model.encode(candidate_contexts, convert_to_tensor=True)
                    
                    # Calculate semantic similarity
                    similarities = util.pytorch_cos_sim(
                        context_embedding, candidate_embeddings
                    )[0]
                    
                    # Find best candidate
                    best_idx = torch.argmax(similarities).item()
                    best_similarity = similarities[best_idx].item()
                    
                    # If the best candidate is significantly better, it's likely a misspelling
                    if best_similarity > threshold:
                        results.append((
                            word,
                            candidates[best_idx],
                            float(best_similarity)
                        ))
                
                except Exception as e:
                    logger.warning(f"Error analyzing potential misspellings: {e}")
        
        return results
    
    def suggest_corrections(self, word: str, context: Optional[str] = None, n: int = 5) -> List[Tuple[str, float]]:
        """Suggest possible corrections for a word, considering context if provided.
        
        Parameters
        ----------
        word : str
            The word to find corrections for
        context : Optional[str], optional
            Context in which the word appears, by default None
        n : int, optional
            Number of suggestions to return, by default 5
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (correction, score) tuples
        """
        lookup_word = word.lower() if self.ignore_case else word
        
        # Get correction candidates
        candidates = self._get_correction_candidates(word, max_candidates=n * 2)
        
        if not candidates:
            return []
        
        # If we have context, use language model to rank
        if context:
            try:
                # Create context snippets with the original word
                original_context = self._extract_context_window(word, context)
                
                # Create context snippets with each candidate
                candidate_contexts = [
                    original_context.replace(word, candidate) for candidate in candidates
                ]
                
                # Add original context to beginning
                all_contexts = [original_context] + candidate_contexts
                
                # Embed all contexts
                embeddings = self.model.encode(all_contexts, convert_to_tensor=True)
                
                # Original context embedding
                original_embedding = embeddings[0]
                
                # Calculate semantic similarity for each candidate
                similarities = []
                
                for i, candidate in enumerate(candidates):
                    # Get similarity to original
                    similarity = util.pytorch_cos_sim(
                        original_embedding, embeddings[i + 1]
                    ).item()
                    
                    similarities.append((candidate, similarity))
                
                # Sort by similarity and return top n
                return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
            
            except Exception as e:
                logger.warning(f"Error computing embedding similarities: {e}")
        
        # Fall back to string similarity if no context or error
        similarities = []
        
        for candidate in candidates:
            score = SequenceMatcher(None, lookup_word, candidate).ratio()
            similarities.append((candidate, score))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]


class LMPreprocessor:
    """Language model enhanced preprocessor.
    
    This class integrates LMAcronymExpander and LMSpellingCorrector into a unified
    preprocessing pipeline with language model support.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the language model to use, by default "sentence-transformers/all-MiniLM-L6-v2"
    acronym_dict : Dict[str, str], optional
        Custom acronym dictionary, by default None
    spelling_dict : Dict[str, str], optional
        Custom spelling dictionary, by default None
    domain : str, optional
        Domain-specific dictionaries to include, by default None
    custom_acronyms : Dict[str, str], optional
        Additional custom acronyms, by default None
    custom_spelling : Dict[str, str], optional
        Additional custom spelling corrections, by default None
    normalize_options : Dict[str, Any], optional
        Options for text normalization, by default None
    use_gpu : bool, optional
        Whether to use GPU if available, by default False
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        acronym_dict: Optional[Dict[str, str]] = None,
        spelling_dict: Optional[Dict[str, str]] = None,
        domain: Optional[str] = None,
        custom_acronyms: Optional[Dict[str, str]] = None,
        custom_spelling: Optional[Dict[str, str]] = None,
        normalize_options: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
    ):
        # Import torch here to avoid early import
        global torch
        import torch
        
        # Initialize language model
        try:
            self.model = SentenceTransformer(model_name)
            if not use_gpu:
                self.model = self.model.to("cpu")
        except Exception as e:
            logger.error(f"Error initializing language model: {e}")
            raise
        
        # Initialize preprocessors
        self.acronym_expander = LMAcronymExpander(
            custom_mappings=acronym_dict,
            domain=domain,
            model=self.model,
            use_gpu=use_gpu
        )
        
        self.spelling_corrector = LMSpellingCorrector(
            dictionary=spelling_dict,
            domain=domain,
            model=self.model,
            use_gpu=use_gpu
        )
        
        # Add custom dictionaries
        if custom_acronyms:
            self.acronym_expander.add_acronyms(custom_acronyms)
        
        if custom_spelling:
            self.spelling_corrector.add_corrections(custom_spelling)
        
        # Set up normalizer with options
        normalize_options = normalize_options or {}
        self.text_normalizer = TextNormalizer(**normalize_options)
    
    def preprocess_text(
        self,
        text: str,
        correct_spelling: bool = True,
        expand_acronyms: bool = True,
        normalize: bool = True,
        context_aware: bool = True,
    ) -> str:
        """Preprocess text with language model enhancement.
        
        Parameters
        ----------
        text : str
            Text to preprocess
        correct_spelling : bool, optional
            Whether to correct spelling, by default True
        expand_acronyms : bool, optional
            Whether to expand acronyms, by default True
        normalize : bool, optional
            Whether to normalize text, by default True
        context_aware : bool, optional
            Whether to use context awareness, by default True
            
        Returns
        -------
        str
            Preprocessed text
        """
        if not text:
            return text
        
        result = text
        
        # Expand acronyms
        if expand_acronyms:
            result = self.acronym_expander.expand_acronyms(result)
        
        # Correct spelling
        if correct_spelling:
            if context_aware:
                result = self.spelling_corrector.correct_text(result)
            else:
                # Use non-context-aware version
                result = correct_spelling(result)
        
        # Normalize text
        if normalize:
            result = self.text_normalizer.normalize_text(result)
        
        return result
    
    def preprocess_batch(
        self,
        texts: Union[List[str], pd.Series],
        correct_spelling: bool = True,
        expand_acronyms: bool = True,
        normalize: bool = True,
        context_aware: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[List[str], pd.Series]:
        """Process a batch of texts with language model enhancement.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Texts to preprocess
        correct_spelling : bool, optional
            Whether to correct spelling, by default True
        expand_acronyms : bool, optional
            Whether to expand acronyms, by default True
        normalize : bool, optional
            Whether to normalize text, by default True
        context_aware : bool, optional
            Whether to use context awareness, by default True
        batch_size : int, optional
            Batch size for processing, by default 32
        show_progress : bool, optional
            Whether to show a progress bar, by default False
            
        Returns
        -------
        Union[List[str], pd.Series]
            Preprocessed texts in the same format as input
        """
        # Handle pandas Series
        is_series = isinstance(texts, pd.Series)
        
        if is_series:
            texts_list = texts.tolist()
        else:
            texts_list = texts
        
        # Process in batches
        results = []
        
        # Set up progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts_list), batch_size))
            except ImportError:
                iterator = range(0, len(texts_list), batch_size)
                logger.warning("tqdm not installed, not showing progress bar")
        else:
            iterator = range(0, len(texts_list), batch_size)
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(texts_list))
            batch = texts_list[start_idx:end_idx]
            
            # Process each text in the batch
            processed_batch = []
            
            for text in batch:
                processed = self.preprocess_text(
                    text,
                    correct_spelling=correct_spelling,
                    expand_acronyms=expand_acronyms,
                    normalize=normalize,
                    context_aware=context_aware
                )
                processed_batch.append(processed)
            
            results.extend(processed_batch)
        
        # Return in the same format as input
        if is_series:
            return pd.Series(results, index=texts.index)
        else:
            return results
    
    def find_document_acronyms(
        self,
        texts: Union[List[str], pd.Series],
        min_count: int = 2,
    ) -> pd.DataFrame:
        """Find and analyze acronyms in a corpus of documents.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Collection of texts to analyze
        min_count : int, optional
            Minimum count to include an acronym, by default 2
            
        Returns
        -------
        pd.DataFrame
            DataFrame with acronym analysis
        """
        # Handle pandas Series
        if isinstance(texts, pd.Series):
            texts_list = texts.tolist()
        else:
            texts_list = texts
        
        # Count acronyms across all texts
        acronym_counts = self.acronym_expander.extract_acronyms_batch(texts_list)
        
        # Filter by minimum count
        acronym_counts = {k: v for k, v in acronym_counts.items() if v >= min_count}
        
        # Get known expansions
        expansions = {}
        for acronym in acronym_counts:
            if acronym in self.acronym_expander.acronym_dict:
                expansions[acronym] = self.acronym_expander.acronym_dict[acronym]
            else:
                # Try to find expansion from the texts
                for text in texts_list:
                    if acronym in text:
                        pattern = re.compile(rf"{re.escape(acronym)}\s*\(([^)]+)\)")
                        matches = pattern.findall(text)
                        if matches:
                            expansions[acronym] = matches[0]
                            break
                
                # If still not found, use "Unknown"
                if acronym not in expansions:
                    expansions[acronym] = "Unknown"
        
        # Create DataFrame
        data = []
        for acronym, count in sorted(acronym_counts.items(), key=lambda x: x[1], reverse=True):
            data.append({
                "Acronym": acronym,
                "Count": count,
                "Expansion": expansions.get(acronym, "Unknown"),
                "Known": acronym in self.acronym_expander.acronym_dict
            })
        
        return pd.DataFrame(data)
    
    def find_misspelled_words(
        self,
        texts: Union[List[str], pd.Series],
        min_count: int = 2,
        confidence_threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Find potential misspellings in a corpus of documents.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Collection of texts to analyze
        min_count : int, optional
            Minimum count to include a misspelling, by default 2
        confidence_threshold : float, optional
            Minimum confidence to flag as misspelled, by default 0.8
            
        Returns
        -------
        pd.DataFrame
            DataFrame with misspelling analysis
        """
        # Handle pandas Series
        if isinstance(texts, pd.Series):
            texts_list = texts.tolist()
        else:
            texts_list = texts
        
        # Track potential misspellings
        misspelled_words = []
        word_counts = Counter()
        
        # Process each text
        for text in texts_list:
            # Get potential misspellings with context
            misspellings = self.spelling_corrector.find_likely_misspellings(
                text, threshold=confidence_threshold
            )
            
            # Track counts and corrections
            for word, correction, confidence in misspellings:
                word_counts[word] += 1
                misspelled_words.append((word, correction, confidence))
        
        # Create DataFrame
        data = []
        
        for word, correction, confidence in misspelled_words:
            # Only include if count meets threshold
            if word_counts[word] >= min_count:
                data.append({
                    "Word": word,
                    "Correction": correction,
                    "Confidence": confidence,
                    "Count": word_counts[word]
                })
        
        # Remove duplicates and sort
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(["Count", "Confidence"], ascending=False)
            df = df.drop_duplicates(subset=["Word"]).reset_index(drop=True)
        
        return df
    
    def save_learned_corrections(self, file_path: str) -> Dict[str, Any]:
        """Save learned corrections and expansions to a file.
        
        Parameters
        ----------
        file_path : str
            Path to save the learned corrections to
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with saved data
        """
        data = {
            "acronyms": self.acronym_expander.acronym_dict,
            "spelling": self.spelling_corrector.learned_corrections
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def load_learned_corrections(self, file_path: str) -> bool:
        """Load learned corrections and expansions from a file.
        
        Parameters
        ----------
        file_path : str
            Path to load the learned corrections from
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Add acronyms
            if "acronyms" in data:
                self.acronym_expander.add_acronyms(data["acronyms"])
            
            # Add spelling corrections
            if "spelling" in data:
                self.spelling_corrector.add_corrections(data["spelling"])
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading learned corrections: {e}")
            return False
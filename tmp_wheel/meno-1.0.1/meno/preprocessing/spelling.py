"""Spelling correction utilities."""

from typing import Dict, List, Optional, Union, Tuple
import re
import pandas as pd
from thefuzz import process


class SpellingCorrector:
    """Class for spelling correction in text data.
    
    This class handles spelling correction using fuzzy matching against
    a dictionary of correct spellings.
    
    Parameters
    ----------
    dictionary : Optional[Dict[str, str]], optional
        Custom dictionary mapping misspelled words to correct ones, by default None
    min_word_length : int, optional
        Minimum length for words to be considered for correction, by default 4
    max_distance : int, optional
        Maximum Levenshtein distance for fuzzy matching, by default 2
    min_score : int, optional
        Minimum similarity score (0-100) to accept a correction, by default 80
    ignore_case : bool, optional
        Whether to ignore case when matching words, by default True
    ignore_words : Optional[List[str]], optional
        List of words to ignore during correction, by default None
    
    Attributes
    ----------
    dictionary : Dict[str, str]
        Dictionary mapping misspelled words to correct ones
    word_list : List[str]
        List of correct spellings for fuzzy matching
    min_word_length : int
        Minimum length for words to be considered for correction
    max_distance : int
        Maximum Levenshtein distance for fuzzy matching
    min_score : int
        Minimum similarity score to accept a correction
    ignore_case : bool
        Whether to ignore case when matching words
    ignore_words : Set[str]
        Set of words to ignore during correction
    """
    
    # Common English words (extend as needed)
    COMMON_WORDS = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    ]
    
    def __init__(
        self,
        dictionary: Optional[Dict[str, str]] = None,
        min_word_length: int = 4,
        max_distance: int = 2,
        min_score: int = 80,
        ignore_case: bool = True,
        ignore_words: Optional[List[str]] = None,
    ):
        """Initialize the spelling corrector with specified options."""
        # Initialize custom dictionary
        self.dictionary = dictionary or {}
        
        # Create word list for fuzzy matching
        self.word_list = list(set(self.COMMON_WORDS + list(self.dictionary.values())))
        
        # Set parameters
        self.min_word_length = min_word_length
        self.max_distance = max_distance
        self.min_score = min_score
        self.ignore_case = ignore_case
        self.ignore_words = set(ignore_words or [])
        
        # Add dictionary keys to ignore words to prevent cyclic corrections
        self.ignore_words.update(self.dictionary.keys())
    
    def correct_word(self, word: str) -> str:
        """Correct a single word using fuzzy matching.
        
        Parameters
        ----------
        word : str
            Word to correct
        
        Returns
        -------
        str
            Corrected word (or original if no correction found)
        """
        # Check if word is in dictionary
        if self.ignore_case:
            lookup_word = word.lower()
            if lookup_word in {w.lower() for w in self.dictionary}:
                # Find the correct case-insensitive match
                for misspelled, correction in self.dictionary.items():
                    if misspelled.lower() == lookup_word:
                        return correction
        else:
            if word in self.dictionary:
                return self.dictionary[word]
        
        # Skip short words or words in ignore list
        if (
            len(word) < self.min_word_length
            or word in self.ignore_words
            or (self.ignore_case and word.lower() in {w.lower() for w in self.ignore_words})
        ):
            return word
        
        # Use fuzzy matching to find closest word
        matches = process.extract(
            word,
            self.word_list,
            limit=1,
        )
        
        # Check if match is good enough
        if matches and matches[0][1] >= self.min_score:
            return matches[0][0]
        
        return word
    
    def correct_text(self, text: str) -> str:
        """Correct spelling errors in a text string.
        
        Parameters
        ----------
        text : str
            Text to correct
        
        Returns
        -------
        str
            Corrected text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Split text into words and non-words
        words = re.findall(r'\b\w+\b', text)
        non_words = re.split(r'\b\w+\b', text)
        
        # Correct each word
        corrected_words = [self.correct_word(word) for word in words]
        
        # Reassemble text
        corrected_text = ""
        for i in range(len(non_words)):
            corrected_text += non_words[i]
            if i < len(corrected_words):
                corrected_text += corrected_words[i]
        
        return corrected_text
    
    def correct_texts(
        self, 
        texts: Union[List[str], pd.Series],
    ) -> Union[List[str], pd.Series]:
        """Correct spelling errors in a batch of texts.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Texts to correct
        
        Returns
        -------
        Union[List[str], pd.Series]
            Corrected texts in the same format as input
        """
        # Convert pandas Series to list if needed
        is_series = isinstance(texts, pd.Series)
        if is_series:
            text_list = texts.tolist()
        else:
            text_list = texts
        
        # Correct each text
        corrected_texts = [self.correct_text(text) for text in text_list]
        
        # Return in the same format as input
        if is_series:
            return pd.Series(corrected_texts, index=texts.index)
        else:
            return corrected_texts
    
    def add_correction(self, misspelled: str, correction: str) -> None:
        """Add a new spelling correction to the dictionary.
        
        Parameters
        ----------
        misspelled : str
            Misspelled word
        correction : str
            Corrected word
        """
        if not misspelled or not isinstance(misspelled, str) or not correction or not isinstance(correction, str):
            return
        
        self.dictionary[misspelled] = correction
        self.word_list = list(set(self.word_list + [correction]))
        self.ignore_words.add(misspelled)
    
    def add_corrections(self, corrections: Dict[str, str]) -> None:
        """Add multiple spelling corrections to the dictionary.
        
        Parameters
        ----------
        corrections : Dict[str, str]
            Dictionary mapping misspelled words to corrections
        """
        if not corrections or not isinstance(corrections, dict):
            return
        
        for misspelled, correction in corrections.items():
            self.add_correction(misspelled, correction)


def correct_spelling(
    text: str,
    dictionary: Optional[Dict[str, str]] = None,
    min_word_length: int = 4,
    max_distance: int = 2,
) -> str:
    """Function for one-off spelling correction without creating a SpellingCorrector instance.
    
    Parameters
    ----------
    text : str
        Text to correct
    dictionary : Optional[Dict[str, str]], optional
        Custom dictionary mapping misspelled words to correct ones, by default None
    min_word_length : int, optional
        Minimum length for words to be considered for correction, by default 4
    max_distance : int, optional
        Maximum Levenshtein distance for fuzzy matching, by default 2
    
    Returns
    -------
    str
        Corrected text
    """
    corrector = SpellingCorrector(
        dictionary=dictionary,
        min_word_length=min_word_length,
        max_distance=max_distance,
    )
    return corrector.correct_text(text)
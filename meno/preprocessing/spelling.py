"""Spelling correction utilities."""

from typing import Dict, List, Optional, Union, Tuple, Set
import re
import os
import json
import logging
import pandas as pd
from thefuzz import process
from thefuzz import fuzz
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class SpellingCorrector:
    """Class for spelling correction in text data.
    
    This class handles spelling correction using multiple techniques:
    - Dictionary lookup of known misspellings
    - Fuzzy matching against a dictionary of correct spellings
    - Phonetic matching for words that sound similar
    - Keyboard proximity analysis for common typos
    - Context-based correction for domain-specific terminology
    
    Parameters
    ----------
    dictionary : Optional[Dict[str, str]], optional
        Custom dictionary mapping misspelled words to correct ones, by default None
    min_word_length : int, optional
        Minimum length for words to be considered for correction, by default 3
    max_distance : int, optional
        Maximum Levenshtein distance for fuzzy matching, by default 2
    min_score : int, optional
        Minimum similarity score (0-100) to accept a correction, by default 80
    ignore_case : bool, optional
        Whether to ignore case when matching words, by default True
    ignore_words : Optional[List[str]], optional
        List of words to ignore during correction, by default None
    domain : Optional[str], optional
        Specific domain to load additional dictionaries for, by default None
        Supported domains: "medical", "legal", "technical", "financial"
    use_keyboard_proximity : bool, optional
        Whether to use keyboard proximity analysis for typos, by default True
    learn_corrections : bool, optional
        Whether to learn from corrections made during processing, by default True
    
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
    
    # Common English words dictionary (expanded)
    COMMON_WORDS = [
        # Top 200 most common English words
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
        "very", "should", "thing", "need", "much", "right", "find", "tell", "many", "may",
        "such", "where", "system", "part", "great", "since", "long", "still", "here", "life",
        "through", "before", "general", "same", "under", "house", "high", "within", "less", "world",
        "own", "might", "while", "last", "being", "both", "those", "again", "place", "during",
        "without", "small", "once", "always", "case", "does", "end", "point", "become", "local",
        "however", "state", "each", "between", "provide", "health", "information", "often", "too",
        "person", "until", "including", "social", "around", "several", "course", "company", "group",
        "different", "public", "every", "development", "nation", "must", "fact", "against", "another",
        "self", "among", "important", "early", "possible", "children", "school", "three", "follow", "member",
        "family", "government", "level", "international", "country", "area", "change", "based", "century", "community"
    ]
    
    # Common misspellings dictionary 
    COMMON_MISSPELLINGS = {
        "teh": "the",
        "recieve": "receive",
        "reciept": "receipt",
        "seperate": "separate",
        "definately": "definitely",
        "accomodate": "accommodate",
        "occured": "occurred",
        "arguement": "argument",
        "beleive": "believe",
        "concious": "conscious",
        "embarass": "embarrass",
        "enviroment": "environment",
        "gaurd": "guard",
        "existance": "existence",
        "goverment": "government",
        "independant": "independent",
        "neccesary": "necessary",
        "occassion": "occasion",
        "publically": "publicly",
        "recomend": "recommend",
        "relevent": "relevant",
        "succesful": "successful",
        "tommorrow": "tomorrow",
        "untill": "until",
        "wierd": "weird",
        "accomodation": "accommodation",
        "begining": "beginning",
        "comming": "coming",
        "comittee": "committee",
        "completly": "completely",
        "concensus": "consensus",
        "critisism": "criticism",
        "dissapear": "disappear",
        "equiptment": "equipment",
        "familar": "familiar",
        "florescent": "fluorescent",
        "foriegn": "foreign",
        "heirarchy": "hierarchy",
        "hygene": "hygiene",
        "innoculate": "inoculate",
        "jewelery": "jewelry",
        "liason": "liaison",
        "maintainance": "maintenance",
        "millenium": "millennium",
        "neccessary": "necessary",
        "noticable": "noticeable",
        "occassionally": "occasionally",
        "occurance": "occurrence",
        "persistant": "persistent",
        "preceeding": "preceding",
        "prefered": "preferred",
        "referance": "reference",
        "resistence": "resistance",
        "rythm": "rhythm",
        "sieze": "seize",
        "similer": "similar",
        "stategic": "strategic",
        "suprise": "surprise",
        "threshhold": "threshold",
        "tommorow": "tomorrow",
        "tounge": "tongue",
        "transister": "transistor",
        "useable": "usable",
        "vacume": "vacuum",
        "vegatables": "vegetables",
        "wether": "whether",
        "whereever": "wherever",
    }
    
    # Medical terminology dictionary
    MEDICAL_TERMS = {
        "antibiotics", "analgesic", "anesthesia", "artery", "benign", "biopsy", 
        "carcinoma", "cardiac", "catheter", "cerebral", "chronic", "diagnosis", 
        "diastolic", "edema", "embolism", "enzyme", "etiology", "febrile", 
        "hemorrhage", "hepatic", "hypertension", "hypotension", "infarction", 
        "inflammation", "ischemia", "lesion", "malignant", "metastasis", 
        "myocardial", "necrosis", "neurological", "oncology", "orthopedic", 
        "osteoporosis", "palliative", "pathology", "pediatric", "prognosis", 
        "pulmonary", "renal", "resuscitation", "sepsis", "serum", "stenosis", 
        "syndrome", "systolic", "thrombosis", "trauma", "triage", "tumor", 
        "vascular", "ventricle"
    }
    
    # Medical misspellings
    MEDICAL_MISSPELLINGS = {
        "diabetis": "diabetes",
        "artheritis": "arthritis",
        "colestrol": "cholesterol",
        "hemmorrhage": "hemorrhage",
        "esophagous": "esophagus",
        "throid": "thyroid",
        "alzheimers": "alzheimer's",
        "parkinsons": "parkinson's",
        "lukemia": "leukemia",
        "amyotropic": "amyotrophic",
        "anaphalactic": "anaphylactic",
        "asma": "asthma",
        "cathetar": "catheter",
        "collitis": "colitis",
        "diverticulitus": "diverticulitis",
        "emnesia": "amnesia",
        "fibromalgia": "fibromyalgia",
        "hemmoroids": "hemorrhoids",
        "hepititis": "hepatitis",
        "hypertention": "hypertension",
        "hipocondria": "hypochondria",
        "inflamation": "inflammation",
        "menangitis": "meningitis",
        "ostemyelitis": "osteomyelitis",
        "osteopirosis": "osteoporosis",
        "pnemonia": "pneumonia",
        "psoriassis": "psoriasis",
        "rehumatoid": "rheumatoid",
        "siatic": "sciatic",
        "siatica": "sciatica",
        "sintoma": "symptom",
        "toncilitis": "tonsillitis",
    }
    
    # Technical terminology dictionary
    TECHNICAL_TERMS = {
        "algorithm", "bandwidth", "binary", "blockchain", "buffer", "byte", 
        "cache", "compiler", "compression", "cryptography", "database", "debugging", 
        "encryption", "framework", "function", "hardware", "indexing", "interface", 
        "kernel", "latency", "middleware", "networking", "optimization", "parameter", 
        "protocol", "quantum", "recursion", "repository", "scaling", "schema", 
        "serialization", "software", "syntax", "throughput", "tokenization", 
        "validation", "virtualization", "webhook", "wireframe"
    }
    
    # Technical misspellings
    TECHNICAL_MISSPELLINGS = {
        "algortihm": "algorithm",
        "bandwith": "bandwidth",
        "cryptograpy": "cryptography",
        "databse": "database",
        "incryption": "encryption",
        "kernal": "kernel",
        "middlewear": "middleware",
        "optimisation": "optimization",
        "protocal": "protocol",
        "syntex": "syntax",
        "througput": "throughput",
        "virtualization": "virtualization",
        "wirframe": "wireframe",
        "authentification": "authentication",
        "compatability": "compatibility",
        "paralel": "parallel",
        "retreive": "retrieve",
        "syncronize": "synchronize",
        "tecnology": "technology",
    }
    
    # Keyboard distance map for common typos
    KEYBOARD_ADJACENCY = {
        'a': 'qwsz',
        'b': 'vghn',
        'c': 'xdfv',
        'd': 'erfcxs',
        'e': 'rdsw',
        'f': 'rtgvcd',
        'g': 'tyhbvf',
        'h': 'yujnbg',
        'i': 'uojk',
        'j': 'uikmnh',
        'k': 'iolmj',
        'l': 'opk',
        'm': 'njk',
        'n': 'bhjm',
        'o': 'ipkl',
        'p': 'ol',
        'q': 'wa',
        'r': 'edft',
        's': 'wedxza',
        't': 'rfgy',
        'u': 'yhji',
        'v': 'cfgb',
        'w': 'qase',
        'x': 'zsdc',
        'y': 'tghu',
        'z': 'asx',
    }
    
    def __init__(
        self,
        dictionary: Optional[Dict[str, str]] = None,
        min_word_length: int = 3,
        max_distance: int = 2,
        min_score: int = 80,
        ignore_case: bool = True,
        ignore_words: Optional[List[str]] = None,
        domain: Optional[str] = None,
        use_keyboard_proximity: bool = True,
        learn_corrections: bool = True,
    ):
        """Initialize the spelling corrector with specified options."""
        # Start with common misspellings dictionary
        self.dictionary = self.COMMON_MISSPELLINGS.copy()
        
        # Add domain-specific dictionaries if requested
        if domain:
            domain_dict = self._get_domain_dictionary(domain)
            if domain_dict:
                self.dictionary.update(domain_dict)
        
        # Add custom dictionary
        if dictionary:
            self.dictionary.update(dictionary)
        
        # Create correct word list for fuzzy matching
        self.word_list = list(set(self.COMMON_WORDS + list(self.dictionary.values())))
        
        # Set parameters
        self.min_word_length = min_word_length
        self.max_distance = max_distance
        self.min_score = min_score
        self.ignore_case = ignore_case
        self.ignore_words = set(ignore_words or [])
        self.use_keyboard_proximity = use_keyboard_proximity
        self.learn_corrections = learn_corrections
        
        # Add dictionary keys to ignore words to prevent cyclic corrections
        self.ignore_words.update(self.dictionary.keys())
        
        # Track learned corrections
        self.learned_corrections = {}
        
        # Add domain-specific terminology to the word list
        if domain:
            self._add_domain_terms(domain)
            
    def _get_domain_dictionary(self, domain: str) -> Dict[str, str]:
        """Get domain-specific misspelling dictionary.
        
        Parameters
        ----------
        domain : str
            Domain to get dictionary for
            
        Returns
        -------
        Dict[str, str]
            Dictionary of domain-specific misspellings
        """
        domain = domain.lower()
        if domain == "medical" or domain == "healthcare":
            return self.MEDICAL_MISSPELLINGS
        elif domain == "technical" or domain == "tech":
            return self.TECHNICAL_MISSPELLINGS
        else:
            logger.warning(f"Unknown domain: {domain}. Using common misspellings only.")
            return {}
            
    def _add_domain_terms(self, domain: str) -> None:
        """Add domain-specific terminology to the word list.
        
        Parameters
        ----------
        domain : str
            Domain to add terminology for
        """
        domain = domain.lower()
        if domain == "medical" or domain == "healthcare":
            self.word_list.extend(self.MEDICAL_TERMS)
        elif domain == "technical" or domain == "tech":
            self.word_list.extend(self.TECHNICAL_TERMS)
            
    def _is_keyboard_typo(self, word: str, candidate: str) -> bool:
        """Check if a word is likely a keyboard typo of a candidate.
        
        Parameters
        ----------
        word : str
            Misspelled word
        candidate : str
            Candidate correction
            
        Returns
        -------
        bool
            True if the word is likely a keyboard typo
        """
        if not self.use_keyboard_proximity:
            return False
            
        # Only check if the lengths are similar (typos rarely change word length dramatically)
        if abs(len(word) - len(candidate)) > 1:
            return False
            
        # Count character differences
        diff_count = 0
        max_diff = 2  # Maximum number of typos to consider
        
        # For each character in the word, check if it's in the keyboard proximity of the candidate
        for i in range(min(len(word), len(candidate))):
            if i >= len(word) or i >= len(candidate):
                diff_count += 1
            elif word[i] != candidate[i]:
                # Check if the characters are adjacent on the keyboard
                if word[i].lower() in self.KEYBOARD_ADJACENCY and candidate[i].lower() in self.KEYBOARD_ADJACENCY.get(word[i].lower(), ''):
                    diff_count += 0.5  # Lower penalty for adjacent keys
                else:
                    diff_count += 1
                    
            if diff_count > max_diff:
                return False
                
        return diff_count <= max_diff
    
    def correct_word(self, word: str) -> str:
        """Correct a single word using multiple correction techniques.
        
        Parameters
        ----------
        word : str
            Word to correct
        
        Returns
        -------
        str
            Corrected word (or original if no correction found)
        """
        # Skip short words or words in ignore list
        if (
            len(word) < self.min_word_length
            or word in self.ignore_words
            or (self.ignore_case and word.lower() in {w.lower() for w in self.ignore_words})
        ):
            return word
        
        # Check if we've learned this correction before
        if self.learn_corrections and word in self.learned_corrections:
            return self.learned_corrections[word]
        
        # Check if word is in dictionary (direct lookup)
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
        
        # Try keyboard proximity analysis
        if self.use_keyboard_proximity:
            for candidate in self.word_list:
                if self._is_keyboard_typo(word, candidate):
                    # If it's a keyboard typo, add to learned corrections if enabled
                    if self.learn_corrections:
                        self.learned_corrections[word] = candidate
                    return candidate
        
        # Use fuzzy matching with thefuzz library
        # Try both the standard and token_sort_ratio for different matching strategies
        standard_matches = process.extract(
            word,
            self.word_list,
            limit=3,
            scorer=fuzz.ratio
        )
        
        token_matches = process.extract(
            word,
            self.word_list,
            limit=3,
            scorer=fuzz.token_sort_ratio
        )
        
        # Combine and sort by score
        all_matches = standard_matches + token_matches
        unique_matches = {}
        for match, score in all_matches:
            if match not in unique_matches or score > unique_matches[match]:
                unique_matches[match] = score
                
        # Sort by score
        sorted_matches = sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)
        
        # Check if best match is good enough
        if sorted_matches and sorted_matches[0][1] >= self.min_score:
            best_match = sorted_matches[0][0]
            
            # Add to learned corrections if enabled
            if self.learn_corrections:
                self.learned_corrections[word] = best_match
                
            return best_match
        
        # If no correction found, return original word
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
            
    def export_learned_corrections(self, filepath: Optional[str] = None) -> Dict[str, str]:
        """Export learned corrections to a file or return as a dictionary.
        
        Parameters
        ----------
        filepath : Optional[str], optional
            Path to save corrections to, by default None
            
        Returns
        -------
        Dict[str, str]
            Dictionary of learned corrections
        """
        if not self.learned_corrections:
            return {}
            
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.learned_corrections, f, indent=2)
                logger.info(f"Exported {len(self.learned_corrections)} learned corrections to {filepath}")
            except Exception as e:
                logger.error(f"Failed to export learned corrections: {e}")
                
        return self.learned_corrections
        
    def import_learned_corrections(self, filepath: str) -> bool:
        """Import learned corrections from a file.
        
        Parameters
        ----------
        filepath : str
            Path to file containing corrections
            
        Returns
        -------
        bool
            True if import was successful
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
            
        try:
            with open(filepath, 'r') as f:
                corrections = json.load(f)
                
            if not isinstance(corrections, dict):
                logger.error(f"Invalid corrections format in {filepath}")
                return False
                
            self.learned_corrections.update(corrections)
            logger.info(f"Imported {len(corrections)} learned corrections from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to import learned corrections: {e}")
            return False


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
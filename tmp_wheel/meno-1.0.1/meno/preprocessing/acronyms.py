"""Acronym expansion and handling utilities."""

from typing import Dict, List, Optional, Union, Tuple, Set
import re
import pandas as pd
from thefuzz import process


class AcronymExpander:
    """Class for expanding acronyms in text data.
    
    This class handles identification and expansion of acronyms based on:
    - Dictionary lookup of common acronyms
    - Custom user-provided mappings
    - Contextual expansion for domain-specific acronyms
    
    Parameters
    ----------
    custom_mappings : Dict[str, str], optional
        Custom acronym to expansion mappings, by default None
    min_length : int, optional
        Minimum length for an acronym to be considered, by default 2
    contextual_expansion : bool, optional
        Whether to attempt contextual expansion for unknown acronyms, by default True
    
    Attributes
    ----------
    acronym_dict : Dict[str, str]
        Dictionary of acronyms and their expansions
    min_length : int
        Minimum length for an acronym to be considered
    contextual_expansion : bool
        Whether to attempt contextual expansion for unknown acronyms
    """
    
    # Common general acronyms
    DEFAULT_ACRONYMS = {
        "CEO": "Chief Executive Officer",
        "CFO": "Chief Financial Officer",
        "CTO": "Chief Technology Officer",
        "COO": "Chief Operating Officer",
        "HR": "Human Resources",
        "IT": "Information Technology",
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning",
        "NLP": "Natural Language Processing",
        "API": "Application Programming Interface",
        "FYI": "For Your Information",
        "ASAP": "As Soon As Possible",
        "FAQ": "Frequently Asked Questions",
        "ROI": "Return On Investment",
        "KPI": "Key Performance Indicator",
        "Q1": "First Quarter",
        "Q2": "Second Quarter",
        "Q3": "Third Quarter",
        "Q4": "Fourth Quarter",
    }
    
    # Insurance-specific acronyms
    INSURANCE_ACRONYMS = {
        "P&C": "Property and Casualty",
        "UW": "Underwriting",
        "NB": "New Business",
        "CM": "Claims Management",
        "DOL": "Date of Loss",
        "POL": "Policy",
        "PH": "Policyholder",
        "LTV": "Lifetime Value",
        "BI": "Bodily Injury",
        "PD": "Property Damage",
        "PIP": "Personal Injury Protection",
        "NCD": "No Claims Discount",
        "LOB": "Line of Business",
        "MOB": "Month of Business",
        "YOB": "Year of Business",
        "EOL": "End of Life",
    }
    
    def __init__(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
        min_length: int = 2,
        contextual_expansion: bool = True,
    ):
        """Initialize the acronym expander with specified options."""
        # Combine default acronyms with insurance-specific ones
        self.acronym_dict = {**self.DEFAULT_ACRONYMS, **self.INSURANCE_ACRONYMS}
        
        # Add custom mappings
        if custom_mappings:
            self.acronym_dict.update(custom_mappings)
        
        self.min_length = min_length
        self.contextual_expansion = contextual_expansion
        
        # Compile regex for acronym detection
        # Matches uppercase words of min_length or more characters
        self.acronym_pattern = re.compile(
            r'\b[A-Z][A-Z0-9&]{' + str(min_length - 1) + r',}\b'
        )
    
    def expand_acronyms(self, text: str) -> str:
        """Expand acronyms in a text string.
        
        Parameters
        ----------
        text : str
            Text containing acronyms to expand
        
        Returns
        -------
        str
            Text with expanded acronyms
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Find all acronyms in the text
        acronyms = self.acronym_pattern.findall(text)
        
        # Replace each acronym with its expansion if available
        for acronym in acronyms:
            if acronym in self.acronym_dict:
                # Replace with format: "ACRONYM (Expansion)"
                expansion = self.acronym_dict[acronym]
                text = text.replace(
                    acronym, 
                    f"{acronym} ({expansion})"
                )
        
        return text
    
    def expand_acronyms_batch(
        self, 
        texts: Union[List[str], pd.Series],
    ) -> Union[List[str], pd.Series]:
        """Expand acronyms in a batch of texts.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Texts containing acronyms to expand
        
        Returns
        -------
        Union[List[str], pd.Series]
            Texts with expanded acronyms in the same format as input
        """
        # Convert pandas Series to list if needed
        is_series = isinstance(texts, pd.Series)
        if is_series:
            text_list = texts.tolist()
        else:
            text_list = texts
        
        # Process each text
        expanded_texts = [self.expand_acronyms(text) for text in text_list]
        
        # Return in the same format as input
        if is_series:
            return pd.Series(expanded_texts, index=texts.index)
        else:
            return expanded_texts
    
    def extract_acronyms(self, text: str) -> List[str]:
        """Extract acronyms from text without expanding them.
        
        Parameters
        ----------
        text : str
            Text to extract acronyms from
        
        Returns
        -------
        List[str]
            List of acronyms found in the text
        """
        if not text or not isinstance(text, str):
            return []
        
        return self.acronym_pattern.findall(text)
    
    def extract_acronyms_batch(
        self, 
        texts: Union[List[str], pd.Series],
    ) -> Dict[str, int]:
        """Extract and count acronyms from a batch of texts.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Texts to extract acronyms from
        
        Returns
        -------
        Dict[str, int]
            Dictionary of acronyms and their frequencies
        """
        # Convert pandas Series to list if needed
        if isinstance(texts, pd.Series):
            text_list = texts.tolist()
        else:
            text_list = texts
        
        # Extract acronyms from all texts
        all_acronyms = []
        for text in text_list:
            all_acronyms.extend(self.extract_acronyms(text))
        
        # Count frequencies
        acronym_counts = {}
        for acronym in all_acronyms:
            if acronym in acronym_counts:
                acronym_counts[acronym] += 1
            else:
                acronym_counts[acronym] = 1
        
        return acronym_counts
    
    def suggest_expansions(
        self, 
        acronym: str, 
        min_score: int = 70,
    ) -> List[Tuple[str, int]]:
        """Suggest possible expansions for an unknown acronym.
        
        Parameters
        ----------
        acronym : str
            Acronym to find expansions for
        min_score : int, optional
            Minimum similarity score (0-100) for suggestions, by default 70
        
        Returns
        -------
        List[Tuple[str, int]]
            List of (expansion, score) tuples for the given acronym
        """
        if not acronym or not isinstance(acronym, str):
            return []
        
        # Check if acronym is already known
        if acronym in self.acronym_dict:
            return [(self.acronym_dict[acronym], 100)]
        
        # Use fuzzy matching to find similar known acronyms
        suggestions = process.extract(
            acronym, 
            self.acronym_dict.keys(),
            limit=5
        )
        
        # Filter by minimum score
        filtered_suggestions = [
            (self.acronym_dict[suggestion], score)
            for suggestion, score in suggestions
            if score >= min_score
        ]
        
        return filtered_suggestions
    
    def add_acronym(self, acronym: str, expansion: str) -> None:
        """Add a new acronym to the dictionary.
        
        Parameters
        ----------
        acronym : str
            Acronym to add
        expansion : str
            Expansion/meaning of the acronym
        """
        if not acronym or not isinstance(acronym, str) or not expansion or not isinstance(expansion, str):
            return
        
        self.acronym_dict[acronym] = expansion
    
    def add_acronyms(self, acronyms: Dict[str, str]) -> None:
        """Add multiple acronyms to the dictionary.
        
        Parameters
        ----------
        acronyms : Dict[str, str]
            Dictionary of acronyms and their expansions
        """
        if not acronyms or not isinstance(acronyms, dict):
            return
        
        self.acronym_dict.update(acronyms)


def expand_acronyms(
    text: str,
    custom_mappings: Optional[Dict[str, str]] = None,
) -> str:
    """Function for one-off acronym expansion without creating an AcronymExpander instance.
    
    Parameters
    ----------
    text : str
        Text containing acronyms to expand
    custom_mappings : Optional[Dict[str, str]], optional
        Custom acronym mappings, by default None
    
    Returns
    -------
    str
        Text with expanded acronyms
    """
    expander = AcronymExpander(custom_mappings=custom_mappings)
    return expander.expand_acronyms(text)
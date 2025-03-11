"""Domain-specific adapters for NLP processing."""

from typing import Dict, List, Optional, Union, Any, Set
import os
import json
import logging
import pandas as pd
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class DomainAdapter:
    """Base class for domain-specific adapters.
    
    This class serves as a foundation for domain-specific adapters
    that provide specialized preprocessing, terminology, and configuration
    for different industries and domains.
    
    Parameters
    ----------
    name : str
        Name of the domain
    terminology_path : Optional[str], optional
        Path to terminology dictionary, by default None
    acronyms_path : Optional[str], optional
        Path to acronyms dictionary, by default None
    misspellings_path : Optional[str], optional
        Path to misspellings dictionary, by default None
    
    Attributes
    ----------
    name : str
        Domain name
    terminology : Dict[str, Any]
        Domain-specific terminology
    acronyms : Dict[str, str]
        Domain-specific acronyms
    misspellings : Dict[str, str]
        Domain-specific misspellings
    """
    
    def __init__(
        self,
        name: str,
        terminology_path: Optional[str] = None,
        acronyms_path: Optional[str] = None,
        misspellings_path: Optional[str] = None,
    ):
        """Initialize the domain adapter."""
        self.name = name
        self.terminology = {}
        self.acronyms = {}
        self.misspellings = {}
        self.stopwords = set()
        
        # Load dictionaries if provided
        if terminology_path and os.path.exists(terminology_path):
            self.load_terminology(terminology_path)
            
        if acronyms_path and os.path.exists(acronyms_path):
            self.load_acronyms(acronyms_path)
            
        if misspellings_path and os.path.exists(misspellings_path):
            self.load_misspellings(misspellings_path)
    
    def load_terminology(self, path: str) -> bool:
        """Load domain-specific terminology from a file.
        
        Parameters
        ----------
        path : str
            Path to terminology file (JSON)
            
        Returns
        -------
        bool
            True if loading was successful
        """
        try:
            with open(path, 'r') as f:
                self.terminology = json.load(f)
            logger.info(f"Loaded {len(self.terminology)} terms for {self.name} domain")
            return True
        except Exception as e:
            logger.error(f"Failed to load terminology from {path}: {e}")
            return False
    
    def load_acronyms(self, path: str) -> bool:
        """Load domain-specific acronyms from a file.
        
        Parameters
        ----------
        path : str
            Path to acronyms file (JSON)
            
        Returns
        -------
        bool
            True if loading was successful
        """
        try:
            with open(path, 'r') as f:
                self.acronyms = json.load(f)
            logger.info(f"Loaded {len(self.acronyms)} acronyms for {self.name} domain")
            return True
        except Exception as e:
            logger.error(f"Failed to load acronyms from {path}: {e}")
            return False
    
    def load_misspellings(self, path: str) -> bool:
        """Load domain-specific misspellings from a file.
        
        Parameters
        ----------
        path : str
            Path to misspellings file (JSON)
            
        Returns
        -------
        bool
            True if loading was successful
        """
        try:
            with open(path, 'r') as f:
                self.misspellings = json.load(f)
            logger.info(f"Loaded {len(self.misspellings)} misspellings for {self.name} domain")
            return True
        except Exception as e:
            logger.error(f"Failed to load misspellings from {path}: {e}")
            return False
    
    def save(self, directory: str) -> bool:
        """Save domain adapter configuration to files.
        
        Parameters
        ----------
        directory : str
            Directory to save files to
            
        Returns
        -------
        bool
            True if saving was successful
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save terminology
            if self.terminology:
                term_path = os.path.join(directory, f"{self.name}_terminology.json")
                with open(term_path, 'w') as f:
                    json.dump(self.terminology, f, indent=2)
            
            # Save acronyms
            if self.acronyms:
                acr_path = os.path.join(directory, f"{self.name}_acronyms.json")
                with open(acr_path, 'w') as f:
                    json.dump(self.acronyms, f, indent=2)
            
            # Save misspellings
            if self.misspellings:
                mis_path = os.path.join(directory, f"{self.name}_misspellings.json")
                with open(mis_path, 'w') as f:
                    json.dump(self.misspellings, f, indent=2)
                    
            logger.info(f"Saved {self.name} domain adapter to {directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to save domain adapter: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Apply domain-specific preprocessing to text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        str
            Preprocessed text
        """
        # Default implementation - subclasses can override
        return text
    
    def get_acronyms(self) -> Dict[str, str]:
        """Get domain-specific acronyms.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of acronyms
        """
        return self.acronyms
    
    def get_misspellings(self) -> Dict[str, str]:
        """Get domain-specific misspellings.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of misspellings
        """
        return self.misspellings
    
    def get_terminology(self) -> Dict[str, Any]:
        """Get domain-specific terminology.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of terminology
        """
        return self.terminology
    
    def get_stopwords(self) -> Set[str]:
        """Get domain-specific stopwords.
        
        Returns
        -------
        Set[str]
            Set of stopwords
        """
        return self.stopwords
    

class MedicalAdapter(DomainAdapter):
    """Domain adapter for medical and healthcare text.
    
    Parameters
    ----------
    terminology_path : Optional[str], optional
        Path to medical terminology dictionary, by default None
    acronyms_path : Optional[str], optional
        Path to medical acronyms dictionary, by default None
    misspellings_path : Optional[str], optional
        Path to medical misspellings dictionary, by default None
    """
    
    # Common medical stopwords that shouldn't be removed in medical contexts
    MEDICAL_PRESERVE_WORDS = {
        "under", "patient", "blood", "positive", "negative", "test", 
        "acute", "chronic", "procedure", "normal", "review", "history"
    }
    
    def __init__(
        self,
        terminology_path: Optional[str] = None,
        acronyms_path: Optional[str] = None,
        misspellings_path: Optional[str] = None,
    ):
        """Initialize the medical domain adapter."""
        super().__init__(
            name="medical",
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
        
        # Add default medical stopwords to preserve
        self.preserve_words = self.MEDICAL_PRESERVE_WORDS
    
    def preprocess_text(self, text: str) -> str:
        """Apply medical-specific preprocessing to text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        str
            Preprocessed text
        """
        # Medical-specific preprocessing could be added here
        return text


class FinancialAdapter(DomainAdapter):
    """Domain adapter for financial and banking text.
    
    Parameters
    ----------
    terminology_path : Optional[str], optional
        Path to financial terminology dictionary, by default None
    acronyms_path : Optional[str], optional
        Path to financial acronyms dictionary, by default None
    misspellings_path : Optional[str], optional
        Path to financial misspellings dictionary, by default None
    """
    
    # Financial terminology-specific stopwords to preserve
    FINANCIAL_PRESERVE_WORDS = {
        "income", "asset", "rate", "return", "market", "fund", "stock",
        "bond", "equity", "credit", "debit", "risk", "account"
    }
    
    def __init__(
        self,
        terminology_path: Optional[str] = None,
        acronyms_path: Optional[str] = None,
        misspellings_path: Optional[str] = None,
    ):
        """Initialize the financial domain adapter."""
        super().__init__(
            name="financial",
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
        
        # Add default financial stopwords to preserve
        self.preserve_words = self.FINANCIAL_PRESERVE_WORDS
    
    def preprocess_text(self, text: str) -> str:
        """Apply financial-specific preprocessing to text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        str
            Preprocessed text
        """
        # Financial-specific preprocessing could be added here
        return text


class TechnicalAdapter(DomainAdapter):
    """Domain adapter for technical and IT text.
    
    Parameters
    ----------
    terminology_path : Optional[str], optional
        Path to technical terminology dictionary, by default None
    acronyms_path : Optional[str], optional
        Path to technical acronyms dictionary, by default None
    misspellings_path : Optional[str], optional
        Path to technical misspellings dictionary, by default None
    """
    
    # Technical terminology-specific stopwords to preserve
    TECHNICAL_PRESERVE_WORDS = {
        "system", "data", "file", "code", "server", "client",
        "network", "api", "interface", "function", "object"
    }
    
    def __init__(
        self,
        terminology_path: Optional[str] = None,
        acronyms_path: Optional[str] = None,
        misspellings_path: Optional[str] = None,
    ):
        """Initialize the technical domain adapter."""
        super().__init__(
            name="technical",
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
        
        # Add default technical stopwords to preserve
        self.preserve_words = self.TECHNICAL_PRESERVE_WORDS
    
    def preprocess_text(self, text: str) -> str:
        """Apply technical-specific preprocessing to text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        str
            Preprocessed text
        """
        # Technical-specific preprocessing could be added here
        return text


class LegalAdapter(DomainAdapter):
    """Domain adapter for legal and compliance text.
    
    Parameters
    ----------
    terminology_path : Optional[str], optional
        Path to legal terminology dictionary, by default None
    acronyms_path : Optional[str], optional
        Path to legal acronyms dictionary, by default None
    misspellings_path : Optional[str], optional
        Path to legal misspellings dictionary, by default None
    """
    
    # Legal terminology-specific stopwords to preserve
    LEGAL_PRESERVE_WORDS = {
        "law", "court", "judge", "legal", "plaintiff", "defendant",
        "action", "case", "claim", "contract", "party", "rights"
    }
    
    def __init__(
        self,
        terminology_path: Optional[str] = None,
        acronyms_path: Optional[str] = None,
        misspellings_path: Optional[str] = None,
    ):
        """Initialize the legal domain adapter."""
        super().__init__(
            name="legal",
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
        
        # Add default legal stopwords to preserve
        self.preserve_words = self.LEGAL_PRESERVE_WORDS
    
    def preprocess_text(self, text: str) -> str:
        """Apply legal-specific preprocessing to text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        str
            Preprocessed text
        """
        # Legal-specific preprocessing could be added here
        return text


def get_domain_adapter(
    domain: str,
    terminology_path: Optional[str] = None,
    acronyms_path: Optional[str] = None,
    misspellings_path: Optional[str] = None,
) -> DomainAdapter:
    """Get a domain-specific adapter.
    
    Parameters
    ----------
    domain : str
        Domain name (medical, financial, technical, legal)
    terminology_path : Optional[str], optional
        Path to terminology dictionary, by default None
    acronyms_path : Optional[str], optional
        Path to acronyms dictionary, by default None
    misspellings_path : Optional[str], optional
        Path to misspellings dictionary, by default None
        
    Returns
    -------
    DomainAdapter
        Domain-specific adapter
    """
    domain = domain.lower()
    
    if domain == "medical" or domain == "healthcare":
        return MedicalAdapter(
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
    elif domain == "financial" or domain == "finance" or domain == "banking":
        return FinancialAdapter(
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
    elif domain == "technical" or domain == "tech" or domain == "it":
        return TechnicalAdapter(
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
    elif domain == "legal" or domain == "law" or domain == "compliance":
        return LegalAdapter(
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
    else:
        logger.warning(f"Unknown domain: {domain}. Using generic adapter.")
        return DomainAdapter(
            name=domain,
            terminology_path=terminology_path,
            acronyms_path=acronyms_path,
            misspellings_path=misspellings_path,
        )
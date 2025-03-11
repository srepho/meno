"""Acronym expansion and handling utilities."""

from typing import Dict, List, Optional, Union, Tuple, Set
import re
import pandas as pd
from thefuzz import process
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AcronymExpander:
    """Class for expanding acronyms in text data.
    
    This class handles identification and expansion of acronyms based on:
    - Dictionary lookup of common acronyms
    - Custom user-provided mappings
    - Contextual expansion for domain-specific acronyms
    - Support for various acronym formats (uppercase, lowercase, dotted)
    
    Parameters
    ----------
    custom_mappings : Dict[str, str], optional
        Custom acronym to expansion mappings, by default None
    min_length : int, optional
        Minimum length for an acronym to be considered, by default 2
    contextual_expansion : bool, optional
        Whether to attempt contextual expansion for unknown acronyms, by default True
    domain : str, optional
        Specific domain to load additional acronyms for, by default None
        Supported domains: "healthcare", "finance", "tech", "legal"
    ignore_case : bool, optional
        Whether to ignore case when matching acronyms, by default False
    
    Attributes
    ----------
    acronym_dict : Dict[str, str]
        Dictionary of acronyms and their expansions
    min_length : int
        Minimum length for an acronym to be considered
    contextual_expansion : bool
        Whether to attempt contextual expansion for unknown acronyms
    ignore_case : bool
        Whether to ignore case when matching acronyms
    """
    
    # Common general acronyms
    DEFAULT_ACRONYMS = {
        "CEO": "Chief Executive Officer",
        "CFO": "Chief Financial Officer",
        "CTO": "Chief Technology Officer",
        "COO": "Chief Operating Officer",
        "CIO": "Chief Information Officer",
        "CMO": "Chief Marketing Officer",
        "CHRO": "Chief Human Resources Officer",
        "CPO": "Chief Product Officer",
        "CSO": "Chief Security Officer",
        "CRO": "Chief Revenue Officer",
        "CDO": "Chief Data Officer",
        "CXO": "Chief Experience Officer",
        "HR": "Human Resources",
        "IT": "Information Technology",
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning",
        "NLP": "Natural Language Processing",
        "API": "Application Programming Interface",
        "UI": "User Interface",
        "UX": "User Experience",
        "GUI": "Graphical User Interface",
        "CLI": "Command Line Interface",
        "FYI": "For Your Information",
        "ASAP": "As Soon As Possible",
        "FAQ": "Frequently Asked Questions",
        "ROI": "Return On Investment",
        "KPI": "Key Performance Indicator",
        "SMART": "Specific, Measurable, Achievable, Relevant, Time-bound",
        "SWOT": "Strengths, Weaknesses, Opportunities, Threats",
        "B2B": "Business to Business",
        "B2C": "Business to Consumer",
        "C2C": "Consumer to Consumer",
        "SaaS": "Software as a Service",
        "PaaS": "Platform as a Service",
        "IaaS": "Infrastructure as a Service",
        "QA": "Quality Assurance",
        "QC": "Quality Control",
        "R&D": "Research and Development",
        "M&A": "Mergers and Acquisitions",
        "Q1": "First Quarter",
        "Q2": "Second Quarter",
        "Q3": "Third Quarter",
        "Q4": "Fourth Quarter",
        "FY": "Fiscal Year",
        "YTD": "Year to Date",
        "MoM": "Month over Month",
        "YoY": "Year over Year",
        "EOD": "End of Day",
        "EOW": "End of Week",
        "EOM": "End of Month",
        "EOY": "End of Year",
        "ETA": "Estimated Time of Arrival",
        "TBD": "To Be Determined",
        "TBA": "To Be Announced",
        "IMO": "In My Opinion",
        "IMHO": "In My Humble Opinion",
        "AKA": "Also Known As",
        "DIY": "Do It Yourself",
        "RSVP": "Répondez S'il Vous Plaît (Please Respond)",
        "BYOD": "Bring Your Own Device",
        "WFH": "Work From Home",
        "OOO": "Out Of Office",
        "RTO": "Return to Office",
        "OKR": "Objectives and Key Results",
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
        "ACV": "Actual Cash Value",
        "RCV": "Replacement Cost Value",
        "FMV": "Fair Market Value",
        "EOB": "Explanation of Benefits",
        "CPT": "Current Procedural Terminology",
        "ICD": "International Classification of Diseases",
        "PPO": "Preferred Provider Organization",
        "HMO": "Health Maintenance Organization",
        "HDHP": "High Deductible Health Plan",
        "HSA": "Health Savings Account",
        "FSA": "Flexible Spending Account",
        "COBRA": "Consolidated Omnibus Budget Reconciliation Act",
        "HIPAA": "Health Insurance Portability and Accountability Act",
        "ACA": "Affordable Care Act",
        "GL": "General Liability",
        "WC": "Workers Compensation",
        "AD&D": "Accidental Death and Dismemberment",
        "LTD": "Long Term Disability",
        "STD": "Short Term Disability",
        "APR": "Annual Percentage Rate",
        "NCB": "No Claim Bonus",
        "GWP": "Gross Written Premium",
        "NWP": "Net Written Premium",
        "CR": "Combined Ratio",
        "LR": "Loss Ratio",
        "ER": "Expense Ratio",
        "MTA": "Mid-Term Adjustment",
        "MER": "Medical Expense Ratio",
        "IBNR": "Incurred But Not Reported",
    }
    
    # Healthcare-specific acronyms
    HEALTHCARE_ACRONYMS = {
        "EHR": "Electronic Health Record",
        "EMR": "Electronic Medical Record",
        "PHI": "Protected Health Information",
        "HIPAA": "Health Insurance Portability and Accountability Act",
        "CDC": "Centers for Disease Control and Prevention",
        "WHO": "World Health Organization",
        "FDA": "Food and Drug Administration",
        "ICU": "Intensive Care Unit",
        "CCU": "Coronary Care Unit",
        "ED": "Emergency Department",
        "BP": "Blood Pressure",
        "HR": "Heart Rate",
        "BMI": "Body Mass Index",
        "CPR": "Cardiopulmonary Resuscitation",
        "DNR": "Do Not Resuscitate",
        "DOB": "Date of Birth",
        "DOA": "Dead on Arrival",
        "MVA": "Motor Vehicle Accident",
        "OTC": "Over the Counter",
        "Rx": "Prescription",
        "NPO": "Nothing by Mouth",
        "PRN": "As Needed",
        "PCP": "Primary Care Physician",
        "PT": "Physical Therapy",
        "OT": "Occupational Therapy",
        "RT": "Respiratory Therapy",
        "SNF": "Skilled Nursing Facility",
        "LTC": "Long Term Care",
        "ADL": "Activities of Daily Living",
        "CABG": "Coronary Artery Bypass Graft",
        "CHF": "Congestive Heart Failure",
        "COPD": "Chronic Obstructive Pulmonary Disease",
        "DVT": "Deep Vein Thrombosis",
        "GERD": "Gastroesophageal Reflux Disease",
        "HTN": "Hypertension",
        "MI": "Myocardial Infarction",
        "RA": "Rheumatoid Arthritis",
        "UTI": "Urinary Tract Infection",
    }
    
    # Finance-specific acronyms
    FINANCE_ACRONYMS = {
        "APR": "Annual Percentage Rate",
        "APY": "Annual Percentage Yield",
        "ATM": "Automated Teller Machine",
        "CD": "Certificate of Deposit",
        "CFPB": "Consumer Financial Protection Bureau",
        "CMA": "Comparative Market Analysis",
        "DTI": "Debt to Income Ratio",
        "EBIT": "Earnings Before Interest and Taxes",
        "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
        "EPS": "Earnings Per Share",
        "ETF": "Exchange-Traded Fund",
        "FDIC": "Federal Deposit Insurance Corporation",
        "FICO": "Fair Isaac Corporation",
        "GDP": "Gross Domestic Product",
        "HOA": "Homeowners Association",
        "IRA": "Individual Retirement Account",
        "LTV": "Loan to Value Ratio",
        "MBS": "Mortgage-Backed Securities",
        "NYSE": "New York Stock Exchange",
        "P/E": "Price to Earnings Ratio",
        "PMI": "Private Mortgage Insurance",
        "REIT": "Real Estate Investment Trust",
        "ROA": "Return on Assets",
        "ROE": "Return on Equity",
        "ROI": "Return on Investment",
        "SEC": "Securities and Exchange Commission",
        "YTM": "Yield to Maturity",
    }
    
    # Technology-specific acronyms
    TECH_ACRONYMS = {
        "AI": "Artificial Intelligence",
        "API": "Application Programming Interface",
        "AWS": "Amazon Web Services",
        "BIOS": "Basic Input/Output System",
        "CAPTCHA": "Completely Automated Public Turing test to tell Computers and Humans Apart",
        "CDN": "Content Delivery Network",
        "CRUD": "Create, Read, Update, Delete",
        "CSS": "Cascading Style Sheets",
        "DNS": "Domain Name System",
        "DRY": "Don't Repeat Yourself",
        "GCP": "Google Cloud Platform",
        "GUI": "Graphical User Interface",
        "HTML": "Hypertext Markup Language",
        "HTTP": "Hypertext Transfer Protocol",
        "HTTPS": "Hypertext Transfer Protocol Secure",
        "IDE": "Integrated Development Environment",
        "IoT": "Internet of Things",
        "IP": "Internet Protocol",
        "JSON": "JavaScript Object Notation",
        "LLM": "Large Language Model",
        "ML": "Machine Learning",
        "NLP": "Natural Language Processing",
        "OOP": "Object-Oriented Programming",
        "OS": "Operating System",
        "RAM": "Random Access Memory",
        "REST": "Representational State Transfer",
        "ROM": "Read-Only Memory",
        "SaaS": "Software as a Service",
        "SDK": "Software Development Kit",
        "SEO": "Search Engine Optimization",
        "SMTP": "Simple Mail Transfer Protocol",
        "SQL": "Structured Query Language",
        "SSL": "Secure Sockets Layer",
        "TCP": "Transmission Control Protocol",
        "TDD": "Test-Driven Development",
        "UI": "User Interface",
        "URL": "Uniform Resource Locator",
        "USB": "Universal Serial Bus",
        "UX": "User Experience",
        "VPN": "Virtual Private Network",
        "XML": "Extensible Markup Language",
    }
    
    # Legal-specific acronyms
    LEGAL_ACRONYMS = {
        "ADR": "Alternative Dispute Resolution",
        "ADA": "Americans with Disabilities Act",
        "BFOQ": "Bona Fide Occupational Qualification",
        "C&D": "Cease and Desist",
        "CBA": "Collective Bargaining Agreement",
        "DOJ": "Department of Justice",
        "DOL": "Department of Labor",
        "EEO": "Equal Employment Opportunity",
        "EEOC": "Equal Employment Opportunity Commission",
        "ERISA": "Employee Retirement Income Security Act",
        "FCRA": "Fair Credit Reporting Act",
        "FMLA": "Family and Medical Leave Act",
        "GDPR": "General Data Protection Regulation",
        "IP": "Intellectual Property",
        "LLC": "Limited Liability Company",
        "M&A": "Mergers and Acquisitions",
        "NDA": "Non-Disclosure Agreement",
        "OSHA": "Occupational Safety and Health Administration",
        "POA": "Power of Attorney",
        "ROI": "Return on Investment",
        "SBA": "Small Business Administration",
        "SOX": "Sarbanes-Oxley Act",
        "T&C": "Terms and Conditions",
        "TM": "Trademark",
    }
    
    def __init__(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
        min_length: int = 2,
        contextual_expansion: bool = True,
        domain: Optional[str] = None,
        ignore_case: bool = False,
    ):
        """Initialize the acronym expander with specified options."""
        # Combine default acronyms with insurance-specific ones
        self.acronym_dict = {**self.DEFAULT_ACRONYMS, **self.INSURANCE_ACRONYMS}
        
        # Add domain-specific acronyms if requested
        if domain:
            domain_acronyms = self._get_domain_acronyms(domain)
            if domain_acronyms:
                self.acronym_dict.update(domain_acronyms)
        
        # Add custom mappings
        if custom_mappings:
            self.acronym_dict.update(custom_mappings)
        
        self.min_length = min_length
        self.contextual_expansion = contextual_expansion
        self.ignore_case = ignore_case
        
        # Create lowercase version of acronym dictionary for case-insensitive matching
        if self.ignore_case:
            self.lowercase_dict = {k.lower(): v for k, v in self.acronym_dict.items()}
        
        # Compile regex patterns for acronym detection
        self._compile_patterns()
        
    def _get_domain_acronyms(self, domain: str) -> Dict[str, str]:
        """Get domain-specific acronym dictionaries.
        
        Parameters
        ----------
        domain : str
            Domain to get acronyms for
            
        Returns
        -------
        Dict[str, str]
            Dictionary of domain-specific acronyms
        """
        domain = domain.lower()
        if domain == "healthcare":
            return self.HEALTHCARE_ACRONYMS
        elif domain == "finance":
            return self.FINANCE_ACRONYMS
        elif domain == "tech" or domain == "technology":
            return self.TECH_ACRONYMS
        elif domain == "legal":
            return self.LEGAL_ACRONYMS
        else:
            logger.warning(f"Unknown domain: {domain}. Using default acronyms only.")
            return {}
            
    def _compile_patterns(self):
        """Compile regex patterns for acronym detection."""
        # Standard uppercase acronym pattern
        standard_pattern = r'\b[A-Z][A-Z0-9&]{' + str(self.min_length - 1) + r',}\b'
        
        # Dotted acronym pattern (e.g., U.S.A., I.B.M.)
        dotted_pattern = r'\b(?:[A-Z]\.){' + str(self.min_length) + r',}\b'
        
        # Mixed-case acronym pattern (e.g., PhD, iOS)
        mixed_pattern = r'\b[A-Z][a-z0-9]{1,}[A-Z][a-zA-Z0-9]*\b'
        
        # Pattern for acronyms in parentheses - commonly found in text
        # e.g., "The World Health Organization (WHO) recommends..."
        paren_pattern = r'\(([A-Z][A-Z0-9&]{' + str(self.min_length - 1) + r',})\)'
        
        # Combine patterns
        self.acronym_pattern = re.compile(
            f"{standard_pattern}|{dotted_pattern}|{mixed_pattern}"
        )
        
        # Separate pattern for parenthetical acronyms
        self.paren_acronym_pattern = re.compile(paren_pattern)
    
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
        
        # Find all acronyms in the text based on patterns
        matches = self.acronym_pattern.findall(text)
        
        # Flatten the list if we got tuples from multiple capture groups
        if matches and isinstance(matches[0], tuple):
            acronyms = [match for group in matches for match in group if match]
        else:
            acronyms = matches
            
        # Look for parenthetical definitions and learn them
        # Example: "World Health Organization (WHO)"
        if self.contextual_expansion:
            self._extract_and_learn_parenthetical_acronyms(text)
        
        # Process each acronym for expansion
        expanded_text = text
        for acronym in acronyms:
            # Skip if the acronym is part of an expansion we already added
            if f"({acronym})" in expanded_text or f"{acronym} (" in expanded_text:
                continue
                
            expansion = self._get_expansion(acronym)
            if expansion:
                # Replace with format: "ACRONYM (Expansion)"
                expanded_text = re.sub(
                    r'\b' + re.escape(acronym) + r'\b',  # Word boundary to avoid partial matches
                    f"{acronym} ({expansion})",
                    expanded_text
                )
        
        return expanded_text
        
    def _extract_and_learn_parenthetical_acronyms(self, text: str) -> None:
        """Extract and learn acronyms defined in parentheses.
        
        Example: "World Health Organization (WHO)" would add WHO -> "World Health Organization"
        to the acronym dictionary.
        
        Parameters
        ----------
        text : str
            Text to extract acronyms from
        """
        # Look for pattern: "Some Longer Name (ACRONYM)"
        pattern = re.compile(r'([A-Z][a-zA-Z\s]+)\s+\(([A-Z][A-Z0-9&]{1,})\)')
        matches = pattern.findall(text)
        
        for full_name, acronym in matches:
            full_name = full_name.strip()
            # Only add if the acronym makes sense for the full name
            if self._is_valid_acronym(full_name, acronym):
                self.add_acronym(acronym, full_name)
                logger.debug(f"Learned acronym from text: {acronym} -> {full_name}")
                
    def _get_expansion(self, acronym: str) -> Optional[str]:
        """Get expansion for an acronym, with support for case sensitivity.
        
        Parameters
        ----------
        acronym : str
            Acronym to get expansion for
            
        Returns
        -------
        Optional[str]
            Expansion if found, None otherwise
        """
        # Direct dictionary lookup
        if acronym in self.acronym_dict:
            return self.acronym_dict[acronym]
            
        # Case-insensitive lookup if enabled
        if self.ignore_case and acronym.lower() in self.lowercase_dict:
            return self.lowercase_dict[acronym.lower()]
            
        # Handle dotted acronyms by removing dots
        if "." in acronym:
            undotted = acronym.replace(".", "")
            if undotted in self.acronym_dict:
                return self.acronym_dict[undotted]
            if self.ignore_case and undotted.lower() in self.lowercase_dict:
                return self.lowercase_dict[undotted.lower()]
                
        return None
    
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
    
    def _is_valid_acronym(self, full_name: str, acronym: str) -> bool:
        """Check if an acronym is valid for a given full name.
        
        Parameters
        ----------
        full_name : str
            Full name or phrase
        acronym : str
            Potential acronym
            
        Returns
        -------
        bool
            True if the acronym is valid for the full name
        """
        # Simple validation: Check if the acronym can be constructed from the words
        # Get first letters of significant words
        words = full_name.split()
        
        # Handle simple cases
        if len(acronym) == 1 and len(words) >= 1 and words[0].startswith(acronym):
            return True
            
        # Case 1: First letter of each word
        first_letters = ''.join(word[0].upper() for word in words if word)
        if acronym in first_letters:
            return True
            
        # Case 2: First letters of significant words (skip common articles, prepositions)
        skip_words = {'a', 'an', 'the', 'of', 'for', 'to', 'in', 'on', 'by', 'at', 'and'}
        sig_first_letters = ''.join(word[0].upper() for word in words 
                                   if word and word.lower() not in skip_words)
        if acronym in sig_first_letters:
            return True
            
        # Case 3: Letters from words (more flexible)
        all_letters = ''.join(full_name.upper().split())
        return len(acronym) <= len(all_letters) and all(letter in all_letters for letter in acronym)
        
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
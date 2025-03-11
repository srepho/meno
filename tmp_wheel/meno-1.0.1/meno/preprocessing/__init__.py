"""Text preprocessing module for cleaning and normalizing text data."""

from typing import List, Dict, Optional, Union

# Re-export key functions
from .spelling import correct_spelling
from .acronyms import expand_acronyms
from .normalization import normalize_text, TextNormalizer

__all__ = [
    "correct_spelling",
    "expand_acronyms",
    "normalize_text",
    "TextNormalizer",
]
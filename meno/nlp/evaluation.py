"""Evaluation metrics for NLP components."""

from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import numpy as np
import logging
import json
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class CorrectionEvaluator:
    """Evaluate the quality of text corrections.
    
    This class provides metrics to evaluate spelling correction and acronym
    expansion quality against ground truth or baseline corrections.
    
    Parameters
    ----------
    reference_data : Optional[Dict[str, str]], optional
        Reference corrections for evaluation, by default None
    
    Attributes
    ----------
    reference_data : Dict[str, str]
        Reference corrections for evaluation
    metrics : Dict[str, Any]
        Computed evaluation metrics
    """
    
    def __init__(
        self,
        reference_data: Optional[Dict[str, str]] = None,
    ):
        """Initialize the evaluator with reference data."""
        self.reference_data = reference_data or {}
        self.metrics = {}
    
    def load_reference_data(self, filepath: str) -> bool:
        """Load reference data from a file.
        
        Parameters
        ----------
        filepath : str
            Path to reference data file (JSON)
            
        Returns
        -------
        bool
            True if loading was successful
        """
        try:
            with open(filepath, 'r') as f:
                self.reference_data = json.load(f)
            logger.info(f"Loaded {len(self.reference_data)} reference corrections")
            return True
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return False
    
    def evaluate_corrections(
        self,
        corrections: Dict[str, str],
        compute_all: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate the quality of corrections against reference data.
        
        Parameters
        ----------
        corrections : Dict[str, str]
            Dictionary of corrections to evaluate
        compute_all : bool, optional
            Whether to compute all metrics, by default True
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics
        """
        if not self.reference_data:
            logger.warning("No reference data available for evaluation")
            return {}
        
        # Initialize metrics
        metrics = {
            "total_corrections": len(corrections),
            "total_reference": len(self.reference_data),
            "matches": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": 0.0,
            "character_similarity": 0.0,
            "levenshtein_reduction": 0.0,
        }
        
        # Count exact matches
        match_count = 0
        character_similarities = []
        levenshtein_reductions = []
        
        for word, correction in corrections.items():
            if word in self.reference_data:
                reference = self.reference_data[word]
                
                # Check for exact match
                if correction.lower() == reference.lower():
                    match_count += 1
                
                # Compute character similarity if requested
                if compute_all:
                    char_sim = self._character_similarity(correction, reference)
                    character_similarities.append(char_sim)
                    
                    # Compute Levenshtein distance reduction
                    orig_distance = self._levenshtein_distance(word, reference)
                    new_distance = self._levenshtein_distance(correction, reference)
                    reduction = (orig_distance - new_distance) / max(1, orig_distance)
                    levenshtein_reductions.append(reduction)
        
        # Set match count
        metrics["matches"] = match_count
        
        # Compute precision, recall, F1
        if len(corrections) > 0:
            metrics["precision"] = match_count / len(corrections)
        
        if len(self.reference_data) > 0:
            metrics["recall"] = match_count / len(self.reference_data)
        
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = (
                2 * metrics["precision"] * metrics["recall"] / 
                (metrics["precision"] + metrics["recall"])
            )
        
        # Compute other metrics if requested
        if compute_all:
            # Character similarity
            if character_similarities:
                metrics["character_similarity"] = np.mean(character_similarities)
            
            # Levenshtein reduction
            if levenshtein_reductions:
                metrics["levenshtein_reduction"] = np.mean(levenshtein_reductions)
            
            # Accuracy on reference set
            total_words = len(self.reference_data)
            metrics["accuracy"] = match_count / total_words if total_words > 0 else 0.0
        
        # Store metrics
        self.metrics = metrics
        
        return metrics
    
    def evaluate_text_corrections(
        self,
        original_texts: Union[List[str], pd.Series],
        corrected_texts: Union[List[str], pd.Series],
        reference_texts: Optional[Union[List[str], pd.Series]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the quality of text corrections.
        
        Parameters
        ----------
        original_texts : Union[List[str], pd.Series]
            Original uncorrected texts
        corrected_texts : Union[List[str], pd.Series]
            Corrected texts
        reference_texts : Optional[Union[List[str], pd.Series]], optional
            Reference texts for evaluation, by default None
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics
        """
        # Convert to lists if needed
        if isinstance(original_texts, pd.Series):
            original_texts = original_texts.tolist()
        
        if isinstance(corrected_texts, pd.Series):
            corrected_texts = corrected_texts.tolist()
        
        if reference_texts is not None:
            if isinstance(reference_texts, pd.Series):
                reference_texts = reference_texts.tolist()
        
        # Check lengths
        if len(original_texts) != len(corrected_texts):
            logger.error("Original and corrected text lists must have the same length")
            return {}
        
        if reference_texts and len(original_texts) != len(reference_texts):
            logger.error("Original and reference text lists must have the same length")
            return {}
        
        # Initialize metrics
        metrics = {
            "total_texts": len(original_texts),
            "changed_texts": 0,
            "word_count": 0,
            "changed_words": 0,
            "text_similarity": 0.0,
            "improvement_rate": 0.0,
        }
        
        # Analyze changes
        word_counts = []
        changed_word_counts = []
        text_similarities = []
        
        for i, (original, corrected) in enumerate(zip(original_texts, corrected_texts)):
            # Skip if either text is empty
            if not original or not corrected:
                continue
                
            # Count words
            original_words = original.split()
            corrected_words = corrected.split()
            word_count = len(original_words)
            word_counts.append(word_count)
            
            # Check if text was changed
            if original != corrected:
                metrics["changed_texts"] += 1
                
                # Count changed words
                changed_count = 0
                for o_word, c_word in zip(original_words, corrected_words):
                    if o_word != c_word:
                        changed_count += 1
                
                changed_word_counts.append(changed_count)
                
                # Compute text similarity
                sim = self._text_similarity(original, corrected)
                text_similarities.append(sim)
                
                # Compute improvement if reference available
                if reference_texts:
                    reference = reference_texts[i]
                    # TODO: Implement improvement metrics
        
        # Compute aggregated metrics
        metrics["word_count"] = sum(word_counts)
        metrics["changed_words"] = sum(changed_word_counts)
        
        if text_similarities:
            metrics["text_similarity"] = np.mean(text_similarities)
        
        # Compute improvement rate if reference available
        if reference_texts:
            # TODO: Compute aggregated improvement metrics
            pass
        
        return metrics
    
    def compare_correction_methods(
        self,
        method_results: Dict[str, Dict[str, str]],
        reference_data: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple correction methods.
        
        Parameters
        ----------
        method_results : Dict[str, Dict[str, str]]
            Dictionary mapping method names to their correction results
        reference_data : Optional[Dict[str, str]], optional
            Reference data for evaluation, by default None (uses self.reference_data)
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of evaluation metrics for each method
        """
        if reference_data:
            self.reference_data = reference_data
        
        if not self.reference_data:
            logger.warning("No reference data available for evaluation")
            return {}
        
        # Evaluate each method
        comparison = {}
        for method_name, corrections in method_results.items():
            metrics = self.evaluate_corrections(corrections)
            comparison[method_name] = metrics
        
        return comparison
    
    def generate_report(
        self,
        output_path: Optional[str] = None,
        include_examples: bool = True,
        max_examples: int = 20,
    ) -> Dict[str, Any]:
        """Generate an evaluation report.
        
        Parameters
        ----------
        output_path : Optional[str], optional
            Path to save the report JSON, by default None
        include_examples : bool, optional
            Whether to include examples in the report, by default True
        max_examples : int, optional
            Maximum number of examples to include, by default 20
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the evaluation report
        """
        if not self.metrics:
            logger.warning("No metrics available for report generation")
            return {}
        
        # Create report structure
        report = {
            "metrics": self.metrics,
            "examples": {
                "correct": [],
                "incorrect": [],
                "missed": [],
            }
        }
        
        # Add examples if requested
        if include_examples and hasattr(self, 'last_corrections'):
            # Add correct and incorrect examples
            correct_count = 0
            incorrect_count = 0
            
            for word, correction in self.last_corrections.items():
                if word in self.reference_data:
                    reference = self.reference_data[word]
                    
                    if correction.lower() == reference.lower():
                        # Correct example
                        if correct_count < max_examples:
                            report["examples"]["correct"].append({
                                "original": word,
                                "correction": correction,
                                "reference": reference
                            })
                            correct_count += 1
                    else:
                        # Incorrect example
                        if incorrect_count < max_examples:
                            report["examples"]["incorrect"].append({
                                "original": word,
                                "correction": correction,
                                "reference": reference
                            })
                            incorrect_count += 1
            
            # Add missed examples
            missed_count = 0
            for word, reference in self.reference_data.items():
                if word not in self.last_corrections and missed_count < max_examples:
                    report["examples"]["missed"].append({
                        "original": word,
                        "reference": reference
                    })
                    missed_count += 1
        
        # Save report if requested
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Saved evaluation report to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save evaluation report: {e}")
        
        return report
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute the Levenshtein distance between two strings.
        
        Parameters
        ----------
        s1 : str
            First string
        s2 : str
            Second string
            
        Returns
        -------
        int
            Levenshtein distance
        """
        if len(s1) < len(s2):
            return CorrectionEvaluator._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def _character_similarity(s1: str, s2: str) -> float:
        """Compute character-level similarity between two strings.
        
        Parameters
        ----------
        s1 : str
            First string
        s2 : str
            Second string
            
        Returns
        -------
        float
            Similarity score (0-1)
        """
        distance = CorrectionEvaluator._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 1.0
            
        return 1.0 - distance / max_len
    
    @staticmethod
    def _text_similarity(t1: str, t2: str) -> float:
        """Compute text similarity based on word overlap.
        
        Parameters
        ----------
        t1 : str
            First text
        t2 : str
            Second text
            
        Returns
        -------
        float
            Similarity score (0-1)
        """
        words1 = set(t1.lower().split())
        words2 = set(t2.lower().split())
        
        if not words1 and not words2:
            return 1.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


def evaluate_acronym_expansion(
    original_texts: Union[List[str], pd.Series],
    expanded_texts: Union[List[str], pd.Series],
    reference_texts: Optional[Union[List[str], pd.Series]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate acronym expansion results.
    
    Parameters
    ----------
    original_texts : Union[List[str], pd.Series]
        Original texts with unexpanded acronyms
    expanded_texts : Union[List[str], pd.Series]
        Texts with expanded acronyms
    reference_texts : Optional[Union[List[str], pd.Series]], optional
        Reference texts with correct expansions, by default None
    output_path : Optional[str], optional
        Path to save evaluation results, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics
    """
    # Create evaluator
    evaluator = CorrectionEvaluator()
    
    # Evaluate text corrections
    metrics = evaluator.evaluate_text_corrections(
        original_texts=original_texts,
        corrected_texts=expanded_texts,
        reference_texts=reference_texts,
    )
    
    # Save results if requested
    if output_path and metrics:
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved acronym expansion evaluation to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    return metrics


def evaluate_spelling_correction(
    original_texts: Union[List[str], pd.Series],
    corrected_texts: Union[List[str], pd.Series],
    reference_texts: Optional[Union[List[str], pd.Series]] = None,
    reference_corrections: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate spelling correction results.
    
    Parameters
    ----------
    original_texts : Union[List[str], pd.Series]
        Original texts with spelling errors
    corrected_texts : Union[List[str], pd.Series]
        Texts with corrected spelling
    reference_texts : Optional[Union[List[str], pd.Series]], optional
        Reference texts with correct spelling, by default None
    reference_corrections : Optional[Dict[str, str]], optional
        Dictionary mapping misspelled words to correct ones, by default None
    output_path : Optional[str], optional
        Path to save evaluation results, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics
    """
    # Create evaluator with reference corrections if available
    evaluator = CorrectionEvaluator(reference_data=reference_corrections)
    
    # Evaluate text corrections
    metrics = evaluator.evaluate_text_corrections(
        original_texts=original_texts,
        corrected_texts=corrected_texts,
        reference_texts=reference_texts,
    )
    
    # Add word-level metrics if reference corrections available
    if reference_corrections and hasattr(evaluator, 'last_corrections'):
        word_metrics = evaluator.evaluate_corrections(evaluator.last_corrections)
        metrics.update(word_metrics)
    
    # Save results if requested
    if output_path and metrics:
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved spelling correction evaluation to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    return metrics
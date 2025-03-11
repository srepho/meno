"""Topic coherence evaluation metrics.

This module provides functions to calculate various topic coherence metrics
based on the underlying corpus and topic word distributions. It implements
standard coherence metrics from the academic literature such as C_v, C_uci,
and C_npmi.
"""

from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging
from collections import Counter

# Gensim imports
try:
    import gensim
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class CoherenceEvaluator:
    """Calculate topic coherence metrics for topic models.
    
    This class provides methods to evaluate topic coherence using different
    metrics including C_v, C_uci, and C_npmi from gensim, as well as simpler
    word co-occurrence-based metrics.
    
    Parameters
    ----------
    texts : List[List[str]]
        Tokenized texts, where each text is a list of tokens
    dictionary : Optional[gensim.corpora.Dictionary], optional
        Gensim dictionary mapping tokens to integer IDs, by default None
        If None, a dictionary will be created from the texts
    
    Attributes
    ----------
    texts : List[List[str]]
        Tokenized texts
    dictionary : gensim.corpora.Dictionary
        Dictionary mapping tokens to integer IDs
    corpus : List[List[Tuple[int, int]]]
        Bag-of-words corpus representation
    """
    
    def __init__(
        self,
        texts: List[List[str]],
        dictionary: Optional[Any] = None,
    ):
        """Initialize the coherence evaluator with texts and optionally a dictionary."""
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "Gensim is required for coherence calculation. "
                "Install with 'pip install gensim>=4.0.0'"
            )
        
        self.texts = texts
        
        # Create dictionary if not provided
        if dictionary is None:
            self.dictionary = Dictionary(texts)
        else:
            self.dictionary = dictionary
            
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        
    def calculate_coherence(
        self,
        topics: List[List[str]],
        coherence: str = "c_v",
        top_n: int = 10,
    ) -> float:
        """Calculate topic coherence.
        
        Parameters
        ----------
        topics : List[List[str]]
            List of topics, where each topic is a list of its top words
        coherence : str, optional
            Coherence metric to use, by default "c_v"
            Options: "c_v", "c_uci", "c_npmi", "u_mass"
        top_n : int, optional
            Number of top words to use for coherence calculation, by default 10
            If topics have fewer than top_n words, all words will be used
            
        Returns
        -------
        float
            Coherence score
        """
        # Ensure topics have at most top_n words
        limited_topics = [topic[:top_n] for topic in topics]
        
        # Calculate coherence
        try:
            cm = CoherenceModel(
                topics=limited_topics,
                texts=self.texts,
                dictionary=self.dictionary,
                corpus=self.corpus,
                coherence=coherence
            )
            coherence_score = cm.get_coherence()
            
            # Normalize highly negative scores
            if coherence_score < -100:
                logger.warning(
                    f"Very low coherence score ({coherence_score}) detected. "
                    f"Setting to minimum of -1.0."
                )
                coherence_score = -1.0
                
            return coherence_score
        
        except Exception as e:
            logger.warning(f"Error calculating coherence: {str(e)}")
            return 0.0
    
    def calculate_pairwise_npmi(
        self,
        topics: List[List[str]],
        top_n: int = 10,
    ) -> float:
        """Calculate pairwise normalized pointwise mutual information (NPMI).
        
        This is a custom implementation of NPMI that calculates pairwise
        word co-occurrences to measure coherence.
        
        Parameters
        ----------
        topics : List[List[str]]
            List of topics, where each topic is a list of its top words
        top_n : int, optional
            Number of top words to use, by default 10
            
        Returns
        -------
        float
            Average NPMI score across all topics
        """
        # Ensure topics have at most top_n words
        limited_topics = [topic[:top_n] for topic in topics]
        
        # Word frequencies
        word_counts = Counter()
        for text in self.texts:
            word_counts.update(text)
            
        # Co-occurrence counts
        cooccur_counts = Counter()
        for text in self.texts:
            # Get unique words in document
            unique_words = set(text)
            # Count all co-occurrences
            for w1 in unique_words:
                for w2 in unique_words:
                    if w1 != w2:
                        cooccur_counts[(w1, w2)] += 1
        
        # Calculate average NPMI for each topic
        topic_scores = []
        
        for topic in limited_topics:
            if len(topic) < 2:
                continue
                
            # Calculate pairwise NPMI
            npmi_values = []
            
            for i, word1 in enumerate(topic):
                for word2 in topic[i+1:]:
                    # Skip words not in corpus
                    if word1 not in word_counts or word2 not in word_counts:
                        continue
                        
                    # Get counts
                    count1 = word_counts[word1]
                    count2 = word_counts[word2]
                    cooccur = cooccur_counts.get((word1, word2), 0) + cooccur_counts.get((word2, word1), 0)
                    
                    # Skip pairs with no co-occurrences
                    if cooccur == 0:
                        continue
                        
                    # Calculate joint and marginal probabilities
                    N = len(self.texts)
                    p_xy = cooccur / N
                    p_x = count1 / N
                    p_y = count2 / N
                    
                    # Calculate PMI and NPMI
                    pmi = np.log(p_xy / (p_x * p_y))
                    npmi = pmi / -np.log(p_xy)
                    
                    npmi_values.append(npmi)
            
            # Calculate average NPMI for this topic
            if npmi_values:
                topic_scores.append(np.mean(npmi_values))
        
        # Return average across all topics
        if topic_scores:
            return np.mean(topic_scores)
        else:
            return 0.0
    
    def evaluate_all_metrics(
        self,
        topics: List[List[str]],
        top_n: int = 10,
    ) -> Dict[str, float]:
        """Calculate multiple coherence metrics.
        
        Parameters
        ----------
        topics : List[List[str]]
            List of topics, where each topic is a list of its top words
        top_n : int, optional
            Number of top words to use, by default 10
            
        Returns
        -------
        Dict[str, float]
            Dictionary with coherence metric names and values
        """
        metrics = {}
        
        # Calculate gensim coherence metrics
        for coherence_type in ["c_v", "c_uci", "c_npmi", "u_mass"]:
            try:
                metrics[coherence_type] = self.calculate_coherence(
                    topics=topics,
                    coherence=coherence_type,
                    top_n=top_n
                )
            except Exception as e:
                logger.warning(f"Error calculating {coherence_type}: {str(e)}")
                metrics[coherence_type] = None
        
        # Calculate custom NPMI
        try:
            metrics["npmi_pairwise"] = self.calculate_pairwise_npmi(
                topics=topics,
                top_n=top_n
            )
        except Exception as e:
            logger.warning(f"Error calculating pairwise NPMI: {str(e)}")
            metrics["npmi_pairwise"] = None
            
        return metrics


# Helper functions for easier usage with different model types

def calculate_bertopic_coherence(
    model: "BERTopic",
    texts: List[List[str]],
    coherence: str = "c_v",
    top_n: int = 10,
) -> Union[float, Dict[str, float]]:
    """Calculate coherence for a BERTopic model.
    
    Parameters
    ----------
    model : BERTopic
        Fitted BERTopic model
    texts : List[List[str]]
        Tokenized texts
    coherence : str, optional
        Coherence metric to use, by default "c_v"
        If "all", returns all metrics in a dictionary
    top_n : int, optional
        Number of top words per topic to use, by default 10
        
    Returns
    -------
    Union[float, Dict[str, float]]
        Coherence score or dictionary of scores if coherence="all"
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Gensim is required for coherence calculation. "
            "Install with 'pip install gensim>=4.0.0'"
        )
        
    # Extract topics from BERTopic model
    topic_list = []
    for topic_id in model.get_topics():
        if topic_id != -1:  # Skip outlier topic
            topic_words = [word for word, _ in model.get_topic(topic_id)[:top_n]]
            topic_list.append(topic_words)
    
    # Create evaluator
    evaluator = CoherenceEvaluator(texts=texts)
    
    # Calculate coherence
    if coherence == "all":
        return evaluator.evaluate_all_metrics(topics=topic_list, top_n=top_n)
    else:
        return evaluator.calculate_coherence(
            topics=topic_list,
            coherence=coherence,
            top_n=top_n
        )


def calculate_lda_coherence(
    model: "LdaModel",
    texts: List[List[str]],
    dictionary: Any,
    coherence: str = "c_v",
    top_n: int = 10,
) -> Union[float, Dict[str, float]]:
    """Calculate coherence for a Gensim LDA model.
    
    Parameters
    ----------
    model : gensim.models.LdaModel
        Fitted LDA model
    texts : List[List[str]]
        Tokenized texts
    dictionary : gensim.corpora.Dictionary
        Dictionary used to train the model
    coherence : str, optional
        Coherence metric to use, by default "c_v"
        If "all", returns all metrics in a dictionary
    top_n : int, optional
        Number of top words per topic to use, by default 10
        
    Returns
    -------
    Union[float, Dict[str, float]]
        Coherence score or dictionary of scores if coherence="all"
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Gensim is required for coherence calculation. "
            "Install with 'pip install gensim>=4.0.0'"
        )
        
    # Extract topics from LDA model
    topic_list = []
    for topic_id in range(model.num_topics):
        topic_words = [word for word, _ in model.show_topic(topic_id, topn=top_n)]
        topic_list.append(topic_words)
    
    # Create evaluator
    evaluator = CoherenceEvaluator(texts=texts, dictionary=dictionary)
    
    # Calculate coherence
    if coherence == "all":
        return evaluator.evaluate_all_metrics(topics=topic_list, top_n=top_n)
    else:
        return evaluator.calculate_coherence(
            topics=topic_list,
            coherence=coherence,
            top_n=top_n
        )


def calculate_generic_coherence(
    topic_words: Dict[Union[int, str], List[str]],
    texts: List[List[str]],
    coherence: str = "c_v",
    top_n: int = 10,
) -> Union[float, Dict[str, float]]:
    """Calculate coherence for generic topic word lists.
    
    Parameters
    ----------
    topic_words : Dict[Union[int, str], List[str]]
        Dictionary mapping topic IDs to lists of topic words
    texts : List[List[str]]
        Tokenized texts
    coherence : str, optional
        Coherence metric to use, by default "c_v"
        If "all", returns all metrics in a dictionary
    top_n : int, optional
        Number of top words per topic to use, by default 10
        
    Returns
    -------
    Union[float, Dict[str, float]]
        Coherence score or dictionary of scores if coherence="all"
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Gensim is required for coherence calculation. "
            "Install with 'pip install gensim>=4.0.0'"
        )
        
    # Extract topics
    topic_list = []
    for topic_id, words in topic_words.items():
        topic_list.append(words[:top_n])
    
    # Create evaluator
    evaluator = CoherenceEvaluator(texts=texts)
    
    # Calculate coherence
    if coherence == "all":
        return evaluator.evaluate_all_metrics(topics=topic_list, top_n=top_n)
    else:
        return evaluator.calculate_coherence(
            topics=topic_list,
            coherence=coherence,
            top_n=top_n
        )
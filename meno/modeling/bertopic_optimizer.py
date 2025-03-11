"""BERTopic hyperparameter optimization module."""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import itertools
import random
import time
import logging
from pathlib import Path

try:
    # Import BERTopic core
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
    
    # Try importing components - different versions have different import paths
    try:
        # BERTopic 0.15+ structure
        from bertopic.vectorizers import ClassTfidfTransformer
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        from bertopic.dimensionality import UMAPReducer
        from bertopic.cluster import HDBSCANClusterer
    except ImportError:
        # BERTopic 0.14 and below structure
        from umap import UMAP
        import hdbscan
        
        # Create adapter classes for backward compatibility
        class UMAPReducer:
            def __init__(self, n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", low_memory=False):
                self.n_neighbors = n_neighbors
                self.n_components = n_components
                self.min_dist = min_dist
                self.metric = metric
                self.low_memory = low_memory
                self.umap_model = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    metric=metric,
                    low_memory=low_memory
                )
            
            def fit_transform(self, X):
                return self.umap_model.fit_transform(X)
                
        class HDBSCANClusterer:
            def __init__(self, min_cluster_size=15, min_samples=None, metric='euclidean', cluster_selection_method='eom'):
                self.min_cluster_size = min_cluster_size
                self.min_samples = min_samples
                self.metric = metric
                self.cluster_selection_method = cluster_selection_method
                self.clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric=metric,
                    cluster_selection_method=cluster_selection_method
                )
                
            def fit(self, X):
                return self.clusterer.fit(X)
                
        # Try importing remaining components directly
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from bertopic.vectorizers import ClassTfidfTransformer
            
            # Define compatible adapter classes
            class KeyBERTInspired:
                def extract_topics(self, documents, embeddings, topic_model):
                    topic_words = topic_model.get_topics()
                    return topic_words
                    
            class MaximalMarginalRelevance:
                def extract_topics(self, documents, embeddings, topic_model):
                    topic_words = topic_model.get_topics()
                    return topic_words
        except ImportError:
            # If still failing, set to minimal implementations
            from sklearn.feature_extraction.text import CountVectorizer as ClassTfidfTransformer
            KeyBERTInspired = lambda: None  # Dummy class
            MaximalMarginalRelevance = lambda: None  # Dummy class
except (ImportError, AttributeError) as e:
    import importlib
    BERTOPIC_AVAILABLE = False
    # Try direct import check using importlib as a fallback
    try:
        bertopic_spec = importlib.util.find_spec("bertopic")
        if bertopic_spec is not None:
            # Module exists but may have import issues, try to load it
            bertopic = importlib.import_module("bertopic")
            if hasattr(bertopic, "BERTopic"):
                BERTOPIC_AVAILABLE = True
    except (ImportError, AttributeError):
        pass

# Set up logging
logger = logging.getLogger(__name__)


class BERTopicOptimizer:
    """Hyperparameter optimizer for BERTopic models.
    
    This class provides methods to optimize BERTopic hyperparameters using
    grid search, random search, or a combination of both.
    
    Parameters
    ----------
    embedding_model : str, optional
        Name of the embedding model to use, by default "all-MiniLM-L6-v2"
    n_trials : int, optional
        Number of hyperparameter combinations to try, by default 10
    random_state : int, optional
        Random seed for reproducibility, by default 42
    metric : str, optional
        Metric to optimize for, by default "combined"
        Options: "n_topics", "coherence", "diversity", "outlier_percentage", "combined"
    verbose : bool, optional
        Whether to print progress information, by default True
    
    Attributes
    ----------
    embedding_model : str
        Name of the embedding model
    n_trials : int
        Number of hyperparameter combinations to try
    random_state : int
        Random seed for reproducibility
    metric : str
        Metric to optimize for
    verbose : bool
        Whether to print progress information
    param_grid : Dict[str, List]
        Grid of hyperparameters to search
    best_params : Dict[str, Any]
        Best hyperparameters found
    best_model : BERTopic
        Best model found
    best_score : float
        Best score achieved
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_trials: int = 10,
        random_state: int = 42,
        metric: str = "combined",
        verbose: bool = True,
    ):
        """Initialize the BERTopicOptimizer."""
        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "BERTopic is required for this optimizer. "
                "Install with 'pip install bertopic>=0.15.0'"
            )
        
        self.embedding_model = embedding_model
        self.n_trials = n_trials
        self.random_state = random_state
        self.metric = metric
        self.verbose = verbose
        
        # Initialize best results
        self.best_params = None
        self.best_model = None
        self.best_score = float('-inf') if metric != "outlier_percentage" else float('inf')
        
        # Set default parameter grid
        self.param_grid = {
            # UMAP parameters
            "n_neighbors": [5, 15, 30],
            "n_components": [5, 10, 15],
            "min_dist": [0.0, 0.1, 0.5],
            
            # HDBSCAN parameters
            "min_cluster_size": [5, 10, 15, 20],
            "min_samples": [None, 5, 10],  # None means min_samples = min_cluster_size
            "cluster_selection_method": ["eom", "leaf"],
            
            # c-TF-IDF parameters
            "reduce_frequent_words": [True, False],
            "bm25_weighting": [True, False],
            
            # Representation parameters
            "representation_model": ["KeyBERTInspired", "MaximalMarginalRelevance", "Both"],
            "diversity": [0.1, 0.3, 0.5],
            
            # BERTopic parameters
            "nr_topics": ["auto", 10, 20, 30, "None"],  # "None" means no topic reduction
        }
    
    def set_param_grid(self, param_grid: Dict[str, List[Any]]) -> None:
        """Set a custom parameter grid.
        
        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of possible values
        """
        self.param_grid = param_grid
    
    def optimize(
        self,
        documents: Union[List[str], pd.Series],
        search_method: str = "random",
        custom_scorer: Optional[Callable] = None,
    ) -> Tuple[Dict[str, Any], BERTopic, float]:
        """Optimize BERTopic hyperparameters.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        search_method : str, optional
            Search method to use, by default "random"
            Options: "grid", "random", "progressive"
        custom_scorer : Optional[Callable], optional
            Custom scoring function, by default None
            If provided, this function should take a BERTopic model and list of
            documents as input and return a float score
        
        Returns
        -------
        Tuple[Dict[str, Any], BERTopic, float]
            Tuple of (best_params, best_model, best_score)
        """
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
            
        # Generate parameter combinations based on search method
        if search_method == "grid":
            param_combinations = self._generate_grid_combinations()
        elif search_method == "random":
            param_combinations = self._generate_random_combinations()
        elif search_method == "progressive":
            param_combinations = self._generate_progressive_combinations()
        else:
            raise ValueError(f"Unknown search method: {search_method}. "
                            "Use 'grid', 'random', or 'progressive'")
            
        # Evaluate each parameter combination
        n_combinations = len(param_combinations)
        if self.verbose:
            logger.info(f"Evaluating {n_combinations} hyperparameter combinations")
            
        for i, params in enumerate(param_combinations):
            if self.verbose:
                logger.info(f"Trial {i+1}/{n_combinations}: {params}")
                
            start_time = time.time()
            
            try:
                # Create BERTopic model with these parameters
                model = self._create_model(params)
                
                # Fit the model
                model.fit_transform(documents)
                
                # Score the model
                if custom_scorer:
                    score = custom_scorer(model, documents)
                else:
                    score = self._evaluate_model(model, documents)
                    
                if self.verbose:
                    logger.info(f"Score: {score}")
                    
                # Update best model if better
                if self._is_better_score(score):
                    self.best_score = score
                    self.best_params = params
                    self.best_model = model
                    
                    if self.verbose:
                        logger.info(f"New best model! Score: {score}")
                        
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error with parameters {params}: {str(e)}")
                    
            if self.verbose:
                logger.info(f"Trial completed in {time.time() - start_time:.1f} seconds")
                
        return self.best_params, self.best_model, self.best_score
    
    def _is_better_score(self, score: float) -> bool:
        """Check if a score is better than the current best.
        
        Parameters
        ----------
        score : float
            Score to compare
            
        Returns
        -------
        bool
            True if the score is better than the current best
        """
        if self.metric == "outlier_percentage":
            # Lower outlier percentage is better
            return score < self.best_score
        else:
            # Higher score is better for all other metrics
            return score > self.best_score
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations from the parameter grid.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of parameter dictionaries
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_dicts = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            # Handle min_samples=None special case
            if param_dict.get("min_samples") is None:
                param_dict["min_samples"] = param_dict["min_cluster_size"]
            param_dicts.append(param_dict)
            
        # If too many combinations, select a random subset
        if len(param_dicts) > self.n_trials:
            random.seed(self.random_state)
            param_dicts = random.sample(param_dicts, self.n_trials)
            
        return param_dicts
    
    def _generate_random_combinations(self) -> List[Dict[str, Any]]:
        """Generate random combinations from the parameter grid.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of parameter dictionaries
        """
        random.seed(self.random_state)
        param_dicts = []
        
        for _ in range(self.n_trials):
            param_dict = {}
            for param_name, param_values in self.param_grid.items():
                param_dict[param_name] = random.choice(param_values)
                
            # Handle min_samples=None special case
            if param_dict.get("min_samples") is None:
                param_dict["min_samples"] = param_dict["min_cluster_size"]
                
            param_dicts.append(param_dict)
            
        return param_dicts
    
    def _generate_progressive_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations progressively.
        
        This method first tries a set of default parameters, then
        iteratively refines the search based on previous results.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of parameter dictionaries
        """
        # Start with a set of reasonable defaults
        defaults = {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.1,
            "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "eom",
            "reduce_frequent_words": True,
            "bm25_weighting": True,
            "representation_model": "Both",
            "diversity": 0.3,
            "nr_topics": "auto",
        }
        
        param_dicts = [defaults]
        
        # Add variations around the defaults
        param_groups = [
            # UMAP focused
            ["n_neighbors", "n_components", "min_dist"],
            # HDBSCAN focused
            ["min_cluster_size", "min_samples", "cluster_selection_method"],
            # Representation focused
            ["representation_model", "diversity"],
            # c-TF-IDF and topic count focused
            ["reduce_frequent_words", "bm25_weighting", "nr_topics"],
        ]
        
        # Allocate remaining trials across parameter groups
        remaining_trials = self.n_trials - 1
        trials_per_group = remaining_trials // len(param_groups)
        
        for group in param_groups:
            for _ in range(trials_per_group):
                param_dict = defaults.copy()
                for param_name in group:
                    param_dict[param_name] = random.choice(self.param_grid[param_name])
                param_dicts.append(param_dict)
                
        # If we didn't use all trials, add some fully random ones
        if len(param_dicts) < self.n_trials:
            param_dicts.extend(self._generate_random_combinations()[:self.n_trials - len(param_dicts)])
            
        return param_dicts
    
    def _create_model(self, params: Dict[str, Any]) -> BERTopic:
        """Create a BERTopic model with the given parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters
            
        Returns
        -------
        BERTopic
            BERTopic model
        """
        # Create UMAP reducer
        umap_model = UMAPReducer(
            n_neighbors=params["n_neighbors"],
            n_components=params["n_components"],
            min_dist=params["min_dist"],
            metric="cosine",
            low_memory=True,
        )
        
        # Create HDBSCAN clusterer
        hdbscan_model = HDBSCANClusterer(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean",
            prediction_data=True,
            cluster_selection_method=params["cluster_selection_method"],
        )
        
        # Create c-TF-IDF transformer
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=params["reduce_frequent_words"],
            bm25_weighting=params["bm25_weighting"],
        )
        
        # Create representation model
        if params["representation_model"] == "KeyBERTInspired":
            representation_model = KeyBERTInspired()
        elif params["representation_model"] == "MaximalMarginalRelevance":
            representation_model = MaximalMarginalRelevance(diversity=params["diversity"])
        elif params["representation_model"] == "Both":
            representation_model = [
                KeyBERTInspired(),
                MaximalMarginalRelevance(diversity=params["diversity"])
            ]
        else:
            representation_model = None
            
        # Handle nr_topics
        if params["nr_topics"] == "None":
            nr_topics = None
        elif params["nr_topics"] == "auto":
            nr_topics = "auto"
        else:
            nr_topics = params["nr_topics"]
            
        # Create BERTopic model
        model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            vectorizer_model=None,  # Use default CountVectorizer
            representation_model=representation_model,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=False,  # Reduce output during optimization
        )
        
        return model
    
    def _evaluate_model(self, model: BERTopic, documents: List[str]) -> float:
        """Evaluate a BERTopic model.
        
        Parameters
        ----------
        model : BERTopic
            The fitted BERTopic model
        documents : List[str]
            List of document texts
            
        Returns
        -------
        float
            Score for the model
        """
        # Get model information
        topic_info = model.get_topic_info()
        
        # Calculate metrics
        n_topics = len(topic_info[topic_info['Topic'] != -1])
        outlier_percentage = topic_info.iloc[0]['Count'] / len(documents) * 100
        
        # Calculate topic diversity based on word overlap
        all_topic_words = []
        for topic_id in model.get_topics():
            if topic_id != -1:  # Skip outlier topic
                words = [word for word, _ in model.get_topic(topic_id)[:10]]
                all_topic_words.append(set(words))
        
        # Calculate average Jaccard similarity (lower is better / more diverse)
        diversity_score = 0
        comparisons = 0
        for i in range(len(all_topic_words)):
            for j in range(i+1, len(all_topic_words)):
                intersection = len(all_topic_words[i].intersection(all_topic_words[j]))
                union = len(all_topic_words[i].union(all_topic_words[j]))
                if union > 0:
                    similarity = intersection / union
                    diversity_score += similarity
                    comparisons += 1
        
        word_diversity = 1.0 - (diversity_score / max(1, comparisons))
        
        # Average topic coherence (higher is better)
        coherence = self._calculate_topic_coherence(model)
        
        # Return the requested metric
        if self.metric == "n_topics":
            return float(n_topics)
        elif self.metric == "coherence":
            return coherence
        elif self.metric == "diversity":
            return word_diversity
        elif self.metric == "outlier_percentage":
            return outlier_percentage
        elif self.metric == "combined":
            # Combined score: balance between number of topics, diversity, and having few outliers
            # The specific formula can be adjusted based on application needs
            return n_topics * word_diversity * coherence * (100 - outlier_percentage) / 100
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _calculate_topic_coherence(self, model: BERTopic) -> float:
        """Calculate average topic coherence.
        
        This uses a simple approximation of coherence based on the
        c-TF-IDF scores of the top words in each topic.
        
        Parameters
        ----------
        model : BERTopic
            The fitted BERTopic model
            
        Returns
        -------
        float
            Average topic coherence score
        """
        # Note: This is a simple approximation using c-TF-IDF scores
        # For more sophisticated coherence measures, see the dedicated
        # coherence module which implements gensim-based metrics
        coherence_scores = []
        
        for topic_id in model.get_topics():
            if topic_id != -1:  # Skip outlier topic
                topic_words = model.get_topic(topic_id)
                # Use c-TF-IDF scores as a simple approximation of coherence
                scores = [score for _, score in topic_words[:10]]
                if scores:
                    coherence_scores.append(np.mean(scores))
        
        if coherence_scores:
            return np.mean(coherence_scores)
        else:
            return 0.0


def optimize_bertopic(
    documents: Union[List[str], pd.Series],
    embedding_model: str = "all-MiniLM-L6-v2",
    n_trials: int = 10,
    search_method: str = "random",
    metric: str = "combined",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], BERTopic, float]:
    """Optimize BERTopic hyperparameters.
    
    Parameters
    ----------
    documents : Union[List[str], pd.Series]
        List or Series of document texts
    embedding_model : str, optional
        Name of the embedding model to use, by default "all-MiniLM-L6-v2"
    n_trials : int, optional
        Number of hyperparameter combinations to try, by default 10
    search_method : str, optional
        Search method to use, by default "random"
        Options: "grid", "random", "progressive"
    metric : str, optional
        Metric to optimize for, by default "combined"
        Options: "n_topics", "coherence", "diversity", "outlier_percentage", "combined"
    random_state : int, optional
        Random seed for reproducibility, by default 42
    verbose : bool, optional
        Whether to print progress information, by default True
        
    Returns
    -------
    Tuple[Dict[str, Any], BERTopic, float]
        Tuple of (best_params, best_model, best_score)
    """
    optimizer = BERTopicOptimizer(
        embedding_model=embedding_model,
        n_trials=n_trials,
        random_state=random_state,
        metric=metric,
        verbose=verbose,
    )
    
    return optimizer.optimize(
        documents=documents,
        search_method=search_method,
    )
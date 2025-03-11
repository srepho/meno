"""Unified interface for topic modeling in Meno.

This module provides a consistent interface for various topic modeling approaches,
making it easier to switch between methods or combine multiple approaches.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import os

# Import Meno components
from .base import BaseTopicModel
from .embeddings import DocumentEmbedding

# Set up logging
logger = logging.getLogger(__name__)


class UnifiedTopicModeler:
    """Unified interface for topic modeling.
    
    This class provides a consistent interface for various topic modeling approaches
    in Meno, supporting swappable modeling methods while keeping the API consistent.
    
    Parameters
    ----------
    method : str, optional
        Topic modeling method to use, by default "embedding_cluster"
        Options:
        - "bertopic": Full BERTopic implementation
        - "embedding_cluster": Embedding-based clustering
        - "simple_kmeans": K-Means clustering on embeddings (lightweight)
        - "nmf": Non-negative Matrix Factorization
        - "lsa": Latent Semantic Analysis (LSA/LSI)
        - "tfidf": TF-IDF vectorization with K-Means
        - "top2vec": Top2Vec algorithm
    num_topics : Optional[int], optional
        Number of topics to extract, by default None
    embedding_model : Optional[DocumentEmbedding], optional
        Model to use for document embeddings, by default None
    random_state : int, optional
        Random seed for reproducibility, by default 42
    config_overrides : Dict[str, Any], optional
        Configuration overrides for the specific model, by default None
    """
    
    def __init__(
        self,
        method: str = "embedding_cluster",
        num_topics: Optional[int] = None,
        embedding_model: Optional[DocumentEmbedding] = None,
        random_state: int = 42,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the unified topic modeler."""
        self.method = method
        self.num_topics = num_topics or 10
        self.embedding_model = embedding_model
        self.random_state = random_state
        self.config_overrides = config_overrides or {}
        
        # Initialize model
        self.model = self._create_model()
    
    def _create_model(self) -> BaseTopicModel:
        """Create the appropriate topic model based on the selected method.
        
        Returns
        -------
        BaseTopicModel
            Topic model instance
        """
        if self.method == "bertopic":
            try:
                from .bertopic_model import MenoBERTopicModel
                return MenoBERTopicModel(
                    num_topics=self.num_topics,
                    embedding_model=self.embedding_model,
                    **self.config_overrides
                )
            except ImportError as e:
                msg = (
                    "To use BERTopic, please install the required dependencies: "
                    "pip install bertopic umap-learn hdbscan"
                )
                logger.error(f"{e}. {msg}")
                raise ImportError(msg) from e
                
        elif self.method == "embedding_cluster":
            from .unsupervised import EmbeddingClusteringModel
            return EmbeddingClusteringModel(
                num_topics=self.num_topics,
                embedding_model=self.embedding_model,
                **self.config_overrides
            )
            
        elif self.method == "top2vec":
            try:
                from .top2vec_model import MenoTop2VecModel
                return MenoTop2VecModel(
                    num_topics=self.num_topics,
                    embedding_model=self.embedding_model,
                    **self.config_overrides
                )
            except ImportError as e:
                msg = (
                    "To use Top2Vec, please install the required dependencies: "
                    "pip install top2vec umap-learn hdbscan"
                )
                logger.error(f"{e}. {msg}")
                raise ImportError(msg) from e
        
        elif self.method == "simple_kmeans":
            try:
                from .simple_models import SimpleTopicModel
                return SimpleTopicModel(
                    num_topics=self.num_topics,
                    embedding_model=self.embedding_model,
                    random_state=self.random_state,
                    **self.config_overrides
                )
            except ImportError as e:
                msg = (
                    "Error loading SimpleTopicModel. Make sure scikit-learn is installed: "
                    "pip install scikit-learn"
                )
                logger.error(f"{e}. {msg}")
                raise ImportError(msg) from e
                
        elif self.method == "nmf":
            try:
                from .simple_models import NMFTopicModel
                return NMFTopicModel(
                    num_topics=self.num_topics,
                    random_state=self.random_state,
                    **self.config_overrides
                )
            except ImportError as e:
                msg = (
                    "Error loading NMFTopicModel. Make sure scikit-learn is installed: "
                    "pip install scikit-learn"
                )
                logger.error(f"{e}. {msg}")
                raise ImportError(msg) from e
                
        elif self.method == "lsa":
            try:
                from .simple_models import LSATopicModel
                return LSATopicModel(
                    num_topics=self.num_topics,
                    random_state=self.random_state,
                    **self.config_overrides
                )
            except ImportError as e:
                msg = (
                    "Error loading LSATopicModel. Make sure scikit-learn is installed: "
                    "pip install scikit-learn"
                )
                logger.error(f"{e}. {msg}")
                raise ImportError(msg) from e
                
        elif self.method == "tfidf":
            try:
                from .simple_models import TFIDFTopicModel
                return TFIDFTopicModel(
                    num_topics=self.num_topics,
                    random_state=self.random_state,
                    **self.config_overrides
                )
            except ImportError as e:
                msg = (
                    "Error loading TFIDFTopicModel. Make sure scikit-learn is installed: "
                    "pip install scikit-learn"
                )
                logger.error(f"{e}. {msg}")
                raise ImportError(msg) from e
                
        else:
            # Default to embedding_cluster
            logger.warning(f"Unknown method '{self.method}'. Falling back to embedding_cluster.")
            from .unsupervised import EmbeddingClusteringModel
            return EmbeddingClusteringModel(
                num_topics=self.num_topics,
                embedding_model=self.embedding_model,
                **self.config_overrides
            )
    
    def fit(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "UnifiedTopicModeler":
        """Fit the topic model to the documents.
        
        Parameters
        ----------
        documents : List[str]
            List of documents to analyze
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        UnifiedTopicModeler
            Fitted model instance
        """
        self.model.fit(documents, embeddings, **kwargs)
        return self
    
    def transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Transform documents to topic vectors.
        
        Parameters
        ----------
        documents : List[str]
            Documents to transform
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        np.ndarray
            Document-topic matrix
        """
        return self.model.transform(documents, embeddings, **kwargs)
    
    def fit_transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : List[str]
            Documents to analyze
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        np.ndarray
            Document-topic matrix
        """
        return self.fit(documents, embeddings, **kwargs).transform(documents, embeddings, **kwargs)
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        return self.model.get_topic_info()
    
    def get_document_info(self, docs: Optional[List[str]] = None) -> pd.DataFrame:
        """Get document topic information.
        
        Parameters
        ----------
        docs : Optional[List[str]], optional
            Documents to analyze, by default None (uses training documents)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with document-topic information
        """
        return self.model.get_document_info(docs)
    
    def get_topics(self) -> Dict[int, List[str]]:
        """Get topic word lists.
        
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of words
        """
        return self.model.get_topics()
    
    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping topic IDs to label strings
        """
        return self.model.get_topic_labels()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model
        """
        # Handle path
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        
        # Save method info
        model_info = {
            "method": self.method,
            "num_topics": self.num_topics,
            "config_overrides": self.config_overrides
        }
        
        # Save model info
        import json
        with open(path / "model_info.json", "w") as f:
            json.dump(model_info, f)
        
        # Save the actual model if it has a save method
        if hasattr(self.model, "save"):
            self.model.save(path / "model")
    
    @classmethod
    def load(cls, path: Union[str, Path], embedding_model: Optional[DocumentEmbedding] = None) -> "UnifiedTopicModeler":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
        embedding_model : Optional[DocumentEmbedding], optional
            Embedding model to use, by default None
            
        Returns
        -------
        UnifiedTopicModeler
            Loaded model instance
        """
        # Handle path
        path = Path(path)
        
        # Load model info
        import json
        with open(path / "model_info.json", "r") as f:
            model_info = json.load(f)
        
        # Create instance
        instance = cls(
            method=model_info["method"],
            num_topics=model_info["num_topics"],
            embedding_model=embedding_model,
            config_overrides=model_info.get("config_overrides", {})
        )
        
        # Load the actual model if it has a load method
        if hasattr(instance.model, "load"):
            model_path = path / "model"
            if os.path.exists(model_path):
                instance.model.load(model_path)
        
        return instance
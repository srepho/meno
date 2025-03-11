"""Unified topic modeling interface for different topic modeling approaches."""

from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os
import pickle

from .base import BaseTopicModel
from .bertopic_model import BERTopicModel
from .top2vec_model import Top2VecModel
from .embeddings import DocumentEmbedding, ModernTextEmbedding
from ..utils.config import get_config, MenoConfig

logger = logging.getLogger(__name__)


class UnifiedTopicModeler(BaseTopicModel):
    """Unified topic modeling interface for different topic modeling approaches.
    
    This class provides a standardized interface to interact with various topic modeling 
    techniques available in the Meno library, including BERTopic and Top2Vec.
    
    Parameters
    ----------
    method : str
        The topic modeling method to use. Options include:
        - "bertopic": BERTopic model
        - "top2vec": Top2Vec model  
        - "embedding_cluster": Embedding-based clustering
    num_topics : int, optional
        The number of topics to discover, by default 10
    config_overrides : Dict[str, Any], optional
        Configuration overrides for the model, by default None
    embedding_model : Union[str, DocumentEmbedding], optional
        The embedding model to use, by default None
    """
    
    def __init__(
        self,
        method: str = "embedding_cluster",
        num_topics: Optional[int] = 10,
        config_overrides: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
        auto_detect_topics: bool = False
    ):
        self.method = method
        self.auto_detect_topics = auto_detect_topics
        
        # Handle automatic topic detection
        if auto_detect_topics:
            num_topics = None  # Force auto-detection
            
            # Add auto-detection configuration to overrides
            if config_overrides is None:
                config_overrides = {}
            config_overrides['auto_detect_topics'] = True
        
        self.num_topics = num_topics
        self.config_overrides = config_overrides or {}
        self.embedding_model = embedding_model
        self.model = self._create_model()
        self.is_fitted = False
        self.topics = {}
        self.topic_sizes = {}
        
    def _create_model(self) -> BaseTopicModel:
        """Create the appropriate topic model based on the specified method.
        
        Returns
        -------
        BaseTopicModel
            The instantiated topic model
        """
        # Create embedding model if specified as string
        if isinstance(self.embedding_model, str):
            self.embedding_model = ModernTextEmbedding(model_name=self.embedding_model)
        
        # Create model based on method
        if self.method == "bertopic":
            return BERTopicModel(
                n_topics=self.num_topics,
                embedding_model=self.embedding_model,
                **self.config_overrides
            )
        elif self.method == "top2vec":
            return Top2VecModel(
                n_topics=self.num_topics,
                embedding_model=self.embedding_model,
                **self.config_overrides
            )
        elif self.method == "embedding_cluster":
            # Default to BERTopic with embedding_model for embedding-based clustering
            return BERTopicModel(
                n_topics=self.num_topics,
                embedding_model=self.embedding_model,
                **self.config_overrides
            )
        else:
            raise ValueError(f"Unknown topic modeling method: {self.method}")
    
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "UnifiedTopicModeler":
        """Fit the topic model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to model
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments passed to the underlying model
        
        Returns
        -------
        UnifiedTopicModeler
            The fitted topic modeler for method chaining
        """
        # Set n_topics parameter appropriately for the underlying model
        # Ensure consistency across different model implementations
        if 'num_topics' in kwargs and 'n_topics' not in kwargs:
            kwargs['n_topics'] = kwargs.pop('num_topics')
        elif self.num_topics is not None and 'n_topics' not in kwargs:
            kwargs['n_topics'] = self.num_topics
        
        # Fit the underlying model
        self.model.fit(documents, embeddings, **kwargs)
        
        # Copy important attributes from the underlying model
        self.topics = getattr(self.model, 'topics', {})
        self.topic_sizes = getattr(self.model, 'topic_sizes', {})
        self.is_fitted = True
        
        return self
    
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Transform documents to topic assignments.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to assign to topics
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments passed to the underlying model
        
        Returns
        -------
        Tuple[Any, Any]
            A tuple containing (topic_assignments, topic_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform can be called.")
        
        # Standardize parameter names for consistency
        if 'top_n' in kwargs:
            kwargs['top_n'] = kwargs['top_n']
        
        # Call the underlying model's transform method
        return self.model.transform(documents, embeddings, **kwargs)
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to model and assign to topics
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments passed to the underlying model
        
        Returns
        -------
        Tuple[Any, Any]
            A tuple containing (topic_assignments, topic_probabilities)
        """
        self.fit(documents, embeddings, **kwargs)
        return self.transform(documents, embeddings, **kwargs)
    
    def visualize_topics(self, **kwargs) -> Any:
        """Visualize discovered topics.
        
        Parameters
        ----------
        **kwargs : Any
            Visualization parameters passed to the underlying model's visualization method
        
        Returns
        -------
        Any
            The visualization object (typically a plotly Figure)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be visualized.")
        
        if hasattr(self.model, 'visualize_topics'):
            return self.model.visualize_topics(**kwargs)
        else:
            raise NotImplementedError(f"Visualization not implemented for {self.method}")
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topic info can be retrieved.")
        
        if hasattr(self.model, 'get_topic_info'):
            return self.model.get_topic_info()
        else:
            # Create a standardized topic info dataframe
            data = []
            for topic_id, topic_words in self.topics.items():
                data.append({
                    'Topic': topic_id,
                    'Count': self.topic_sizes.get(topic_id, 0),
                    'Name': f"Topic {topic_id}",
                    'Representation': str(topic_words)
                })
            return pd.DataFrame(data)
    
    def get_document_topics(self, documents: Union[List[str], pd.Series]) -> pd.DataFrame:
        """Get topic assignments for documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to assign to topics
        
        Returns
        -------
        pd.DataFrame
            DataFrame with document-topic assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be assigned.")
        
        topic_assignments, probabilities = self.transform(documents)
        
        # Create a standardized document-topic dataframe
        result = pd.DataFrame({
            'document_id': range(len(documents)),
            'topic': topic_assignments,
            'probability': np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else probabilities
        })
        
        return result
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save attributes
        model_data = {
            'method': self.method,
            'num_topics': self.num_topics,
            'config_overrides': self.config_overrides,
            'is_fitted': self.is_fitted,
            'topics': self.topics,
            'topic_sizes': self.topic_sizes
        }
        
        # Save underlying model if it has a save method
        model_path = f"{path}_underlying_model"
        if hasattr(self.model, 'save'):
            self.model.save(model_path)
            model_data['underlying_model_path'] = model_path
        else:
            # Fallback to pickle if no save method
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            model_data['underlying_model_pickle'] = f"{model_path}.pkl"
        
        # Save model data
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> "UnifiedTopicModeler":
        """Load a model from disk.
        
        Parameters
        ----------
        path : str
            Path to load the model from
        
        Returns
        -------
        UnifiedTopicModeler
            Loaded model
        """
        # Load model data
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            method=model_data['method'],
            num_topics=model_data['num_topics'],
            config_overrides=model_data['config_overrides']
        )
        
        # Load underlying model
        if 'underlying_model_path' in model_data:
            if instance.method == 'bertopic':
                instance.model = BERTopicModel.load(model_data['underlying_model_path'])
            elif instance.method == 'top2vec':
                instance.model = Top2VecModel.load(model_data['underlying_model_path'])
            else:
                # Fallback to using the appropriate class's load method
                model_class = instance.model.__class__
                instance.model = model_class.load(model_data['underlying_model_path'])
        elif 'underlying_model_pickle' in model_data:
            with open(model_data['underlying_model_pickle'], 'rb') as f:
                instance.model = pickle.load(f)
        
        # Set attributes
        instance.is_fitted = model_data['is_fitted']
        instance.topics = model_data['topics']
        instance.topic_sizes = model_data['topic_sizes']
        
        return instance


def create_topic_modeler(
    method: str = "embedding_cluster",
    num_topics: Optional[int] = 10,
    config_overrides: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
    auto_detect_topics: bool = False
) -> BaseTopicModel:
    """Create a topic modeler with the specified method.
    
    This factory function creates and returns an appropriate topic modeler
    based on the specified method and configuration.
    
    Parameters
    ----------
    method : str, optional
        The topic modeling method to use, by default "embedding_cluster"
        Options: "bertopic", "top2vec", "embedding_cluster"
    num_topics : Optional[int], optional
        The number of topics to discover, by default 10
        If None or if auto_detect_topics=True, the model will automatically 
        determine the optimal number of topics
    config_overrides : Optional[Dict[str, Any]], optional
        Configuration overrides for the model, by default None
    embedding_model : Optional[Union[str, DocumentEmbedding]], optional
        The embedding model to use, by default None
    auto_detect_topics : bool, optional
        Whether to automatically detect the optimal number of topics, by default False
        If True, num_topics will be ignored and the model will determine the best
        number of topics based on the data
    
    Returns
    -------
    BaseTopicModel
        An instance of the appropriate topic model
    """
    config = get_config()
    config_overrides = config_overrides or {}
    
    # Handle auto-detection configuration
    if auto_detect_topics:
        if config_overrides is None:
            config_overrides = {}
        config_overrides['auto_detect_topics'] = True
        
        # Force num_topics to None for auto-detection
        num_topics = None
    
    # Special case for directly creating a specific model type
    if method == "bertopic":
        return BERTopicModel(
            num_topics=num_topics,  # Use standardized parameter name
            embedding_model=embedding_model,
            auto_detect_topics=auto_detect_topics,
            **config_overrides
        )
    elif method == "top2vec":
        return Top2VecModel(
            num_topics=num_topics,  # Use standardized parameter name
            embedding_model=embedding_model,
            auto_detect_topics=auto_detect_topics,
            **config_overrides
        )
    
    # Use the unified interface for other methods
    return UnifiedTopicModeler(
        method=method,
        num_topics=num_topics,
        config_overrides=config_overrides,
        embedding_model=embedding_model,
        auto_detect_topics=auto_detect_topics
    )
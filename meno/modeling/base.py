"""Base classes for topic modeling in Meno."""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseTopicModel(ABC):
    """Abstract base class for topic models in Meno.
    
    This class defines the standard interface that all topic models must implement.
    By adhering to this interface, models can be used interchangeably throughout
    the Meno library.
    
    Attributes
    ----------
    topics : Dict[int, str]
        Mapping of topic IDs to topic descriptions
    topic_sizes : Dict[int, int]
        Mapping of topic IDs to topic sizes
    is_fitted : bool
        Whether the model has been fitted
    num_topics : int
        The number of topics in the model (standardized parameter name)
    API_VERSION : ClassVar[str]
        The API version this model implements
    """
    
    # API version for compatibility checks
    API_VERSION: ClassVar[str] = "1.0.0"
    
    @abstractmethod
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseTopicModel":
        """Fit the model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments for model-specific configurations
            
        Returns
        -------
        BaseTopicModel
            Fitted model
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform documents to topics.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments for model-specific configurations
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_assignments, topic_probabilities)
            - topic_assignments: 1D array of shape (n_documents,) with integer topic IDs
            - topic_probabilities: 2D array of shape (n_documents, n_topics) with probability scores
        """
        pass
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments for model-specific configurations
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_assignments, topic_probabilities)
            - topic_assignments: 1D array of shape (n_documents,) with integer topic IDs
            - topic_probabilities: 2D array of shape (n_documents, n_topics) with probability scores
        """
        self.fit(documents, embeddings, **kwargs)
        return self.transform(documents, embeddings, **kwargs)
    
    @abstractmethod
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized topic information containing at minimum:
            - Topic: The topic ID
            - Count: Number of documents in the topic
            - Name: Human-readable topic name
            - Representation: Keywords or representation of the topic content
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model to
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : str
            Path to load the model from
            
        Returns
        -------
        BaseTopicModel
            Loaded model
        """
        pass
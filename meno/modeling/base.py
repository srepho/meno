"""Base classes for topic modeling in Meno."""

from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseTopicModel(ABC):
    """Abstract base class for topic models in Meno.
    
    This class defines the interface that all topic models should implement.
    
    Attributes
    ----------
    topics : Dict[int, str]
        Mapping of topic IDs to topic descriptions
    topic_sizes : Dict[int, int]
        Mapping of topic IDs to topic sizes
    is_fitted : bool
        Whether the model has been fitted
    """
    
    @abstractmethod
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> "BaseTopicModel":
        """Fit the model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
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
    ) -> Tuple[Any, Any]:
        """Transform documents to topics.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        Tuple[Any, Any]
            Tuple of (topic_assignments, topic_probabilities)
        """
        pass
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any]:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        Tuple[Any, Any]
            Tuple of (topic_assignments, topic_probabilities)
        """
        self.fit(documents, embeddings)
        return self.transform(documents, embeddings)
    
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
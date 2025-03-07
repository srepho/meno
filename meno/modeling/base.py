"""Base classes for topic modeling in Meno."""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar
from pathlib import Path
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
    num_topics : Optional[int]
        The number of topics in the model. If None, the model will 
        automatically determine the optimal number of topics based on the data.
    auto_detect_topics : bool
        Whether the model should automatically detect the optimal number of topics
    embedding_model : Optional[Any]
        The embedding model used to generate document vectors
    document_embeddings : Optional[np.ndarray]
        The embeddings for the documents used to train the model
    API_VERSION : ClassVar[str]
        The API version this model implements
    """
    
    # API version for compatibility checks 
    API_VERSION: ClassVar[str] = "1.0.0"
    
    def __init__(
        self,
        num_topics: Optional[int] = 10,
        auto_detect_topics: bool = False,
        **kwargs
    ):
        """Initialize the base topic model.
        
        Parameters
        ----------
        num_topics : Optional[int], optional
            Number of topics to extract, by default 10
            If None or auto_detect_topics=True, the model will automatically
            determine the optimal number of topics
        auto_detect_topics : bool, optional
            Whether to automatically detect the optimal number of topics, by default False
        **kwargs : Any
            Additional keyword arguments for model-specific configurations
        """
        self.num_topics = None if auto_detect_topics else num_topics
        self.auto_detect_topics = auto_detect_topics
        self.is_fitted = False
        self.topics = {}
        self.topic_sizes = {}
    
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
    def visualize_topics(
        self,
        width: int = 800,
        height: int = 600,
        **kwargs
    ) -> Any:
        """Visualize discovered topics.
        
        Parameters
        ----------
        width : int, optional
            Width of the visualization in pixels, by default 800
        height : int, optional
            Height of the visualization in pixels, by default 600
        **kwargs : Any
            Additional visualization parameters
            
        Returns
        -------
        Any
            Visualization object (typically a plotly Figure)
        """
        pass
    
    def add_documents(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseTopicModel":
        """Add new documents to the model.
        
        This method is optional - models that don't support adding documents
        after initialization will raise NotImplementedError.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts to add
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments for model-specific configurations
            
        Returns
        -------
        BaseTopicModel
            Updated model
            
        Raises
        ------
        NotImplementedError
            If the model doesn't support adding documents after initialization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support adding documents after initialization."
        )
    
    def get_document_embeddings(self) -> np.ndarray:
        """Get document embeddings used by the model.
        
        Returns
        -------
        np.ndarray
            Document embeddings of shape (n_documents, embedding_dim)
            
        Raises
        ------
        NotImplementedError
            If the model doesn't store document embeddings
        """
        if not hasattr(self, "document_embeddings") or self.document_embeddings is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not store document embeddings."
            )
        return self.document_embeddings
    
    def find_similar_topics(
        self,
        query: str,
        n_topics: int = 5,
        **kwargs
    ) -> List[Tuple[int, str, float]]:
        """Find topics similar to a query string.
        
        Parameters
        ----------
        query : str
            Query string to find similar topics for
        n_topics : int, optional
            Number of similar topics to return, by default 5
        **kwargs : Any
            Additional keyword arguments for model-specific configurations
            
        Returns
        -------
        List[Tuple[int, str, float]]
            List of tuples (topic_id, topic_description, similarity_score)
            
        Raises
        ------
        NotImplementedError
            If the model doesn't support finding similar topics
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support finding similar topics."
        )
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
            
        Returns
        -------
        BaseTopicModel
            Loaded model
        """
        pass
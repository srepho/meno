"""BERTopic model for topic modeling."""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import pickle

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.dimensionality import UMAPReducer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.base import BaseTopicModel


class BERTopicModel(BaseTopicModel):
    """BERTopic model for topic modeling using BERT embeddings with UMAP and HDBSCAN.
    
    Parameters
    ----------
    num_topics : Optional[int], optional
        Number of topics to extract, by default None
        If None, BERTopic automatically determines the number of topics
        (Standardized parameter name, internally mapped to n_topics for BERTopic)
    embedding_model : Optional[DocumentEmbedding], optional
        Document embedding model to use, by default None
        If None, a default DocumentEmbedding will be created
    min_topic_size : int, optional
        Minimum size of topics, by default 10
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
        Setting to False (default) ensures CPU-only operation and avoids CUDA dependencies
    n_neighbors : int, optional
        Number of neighbors for UMAP, by default 15
    n_components : int, optional
        Number of dimensions for UMAP, by default 5
    verbose : bool, optional
        Whether to show verbose output, by default True
    
    Attributes
    ----------
    model : BERTopic
        Trained BERTopic model
    topics : Dict[int, str]
        Mapping of topic IDs to topic descriptions
    topic_sizes : Dict[int, int]
        Mapping of topic IDs to topic sizes
    """
    
    # API version for compatibility checks
    API_VERSION: ClassVar[str] = "1.0.0"
    
    def __init__(
        self,
        num_topics: Optional[int] = None,
        embedding_model: Optional[DocumentEmbedding] = None,
        min_topic_size: int = 10,
        use_gpu: bool = False,
        n_neighbors: int = 15,
        n_components: int = 5,
        verbose: bool = True,
        auto_detect_topics: bool = False,
        **kwargs
    ):
        """Initialize the BERTopic model."""
        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "BERTopic is required for this model. "
                "Install with 'pip install bertopic>=0.15.0'"
            )
        
        # Handle automatic topic detection
        self.auto_detect_topics = auto_detect_topics
        if auto_detect_topics:
            num_topics = None  # Force automatic detection
        
        # Map standardized parameter name to BERTopic parameter
        n_topics = num_topics
        
        self.num_topics = num_topics  # Standardized name
        self.n_topics = n_topics      # For backward compatibility
        self.min_topic_size = min_topic_size
        self.use_gpu = use_gpu
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.verbose = verbose
        
        # Set up embedding model if not provided - default to CPU
        if embedding_model is None:
            self.embedding_model = DocumentEmbedding(use_gpu=False)  # Default to CPU
        else:
            self.embedding_model = embedding_model
            
        # Set up UMAP reducer
        self.umap_model = UMAPReducer(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            low_memory=True,
        )
        
        # Set up representation model
        self.representation_model = KeyBERTInspired()
        
        # Set up vectorizer
        self.vectorizer_model = ClassTfidfTransformer()
        
        # Initialize BERTopic model
        self.model = BERTopic(
            nr_topics=n_topics,
            min_topic_size=min_topic_size,
            umap_model=self.umap_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
            verbose=verbose,
            **kwargs
        )
        
        # Initialize empty attributes
        self.topics = {}
        self.topic_sizes = {}
        self.topic_embeddings = None
        self.is_fitted = False
    
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BERTopicModel":
        """Fit the BERTopic model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            If None, embeddings will be computed using the embedding model
        **kwargs : Any
            Additional keyword arguments for model-specific configurations:
            - num_topics: Override the number of topics from initialization
            - min_topic_size: Minimum size of topics
            - seed: Random seed for reproducibility
        
        Returns
        -------
        BERTopicModel
            Fitted model
        """
        # Process kwargs for standardized parameters
        if 'num_topics' in kwargs:
            self.num_topics = kwargs.pop('num_topics')
            self.n_topics = self.num_topics
            self.model.nr_topics = self.n_topics
            
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
            
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.embedding_model.embed_documents(documents)
            
        # Fit BERTopic model with any additional kwargs
        topics, probs = self.model.fit_transform(documents, embeddings=embeddings, **kwargs)
        
        # Store topic information
        self.topics = {i: f"Topic {i}" for i in set(topics) if i != -1}
        self.topic_sizes = {
            topic: (topics == topic).sum() for topic in self.topics.keys()
        }
        
        # Add -1 for outliers
        if -1 in set(topics):
            self.topics[-1] = "Other"
            self.topic_sizes[-1] = (topics == -1).sum()
            
        # Update topic descriptions with top words
        topic_info = self.model.get_topic_info()
        for topic_id, row in topic_info.iterrows():
            if topic_id in self.topics and topic_id != -1:
                words = [word for word, _ in self.model.get_topic(topic_id)][:5]
                self.topics[topic_id] = f"Topic {topic_id}: {', '.join(words)}"
                
        # Compute topic embeddings
        self._compute_topic_embeddings()
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform documents to topic assignments and probabilities.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            If None, embeddings will be computed using the embedding model
        **kwargs : Any
            Additional keyword arguments:
            - top_n: Number of top topics to return per document
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_assignments, topic_probabilities)
            - topic_assignments: 1D array of shape (n_documents,) with integer topic IDs
            - topic_probabilities: 2D array of shape (n_documents, n_topics) with probability scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform can be called")
            
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
            
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.embedding_model.embed_documents(documents)
            
        # Transform documents
        topics, probs = self.model.transform(documents, embeddings=embeddings, **kwargs)
        
        # Ensure output is numpy arrays with consistent shape for API compliance
        topics_array = np.array(topics)
        probs_array = np.array(probs)
        
        # Ensure 2D shape for probabilities
        if len(probs_array.shape) == 1:
            probs_array = probs_array.reshape(-1, 1)
        
        return topics_array, probs_array
    
    def _compute_topic_embeddings(self) -> None:
        """Compute embeddings for all topics."""
        # Get topic descriptions with top words
        topic_info = self.model.get_topic_info()
        
        # Create descriptions for each topic
        descriptions = []
        topic_ids = []
        
        for topic_id, row in topic_info.iterrows():
            if topic_id != -1:  # Skip outlier topic
                words = [word for word, _ in self.model.get_topic(topic_id)][:10]
                description = " ".join(words)
                descriptions.append(description)
                topic_ids.append(topic_id)
                
        # Compute embeddings for descriptions
        if descriptions:
            self.topic_embeddings = self.embedding_model.embed_documents(descriptions)
            self.topic_id_mapping = {i: topic_id for i, topic_id in enumerate(topic_ids)}
    
    def find_similar_topics(
        self,
        query: str,
        n_topics: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """Find topics similar to a query string.
        
        Parameters
        ----------
        query : str
            Query string to find similar topics for
        n_topics : int, optional
            Number of similar topics to return, by default 5
        
        Returns
        -------
        List[Tuple[int, str, float]]
            List of tuples (topic_id, topic_description, similarity_score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar topics")
            
        if self.topic_embeddings is None:
            self._compute_topic_embeddings()
            
        if len(self.topic_embeddings) == 0:
            return []
            
        # Compute query embedding
        query_embedding = self.embedding_model.embed_documents([query])[0]
        
        # Compute similarities
        similarities = np.dot(self.topic_embeddings, query_embedding)
        
        # Get top n_topics
        top_indices = similarities.argsort()[-n_topics:][::-1]
        
        # Return topic IDs, descriptions, and similarity scores
        return [
            (
                self.topic_id_mapping[i],
                self.topics[self.topic_id_mapping[i]],
                float(similarities[i])
            )
            for i in top_indices
        ]
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized topic information containing:
            - Topic: The topic ID
            - Count: Number of documents in the topic
            - Name: Human-readable topic name
            - Representation: Keywords or representation of the topic content
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topic info can be retrieved")
            
        # Get BERTopic's topic information
        base_info = self.model.get_topic_info()
        
        # Standardize format for API compliance
        topic_info = base_info.copy()
        
        # Ensure required columns are present with standard names
        if 'Topic' not in topic_info.columns:
            topic_info['Topic'] = topic_info.index
            
        if 'Count' not in topic_info.columns and 'Count' in base_info.columns:
            topic_info['Count'] = base_info['Count']
        elif 'Count' not in topic_info.columns and 'Size' in base_info.columns:
            topic_info['Count'] = base_info['Size']
            
        if 'Name' not in topic_info.columns:
            topic_info['Name'] = [f"Topic {i}" if i != -1 else "Other" for i in topic_info['Topic']]
            
        if 'Representation' not in topic_info.columns:
            # Create representations with top words from each topic
            representations = []
            for topic_id in topic_info['Topic']:
                if topic_id != -1:
                    words = [word for word, _ in self.model.get_topic(topic_id)][:5]
                    representations.append(", ".join(words))
                else:
                    representations.append("Other/Outlier documents")
            topic_info['Representation'] = representations
            
        # Select only standardized columns in the correct order
        return topic_info[['Topic', 'Count', 'Name', 'Representation']]
        
    def visualize_topics(
        self,
        width: int = 800,
        height: int = 600,
    ) -> Any:
        """Visualize topics using BERTopic's visualization tools.
        
        Parameters
        ----------
        width : int, optional
            Width of the visualization, by default 800
        height : int, optional
            Height of the visualization, by default 600
        
        Returns
        -------
        Any
            Plotly figure for topic visualization
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")
            
        # Use BERTopic's topic visualization
        return self.model.visualize_topics(width=width, height=height)
    
    def visualize_hierarchy(
        self,
        width: int = 1000,
        height: int = 600,
    ) -> Any:
        """Visualize topic hierarchy.
        
        Parameters
        ----------
        width : int, optional
            Width of the visualization, by default 1000
        height : int, optional
            Height of the visualization, by default 600
        
        Returns
        -------
        Any
            Plotly figure for hierarchy visualization
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")
            
        # Use BERTopic's hierarchical visualization
        return self.model.visualize_hierarchy(width=width, height=height)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save BERTopic model
        self.model.save(path / "bertopic_model")
        
        # Save other attributes
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "n_topics": self.n_topics,
                "min_topic_size": self.min_topic_size,
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "topics": {str(k): v for k, v in self.topics.items()},
                "topic_sizes": {str(k): v for k, v in self.topic_sizes.items()},
                "is_fitted": self.is_fitted,
                "topic_id_mapping": {str(k): v for k, v in self.topic_id_mapping.items()} if hasattr(self, "topic_id_mapping") else None,
            }, f)
            
        # Save topic embeddings
        if self.topic_embeddings is not None:
            np.save(path / "topic_embeddings.npy", self.topic_embeddings)
            
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        embedding_model: Optional[DocumentEmbedding] = None,
    ) -> "BERTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
        embedding_model : Optional[DocumentEmbedding], optional
            Document embedding model to use, by default None
            
        Returns
        -------
        BERTopicModel
            Loaded model
        """
        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "BERTopic is required for this model. "
                "Install with 'pip install bertopic>=0.15.0'"
            )
            
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Create instance with loaded parameters
        instance = cls(
            n_topics=metadata["n_topics"],
            min_topic_size=metadata["min_topic_size"],
            n_neighbors=metadata["n_neighbors"],
            n_components=metadata["n_components"],
            embedding_model=embedding_model,
        )
        
        # Load BERTopic model
        instance.model = BERTopic.load(path / "bertopic_model")
        
        # Load other attributes
        instance.topics = {int(k): v for k, v in metadata["topics"].items()}
        instance.topic_sizes = {int(k): v for k, v in metadata["topic_sizes"].items()}
        instance.is_fitted = metadata["is_fitted"]
        
        if metadata["topic_id_mapping"] is not None:
            instance.topic_id_mapping = {int(k): v for k, v in metadata["topic_id_mapping"].items()}
        
        # Load topic embeddings if they exist
        topic_embeddings_path = path / "topic_embeddings.npy"
        if topic_embeddings_path.exists():
            instance.topic_embeddings = np.load(topic_embeddings_path)
            
        return instance
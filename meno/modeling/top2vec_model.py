"""Top2Vec model for topic modeling."""

from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import pickle
import os

try:
    from top2vec import Top2Vec
    TOP2VEC_AVAILABLE = True
except ImportError:
    TOP2VEC_AVAILABLE = False

from meno.modeling.base import BaseTopicModel
from meno.modeling.embeddings import DocumentEmbedding


class Top2VecModel(BaseTopicModel):
    """Top2Vec model for topic modeling.
    
    Parameters
    ----------
    embedding_model : Optional[DocumentEmbedding], optional
        Document embedding model to use, by default None
        If None, Top2Vec will use internal embeddings
    n_topics : Optional[int], optional
        Number of topics to extract, by default None
        If None, Top2Vec automatically determines the number of topics
    min_topic_size : int, optional
        Minimum size of topics, by default 10
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
        Setting to False (default) ensures CPU-only operation and avoids CUDA dependencies
    umap_args : Optional[Dict], optional
        Arguments for UMAP, by default None
    hdbscan_args : Optional[Dict], optional
        Arguments for HDBSCAN, by default None
    use_custom_embeddings : bool, optional
        Whether to use custom embeddings from DocumentEmbedding, by default True
        If False, Top2Vec will use its own embeddings
    verbose : bool, optional
        Whether to show verbose output, by default True
    
    Attributes
    ----------
    model : Top2Vec
        Trained Top2Vec model
    topics : Dict[int, str]
        Mapping of topic IDs to topic descriptions
    topic_sizes : Dict[int, int]
        Mapping of topic IDs to topic sizes
    """
    
    def __init__(
        self,
        embedding_model: Optional[DocumentEmbedding] = None,
        n_topics: Optional[int] = None,
        min_topic_size: int = 10,
        use_gpu: bool = False,
        umap_args: Optional[Dict] = None,
        hdbscan_args: Optional[Dict] = None,
        use_custom_embeddings: bool = True,
        verbose: bool = True,
    ):
        """Initialize the Top2Vec model."""
        if not TOP2VEC_AVAILABLE:
            raise ImportError(
                "Top2Vec is required for this model. "
                "Install with 'pip install top2vec>=1.0.27'"
            )
            
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.use_gpu = use_gpu
        self.umap_args = umap_args or {}
        self.hdbscan_args = hdbscan_args or {
            "min_cluster_size": min_topic_size,
            "min_samples": 5,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }
        self.use_custom_embeddings = use_custom_embeddings
        self.verbose = verbose
        
        # Set up embedding model if not provided and using custom embeddings
        if embedding_model is None and use_custom_embeddings:
            self.embedding_model = DocumentEmbedding(use_gpu=False)  # Default to CPU
        else:
            self.embedding_model = embedding_model
            
        # Initialize model to None
        self.model = None
        self.topics = {}
        self.topic_sizes = {}
        self.is_fitted = False
    
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> "Top2VecModel":
        """Fit the Top2Vec model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            If None and use_custom_embeddings=True, embeddings will be computed
            
        Returns
        -------
        Top2VecModel
            Fitted model
        """
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        # Handle custom embeddings
        if self.use_custom_embeddings:
            if embeddings is None and self.embedding_model is not None:
                embeddings = self.embedding_model.embed_documents(documents)
                
            # Initialize and fit Top2Vec with custom embeddings
            self.model = Top2Vec(
                documents=documents, 
                document_vectors=embeddings,
                embedding_model=None,  # Custom embeddings already provided
                min_count=1,  # Not used with custom embeddings
                keep_documents=True,
                workers=os.cpu_count() or 1,
                umap_args=self.umap_args,
                hdbscan_args=self.hdbscan_args,
                verbose=self.verbose,
            )
        else:
            # Let Top2Vec handle embeddings
            use_embedding = "doc2vec" if self.embedding_model is None else "universal-sentence-encoder"
            speed = "learn" if self.embedding_model is None else "deep-learn"
            
            self.model = Top2Vec(
                documents=documents,
                embedding_model=use_embedding,
                speed=speed,
                min_count=1,
                keep_documents=True,
                workers=os.cpu_count() or 1,
                umap_args=self.umap_args,
                hdbscan_args=self.hdbscan_args,
                verbose=self.verbose,
            )
            
        # Reduce topics if specified
        if self.n_topics is not None and self.n_topics < len(self.model.topic_sizes):
            self.model.hierarchical_topic_reduction(self.n_topics)
            
        # Store topic information
        self._extract_topic_info()
        
        self.is_fitted = True
        return self
    
    def _extract_topic_info(self) -> None:
        """Extract topic information from the fitted model."""
        # Get topic words and sizes
        topic_words, _, _ = self.model.get_topics()
        topic_sizes = self.model.topic_sizes
        
        # Create topic descriptions
        self.topics = {}
        self.topic_sizes = {}
        
        for i, words in enumerate(topic_words):
            # Skip outlier topic (-1)
            if i < len(topic_sizes):
                # Create topic description from top words
                top_words = words[:5]
                topic_name = f"Topic {i}: {', '.join(top_words)}"
                self.topics[i] = topic_name
                self.topic_sizes[i] = topic_sizes[i]
    
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        top_n: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign documents to topics.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        top_n : int, optional
            Number of top topics to return for each document, by default 1
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_nums, topic_scores)
            topic_nums has shape (n_documents, top_n)
            topic_scores has shape (n_documents, top_n)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform can be called")
            
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
            
        # Handle custom embeddings
        if self.use_custom_embeddings and embeddings is None and self.embedding_model is not None:
            embeddings = self.embedding_model.embed_documents(documents)
            
        # Get document vectors if using custom embeddings
        if self.use_custom_embeddings and embeddings is not None:
            topic_nums, topic_scores = self.model.get_documents_topics(
                doc_vectors=embeddings,
                num_topics=top_n
            )
        else:
            # Use Top2Vec's built-in methods
            topic_nums, topic_scores = self.model.get_documents_topics(
                doc_ids=None,  # Will use the documents passed to add_documents
                num_topics=top_n
            )
            
        return topic_nums, topic_scores
    
    def add_documents(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Add new documents to the trained model.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts to add
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before adding documents")
            
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
            
        # Handle custom embeddings
        if self.use_custom_embeddings and embeddings is None and self.embedding_model is not None:
            embeddings = self.embedding_model.embed_documents(documents)
            
        # Add documents to the model
        if self.use_custom_embeddings and embeddings is not None:
            self.model.add_documents(
                documents=documents,
                doc_vectors=embeddings
            )
        else:
            self.model.add_documents(documents=documents)
            
        # Update topic information
        self._extract_topic_info()
    
    def search_topics(
        self,
        query: str,
        n_topics: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """Search for topics similar to a query.
        
        Parameters
        ----------
        query : str
            Query string to search for
        n_topics : int, optional
            Number of topics to return, by default 5
            
        Returns
        -------
        List[Tuple[int, str, float]]
            List of tuples (topic_id, topic_name, score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before searching topics")
            
        # Use Top2Vec's search_topics method
        topic_nums, topic_scores, _ = self.model.search_topics(
            query=query,
            num_topics=n_topics
        )
        
        results = []
        for i, (topic_num, score) in enumerate(zip(topic_nums, topic_scores)):
            if topic_num in self.topics:
                results.append((topic_num, self.topics[topic_num], float(score)))
                
        return results
    
    def search_documents(
        self,
        query: str,
        n_docs: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search for documents similar to a query.
        
        Parameters
        ----------
        query : str
            Query string to search for
        n_docs : int, optional
            Number of documents to return, by default 10
            
        Returns
        -------
        List[Tuple[str, float]]
            List of tuples (document, score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before searching documents")
            
        # Use Top2Vec's search_documents method
        doc_ids, doc_scores = self.model.search_documents_by_keywords(
            keywords=[query],
            num_docs=n_docs
        )
        
        # Get the documents
        documents = self.model.documents[doc_ids]
        
        # Create the result list
        results = []
        for doc, score in zip(documents, doc_scores):
            results.append((doc, float(score)))
            
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Top2Vec model
        self.model.save(str(path / "top2vec_model"))
        
        # Save other attributes
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump({
                "n_topics": self.n_topics,
                "min_topic_size": self.min_topic_size,
                "use_gpu": self.use_gpu,
                "umap_args": self.umap_args,
                "hdbscan_args": self.hdbscan_args,
                "use_custom_embeddings": self.use_custom_embeddings,
                "verbose": self.verbose,
                "topics": self.topics,
                "topic_sizes": self.topic_sizes,
                "is_fitted": self.is_fitted,
            }, f)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        embedding_model: Optional[DocumentEmbedding] = None,
    ) -> "Top2VecModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
        embedding_model : Optional[DocumentEmbedding], optional
            Document embedding model to use, by default None
            
        Returns
        -------
        Top2VecModel
            Loaded model
        """
        if not TOP2VEC_AVAILABLE:
            raise ImportError(
                "Top2Vec is required for this model. "
                "Install with 'pip install top2vec>=1.0.27'"
            )
            
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            
        # Create instance with loaded parameters
        instance = cls(
            embedding_model=embedding_model,
            n_topics=metadata["n_topics"],
            min_topic_size=metadata["min_topic_size"],
            use_gpu=metadata["use_gpu"],
            umap_args=metadata["umap_args"],
            hdbscan_args=metadata["hdbscan_args"],
            use_custom_embeddings=metadata["use_custom_embeddings"],
            verbose=metadata["verbose"],
        )
        
        # Load Top2Vec model
        instance.model = Top2Vec.load(str(path / "top2vec_model"))
        
        # Load other attributes
        instance.topics = metadata["topics"]
        instance.topic_sizes = metadata["topic_sizes"]
        instance.is_fitted = metadata["is_fitted"]
        
        return instance
"""Top2Vec model implementation for topic modeling in Meno."""

from typing import List, Dict, Optional, Union, Any, Tuple, Callable, ClassVar
import numpy as np
import pandas as pd
import logging
import os
import pickle
from pathlib import Path
import warnings

from .base import BaseTopicModel
from .embeddings import DocumentEmbedding

logger = logging.getLogger(__name__)

try:
    import umap
    import hdbscan
    from sklearn.cluster import KMeans
    from top2vec import Top2Vec
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
    warnings.warn(
        "Top2Vec dependencies not installed. "
        "To use Top2VecModel, install with: pip install meno[top2vec]"
    )


class Top2VecModel(BaseTopicModel):
    """Top2Vec model for topic discovery.
    
    This class provides a wrapper around the Top2Vec algorithm for topic discovery,
    with integration into the Meno framework.
    
    Parameters
    ----------
    num_topics : int, optional
        Number of topics to discover, by default 10
        (Standardized parameter name, internally mapped to n_topics for Top2Vec)
    embedding_model : Union[str, DocumentEmbedding], optional
        Model to use for document embeddings, by default None
    umap_args : Dict[str, Any], optional
        Arguments to pass to UMAP, by default None
    hdbscan_args : Dict[str, Any], optional
        Arguments to pass to HDBSCAN, by default None
    low_memory : bool, optional
        Whether to use low memory mode, by default False
    use_gpu : bool, optional
        Whether to use GPU for embedding computation, by default False
    **kwargs : Any
        Additional arguments to pass to Top2Vec
    """
    
    # API version for compatibility checks
    API_VERSION: ClassVar[str] = "1.0.0"
    
    def __init__(
        self,
        num_topics: Optional[int] = 10,
        embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
        umap_args: Optional[Dict[str, Any]] = None,
        hdbscan_args: Optional[Dict[str, Any]] = None,
        low_memory: bool = False,
        use_gpu: bool = False,
        auto_detect_topics: bool = False,
        **kwargs
    ):
        if not HAVE_DEPS:
            raise ImportError(
                "Top2Vec dependencies not installed. "
                "To use Top2VecModel, install with: pip install meno[top2vec]"
            )
        
        # Handle automatic topic detection
        self.auto_detect_topics = auto_detect_topics
        if auto_detect_topics:
            # Top2Vec can automatically detect topics using HDBSCAN
            num_topics = None  # Set to None for automatic detection
            
            # Ensure we have appropriate HDBSCAN args for auto-detection
            if hdbscan_args is None:
                hdbscan_args = {}
            if 'min_cluster_size' not in hdbscan_args:
                hdbscan_args['min_cluster_size'] = 15  # Default for reasonable clusters
        
        # Map standardized parameter name to model-specific parameter
        n_topics = num_topics
        
        self.num_topics = num_topics  # Standardized name
        self.n_topics = n_topics      # For backward compatibility
        self.embedding_model = embedding_model
        self.umap_args = umap_args or {}
        self.hdbscan_args = hdbscan_args or {}
        self.low_memory = low_memory
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        
        # Initialize empty model and state flags
        self.model = None
        self.is_fitted = False
        self.topics = {}
        self.topic_sizes = {}
        self.document_embeddings = None
        self.document_ids = None
        
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "Top2VecModel":
        """Fit the Top2Vec model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional arguments to pass to Top2Vec
            
        Returns
        -------
        Top2VecModel
            Fitted model
        """
        # Use specific num_topics if provided in kwargs
        if 'num_topics' in kwargs:
            self.n_topics = kwargs.pop('num_topics')
        
        # Convert pandas Series to list
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        # Define Top2Vec parameters
        params = {
            "documents": documents,
            "embedding_model": self.embedding_model,
            "umap_args": self.umap_args,
            "hdbscan_args": self.hdbscan_args,
            "use_embedding_model_tokenizer": True,
            "verbose": True
        }
        
        # Update with instance kwargs and method kwargs
        params.update(self.kwargs)
        params.update(kwargs)
        
        # Use pre-computed embeddings if provided
        if embeddings is not None:
            params["document_vectors"] = embeddings
        
        # Remove n_topics parameter as it's not accepted by Top2Vec
        if 'n_topics' in params:
            del params['n_topics']
        
        # Fit Top2Vec model
        logger.info(f"Fitting Top2Vec model with parameters: {params.keys()}")
        self.model = Top2Vec(**params)
        
        # Reduce to target number of topics if needed and if n_topics is not None
        if self.n_topics is not None and hasattr(self.model, 'get_num_topics'):
            current_topics = self.model.get_num_topics()
            if current_topics > self.n_topics:
                logger.info(f"Reducing from {current_topics} to {self.n_topics} topics...")
                self.model.hierarchical_topic_reduction(self.n_topics)
        
        # Set instance attributes
        self._update_topic_info()
        self.document_embeddings = self.model.document_vectors
        self.document_ids = list(range(len(documents)))
        self.is_fitted = True
        
        return self
    
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
            Additional arguments to pass to the underlying model
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_assignments, topic_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform can be called.")
        
        # Convert pandas Series to list
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        # Use document embedding model if provided
        if embeddings is None and hasattr(self.model, 'embed_documents'):
            vectors = self.model.embed_documents(documents)
        else:
            vectors = embeddings
        
        # Get document scores
        doc_topics, doc_scores = self.model.get_documents_topics(
            doc_vectors=vectors, 
            num_topics=kwargs.get('top_n', self.n_topics)
        )
        
        # Convert to numpy arrays
        topic_assignments = np.array(doc_topics)
        topic_probabilities = np.array(doc_scores)
        
        # Ensure 2D array for probabilities (for consistency)
        if len(topic_probabilities.shape) == 1:
            topic_probabilities = topic_probabilities.reshape(-1, 1)
        
        return topic_assignments, topic_probabilities
    
    def _update_topic_info(self) -> None:
        """Update topic information from the model."""
        if not hasattr(self.model, 'topic_sizes'):
            return
        
        # Get topic sizes
        self.topic_sizes = {i: size for i, size in enumerate(self.model.topic_sizes)}
        
        # Get topic words
        word_per_topic = 10
        topic_words = self.model.get_topics(word_per_topic)
        
        # Format topic descriptions
        self.topics = {}
        for i, (words, _) in enumerate(topic_words):
            self.topics[i] = ", ".join(words[:5])
    
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
            raise ValueError("Model must be fitted before topic info can be retrieved.")
        
        # Get topic words and scores
        word_per_topic = 10
        topic_words = self.model.get_topics(word_per_topic)
        
        # Create DataFrame with standardized columns
        data = []
        for i, (words, scores) in enumerate(topic_words):
            data.append({
                'Topic': i,
                'Count': self.topic_sizes.get(i, 0),
                'Name': f"Topic {i}",
                'Representation': ", ".join(words[:5]),
                # Additional data (not part of standard API)
                'Words': words,
                'Scores': scores
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure standard columns are present and in the right order
        standard_columns = ['Topic', 'Count', 'Name', 'Representation']
        all_columns = standard_columns + [col for col in df.columns if col not in standard_columns]
        
        return df[all_columns]
    
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
            The query string to find similar topics for
        n_topics : int, optional
            Number of similar topics to return, by default 5
        **kwargs : Any
            Additional keyword arguments, not used
            
        Returns
        -------
        List[Tuple[int, str, float]]
            List of tuples (topic_id, topic_description, similarity_score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be searched.")
        
        # Find topics related to query
        topic_nums, topic_scores, topic_words = self.model.search_topics(
            query, n_topics
        )
        
        # Format the results to match BaseTopicModel's standard format
        results = []
        for i, (topic_id, score, words) in enumerate(zip(topic_nums, topic_scores, topic_words)):
            topic_desc = self.topics.get(topic_id, f"Topic {topic_id}")
            results.append((topic_id, topic_desc, float(score)))
        
        return results
        
    def search_topics(
        self,
        search_term: str,
        num_topics: int = 5
    ) -> Tuple[List[int], List[float], List[List[str]]]:
        """Search for topics related to a search term (deprecated, use find_similar_topics).
        
        Parameters
        ----------
        search_term : str
            The search term to find related topics
        num_topics : int, optional
            Number of topics to return, by default 5
            
        Returns
        -------
        Tuple[List[int], List[float], List[List[str]]]
            Tuple of (topic_ids, topic_scores, topic_words)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be searched.")
        
        # Find topics related to search term
        topic_nums, topic_scores, topic_words = self.model.search_topics(
            search_term, num_topics
        )
        
        return topic_nums, topic_scores, topic_words
    
    def add_documents(
        self,
        documents: Union[List[str], pd.Series],
        doc_ids: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """Add new documents to the model.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        doc_ids : Optional[List[Any]], optional
            List of document IDs, by default None
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before documents can be added.")
        
        # Convert pandas Series to list
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        # Generate sequential IDs if not provided
        if doc_ids is None:
            last_id = max(self.document_ids) if self.document_ids else -1
            doc_ids = [last_id + i + 1 for i in range(len(documents))]
        
        # Add documents to model
        self.model.add_documents(
            documents=documents,
            doc_ids=doc_ids,
            document_vectors=embeddings
        )
        
        # Update instance attributes
        self._update_topic_info()
        self.document_embeddings = self.model.document_vectors
        self.document_ids = self.model.doc_ids
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        # Convert to Path object
        path = Path(path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model attributes
        model_data = {
            'n_topics': self.n_topics,
            'num_topics': self.num_topics,  # Include standardized name
            'umap_args': self.umap_args,
            'hdbscan_args': self.hdbscan_args,
            'low_memory': self.low_memory,
            'use_gpu': self.use_gpu,
            'kwargs': self.kwargs,
            'is_fitted': self.is_fitted,
            'topics': self.topics,
            'topic_sizes': self.topic_sizes,
            'document_ids': self.document_ids,
            'auto_detect_topics': self.auto_detect_topics
        }
        
        # Save Top2Vec model separately
        if self.model is not None:
            model_path = str(path) + "_top2vec_model"
            # Use Top2Vec's save method
            self.model.save(model_path)
            model_data['model_path'] = model_path
        
        # Save model data
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Top2VecModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
            
        Returns
        -------
        Top2VecModel
            Loaded model
        """
        # Convert to Path object
        path = Path(path)
        
        # Load model data
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle standardized parameter names
        if 'num_topics' in model_data:
            num_topics = model_data['num_topics']
            auto_detect_topics = model_data.get('auto_detect_topics', False)
        else:
            # Backward compatibility
            num_topics = model_data['n_topics']
            auto_detect_topics = False
        
        # Create instance
        instance = cls(
            num_topics=num_topics,
            auto_detect_topics=auto_detect_topics,
            umap_args=model_data['umap_args'],
            hdbscan_args=model_data['hdbscan_args'],
            low_memory=model_data['low_memory'],
            use_gpu=model_data['use_gpu'],
            **model_data['kwargs']
        )
        
        # Load Top2Vec model
        if 'model_path' in model_data:
            from top2vec import Top2Vec
            instance.model = Top2Vec.load(model_data['model_path'])
        
        # Set instance attributes
        instance.is_fitted = model_data['is_fitted']
        instance.topics = model_data['topics']
        instance.topic_sizes = model_data['topic_sizes']
        instance.document_ids = model_data['document_ids']
        if instance.model is not None:
            instance.document_embeddings = instance.model.document_vectors
        
        return instance
    
    def visualize_topics(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "Topic Distribution",
        return_fig: bool = False,
        colorscale: str = "Viridis",
        **kwargs
    ) -> Any:
        """Visualize discovered topics.
        
        Parameters
        ----------
        width : int, optional
            Width of the plot, by default 800
        height : int, optional
            Height of the plot, by default 600
        title : str, optional
            Title of the plot, by default "Topic Distribution"
        return_fig : bool, optional
            Whether to return the figure, by default False
        colorscale : str, optional
            Colorscale to use, by default "Viridis"
        **kwargs : Any
            Additional parameters for the visualization
            
        Returns
        -------
        Any
            Visualization object if return_fig is True
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be visualized.")
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from sklearn.manifold import TSNE
            
            # Get topic word distributions
            topic_info = self.get_topic_info()
            
            # Get topic vectors
            topic_vectors = self.model.topic_vectors
            
            # Reduce dimensions with t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            topic_vectors_2d = tsne.fit_transform(topic_vectors)
            
            # Create DataFrame for visualization
            viz_df = pd.DataFrame({
                'Topic': topic_info['Topic'],
                'Size': topic_info['Count'],
                'X': topic_vectors_2d[:, 0],
                'Y': topic_vectors_2d[:, 1],
                'Description': [", ".join(words[:5]) for words in topic_info['Words']]
            })
            
            # Create figure
            fig = px.scatter(
                viz_df,
                x='X',
                y='Y',
                size='Size',
                color='Topic',
                hover_data=['Description'],
                color_continuous_scale=colorscale,
                title=title,
                width=width,
                height=height
            )
            
            # Update layout
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            if return_fig:
                return fig
            else:
                fig.show()
                
        except ImportError:
            warnings.warn(
                "Plotly not installed. To use visualize_topics, "
                "install with: pip install plotly"
            )
            return None
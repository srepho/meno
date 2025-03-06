"""Unified topic modeling API for Meno."""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import pickle
import os

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.dimensionality import UMAPReducer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    from top2vec import Top2Vec
    TOP2VEC_AVAILABLE = True
except ImportError:
    TOP2VEC_AVAILABLE = False

from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.bertopic_model import BERTopicModel
from meno.modeling.top2vec_model import Top2VecModel
from meno.modeling.bertopic_optimizer import BERTopicOptimizer

# Set up logging
logger = logging.getLogger(__name__)


class UnifiedTopicModeler:
    """Unified API for different topic modeling approaches in Meno.
    
    This class provides a common interface for using different topic modeling
    backends (BERTopic, Top2Vec, etc.) with consistent functionality.
    
    Parameters
    ----------
    method : str
        Topic modeling method to use.
        Options: "bertopic", "top2vec"
    n_topics : Optional[int], optional
        Number of topics to extract, by default None
        If None, automatic determination will be used when supported
    embedding_model : Optional[Union[str, DocumentEmbedding]], optional
        Embedding model to use, by default None
        Can be a DocumentEmbedding instance or a string name for a model
    min_topic_size : int, optional
        Minimum size of topics, by default 10
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    advanced_config : Optional[Dict[str, Any]], optional
        Advanced configuration options specific to the selected method, by default None
    optimizer_config : Optional[Dict[str, Any]], optional
        Configuration for hyperparameter optimization, by default None
        If provided, optimization will be performed when fitting the model
    verbose : bool, optional
        Whether to display verbose output, by default True
    
    Attributes
    ----------
    model : Union[BERTopicModel, Top2VecModel]
        The underlying topic model instance
    method : str
        Topic modeling method being used
    topics : Dict[int, str]
        Mapping of topic IDs to topic descriptions
    topic_sizes : Dict[int, int]
        Mapping of topic IDs to topic sizes
    is_fitted : bool
        Whether the model has been fitted
    """
    
    def __init__(
        self,
        method: str,
        n_topics: Optional[int] = None,
        embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
        min_topic_size: int = 10,
        use_gpu: bool = False,
        advanced_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """Initialize the unified topic modeler."""
        self.method = method.lower()
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.use_gpu = use_gpu
        self.advanced_config = advanced_config or {}
        self.optimizer_config = optimizer_config
        self.verbose = verbose
        
        # Set up embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = DocumentEmbedding(
                model_name=embedding_model,
                use_gpu=use_gpu
            )
        else:
            self.embedding_model = embedding_model
            
        # Initialize method-specific components
        if self.method == "bertopic":
            if not BERTOPIC_AVAILABLE:
                raise ImportError(
                    "BERTopic is required for this method. "
                    "Install with 'pip install bertopic>=0.15.0'"
                )
            
            # Set up UMAP reducer
            umap_args = self.advanced_config.get("umap", {})
            n_neighbors = umap_args.get("n_neighbors", 15)
            n_components = umap_args.get("n_components", 5)
            
            # Set up representation model
            representation_args = self.advanced_config.get("representation", {})
            representation_type = representation_args.get("type", "keybert")
            
            # Initialize BERTopicModel
            self.model = BERTopicModel(
                n_topics=n_topics,
                embedding_model=self.embedding_model,
                min_topic_size=min_topic_size,
                use_gpu=use_gpu,
                n_neighbors=n_neighbors,
                n_components=n_components,
                verbose=verbose,
            )
            
        elif self.method == "top2vec":
            if not TOP2VEC_AVAILABLE:
                raise ImportError(
                    "Top2Vec is required for this method. "
                    "Install with 'pip install top2vec>=1.0.27'"
                )
            
            # Set up UMAP and HDBSCAN arguments
            umap_args = self.advanced_config.get("umap", {})
            hdbscan_args = self.advanced_config.get("hdbscan", {
                "min_cluster_size": min_topic_size,
                "min_samples": 5,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            })
            
            # Use custom embeddings or not
            use_custom_embeddings = self.advanced_config.get("use_custom_embeddings", True)
            
            # Initialize Top2VecModel
            self.model = Top2VecModel(
                embedding_model=self.embedding_model,
                n_topics=n_topics,
                min_topic_size=min_topic_size,
                use_gpu=use_gpu,
                umap_args=umap_args,
                hdbscan_args=hdbscan_args,
                use_custom_embeddings=use_custom_embeddings,
                verbose=verbose,
            )
            
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Available methods: 'bertopic', 'top2vec'"
            )
            
        # Initialize state
        self.topics = {}
        self.topic_sizes = {}
        self.is_fitted = False
    
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> "UnifiedTopicModeler":
        """Fit the topic model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        UnifiedTopicModeler
            Fitted model instance
        """
        # If optimizer config is provided, perform hyperparameter optimization
        if self.optimizer_config is not None and self.method == "bertopic":
            if self.verbose:
                logger.info("Performing hyperparameter optimization...")
                
            # Create optimizer
            optimizer = BERTopicOptimizer(
                embedding_model=self.embedding_model.model_name if self.embedding_model else "all-MiniLM-L6-v2",
                n_trials=self.optimizer_config.get("n_trials", 10),
                random_state=self.optimizer_config.get("random_state", 42),
                metric=self.optimizer_config.get("metric", "combined"),
                verbose=self.verbose,
            )
            
            # Set custom parameter grid if provided
            if "param_grid" in self.optimizer_config:
                optimizer.set_param_grid(self.optimizer_config["param_grid"])
                
            # Optimize
            best_params, best_model, best_score = optimizer.optimize(
                documents=documents,
                search_method=self.optimizer_config.get("search_method", "random"),
            )
            
            if self.verbose:
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best score: {best_score}")
                
            # Use the optimized model directly
            self.model.model = best_model
            
            # Extract topic information
            topic_info = best_model.get_topic_info()
            
            # Store topic information
            self.topics = {}
            self.topic_sizes = {}
            
            for i, row in topic_info.iterrows():
                topic_id = row["Topic"]
                if topic_id != -1:
                    words = [word for word, _ in best_model.get_topic(topic_id)][:5]
                    self.topics[topic_id] = f"Topic {topic_id}: {', '.join(words)}"
                    self.topic_sizes[topic_id] = row["Count"]
                    
            # Add -1 for outliers
            if -1 in set(topic_info["Topic"]):
                row = topic_info[topic_info["Topic"] == -1].iloc[0]
                self.topics[-1] = "Other"
                self.topic_sizes[-1] = row["Count"]
                
            # Update state
            self.is_fitted = True
            
        else:
            # Fit the model directly
            self.model.fit(documents, embeddings)
            
            # Store topic information
            self.topics = self.model.topics
            self.topic_sizes = self.model.topic_sizes
            self.is_fitted = self.model.is_fitted
            
        return self
    
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        top_n: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform documents to topic assignments.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        top_n : int, optional
            Number of top topics to return per document, by default 1
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_ids, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform can be called")
            
        if self.method == "bertopic":
            return self.model.transform(documents, embeddings)
        elif self.method == "top2vec":
            return self.model.transform(documents, embeddings, top_n=top_n)
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        top_n: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        top_n : int, optional
            Number of top topics to return per document, by default 1
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_ids, probabilities)
        """
        self.fit(documents, embeddings)
        return self.transform(documents, embeddings, top_n)
    
    def find_similar_topics(
        self,
        query: str,
        n_topics: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """Find topics similar to a query.
        
        Parameters
        ----------
        query : str
            Query text to find similar topics for
        n_topics : int, optional
            Number of similar topics to return, by default 5
            
        Returns
        -------
        List[Tuple[int, str, float]]
            List of tuples (topic_id, topic_description, similarity_score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar topics")
            
        if self.method == "bertopic":
            return self.model.find_similar_topics(query, n_topics)
        elif self.method == "top2vec":
            return self.model.search_topics(query, n_topics)
    
    def get_topic_words(
        self,
        topic_id: int,
        n_words: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get the top words for a topic.
        
        Parameters
        ----------
        topic_id : int
            Topic ID to get words for
        n_words : int, optional
            Number of top words to return, by default 10
            
        Returns
        -------
        List[Tuple[str, float]]
            List of tuples (word, score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting topic words")
            
        if self.method == "bertopic":
            return self.model.model.get_topic(topic_id)[:n_words]
        elif self.method == "top2vec":
            # Top2Vec uses a different API
            topic_words, word_scores, _ = self.model.model.get_topics()
            if topic_id < len(topic_words):
                return [(word, score) for word, score in 
                        zip(topic_words[topic_id][:n_words], word_scores[topic_id][:n_words])]
            else:
                return []
    
    def visualize_topics(
        self,
        width: int = 800,
        height: int = 600,
    ) -> Any:
        """Visualize topics.
        
        Parameters
        ----------
        width : int, optional
            Width of the visualization, by default 800
        height : int, optional
            Height of the visualization, by default 600
            
        Returns
        -------
        Any
            Visualization object (typically a plotly Figure)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")
            
        if self.method == "bertopic":
            return self.model.visualize_topics(width, height)
        elif self.method == "top2vec":
            # Top2Vec doesn't have an equivalent method, so we create a similar visualization
            if hasattr(self.model.model, "topic_vectors") and self.model.model.topic_vectors is not None:
                from sklearn.manifold import TSNE
                import plotly.graph_objects as go
                
                # Get topic vectors
                topic_vectors = self.model.model.topic_vectors
                
                # Reduce to 2D for visualization
                tsne = TSNE(n_components=2, random_state=42)
                topic_coords = tsne.fit_transform(topic_vectors)
                
                # Create visualization
                fig = go.Figure()
                
                # Add points for each topic
                for i, (x, y) in enumerate(topic_coords):
                    if i in self.topics:
                        topic_name = self.topics[i]
                        size = self.topic_sizes.get(i, 10)
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            marker=dict(size=size / 5 + 10, opacity=0.8),
                            text=[topic_name],
                            name=topic_name,
                            textposition="bottom center",
                        ))
                
                # Update layout
                fig.update_layout(
                    title="Topic Visualization",
                    width=width,
                    height=height,
                    xaxis=dict(title="Dimension 1", showticklabels=False),
                    yaxis=dict(title="Dimension 2", showticklabels=False),
                )
                
                return fig
            else:
                logger.warning("Topic vectors not available for visualization")
                return None
    
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
            Visualization object (typically a plotly Figure)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")
            
        if self.method == "bertopic":
            return self.model.visualize_hierarchy(width, height)
        elif self.method == "top2vec":
            logger.warning("Hierarchy visualization not available for Top2Vec")
            return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the underlying model
        self.model.save(path / f"{self.method}_model")
        
        # Save the unified model metadata
        metadata = {
            "method": self.method,
            "n_topics": self.n_topics,
            "min_topic_size": self.min_topic_size,
            "use_gpu": self.use_gpu,
            "advanced_config": self.advanced_config,
            "is_fitted": self.is_fitted,
            "topics": {str(k): v for k, v in self.topics.items()},
            "topic_sizes": {str(k): v for k, v in self.topic_sizes.items()},
        }
        
        with open(path / "unified_metadata.json", "w") as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        embedding_model: Optional[DocumentEmbedding] = None,
    ) -> "UnifiedTopicModeler":
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
            Loaded model
        """
        path = Path(path)
        
        # Load metadata
        with open(path / "unified_metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Create instance
        instance = cls(
            method=metadata["method"],
            n_topics=metadata["n_topics"],
            embedding_model=embedding_model,
            min_topic_size=metadata["min_topic_size"],
            use_gpu=metadata["use_gpu"],
            advanced_config=metadata["advanced_config"],
            verbose=True,
        )
        
        # Load the underlying model
        if metadata["method"] == "bertopic":
            from meno.modeling.bertopic_model import BERTopicModel
            instance.model = BERTopicModel.load(
                path / "bertopic_model",
                embedding_model=embedding_model,
            )
        elif metadata["method"] == "top2vec":
            from meno.modeling.top2vec_model import Top2VecModel
            instance.model = Top2VecModel.load(
                path / "top2vec_model",
                embedding_model=embedding_model,
            )
            
        # Load other attributes
        instance.topics = {int(k): v for k, v in metadata["topics"].items()}
        instance.topic_sizes = {int(k): v for k, v in metadata["topic_sizes"].items()}
        instance.is_fitted = metadata["is_fitted"]
        
        return instance


# Convenience function
def create_topic_modeler(
    method: str = "bertopic",
    n_topics: Optional[int] = None,
    embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
    min_topic_size: int = 10,
    use_gpu: bool = False,
    advanced_config: Optional[Dict[str, Any]] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> UnifiedTopicModeler:
    """Create a unified topic modeler.
    
    Parameters
    ----------
    method : str, optional
        Topic modeling method to use, by default "bertopic"
        Options: "bertopic", "top2vec"
    n_topics : Optional[int], optional
        Number of topics to extract, by default None
    embedding_model : Optional[Union[str, DocumentEmbedding]], optional
        Embedding model to use, by default None
    min_topic_size : int, optional
        Minimum size of topics, by default 10
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    advanced_config : Optional[Dict[str, Any]], optional
        Advanced configuration options specific to the selected method, by default None
    optimizer_config : Optional[Dict[str, Any]], optional
        Configuration for hyperparameter optimization, by default None
    verbose : bool, optional
        Whether to display verbose output, by default True
        
    Returns
    -------
    UnifiedTopicModeler
        Initialized topic modeler
    """
    return UnifiedTopicModeler(
        method=method,
        n_topics=n_topics,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        use_gpu=use_gpu,
        advanced_config=advanced_config,
        optimizer_config=optimizer_config,
        verbose=verbose,
    )
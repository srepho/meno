"""Lightweight topic modeling implementations that don't require heavy dependencies.

These models provide alternative topic modeling approaches that rely only on
scikit-learn rather than more complex libraries like UMAP, HDBSCAN, etc.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Union, Tuple, Any, Callable, Set
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from meno.modeling.base import BaseTopicModel
from meno.modeling.embeddings import DocumentEmbedding

logger = logging.getLogger(__name__)


class SimpleTopicModel(BaseTopicModel):
    """Lightweight topic modeling using K-Means clustering on document embeddings.
    
    This model uses sentence embeddings combined with K-Means clustering to
    discover topics in text. It avoids the need for UMAP and HDBSCAN dependencies,
    making it suitable for larger datasets or environments with limited resources.
    
    Parameters
    ----------
    num_topics : int, optional
        Number of topics to extract, by default 10
    embedding_model : Optional[DocumentEmbedding], optional
        Model to use for document embeddings, by default None (creates a new instance)
    random_state : int, optional
        Random seed for reproducibility, by default 42
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        embedding_model: Optional[DocumentEmbedding] = None,
        random_state: int = 42,
        min_topic_similarity_threshold: float = 0.75,
        auto_reduce_topics: bool = True,
        **kwargs
    ):
        """Initialize the simple topic model.
        
        Parameters
        ----------
        num_topics : int, optional
            Number of topics to extract, by default 10
        embedding_model : Optional[DocumentEmbedding], optional
            Model to use for document embeddings, by default None (creates a new instance)
        random_state : int, optional
            Random seed for reproducibility, by default 42
        min_topic_similarity_threshold : float, optional
            Threshold for detecting duplicate topics (0.0-1.0), by default 0.75
            Higher values mean topics need to be more similar to be considered duplicates
        auto_reduce_topics : bool, optional
            Whether to automatically reduce the number of topics by merging similar ones, by default True
        """
        super().__init__(num_topics=num_topics, **kwargs)
        self.num_topics = num_topics
        self.embedding_model = embedding_model or DocumentEmbedding()
        self.random_state = random_state
        self.min_topic_similarity_threshold = min_topic_similarity_threshold
        self.auto_reduce_topics = auto_reduce_topics
        self.model = None
        self.vectorizer = None
        self.topics = {}
        self.topic_words = {}
        self.topic_sizes = {}
        self.is_fitted = False
        self.merged_topics = {}
        
    def fit(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "SimpleTopicModel":
        """Fit the topic model using K-Means clustering on document embeddings.
        
        Parameters
        ----------
        documents : List[str]
            List of text documents to analyze
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        SimpleTopicModel
            Fitted model instance
        """
        if len(documents) == 0:
            logger.warning("Empty document list provided. Can't fit the model.")
            return self
            
        # Compute embeddings if not provided
        if embeddings is None:
            logger.info("Computing document embeddings...")
            embeddings = self.embedding_model.embed_documents(documents)
        
        # If requested num_topics is too large for the dataset, reduce it
        actual_num_topics = min(self.num_topics, len(documents) - 1)
        if actual_num_topics < self.num_topics:
            logger.warning(f"Reducing requested topics from {self.num_topics} to {actual_num_topics} due to small dataset size")
            self.num_topics = actual_num_topics
            
        # Train KMeans
        logger.info(f"Clustering documents into {self.num_topics} topics...")
        self.model = KMeans(
            n_clusters=self.num_topics,
            random_state=self.random_state,
            n_init="auto"
        )
        self.clusters = self.model.fit_predict(embeddings)
        
        # Extract keywords for each cluster
        logger.info("Extracting topic keywords...")
        self.vectorizer = CountVectorizer(max_features=1000)
        document_term_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get topic words and sizes
        self.topic_words = {}
        self.topic_sizes = {}
        self.topics = {}
        
        # First pass: extract topic keywords and create initial topic labels
        for topic_id in range(self.num_topics):
            # Get documents in this cluster
            cluster_docs = [i for i, cluster in enumerate(self.clusters) if cluster == topic_id]
            self.topic_sizes[topic_id] = len(cluster_docs)
            
            if not cluster_docs:
                self.topics[topic_id] = f"Topic {topic_id}"
                self.topic_words[topic_id] = []
                continue
                
            # Get top terms for this cluster
            if len(cluster_docs) > 0:
                cluster_terms = document_term_matrix[cluster_docs].sum(axis=0)
                top_term_indices = cluster_terms.argsort().flatten()[-20:][::-1]
                top_terms = [feature_names[i] for i in top_term_indices]
                self.topic_words[topic_id] = top_terms
                
                # Create topic label
                if top_terms:
                    # Handle case when top_terms might be numpy array
                    if isinstance(top_terms[0], np.ndarray) or not hasattr(top_terms[0], 'title'):
                        first_term = str(top_terms[0])
                        other_terms = [str(t) for t in top_terms[1:4]]
                    else:
                        first_term = top_terms[0].title()
                        other_terms = top_terms[1:4]
                    
                    self.topics[topic_id] = f"{first_term}: {', '.join(other_terms)}"
                else:
                    self.topics[topic_id] = f"Topic {topic_id}"
            else:
                self.topics[topic_id] = f"Topic {topic_id}"
                self.topic_words[topic_id] = []
        
        # Second pass: identify and merge similar topics if auto_reduce_topics is enabled
        if self.auto_reduce_topics:
            self._merge_similar_topics()
        
        self.is_fitted = True
        logger.info("Simple topic model fitting complete.")
        return self
    
    def _merge_similar_topics(self) -> None:
        """Identify and merge similar topics to reduce topic count.
        
        This method compares topic vectors using cosine similarity to find and merge
        duplicate topics that exceed the similarity threshold.
        """
        # Skip if no topics or only one topic
        if len(self.topic_words) <= 1:
            return
        
        # Create topic term vectors from topic words
        topic_term_vectors = {}
        all_terms = set()
        
        # Collect all unique terms across topics
        for topic_id, words in self.topic_words.items():
            # Convert any numpy arrays to strings
            string_words = [str(word) for word in words]
            all_terms.update(string_words)
        
        all_terms = list(all_terms)
        term_index = {term: i for i, term in enumerate(all_terms)}
        
        # Create term vectors for each topic
        for topic_id, words in self.topic_words.items():
            vector = np.zeros(len(all_terms))
            for i, word in enumerate(words):
                # Convert word to string if it's not already
                word_str = str(word)
                # Weight by position (descending)
                weight = 1.0 - (i / len(words))
                if word_str in term_index:  # Ensure the string is in the index
                    vector[term_index[word_str]] = weight
            topic_term_vectors[topic_id] = vector
        
        # Compute similarity between topics
        topic_ids = list(topic_term_vectors.keys())
        merged_topics = {}  # Map of {merged_topic_id: primary_topic_id}
        merged_topic_ids = set()  # Topics that have been merged into others
        
        for i in range(len(topic_ids)):
            topic_i = topic_ids[i]
            
            # Skip if this topic has already been merged into another
            if topic_i in merged_topic_ids:
                continue
                
            for j in range(i + 1, len(topic_ids)):
                topic_j = topic_ids[j]
                
                # Skip if this topic has already been merged into another
                if topic_j in merged_topic_ids:
                    continue
                
                # Calculate cosine similarity
                vec_i = topic_term_vectors[topic_i]
                vec_j = topic_term_vectors[topic_j]
                
                # Handle zero vectors
                if np.all(vec_i == 0) or np.all(vec_j == 0):
                    continue
                    
                similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                
                # If similarity exceeds threshold, merge topics
                if similarity >= self.min_topic_similarity_threshold:
                    # Keep the topic with more documents
                    if self.topic_sizes.get(topic_i, 0) >= self.topic_sizes.get(topic_j, 0):
                        primary_topic, merged_topic = topic_i, topic_j
                    else:
                        primary_topic, merged_topic = topic_j, topic_i
                    
                    logger.info(f"Merging similar topics: {self.topics[merged_topic]} -> {self.topics[primary_topic]} (similarity: {similarity:.2f})")
                    
                    # Record the merge
                    merged_topics[merged_topic] = primary_topic
                    merged_topic_ids.add(merged_topic)
                    
                    # Combine topic sizes
                    self.topic_sizes[primary_topic] = self.topic_sizes.get(primary_topic, 0) + self.topic_sizes.get(merged_topic, 0)
                    
                    # Update cluster assignments
                    self.clusters = np.array([
                        primary_topic if c == merged_topic else c
                        for c in self.clusters
                    ])
        
        # Store merged topics for reference
        self.merged_topics = merged_topics
        
        # If topics were merged, update topic count
        if merged_topics:
            # Count actual topics after merging
            active_topics = set(self.topics.keys()) - merged_topic_ids
            logger.info(f"Reduced topics from {self.num_topics} to {len(active_topics)} by merging similar topics")
            
            # Create a mapping for topic IDs to make them sequential
            id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(active_topics))}
            
            # Update topic attributes with new IDs
            new_topics = {}
            new_topic_words = {}
            new_topic_sizes = {}
            
            for old_id in active_topics:
                new_id = id_map[old_id]
                new_topics[new_id] = self.topics[old_id]
                new_topic_words[new_id] = self.topic_words[old_id]
                new_topic_sizes[new_id] = self.topic_sizes[old_id]
            
            # Update instance variables
            self.topics = new_topics
            self.topic_words = new_topic_words
            self.topic_sizes = new_topic_sizes
            
            # Update num_topics attribute
            self.num_topics = len(active_topics)
            
            # Update cluster assignments with new topic IDs
            id_map_with_merged = {}
            for old_id in range(max(self.clusters) + 1):
                # If topic was merged, map to the new ID of its primary topic
                if old_id in merged_topics:
                    primary_old_id = merged_topics[old_id]
                    # The primary topic might itself have been remapped
                    id_map_with_merged[old_id] = id_map.get(primary_old_id, 0)
                else:
                    id_map_with_merged[old_id] = id_map.get(old_id, 0)
            
            # Apply the mapping to clusters
            self.clusters = np.array([id_map_with_merged.get(c, 0) for c in self.clusters])
    
    def transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform documents to topic vector representation.
        
        Parameters
        ----------
        documents : List[str]
            Documents to transform
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
            
        Returns
        -------
        np.ndarray
            Document-topic matrix of shape (n_documents, n_topics)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return np.zeros((len(documents), self.num_topics))
            
        # Compute embeddings if not provided
        if embeddings is None and documents:
            embeddings = self.embedding_model.embed_documents(documents)
            
        # Predict clusters
        if embeddings is not None and embeddings.shape[0] > 0:
            clusters = self.model.predict(embeddings)
            
            # Convert to document-topic matrix
            doc_topic = np.zeros((len(documents), self.num_topics))
            for i, cluster in enumerate(clusters):
                doc_topic[i, cluster] = 1.0
                
            return clusters, doc_topic
        else:
            empty_clusters = np.zeros(len(documents), dtype=int)
            empty_matrix = np.zeros((len(documents), self.num_topics))
            return empty_clusters, empty_matrix
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        data = []
        for topic_id, topic_label in self.topics.items():
            size = self.topic_sizes.get(topic_id, 0)
            data.append({
                "Topic": topic_id,
                "Name": topic_label,
                "Size": size,
                "Count": size,
                "Words": self.topic_words.get(topic_id, [])
            })
            
        return pd.DataFrame(data)
    
    def get_document_info(self, docs: Optional[List[str]] = None) -> pd.DataFrame:
        """Get document clustering information.
        
        Parameters
        ----------
        docs : Optional[List[str]], optional
            Documents to analyze, by default None (uses training documents)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with document-topic information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        if docs is None:
            clusters = self.clusters
        else:
            embeddings = self.embedding_model.embed_documents(docs)
            clusters = self.model.predict(embeddings)
            
        data = []
        for i, cluster in enumerate(clusters):
            data.append({
                "Document": i,
                "Topic": int(cluster),
                "Name": self.topics.get(int(cluster), f"Topic {cluster}")
            })
            
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get the top words for a given topic with their weights.
        
        Parameters
        ----------
        topic_id : int
            The ID of the topic
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (word, weight) tuples for the topic
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return []
            
        if topic_id not in self.topic_words:
            return []
            
        words = self.topic_words[topic_id]
        
        # Create dummy weights (decreasing from 1.0)
        weights = [1.0 - 0.05 * i for i in range(len(words))]
        
        return list(zip(words, weights))
    
    def visualize_topics(self, width: int = 800, height: int = 600, **kwargs) -> Any:
        """Visualize topics as a word cloud.
        
        Parameters
        ----------
        width : int, optional
            Width of the plot in pixels, by default 800
        height : int, optional
            Height of the plot in pixels, by default 600
        **kwargs : Any
            Additional arguments passed to plotting functions
            
        Returns
        -------
        Any
            Visualization object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots for each topic
        n_topics = len(self.topics)
        cols = min(3, n_topics)
        rows = (n_topics + cols - 1) // cols
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[self.topics[i] for i in range(n_topics)])
        
        for topic_id in range(n_topics):
            words = self.get_topic(topic_id)
            
            if not words:
                continue
                
            # Extract words and weights
            word_texts = [word for word, _ in words[:10]]
            word_weights = [weight for _, weight in words[:10]]
            
            # Calculate row and column for this topic
            row = topic_id // cols + 1
            col = topic_id % cols + 1
            
            # Add bar chart for this topic
            fig.add_trace(
                go.Bar(
                    x=word_texts,
                    y=word_weights,
                    name=f"Topic {topic_id}"
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Topic Keywords",
            showlegend=False,
            width=width,
            height=height
        )
        
        return fig
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        import pickle
        import os
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model attributes
        model_data = {
            "num_topics": self.num_topics,
            "random_state": self.random_state,
            "topics": self.topics,
            "topic_words": self.topic_words,
            "topic_sizes": self.topic_sizes,
            "is_fitted": self.is_fitted
        }
        
        with open(path / "model_data.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        # Save K-Means model
        if self.model is not None:
            with open(path / "kmeans_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
        
        # Save vectorizer
        if self.vectorizer is not None:
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
        
        # Save clusters
        if hasattr(self, "clusters"):
            np.save(path / "clusters.npy", self.clusters)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimpleTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
            
        Returns
        -------
        SimpleTopicModel
            Loaded model
        """
        import pickle
        
        path = Path(path)
        
        # Load model attributes
        with open(path / "model_data.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            num_topics=model_data["num_topics"],
            random_state=model_data["random_state"]
        )
        
        # Load attributes
        instance.topics = model_data["topics"]
        instance.topic_words = model_data["topic_words"]
        instance.topic_sizes = model_data["topic_sizes"]
        instance.is_fitted = model_data["is_fitted"]
        
        # Load K-Means model
        if (path / "kmeans_model.pkl").exists():
            with open(path / "kmeans_model.pkl", "rb") as f:
                instance.model = pickle.load(f)
        
        # Load vectorizer
        if (path / "vectorizer.pkl").exists():
            with open(path / "vectorizer.pkl", "rb") as f:
                instance.vectorizer = pickle.load(f)
        
        # Load clusters
        if (path / "clusters.npy").exists():
            instance.clusters = np.load(path / "clusters.npy")
        
        return instance


class TFIDFTopicModel(BaseTopicModel):
    """Extremely lightweight topic modeling using TF-IDF and clustering.
    
    This model uses TF-IDF vectorization and K-Means clustering for topic discovery,
    without requiring document embeddings at all. It's the most lightweight approach,
    suitable for very large datasets.
    
    Parameters
    ----------
    num_topics : int, optional
        Number of topics to extract, by default 10
    max_features : int, optional
        Maximum number of features to use in TF-IDF vectorization, by default 1000
    random_state : int, optional
        Random seed for reproducibility, by default 42
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        max_features: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize the TF-IDF topic model."""
        super().__init__(num_topics=num_topics, **kwargs)
        self.num_topics = num_topics
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.topics = {}
        self.topic_words = {}
        self.topic_sizes = {}
        self.is_fitted = False
        
    def fit(
        self,
        documents: List[str],
        **kwargs
    ) -> "TFIDFTopicModel":
        """Fit the topic model using TF-IDF and K-Means clustering.
        
        Parameters
        ----------
        documents : List[str]
            List of text documents to analyze
            
        Returns
        -------
        TFIDFTopicModel
            Fitted model instance
        """
        if len(documents) == 0:
            logger.warning("Empty document list provided. Can't fit the model.")
            return self
            
        # Create TF-IDF matrix
        logger.info("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Cluster documents
        logger.info(f"Clustering documents into {self.num_topics} topics...")
        self.model = KMeans(
            n_clusters=self.num_topics,
            random_state=self.random_state,
            n_init="auto"
        )
        self.clusters = self.model.fit_predict(tfidf_matrix)
        
        # Get topic words and sizes
        self.topic_words = {}
        self.topic_sizes = {}
        self.topics = {}
        
        for topic_id in range(self.num_topics):
            # Get documents in this cluster
            cluster_docs = [i for i, cluster in enumerate(self.clusters) if cluster == topic_id]
            self.topic_sizes[topic_id] = len(cluster_docs)
            
            if not cluster_docs:
                self.topics[topic_id] = f"Topic {topic_id}"
                self.topic_words[topic_id] = []
                continue
                
            # Get top terms for this cluster
            if len(cluster_docs) > 0:
                cluster_terms = tfidf_matrix[cluster_docs].sum(axis=0)
                top_term_indices = cluster_terms.argsort().flatten()[-20:][::-1]
                top_terms = [feature_names[i] for i in top_term_indices]
                self.topic_words[topic_id] = top_terms
                
                # Create topic label
                if top_terms:
                    # Handle case when top_terms might be numpy array
                    if isinstance(top_terms[0], np.ndarray) or not hasattr(top_terms[0], 'title'):
                        first_term = str(top_terms[0])
                        other_terms = [str(t) for t in top_terms[1:4]]
                    else:
                        first_term = top_terms[0].title()
                        other_terms = top_terms[1:4]
                    
                    self.topics[topic_id] = f"{first_term}: {', '.join(other_terms)}"
                else:
                    self.topics[topic_id] = f"Topic {topic_id}"
            else:
                self.topics[topic_id] = f"Topic {topic_id}"
                self.topic_words[topic_id] = []
        
        self.is_fitted = True
        logger.info("TF-IDF topic model fitting complete.")
        return self
    
    def transform(
        self,
        documents: List[str],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform documents to topic vector representation.
        
        Parameters
        ----------
        documents : List[str]
            Documents to transform
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (topic_assignments, topic_probabilities)
            - topic_assignments: 1D array of shape (n_documents,) with integer topic IDs
            - topic_probabilities: 2D array of shape (n_documents, n_topics) with probability scores
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            empty_clusters = np.zeros(len(documents), dtype=int)
            empty_matrix = np.zeros((len(documents), self.num_topics))
            return empty_clusters, empty_matrix
            
        # Transform documents to TF-IDF
        if documents:
            tfidf_matrix = self.vectorizer.transform(documents)
            
            # Predict clusters
            clusters = self.model.predict(tfidf_matrix)
            
            # Convert to document-topic matrix
            doc_topic = np.zeros((len(documents), self.num_topics))
            for i, cluster in enumerate(clusters):
                doc_topic[i, cluster] = 1.0
                
            return clusters, doc_topic
        else:
            empty_clusters = np.zeros(len(documents), dtype=int)
            empty_matrix = np.zeros((len(documents), self.num_topics))
            return empty_clusters, empty_matrix
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        data = []
        for topic_id, topic_label in self.topics.items():
            size = self.topic_sizes.get(topic_id, 0)
            data.append({
                "Topic": topic_id,
                "Name": topic_label,
                "Size": size,
                "Count": size,
                "Words": self.topic_words.get(topic_id, [])
            })
            
        return pd.DataFrame(data)
    
    def get_document_info(self, docs: Optional[List[str]] = None) -> pd.DataFrame:
        """Get document clustering information.
        
        Parameters
        ----------
        docs : Optional[List[str]], optional
            Documents to analyze, by default None (uses training documents)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with document-topic information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        if docs is None:
            clusters = self.clusters
        else:
            tfidf_matrix = self.vectorizer.transform(docs)
            clusters = self.model.predict(tfidf_matrix)
            
        data = []
        for i, cluster in enumerate(clusters):
            data.append({
                "Document": i,
                "Topic": int(cluster),
                "Name": self.topics.get(int(cluster), f"Topic {cluster}")
            })
            
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get the top words for a given topic with their weights.
        
        Parameters
        ----------
        topic_id : int
            The ID of the topic
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (word, weight) tuples for the topic
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return []
            
        if topic_id not in self.topic_words:
            return []
            
        words = self.topic_words[topic_id]
        
        # Create dummy weights (decreasing from 1.0)
        weights = [1.0 - 0.05 * i for i in range(len(words))]
        
        return list(zip(words, weights))
    
    def visualize_topics(self, width: int = 800, height: int = 600, **kwargs) -> Any:
        """Visualize topics as a word cloud.
        
        Parameters
        ----------
        width : int, optional
            Width of the plot in pixels, by default 800
        height : int, optional
            Height of the plot in pixels, by default 600
        **kwargs : Any
            Additional arguments passed to plotting functions
            
        Returns
        -------
        Any
            Visualization object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots for each topic
        n_topics = len(self.topics)
        cols = min(3, n_topics)
        rows = (n_topics + cols - 1) // cols
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[self.topics[i] for i in range(n_topics)])
        
        for topic_id in range(n_topics):
            words = self.get_topic(topic_id)
            
            if not words:
                continue
                
            # Extract words and weights
            word_texts = [word for word, _ in words[:10]]
            word_weights = [weight for _, weight in words[:10]]
            
            # Calculate row and column for this topic
            row = topic_id // cols + 1
            col = topic_id % cols + 1
            
            # Add bar chart for this topic
            fig.add_trace(
                go.Bar(
                    x=word_texts,
                    y=word_weights,
                    name=f"Topic {topic_id}"
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Topic Keywords",
            showlegend=False,
            width=width,
            height=height
        )
        
        return fig
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        import pickle
        import os
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model attributes
        model_data = {
            "num_topics": self.num_topics,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "topics": self.topics,
            "topic_words": self.topic_words,
            "topic_sizes": self.topic_sizes,
            "is_fitted": self.is_fitted
        }
        
        with open(path / "model_data.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        # Save K-Means model
        if self.model is not None:
            with open(path / "kmeans_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
        
        # Save vectorizer
        if self.vectorizer is not None:
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
        
        # Save clusters
        if hasattr(self, "clusters"):
            np.save(path / "clusters.npy", self.clusters)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TFIDFTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
            
        Returns
        -------
        TFIDFTopicModel
            Loaded model
        """
        import pickle
        
        path = Path(path)
        
        # Load model attributes
        with open(path / "model_data.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            num_topics=model_data["num_topics"],
            max_features=model_data["max_features"],
            random_state=model_data["random_state"]
        )
        
        # Load attributes
        instance.topics = model_data["topics"]
        instance.topic_words = model_data["topic_words"]
        instance.topic_sizes = model_data["topic_sizes"]
        instance.is_fitted = model_data["is_fitted"]
        
        # Load K-Means model
        if (path / "kmeans_model.pkl").exists():
            with open(path / "kmeans_model.pkl", "rb") as f:
                instance.model = pickle.load(f)
        
        # Load vectorizer
        if (path / "vectorizer.pkl").exists():
            with open(path / "vectorizer.pkl", "rb") as f:
                instance.vectorizer = pickle.load(f)
        
        # Load clusters
        if (path / "clusters.npy").exists():
            instance.clusters = np.load(path / "clusters.npy")
        
        return instance


class NMFTopicModel(BaseTopicModel):
    """Topic modeling using Non-negative Matrix Factorization (NMF).
    
    This model uses TF-IDF vectorization with NMF decomposition to discover topics.
    It's similar to classical LDA but uses a different algorithm that can be faster
    and produce more coherent topics in many cases.
    
    Parameters
    ----------
    num_topics : int, optional
        Number of topics to extract, by default 10
    max_features : int, optional
        Maximum number of features to use in vectorization, by default 1000
    random_state : int, optional
        Random seed for reproducibility, by default 42
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        max_features: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize the NMF topic model."""
        super().__init__(num_topics=num_topics, **kwargs)
        self.num_topics = num_topics
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.topics = {}
        self.topic_words = {}
        self.topic_sizes = {}
        self.is_fitted = False
        
    def fit(
        self,
        documents: List[str],
        **kwargs
    ) -> "NMFTopicModel":
        """Fit the topic model using NMF on TF-IDF matrix.
        
        Parameters
        ----------
        documents : List[str]
            List of text documents to analyze
            
        Returns
        -------
        NMFTopicModel
            Fitted model instance
        """
        if len(documents) == 0:
            logger.warning("Empty document list provided. Can't fit the model.")
            return self
            
        # Create TF-IDF matrix
        logger.info("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply NMF
        logger.info(f"Extracting {self.num_topics} topics using NMF...")
        self.model = NMF(
            n_components=self.num_topics,
            random_state=self.random_state
        )
        self.doc_topic_matrix = self.model.fit_transform(tfidf_matrix)
        self.topic_word_matrix = self.model.components_
        
        # Get topic words
        self.topic_words = {}
        self.topics = {}
        self.topic_sizes = {}
        
        # Assign documents to primary topics
        dominant_topics = np.argmax(self.doc_topic_matrix, axis=1)
        for topic_id in range(self.num_topics):
            # Count documents primarily in this topic
            self.topic_sizes[topic_id] = np.sum(dominant_topics == topic_id)
            
            # Get top terms for this topic
            top_term_indices = self.topic_word_matrix[topic_id].argsort()[-20:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            self.topic_words[topic_id] = top_terms
            
            # Create topic label
            if top_terms:
                # Handle case when top_terms might be numpy array
                if isinstance(top_terms[0], np.ndarray) or not hasattr(top_terms[0], 'title'):
                    first_term = str(top_terms[0])
                    other_terms = [str(t) for t in top_terms[1:4]]
                else:
                    first_term = top_terms[0].title()
                    other_terms = top_terms[1:4]
                
                self.topics[topic_id] = f"{first_term}: {', '.join(other_terms)}"
            else:
                self.topics[topic_id] = f"Topic {topic_id}"
        
        self.is_fitted = True
        logger.info("NMF topic model fitting complete.")
        return self
    
    def transform(
        self,
        documents: List[str],
        **kwargs
    ) -> np.ndarray:
        """Transform documents to topic vector representation.
        
        Parameters
        ----------
        documents : List[str]
            Documents to transform
            
        Returns
        -------
        np.ndarray
            Document-topic matrix
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return np.zeros((len(documents), self.num_topics))
            
        # Transform documents to TF-IDF and then to topic space
        if documents:
            tfidf_matrix = self.vectorizer.transform(documents)
            return self.model.transform(tfidf_matrix)
        else:
            return np.zeros((len(documents), self.num_topics))
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        data = []
        for topic_id, topic_label in self.topics.items():
            size = self.topic_sizes.get(topic_id, 0)
            data.append({
                "Topic": topic_id,
                "Name": topic_label,
                "Size": size,
                "Count": size,
                "Words": self.topic_words.get(topic_id, [])
            })
            
        return pd.DataFrame(data)
    
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
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        if docs is None:
            doc_topic_matrix = self.doc_topic_matrix
        else:
            doc_topic_matrix = self.transform(docs)
            
        # Get dominant topic for each document
        dominant_topics = np.argmax(doc_topic_matrix, axis=1)
        
        data = []
        for i, topic_id in enumerate(dominant_topics):
            data.append({
                "Document": i,
                "Topic": int(topic_id),
                "Name": self.topics.get(int(topic_id), f"Topic {topic_id}"),
                "Weight": doc_topic_matrix[i, topic_id]
            })
            
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get the top words for a given topic with their weights.
        
        Parameters
        ----------
        topic_id : int
            The ID of the topic
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (word, weight) tuples for the topic
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return []
            
        if not (0 <= topic_id < self.num_topics):
            return []
            
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top terms and weights for this topic
        top_indices = self.topic_word_matrix[topic_id].argsort()[-20:][::-1]
        top_words = [(feature_names[i], self.topic_word_matrix[topic_id, i]) for i in top_indices]
        
        return top_words
    
    def visualize_topics(self, width: int = 800, height: int = 600, **kwargs) -> Any:
        """Visualize topics as a bar chart.
        
        Parameters
        ----------
        width : int, optional
            Width of the plot in pixels, by default 800
        height : int, optional
            Height of the plot in pixels, by default 600
        **kwargs : Any
            Additional arguments passed to plotting functions
            
        Returns
        -------
        Any
            Visualization object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots for each topic
        n_topics = min(self.num_topics, 9)  # Limit to 9 topics for readability
        cols = min(3, n_topics)
        rows = (min(n_topics, 9) + cols - 1) // cols
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[self.topics[i] for i in range(n_topics)])
        
        for topic_id in range(n_topics):
            words = self.get_topic(topic_id)
            
            if not words:
                continue
                
            # Extract words and weights
            word_texts = [word for word, _ in words[:10]]
            word_weights = [weight for _, weight in words[:10]]
            
            # Calculate row and column for this topic
            row = topic_id // cols + 1
            col = topic_id % cols + 1
            
            # Add bar chart for this topic
            fig.add_trace(
                go.Bar(
                    x=word_texts,
                    y=word_weights,
                    name=f"Topic {topic_id}"
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Top Topic Terms (NMF)",
            showlegend=False,
            width=width,
            height=height
        )
        
        return fig
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        import pickle
        import os
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model attributes
        model_data = {
            "num_topics": self.num_topics,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "topics": self.topics,
            "topic_words": self.topic_words,
            "topic_sizes": self.topic_sizes,
            "is_fitted": self.is_fitted
        }
        
        with open(path / "model_data.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        # Save NMF model
        if self.model is not None:
            with open(path / "nmf_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
        
        # Save vectorizer
        if self.vectorizer is not None:
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
        
        # Save document-topic matrix
        if hasattr(self, "doc_topic_matrix"):
            np.save(path / "doc_topic_matrix.npy", self.doc_topic_matrix)
        
        # Save topic-word matrix
        if hasattr(self, "topic_word_matrix"):
            np.save(path / "topic_word_matrix.npy", self.topic_word_matrix)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "NMFTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
            
        Returns
        -------
        NMFTopicModel
            Loaded model
        """
        import pickle
        
        path = Path(path)
        
        # Load model attributes
        with open(path / "model_data.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            num_topics=model_data["num_topics"],
            max_features=model_data["max_features"],
            random_state=model_data["random_state"]
        )
        
        # Load attributes
        instance.topics = model_data["topics"]
        instance.topic_words = model_data["topic_words"]
        instance.topic_sizes = model_data["topic_sizes"]
        instance.is_fitted = model_data["is_fitted"]
        
        # Load NMF model
        if (path / "nmf_model.pkl").exists():
            with open(path / "nmf_model.pkl", "rb") as f:
                instance.model = pickle.load(f)
        
        # Load vectorizer
        if (path / "vectorizer.pkl").exists():
            with open(path / "vectorizer.pkl", "rb") as f:
                instance.vectorizer = pickle.load(f)
        
        # Load document-topic matrix
        if (path / "doc_topic_matrix.npy").exists():
            instance.doc_topic_matrix = np.load(path / "doc_topic_matrix.npy")
        
        # Load topic-word matrix
        if (path / "topic_word_matrix.npy").exists():
            instance.topic_word_matrix = np.load(path / "topic_word_matrix.npy")
        
        return instance


class LSATopicModel(BaseTopicModel):
    """Topic modeling using Latent Semantic Analysis (LSA/LSI).
    
    This model uses TF-IDF vectorization with truncated SVD (aka LSA) to discover topics.
    It's particularly good for capturing the semantic structure of text documents and
    is very fast compared to probabilistic models.
    
    Parameters
    ----------
    num_topics : int, optional
        Number of topics to extract, by default 10
    max_features : int, optional
        Maximum number of features to use in vectorization, by default 1000
    random_state : int, optional
        Random seed for reproducibility, by default 42
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        max_features: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize the LSA topic model."""
        super().__init__(num_topics=num_topics, **kwargs)
        self.num_topics = num_topics
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.topics = {}
        self.topic_words = {}
        self.topic_sizes = {}
        self.is_fitted = False
        
    def fit(
        self,
        documents: List[str],
        **kwargs
    ) -> "LSATopicModel":
        """Fit the topic model using LSA on TF-IDF matrix.
        
        Parameters
        ----------
        documents : List[str]
            List of text documents to analyze
            
        Returns
        -------
        LSATopicModel
            Fitted model instance
        """
        if len(documents) == 0:
            logger.warning("Empty document list provided. Can't fit the model.")
            return self
            
        # Create TF-IDF matrix
        logger.info("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply LSA (TruncatedSVD)
        logger.info(f"Extracting {self.num_topics} topics using LSA...")
        self.model = TruncatedSVD(
            n_components=self.num_topics,
            random_state=self.random_state
        )
        self.doc_topic_matrix = self.model.fit_transform(tfidf_matrix)
        self.topic_word_matrix = self.model.components_
        
        # Get topic words
        self.topic_words = {}
        self.topics = {}
        self.topic_sizes = {}
        
        # Assign documents to primary topics
        # For LSA, we take the absolute largest component since they can be negative
        dominant_topics = np.argmax(np.abs(self.doc_topic_matrix), axis=1)
        for topic_id in range(self.num_topics):
            # Count documents primarily in this topic
            self.topic_sizes[topic_id] = np.sum(dominant_topics == topic_id)
            
            # Get top terms for this topic by absolute coefficient value
            # First get absolute values of coefficients
            abs_coefficients = np.abs(self.topic_word_matrix[topic_id])
            # Get indices of top terms by absolute value
            top_term_indices = abs_coefficients.argsort()[-20:][::-1]
            # Get the actual terms
            top_terms = [feature_names[i] for i in top_term_indices]
            self.topic_words[topic_id] = top_terms
            
            # Create topic label
            if top_terms:
                self.topics[topic_id] = f"{top_terms[0].title()}: {', '.join(top_terms[1:4])}"
            else:
                self.topics[topic_id] = f"Topic {topic_id}"
        
        self.is_fitted = True
        logger.info("LSA topic model fitting complete.")
        return self
    
    def transform(
        self,
        documents: List[str],
        **kwargs
    ) -> np.ndarray:
        """Transform documents to topic vector representation.
        
        Parameters
        ----------
        documents : List[str]
            Documents to transform
            
        Returns
        -------
        np.ndarray
            Document-topic matrix
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return np.zeros((len(documents), self.num_topics))
            
        # Transform documents to TF-IDF and then to topic space
        if documents:
            tfidf_matrix = self.vectorizer.transform(documents)
            return self.model.transform(tfidf_matrix)
        else:
            return np.zeros((len(documents), self.num_topics))
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        data = []
        for topic_id, topic_label in self.topics.items():
            size = self.topic_sizes.get(topic_id, 0)
            data.append({
                "Topic": topic_id,
                "Name": topic_label,
                "Size": size,
                "Count": size,
                "Words": self.topic_words.get(topic_id, [])
            })
            
        return pd.DataFrame(data)
    
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
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return pd.DataFrame()
            
        if docs is None:
            doc_topic_matrix = self.doc_topic_matrix
        else:
            doc_topic_matrix = self.transform(docs)
            
        # Get dominant topic for each document (by absolute value for LSA)
        dominant_topics = np.argmax(np.abs(doc_topic_matrix), axis=1)
        
        data = []
        for i, topic_id in enumerate(dominant_topics):
            data.append({
                "Document": i,
                "Topic": int(topic_id),
                "Name": self.topics.get(int(topic_id), f"Topic {topic_id}"),
                "Weight": abs(doc_topic_matrix[i, topic_id])
            })
            
        return pd.DataFrame(data)
    
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get the top words for a given topic with their weights.
        
        Parameters
        ----------
        topic_id : int
            The ID of the topic
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (word, weight) tuples for the topic
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return []
            
        if not (0 <= topic_id < self.num_topics):
            return []
            
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top terms by absolute value
        abs_coefficients = np.abs(self.topic_word_matrix[topic_id])
        top_indices = abs_coefficients.argsort()[-20:][::-1]
        
        # Get words and original (not absolute) weights
        top_words = [(feature_names[i], self.topic_word_matrix[topic_id, i]) for i in top_indices]
        
        return top_words
    
    def visualize_topics(self, width: int = 800, height: int = 600, **kwargs) -> Any:
        """Visualize topics as a bar chart.
        
        Parameters
        ----------
        width : int, optional
            Width of the plot in pixels, by default 800
        height : int, optional
            Height of the plot in pixels, by default 600
        **kwargs : Any
            Additional arguments passed to plotting functions
            
        Returns
        -------
        Any
            Visualization object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots for each topic
        n_topics = min(self.num_topics, 9)  # Limit to 9 topics for readability
        cols = min(3, n_topics)
        rows = (min(n_topics, 9) + cols - 1) // cols
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[self.topics[i] for i in range(n_topics)])
        
        for topic_id in range(n_topics):
            words = self.get_topic(topic_id)
            
            if not words:
                continue
                
            # Extract words and weights
            word_texts = [word for word, _ in words[:10]]
            word_weights = [abs(weight) for _, weight in words[:10]]  # Use absolute weights for visualization
            
            # Calculate row and column for this topic
            row = topic_id // cols + 1
            col = topic_id % cols + 1
            
            # Add bar chart for this topic
            fig.add_trace(
                go.Bar(
                    x=word_texts,
                    y=word_weights,
                    name=f"Topic {topic_id}"
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Top Topic Terms (LSA)",
            showlegend=False,
            width=width,
            height=height
        )
        
        return fig
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model to
        """
        import pickle
        import os
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model attributes
        model_data = {
            "num_topics": self.num_topics,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "topics": self.topics,
            "topic_words": self.topic_words,
            "topic_sizes": self.topic_sizes,
            "is_fitted": self.is_fitted
        }
        
        with open(path / "model_data.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        # Save LSA model
        if self.model is not None:
            with open(path / "lsa_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
        
        # Save vectorizer
        if self.vectorizer is not None:
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
        
        # Save document-topic matrix
        if hasattr(self, "doc_topic_matrix"):
            np.save(path / "doc_topic_matrix.npy", self.doc_topic_matrix)
        
        # Save topic-word matrix
        if hasattr(self, "topic_word_matrix"):
            np.save(path / "topic_word_matrix.npy", self.topic_word_matrix)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSATopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
            
        Returns
        -------
        LSATopicModel
            Loaded model
        """
        import pickle
        
        path = Path(path)
        
        # Load model attributes
        with open(path / "model_data.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            num_topics=model_data["num_topics"],
            max_features=model_data["max_features"],
            random_state=model_data["random_state"]
        )
        
        # Load attributes
        instance.topics = model_data["topics"]
        instance.topic_words = model_data["topic_words"]
        instance.topic_sizes = model_data["topic_sizes"]
        instance.is_fitted = model_data["is_fitted"]
        
        # Load LSA model
        if (path / "lsa_model.pkl").exists():
            with open(path / "lsa_model.pkl", "rb") as f:
                instance.model = pickle.load(f)
        
        # Load vectorizer
        if (path / "vectorizer.pkl").exists():
            with open(path / "vectorizer.pkl", "rb") as f:
                instance.vectorizer = pickle.load(f)
        
        # Load document-topic matrix
        if (path / "doc_topic_matrix.npy").exists():
            instance.doc_topic_matrix = np.load(path / "doc_topic_matrix.npy")
        
        # Load topic-word matrix
        if (path / "topic_word_matrix.npy").exists():
            instance.topic_word_matrix = np.load(path / "topic_word_matrix.npy")
        
        return instance
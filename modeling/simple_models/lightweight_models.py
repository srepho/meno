"""Lightweight topic modeling implementations that don't require heavy dependencies.

These models provide alternative topic modeling approaches that rely only on
scikit-learn rather than more complex libraries like UMAP, HDBSCAN, etc.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD

from ..base import BaseTopicModel
from ..embeddings import DocumentEmbedding

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
        **kwargs
    ):
        """Initialize the simple topic model."""
        self.num_topics = num_topics
        self.embedding_model = embedding_model or DocumentEmbedding()
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
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ):
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
                    self.topics[topic_id] = f"{top_terms[0].title()}: {', '.join(top_terms[1:4])}"
                else:
                    self.topics[topic_id] = f"Topic {topic_id}"
            else:
                self.topics[topic_id] = f"Topic {topic_id}"
                self.topic_words[topic_id] = []
        
        self.is_fitted = True
        logger.info("Simple topic model fitting complete.")
        return self
    
    def transform(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
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
                
            return doc_topic
        else:
            return np.zeros((len(documents), self.num_topics))
    
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
            Document-topic matrix of shape (n_documents, n_topics)
        """
        return self.fit(documents, embeddings, **kwargs).transform(documents, embeddings)
    
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
    
    def get_topics(self) -> Dict[int, List[str]]:
        """Get topic word lists.
        
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of words
        """
        return self.topic_words
    
    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping topic IDs to label strings
        """
        return self.topics


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
    ):
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
                    self.topics[topic_id] = f"{top_terms[0].title()}: {', '.join(top_terms[1:4])}"
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
    ) -> np.ndarray:
        """Transform documents to topic vector representation.
        
        Parameters
        ----------
        documents : List[str]
            Documents to transform
            
        Returns
        -------
        np.ndarray
            Document-topic matrix of shape (n_documents, n_topics)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Please fit the model first.")
            return np.zeros((len(documents), self.num_topics))
            
        # Transform documents to TF-IDF
        if documents:
            tfidf_matrix = self.vectorizer.transform(documents)
            
            # Predict clusters
            clusters = self.model.predict(tfidf_matrix)
            
            # Convert to document-topic matrix
            doc_topic = np.zeros((len(documents), self.num_topics))
            for i, cluster in enumerate(clusters):
                doc_topic[i, cluster] = 1.0
                
            return doc_topic
        else:
            return np.zeros((len(documents), self.num_topics))
    
    def fit_transform(
        self,
        documents: List[str],
        **kwargs
    ) -> np.ndarray:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : List[str]
            Documents to analyze
            
        Returns
        -------
        np.ndarray
            Document-topic matrix of shape (n_documents, n_topics)
        """
        return self.fit(documents, **kwargs).transform(documents)
    
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
    
    def get_topics(self) -> Dict[int, List[str]]:
        """Get topic word lists.
        
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of words
        """
        return self.topic_words
    
    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping topic IDs to label strings
        """
        return self.topics


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
    ):
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
                self.topics[topic_id] = f"{top_terms[0].title()}: {', '.join(top_terms[1:4])}"
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
    
    def fit_transform(
        self,
        documents: List[str],
        **kwargs
    ) -> np.ndarray:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : List[str]
            Documents to analyze
            
        Returns
        -------
        np.ndarray
            Document-topic matrix
        """
        self.fit(documents, **kwargs)
        return self.doc_topic_matrix
    
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
    
    def get_topics(self) -> Dict[int, List[str]]:
        """Get topic word lists.
        
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of words
        """
        return self.topic_words
    
    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping topic IDs to label strings
        """
        return self.topics


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
    ):
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
    
    def fit_transform(
        self,
        documents: List[str],
        **kwargs
    ) -> np.ndarray:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : List[str]
            Documents to analyze
            
        Returns
        -------
        np.ndarray
            Document-topic matrix
        """
        self.fit(documents, **kwargs)
        return self.doc_topic_matrix
    
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
    
    def get_topics(self) -> Dict[int, List[str]]:
        """Get topic word lists.
        
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of words
        """
        return self.topic_words
    
    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping topic IDs to label strings
        """
        return self.topics
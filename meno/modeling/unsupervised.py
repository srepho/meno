"""Unsupervised topic modeling modules."""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import hdbscan
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from collections import defaultdict, Counter


class EmbeddingClusterModel:
    """Cluster document embeddings for unsupervised topic modeling.
    
    This class provides methods for clustering document embeddings to discover
    topics without supervision. It supports multiple clustering algorithms.
    
    Parameters
    ----------
    algorithm : str, optional
        Clustering algorithm to use, by default "hdbscan"
        Options: "hdbscan", "kmeans", "agglomerative"
    n_clusters : Optional[int], optional
        Number of clusters/topics to use, by default None
        Required for "kmeans" and "agglomerative", ignored for "hdbscan"
    min_cluster_size : int, optional
        Minimum cluster size for HDBSCAN, by default 15
    min_samples : int, optional
        Minimum samples parameter for HDBSCAN, by default 5
    cluster_selection_method : str, optional
        Cluster selection method for HDBSCAN, by default "eom"
        Options: "eom", "leaf"
    
    Attributes
    ----------
    algorithm : str
        Selected clustering algorithm
    n_clusters : Optional[int]
        Number of clusters/topics to use
    model : Any
        Fitted clustering model
    topic_words : Dict[int, List[str]]
        Top words for each topic (if available)
    """
    
    def __init__(
        self,
        algorithm: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        cluster_selection_method: str = "eom",
    ):
        """Initialize the embedding cluster model with specified options."""
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize model as None
        self.model = None
        self.topic_words = {}
        self.embeddings = None
    
    def _validate_parameters(self) -> None:
        """Validate model parameters.
        
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        valid_algorithms = ["hdbscan", "kmeans", "agglomerative"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. "
                f"Must be one of: {', '.join(valid_algorithms)}"
            )
        
        if self.algorithm in ["kmeans", "agglomerative"] and self.n_clusters is None:
            raise ValueError(
                f"n_clusters must be specified for {self.algorithm} algorithm"
            )
        
        if self.algorithm == "hdbscan":
            valid_selection_methods = ["eom", "leaf"]
            if self.cluster_selection_method not in valid_selection_methods:
                raise ValueError(
                    f"Invalid cluster_selection_method: {self.cluster_selection_method}. "
                    f"Must be one of: {', '.join(valid_selection_methods)}"
                )
    
    def _create_model(self) -> Any:
        """Create the appropriate clustering model based on algorithm.
        
        Returns
        -------
        Any
            Clustering model instance
        """
        if self.algorithm == "hdbscan":
            return hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_method=self.cluster_selection_method,
                metric="euclidean",
                core_dist_n_jobs=-1,  # Use all CPU cores
            )
        elif self.algorithm == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
        elif self.algorithm == "agglomerative":
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                affinity="euclidean",
                linkage="ward",
            )
    
    def fit(self, embeddings: np.ndarray) -> "EmbeddingClusterModel":
        """Fit the clustering model to document embeddings.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings with shape (n_documents, embedding_dim)
        
        Returns
        -------
        EmbeddingClusterModel
            Fitted model instance
        """
        # Save embeddings for later use
        self.embeddings = embeddings
        
        # Create model
        self.model = self._create_model()
        
        # Fit model
        if self.algorithm in ["kmeans", "hdbscan"]:
            self.model.fit(embeddings)
        else:  # agglomerative
            self.model.fit(embeddings)
        
        return self
    
    def transform(self, embeddings: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Assign cluster/topic labels to embeddings.
        
        Parameters
        ----------
        embeddings : Optional[np.ndarray], optional
            Document embeddings to cluster, by default None
            If None, uses the embeddings from fit
        
        Returns
        -------
        pd.DataFrame
            DataFrame with cluster/topic assignments and probabilities
        
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before transform")
        
        # Use embeddings from fit if not provided
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings provided and none saved from fit")
            embeddings = self.embeddings
        
        # Get cluster labels
        if self.algorithm == "hdbscan":
            labels = self.model.labels_
            if hasattr(self.model, "probabilities_"):
                probabilities = self.model.probabilities_
            else:
                probabilities = None
        elif self.algorithm == "kmeans":
            labels = self.model.predict(embeddings)
            # Calculate distance to cluster centers for probability
            distances = np.zeros((embeddings.shape[0], self.n_clusters))
            for i in range(self.n_clusters):
                center = self.model.cluster_centers_[i]
                # Euclidean distance
                distances[:, i] = np.sqrt(np.sum((embeddings - center) ** 2, axis=1))
            # Convert to probabilities
            probabilities = 1 / (1 + distances)
            # Normalize probabilities
            row_sums = probabilities.sum(axis=1)[:, np.newaxis]
            probabilities = probabilities / row_sums
        else:  # agglomerative
            labels = self.model.fit_predict(embeddings)
            probabilities = None
        
        # Create result DataFrame
        result = pd.DataFrame({"topic": labels})
        
        # Add probabilities if available
        if probabilities is not None:
            # For HDBSCAN, just one probability value per document
            if self.algorithm == "hdbscan":
                result["topic_probability"] = probabilities
            # For KMeans, add a column for each cluster probability
            elif self.algorithm == "kmeans":
                for i in range(self.n_clusters):
                    result[f"topic_{i}_probability"] = probabilities[:, i]
        
        # Convert -1 labels (noise in HDBSCAN) to a string for consistency
        if self.algorithm == "hdbscan":
            result["topic"] = result["topic"].apply(
                lambda x: f"Topic_{x}" if x >= 0 else "Noise"
            )
        else:
            result["topic"] = result["topic"].apply(lambda x: f"Topic_{x}")
        
        return result
    
    def fit_transform(self, embeddings: np.ndarray) -> pd.DataFrame:
        """Fit the model and transform the embeddings in one step.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings to cluster
        
        Returns
        -------
        pd.DataFrame
            DataFrame with cluster/topic assignments and probabilities
        """
        self.fit(embeddings)
        return self.transform()
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get the cluster centers.
        
        Returns
        -------
        Optional[np.ndarray]
            Cluster centers with shape (n_clusters, embedding_dim)
            or None if not available
        """
        if self.model is None:
            return None
        
        if self.algorithm == "kmeans":
            return self.model.cluster_centers_
        elif self.algorithm == "hdbscan" and self.embeddings is not None:
            # Calculate cluster centers for HDBSCAN
            labels = self.model.labels_
            centers = []
            for i in range(max(labels) + 1):  # Exclude noise cluster (-1)
                cluster_embeds = self.embeddings[labels == i]
                if len(cluster_embeds) > 0:
                    centers.append(np.mean(cluster_embeds, axis=0))
            if centers:
                return np.vstack(centers)
        
        return None
    
    def get_topics_df(self) -> pd.DataFrame:
        """Get a DataFrame with topic information.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting topics")
        
        # For HDBSCAN
        if self.algorithm == "hdbscan":
            labels = self.model.labels_
            n_clusters = max(labels) + 1
            
            # Count documents per topic
            topic_counts = pd.Series(labels).value_counts().sort_index()
            
            # Remove noise cluster if present
            if -1 in topic_counts.index:
                noise_count = topic_counts[-1]
                topic_counts = topic_counts[topic_counts.index >= 0]
            else:
                noise_count = 0
            
            # Create topic DataFrame
            topics_df = pd.DataFrame({
                "topic": [f"Topic_{i}" for i in range(n_clusters)],
                "document_count": topic_counts.values,
            })
            
            # Add noise cluster
            if noise_count > 0:
                noise_df = pd.DataFrame({
                    "topic": ["Noise"],
                    "document_count": [noise_count],
                })
                topics_df = pd.concat([topics_df, noise_df], ignore_index=True)
        
        # For KMeans and Agglomerative
        else:
            counts = defaultdict(int)
            for label in self.transform()["topic"]:
                counts[label] += 1
            
            topics_df = pd.DataFrame({
                "topic": list(counts.keys()),
                "document_count": list(counts.values()),
            })
        
        return topics_df.sort_values("document_count", ascending=False)


class LDAModel:
    """Latent Dirichlet Allocation for topic modeling.
    
    This class provides a wrapper around gensim's LDA model for topic modeling.
    
    Parameters
    ----------
    num_topics : int, optional
        Number of topics to extract, by default 10
    passes : int, optional
        Number of passes through the corpus during training, by default 20
    iterations : int, optional
        Maximum number of iterations through the corpus, by default 400
    alpha : Union[str, float], optional
        Prior on document-topic distribution, by default "auto"
        Options: "auto", "symmetric", "asymmetric", or a float
    eta : Union[str, float], optional
        Prior on topic-word distribution, by default "auto"
        Options: "auto", "symmetric", or a float
    
    Attributes
    ----------
    num_topics : int
        Number of topics to extract
    dictionary : gensim.corpora.Dictionary
        Dictionary mapping words to IDs
    corpus : List
        Bag-of-words corpus for LDA
    model : gensim.models.LdaModel
        Trained LDA model
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 20,
        iterations: int = 400,
        alpha: Union[str, float] = "auto",
        eta: Union[str, float] = "auto",
    ):
        """Initialize the LDA model with specified options."""
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.alpha = alpha
        self.eta = eta
        
        # Initialize model components as None
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.topic_words = None
    
    def _preprocess_for_lda(self, texts: List[str]) -> Tuple[Dictionary, List]:
        """Preprocess text data for LDA.
        
        Parameters
        ----------
        texts : List[str]
            Preprocessed texts (already tokenized by space)
        
        Returns
        -------
        Tuple[Dictionary, List]
            Dictionary and bag-of-words corpus
        """
        # Tokenize (split by whitespace since texts should already be preprocessed)
        tokenized_texts = [text.split() for text in texts]
        
        # Create dictionary
        dictionary = Dictionary(tokenized_texts)
        
        # Filter out extreme values
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create bag-of-words corpus
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        return dictionary, corpus
    
    def fit(self, texts: Union[List[str], pd.Series]) -> "LDAModel":
        """Fit the LDA model to text data.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Preprocessed text data
        
        Returns
        -------
        LDAModel
            Fitted model instance
        """
        # Convert pandas Series to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Preprocess for LDA
        self.dictionary, self.corpus = self._preprocess_for_lda(texts)
        
        # Train LDA model
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            alpha=self.alpha,
            eta=self.eta,
            random_state=42,
        )
        
        # Extract top words for each topic
        self.topic_words = {}
        for topic_id in range(self.num_topics):
            words = self.model.show_topic(topic_id, topn=20)
            self.topic_words[topic_id] = [(word, prob) for word, prob in words]
        
        return self
    
    def transform(self, texts: Optional[Union[List[str], pd.Series]] = None) -> pd.DataFrame:
        """Transform texts to topic distributions.
        
        Parameters
        ----------
        texts : Optional[Union[List[str], pd.Series]], optional
            Preprocessed text data, by default None
            If None, uses the texts from fit
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic distributions for each document
        
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model must be fitted before transform")
        
        # Use corpus from fit if texts not provided
        if texts is None:
            if self.corpus is None:
                raise ValueError("No texts provided and none saved from fit")
            corpus = self.corpus
        else:
            # Convert pandas Series to list if needed
            if isinstance(texts, pd.Series):
                texts = texts.tolist()
            
            # Tokenize and convert to bag-of-words
            tokenized_texts = [text.split() for text in texts]
            corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Transform to topic distributions
        topic_distributions = []
        for doc in corpus:
            doc_topics = self.model.get_document_topics(doc, minimum_probability=0)
            # Convert to a full vector of topic probabilities
            topics_vec = [0] * self.num_topics
            for topic_id, prob in doc_topics:
                topics_vec[topic_id] = prob
            topic_distributions.append(topics_vec)
        
        # Create DataFrame with topic distributions
        df = pd.DataFrame(
            topic_distributions,
            columns=[f"Topic_{i}" for i in range(self.num_topics)]
        )
        
        # Add dominant topic column
        df["topic"] = df.idxmax(axis=1)
        
        # Add dominant topic probability
        df["topic_probability"] = df.max(axis=1)
        
        return df
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """Fit the model and transform the texts in one step.
        
        Parameters
        ----------
        texts : Union[List[str], pd.Series]
            Preprocessed text data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic distributions for each document
        """
        self.fit(texts)
        return self.transform()
    
    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top words for a specific topic.
        
        Parameters
        ----------
        topic_id : int
            Topic ID
        top_n : int, optional
            Number of top words to return, by default 10
        
        Returns
        -------
        List[Tuple[str, float]]
            List of (word, probability) tuples
        
        Raises
        ------
        ValueError
            If model has not been fitted or topic_id is invalid
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting topic words")
        
        if topic_id < 0 or topic_id >= self.num_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}")
        
        return self.model.show_topic(topic_id, topn=top_n)
    
    def get_topics_df(self) -> pd.DataFrame:
        """Get a DataFrame with topic information.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information and top words
        
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting topics")
        
        # Collect topic data
        topic_data = []
        for topic_id in range(self.num_topics):
            top_words = self.get_topic_words(topic_id, top_n=10)
            top_words_str = ", ".join([word for word, _ in top_words])
            
            # Count documents per topic if corpus is available
            if self.corpus is not None:
                doc_count = sum(
                    1 for doc in self.corpus 
                    if self.model.get_document_topics(doc, minimum_probability=0.2)[0][0] == topic_id
                )
            else:
                doc_count = 0
            
            topic_data.append({
                "topic": f"Topic_{topic_id}",
                "top_words": top_words_str,
                "document_count": doc_count,
            })
        
        return pd.DataFrame(topic_data)
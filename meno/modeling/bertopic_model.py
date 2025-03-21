"""BERTopic model for topic modeling."""

from typing import List, Dict, Optional, Union, Any, Tuple, ClassVar
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import pickle

# Create logger
logger = logging.getLogger(__name__)

# Check for BERTopic availability
import importlib.util

# Direct import check using importlib
bertopic_spec = importlib.util.find_spec("bertopic")
BERTOPIC_AVAILABLE = bertopic_spec is not None

if BERTOPIC_AVAILABLE:
    try:
        # Import BERTopic core
        from bertopic import BERTopic
        
        # Try importing components for BERTopic 0.15+
        try:
            from bertopic.representation import KeyBERTInspired
            from bertopic.vectorizers import ClassTfidfTransformer
            from bertopic.dimensionality import UMAPReducer
            logger.info("Using BERTopic 0.15+ components")
        except ImportError:
            # Handle BERTopic 0.14 and below structure
            logger.info("Using BERTopic 0.14 compatibility mode")
            
            # Import UMAP directly
            try:
                from umap import UMAP
                
                # Create a compatible UMAPReducer class
                class UMAPReducer:
                    def __init__(self, n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", low_memory=False):
                        self.n_neighbors = n_neighbors
                        self.n_components = n_components
                        self.min_dist = min_dist
                        self.metric = metric
                        self.low_memory = low_memory
                        self.umap_model = UMAP(
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=min_dist,
                            metric=metric,
                            low_memory=low_memory
                        )
                    
                    def fit(self, X, y=None):
                        self.umap_model.fit(X, y=y)
                        return self
                    
                    def fit_transform(self, X, y=None):
                        return self.umap_model.fit_transform(X, y=y)
                
                # Import remaining components
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Import ClassTfidfTransformer if available, otherwise use CountVectorizer as fallback
                try:
                    from bertopic.vectorizers import ClassTfidfTransformer
                except ImportError:
                    logger.warning("ClassTfidfTransformer not found, using CountVectorizer as fallback")
                    ClassTfidfTransformer = CountVectorizer
                
                # Create a compatible KeyBERTInspired class
                class KeyBERTInspired:
                    def extract_topics(self, documents, embeddings, topic_model):
                        topic_words = topic_model.get_topics()
                        return topic_words
                        
            except ImportError as e:
                logger.error(f"Failed to import UMAP: {e}")
                BERTOPIC_AVAILABLE = False
                
    except ImportError as e:
        logger.error(f"Failed to import BERTopic: {e}")
        BERTOPIC_AVAILABLE = False

from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.base import BaseTopicModel
from meno.modeling.coherence import calculate_bertopic_coherence, GENSIM_AVAILABLE


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
    use_llm_labeling : bool, optional
        Whether to use LLM-based topic labeling, by default False
    llm_model_type : str, optional
        Type of LLM to use for topic labeling, by default "local"
        Options: "local", "openai", "auto"
    llm_model_name : Optional[str], optional
        Name of the LLM to use for topic labeling, by default None
        If None, defaults to "google/flan-t5-small" for local models or "gpt-3.5-turbo" for OpenAI
    
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
        use_llm_labeling: bool = False,
        llm_model_type: str = "local",
        llm_model_name: Optional[str] = None,
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
        
        # Store LLM labeling parameters
        self.use_llm_labeling = use_llm_labeling
        self.llm_model_type = llm_model_type
        self.llm_model_name = llm_model_name
        
        # Set up embedding model if not provided - default to CPU
        if embedding_model is None:
            self.embedding_model = DocumentEmbedding(use_gpu=False)  # Default to CPU
        else:
            self.embedding_model = embedding_model
            
        # Set up dimensionality reduction
        try:
            # For BERTopic 0.15+
            self.umap_model = UMAPReducer(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric="cosine",
                low_memory=True,
            )
        except Exception as e:
            # For compatibility with older versions or direct UMAP usage
            from umap import UMAP
            self.umap_model = UMAP(
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
        try:
            self.model = BERTopic(
                nr_topics=n_topics,
                min_topic_size=min_topic_size,
                umap_model=self.umap_model,
                vectorizer_model=self.vectorizer_model,
                representation_model=self.representation_model,
                verbose=verbose,
                **kwargs
            )
        except Exception as e:
            # For older BERTopic versions
            try:
                from hdbscan import HDBSCAN
                import shutil
                
                # For older versions we have to set up a custom pipeline
                self.clusterer = HDBSCAN(
                    min_cluster_size=min_topic_size,
                    min_samples=min_topic_size-1,
                    metric='euclidean',
                    prediction_data=True
                )
                
                # Try again with minimal parameters
                self.model = BERTopic(
                    nr_topics=n_topics,
                    verbose=verbose,
                    **kwargs
                )
                
                # Override the model's pipeline components manually
                self.model.umap_model = self.umap_model
                self.model.hdbscan_model = self.clusterer
            except Exception as nested_e:
                raise ImportError(
                    f"Failed to initialize BERTopic model: {e}. Nested error: {nested_e}. "
                    f"Try installing packages: pip install bertopic>=0.15.0 umap-learn hdbscan"
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
            
        # Update topic descriptions with more meaningful names
        topic_info = self.model.get_topic_info()
        
        # First collect all topics and their keywords
        all_topics_words = {}
        for topic_id, row in topic_info.iterrows():
            if topic_id in self.topics and topic_id != -1:
                # Get top 10 words with scores for each topic
                words_with_scores = self.model.get_topic(topic_id)[:15]
                all_topics_words[topic_id] = words_with_scores
        
        # Create a set of already used primary words to avoid duplicates
        used_primary_words = set()
        
        # First pass: Assign the most distinctive word to each topic
        for topic_id, words_with_scores in all_topics_words.items():
            # Find first word that hasn't been used as a primary word yet
            for word, score in words_with_scores:
                if word not in used_primary_words:
                    # Capitalize the primary word to create a theme name
                    theme_word = word.title()
                    self.topics[topic_id] = f"{theme_word}"
                    used_primary_words.add(word)
                    break
            
            # If we couldn't find an unused word, use topic ID with a fallback
            if topic_id not in self.topics or self.topics[topic_id] == f"Topic {topic_id}":
                self.topics[topic_id] = f"Topic {topic_id}"
        
        # Second pass: Add supporting words to each topic name, avoiding duplicates
        for topic_id, words_with_scores in all_topics_words.items():
            # Get primary theme already assigned
            theme = self.topics[topic_id]
            
            # Find 2-3 distinctive supporting words
            supporting_words = []
            word_count = 0
            
            for word, score in words_with_scores:
                # Skip the primary word (it's already in the theme)
                if word.title() in theme:
                    continue
                
                # Only use words with good scores to ensure they're relevant
                if score > 0.1 and word_count < 3:
                    supporting_words.append(word)
                    word_count += 1
                
                if word_count >= 3:
                    break
            
            # Add supporting words to create meaningful name
            if supporting_words:
                self.topics[topic_id] = f"{theme}: {', '.join(supporting_words)}"
        
        # Try to apply LLM topic labeling if enabled
        if self.use_llm_labeling:
            try:
                # Dynamically import to avoid dependency issues
                from meno.modeling.llm_topic_labeling import LLMTopicLabeler
                
                # Extract example documents for each topic for better labeling
                example_docs_per_topic = self._extract_example_docs(documents, topics)
                
                # Create LLM labeler
                labeler = LLMTopicLabeler(
                    model_type=self.llm_model_type,
                    model_name=self.llm_model_name,
                    temperature=0.5,
                    enable_fallback=True
                )
                
                # Apply LLM topic labeling
                labeler.update_model_topic_names(
                    topic_model=self,
                    example_docs_per_topic=example_docs_per_topic,
                    detailed=True
                )
                
                logger.info("Applied LLM-based topic labeling")
                
            except Exception as e:
                logger.warning(f"LLM topic labeling failed: {e}")
                logger.warning("Falling back to keyword-based topic names")
                
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
    
    def _extract_example_docs(
        self,
        documents: Union[List[str], pd.Series],
        topics: List[int],
        examples_per_topic: int = 5
    ) -> Dict[int, List[str]]:
        """Extract example documents for each topic.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            Original documents used to fit the model
        topics : List[int]
            Topic assignments for each document
        examples_per_topic : int, optional
            Number of examples to extract per topic, by default 5
            
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of example documents
        """
        # Convert to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
            
        # Create dictionary for topic examples
        example_docs = {}
        
        # Get unique topic IDs
        unique_topics = set(topics)
        
        # For each topic, collect example documents
        for topic_id in unique_topics:
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Find documents assigned to this topic
            topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
            
            # Sample documents if there are more than needed
            if len(topic_doc_indices) > examples_per_topic:
                import random
                sample_indices = random.sample(topic_doc_indices, examples_per_topic)
            else:
                sample_indices = topic_doc_indices
                
            # Get the actual document texts
            example_docs[topic_id] = [documents[i] for i in sample_indices]
            
        return example_docs

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
        
    def apply_llm_labeling(
        self,
        documents: Union[List[str], pd.Series],
        model_type: str = "local",
        model_name: Optional[str] = None,
        detailed: bool = True
    ) -> "BERTopicModel":
        """Apply LLM-based topic labeling to improve topic names.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            The original documents used for topic modeling
        model_type : str, optional
            Type of LLM to use, by default "local"
            Options: "local", "openai", "auto"
        model_name : Optional[str], optional
            Name of the model to use, by default None
            If None, defaults to "google/flan-t5-small" for local or "gpt-3.5-turbo" for OpenAI
        detailed : bool, optional
            Whether to generate detailed topic descriptions, by default True
            
        Returns
        -------
        BERTopicModel
            The model with updated topic names
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet
        ImportError
            If LLM topic labeling is not available
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before applying LLM labeling")
            
        try:
            # Import LLM topic labeling
            from meno.modeling.llm_topic_labeling import LLMTopicLabeler
            
            # Convert documents to list if needed
            if isinstance(documents, pd.Series):
                documents = documents.tolist()
                
            # Get document topic assignments
            topics, _ = self.transform(documents)
            
            # Extract example documents for each topic
            example_docs_per_topic = self._extract_example_docs(documents, topics)
            
            # Create LLM labeler
            labeler = LLMTopicLabeler(
                model_type=model_type,
                model_name=model_name,
                temperature=0.5,
                enable_fallback=True
            )
            
            # Apply LLM topic labeling
            labeler.update_model_topic_names(
                topic_model=self,
                example_docs_per_topic=example_docs_per_topic,
                detailed=detailed
            )
            
            return self
            
        except ImportError as e:
            raise ImportError(
                "LLM topic labeling is not available. "
                "Install with 'pip install transformers' or 'pip install openai'"
            ) from e
            
    def merge_models(
        self,
        models: List["BERTopicModel"],
        documents: Optional[Union[List[str], pd.Series]] = None,
        embeddings: Optional[np.ndarray] = None,
        min_similarity: float = 0.7
    ) -> "BERTopicModel":
        """Merge multiple BERTopic models into a single model.
        
        This method allows combining multiple topic models trained on different
        datasets, creating an ensemble model with unified topics. Topic clusters
        that are sufficiently similar will be merged.
        
        Parameters
        ----------
        models : List[BERTopicModel]
            List of BERTopicModel instances to merge
        documents : Optional[Union[List[str], pd.Series]], optional
            New documents to fit the merged model on, by default None
            If None, merging is done based on topic similarity only
        embeddings : Optional[np.ndarray], optional
            Pre-computed embeddings for documents, by default None
        min_similarity : float, optional
            Minimum similarity threshold for merging topics, by default 0.7
            
        Returns
        -------
        BERTopicModel
            A new model with merged topics
            
        Raises
        ------
        ValueError
            If any of the models is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Base model must be fitted before merging")
            
        for i, model in enumerate(models):
            if not hasattr(model, 'is_fitted') or not model.is_fitted:
                raise ValueError(f"Model at index {i} is not fitted")
                
        # Ensure all models are BERTopic instances
        for model in models:
            if not isinstance(model, BERTopicModel):
                raise ValueError("All models must be BERTopicModel instances")
        
        # Create a new merged model with the same parameters as this model
        merged_model = self.__class__(
            num_topics=self.num_topics,
            embedding_model=self.embedding_model,
            min_topic_size=self.min_topic_size,
            use_gpu=self.use_gpu,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            verbose=self.verbose
        )
        
        # Use BERTopic's merge_models method to perform the merge
        try:
            # Approach 1: If models is a list of BERTopic instances
            merged_bertopic = self.model.merge_models(
                [model.model for model in models],
                min_similarity=min_similarity
            )
            
            # Copy the merged BERTopic model to our wrapper
            merged_model.model = merged_bertopic
            merged_model.is_fitted = True
            
            # Update topic information
            if hasattr(merged_bertopic, 'topics_'):
                merged_topics = {}
                merged_sizes = {}
                
                # Extract topics from the merged model
                for topic_id in set(merged_bertopic.topics_):
                    if topic_id != -1:
                        # Get topic representation from the merged model
                        if hasattr(merged_bertopic, 'get_topic') and callable(merged_bertopic.get_topic):
                            words = [word for word, _ in merged_bertopic.get_topic(topic_id)][:5]
                            topic_repr = f"Topic {topic_id}: {', '.join(words)}"
                        else:
                            topic_repr = f"Topic {topic_id}"
                            
                        merged_topics[topic_id] = topic_repr
                        merged_sizes[topic_id] = (merged_bertopic.topics_ == topic_id).sum()
                
                # Add -1 for outliers
                if -1 in set(merged_bertopic.topics_):
                    merged_topics[-1] = "Other"
                    merged_sizes[-1] = (merged_bertopic.topics_ == -1).sum()
                
                merged_model.topics = merged_topics
                merged_model.topic_sizes = merged_sizes
                
            else:
                # If topics_ attribute not found, try to adapt to the model structure
                topic_info = merged_bertopic.get_topic_info()
                merged_model.topics = {
                    row['Topic']: row.get('Name', f"Topic {row['Topic']}") if row['Topic'] != -1 else "Other"
                    for _, row in topic_info.iterrows()
                }
                merged_model.topic_sizes = {
                    row['Topic']: row['Count'] for _, row in topic_info.iterrows()
                }
            
            # Recompute topic embeddings
            merged_model._compute_topic_embeddings()
            
            # If documents provided, fit the merged model on the new documents
            if documents is not None:
                if embeddings is None:
                    embeddings = merged_model.embedding_model.embed_documents(documents)
                
                topics, probs = merged_model.model.transform(documents, embeddings=embeddings)
                
                # Optionally refine the model with the new documents...
                # For now, just update topic sizes if needed
                if hasattr(merged_bertopic, 'topics_'):
                    for topic_id in set(topics):
                        if topic_id in merged_model.topic_sizes:
                            merged_model.topic_sizes[topic_id] = (topics == topic_id).sum()
                
            return merged_model
            
        except Exception as e:
            logger.error(f"Error merging models: {e}")
            raise ValueError(f"Failed to merge models: {e}")
            
    def merge_topics(
        self,
        topics_to_merge: List[List[int]],
        documents: Optional[Union[List[str], pd.Series]] = None
    ) -> "BERTopicModel":
        """Merge specific topics within the model.
        
        Parameters
        ----------
        topics_to_merge : List[List[int]]
            List of topic groups to merge, where each group is a list of topic IDs
            Example: [[1, 2, 3], [4, 5]] would merge topics 1, 2, 3 into one topic
            and topics 4, 5 into another
        documents : Optional[Union[List[str], pd.Series]], optional
            Documents used to update the topic representations after merging,
            by default None
            
        Returns
        -------
        BERTopicModel
            The model with merged topics
            
        Raises
        ------
        ValueError
            If the model is not fitted or topics are invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before merging topics")
            
        try:
            # Verify all topics exist in the model
            all_topics = set()
            for group in topics_to_merge:
                all_topics.update(group)
                
            for topic in all_topics:
                if topic not in self.topics and topic != -1:
                    raise ValueError(f"Topic {topic} not found in the model")
            
            # Use BERTopic's merge_topics method
            if hasattr(self.model, 'merge_topics') and callable(self.model.merge_topics):
                # Create document list if provided
                docs = None
                if documents is not None:
                    if isinstance(documents, pd.Series):
                        docs = documents.tolist()
                    else:
                        docs = documents
                
                # Merge topics in the underlying BERTopic model
                self.model.merge_topics(docs, topics_to_merge)
                
                # Update our wrapper's topic information
                topic_info = self.model.get_topic_info()
                self.topics = {
                    row['Topic']: row.get('Name', f"Topic {row['Topic']}") if row['Topic'] != -1 else "Other"
                    for _, row in topic_info.iterrows()
                }
                self.topic_sizes = {
                    row['Topic']: row['Count'] for _, row in topic_info.iterrows()
                }
                
                # Recompute topic embeddings
                self._compute_topic_embeddings()
                
                return self
            else:
                raise NotImplementedError("The underlying BERTopic model does not support topic merging")
                
        except Exception as e:
            logger.error(f"Error merging topics: {e}")
            raise ValueError(f"Failed to merge topics: {e}")
            
    def reduce_topics(
        self,
        documents: Union[List[str], pd.Series],
        nr_topics: int
    ) -> "BERTopicModel":
        """Reduce the number of topics to a specified amount.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            Documents used for refitting with reduced topics
        nr_topics : int
            The number of topics to reduce to
            
        Returns
        -------
        BERTopicModel
            The model with reduced topics
            
        Raises
        ------
        ValueError
            If the model is not fitted or if nr_topics is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reducing topics")
            
        if nr_topics <= 0:
            raise ValueError("Number of topics must be positive")
            
        if len(self.topics) - 1 <= nr_topics:  # -1 for outlier topic
            logger.warning(f"Model already has {len(self.topics) - 1} topics, which is <= {nr_topics}")
            return self
            
        try:
            # Convert documents to list if needed
            if isinstance(documents, pd.Series):
                documents = documents.tolist()
                
            # Use BERTopic's reduce_topics method
            if hasattr(self.model, 'reduce_topics') and callable(self.model.reduce_topics):
                # Reduce topics in the underlying BERTopic model
                self.model.reduce_topics(documents, nr_topics=nr_topics)
                
                # Update our wrapper's topic information
                topic_info = self.model.get_topic_info()
                self.topics = {
                    row['Topic']: row.get('Name', f"Topic {row['Topic']}") if row['Topic'] != -1 else "Other"
                    for _, row in topic_info.iterrows()
                }
                self.topic_sizes = {
                    row['Topic']: row['Count'] for _, row in topic_info.iterrows()
                }
                
                # Recompute topic embeddings
                self._compute_topic_embeddings()
                
                return self
            else:
                raise NotImplementedError("The underlying BERTopic model does not support topic reduction")
                
        except Exception as e:
            logger.error(f"Error reducing topics: {e}")
            raise ValueError(f"Failed to reduce topics: {e}")
    
    def find_topics(
        self,
        search_term: str,
        top_n: int = 5
    ) -> List[Tuple[int, str, float]]:
        """Find topics that are similar to a search term.
        
        Parameters
        ----------
        search_term : str
            Search query to find similar topics
        top_n : int, optional
            Number of topics to return, by default 5
            
        Returns
        -------
        List[Tuple[int, str, float]]
            List of tuples with (topic_id, topic_name, similarity_score)
            
        Raises
        ------
        ValueError
            If the model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before searching for topics")
            
        try:
            # Use BERTopic's find_topics method if available
            if hasattr(self.model, 'find_topics') and callable(self.model.find_topics):
                # Find topics using the underlying BERTopic model
                topic_ids, similarities = self.model.find_topics(search_term, top_n=top_n)
                
                # Convert to expected format
                return [
                    (topic_id, self.topics.get(topic_id, f"Topic {topic_id}"), float(similarity))
                    for topic_id, similarity in zip(topic_ids, similarities)
                ]
            else:
                # Fall back to our own implementation
                return self.find_similar_topics(search_term, n_topics=top_n)
                
        except Exception as e:
            logger.error(f"Error finding topics: {e}")
            raise ValueError(f"Failed to find topics: {e}")
    
    def update_topics(
        self,
        documents: Union[List[str], pd.Series],
        topics: List[int],
        embeddings: Optional[np.ndarray] = None,
        vectorizer_model: Optional[Any] = None,
        representation_model: Optional[Any] = None
    ) -> "BERTopicModel":
        """Update topic representations based on new documents.
        
        This method allows updating topic representations without having to retrain
        the entire model, which is useful for incremental learning.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            New documents to use for updating topic representations
        topics : List[int]
            Topic assignments for the documents
        embeddings : Optional[np.ndarray], optional
            Pre-computed embeddings for the documents, by default None
        vectorizer_model : Optional[Any], optional
            Custom vectorizer model, by default None
        representation_model : Optional[Any], optional
            Custom representation model, by default None
            
        Returns
        -------
        BERTopicModel
            The model with updated topic representations
            
        Raises
        ------
        ValueError
            If the model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating topics")
            
        try:
            # Convert documents to list if needed
            if isinstance(documents, pd.Series):
                documents = documents.tolist()
                
            # Generate embeddings if not provided
            if embeddings is None:
                embeddings = self.embedding_model.embed_documents(documents)
                
            # Use BERTopic's update_topics method
            if hasattr(self.model, 'update_topics') and callable(self.model.update_topics):
                # Update topics in the underlying BERTopic model
                self.model.update_topics(
                    documents=documents,
                    topics=topics,
                    embeddings=embeddings,
                    vectorizer_model=vectorizer_model,
                    representation_model=representation_model
                )
                
                # Update our wrapper's topic information
                topic_info = self.model.get_topic_info()
                self.topics = {
                    row['Topic']: row.get('Name', f"Topic {row['Topic']}") if row['Topic'] != -1 else "Other"
                    for _, row in topic_info.iterrows()
                }
                
                # Recompute topic embeddings
                self._compute_topic_embeddings()
                
                return self
            else:
                raise NotImplementedError("The underlying BERTopic model does not support topic updating")
                
        except Exception as e:
            logger.error(f"Error updating topics: {e}")
            raise ValueError(f"Failed to update topics: {e}")
    
    def fit_transform_with_timestamps(
        self,
        documents: Union[List[str], pd.Series],
        timestamps: Union[List[str], List[int], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        global_tuning: bool = True,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Fit the model to documents with timestamps for dynamic topic modeling.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            The documents to fit the model on
        timestamps : Union[List[str], List[int], pd.Series]
            Timestamps corresponding to each document
        embeddings : Optional[np.ndarray], optional
            Pre-computed embeddings for the documents, by default None
        global_tuning : bool, optional
            Whether to apply global tuning of topics, by default True
        **kwargs : Any
            Additional arguments to pass to the model
            
        Returns
        -------
        Tuple[List[int], np.ndarray, np.ndarray]
            Tuple of (topic_assignments, probabilities, timestamps)
            
        Raises
        ------
        ValueError
            If the model cannot be fitted with timestamps
        """
        try:
            # Convert documents and timestamps to lists if needed
            if isinstance(documents, pd.Series):
                documents = documents.tolist()
                
            if isinstance(timestamps, pd.Series):
                timestamps = timestamps.tolist()
                
            # Generate embeddings if not provided
            if embeddings is None:
                embeddings = self.embedding_model.embed_documents(documents)
                
            # Use BERTopic's fit_transform method with timestamps
            if hasattr(self.model, 'fit_transform') and callable(self.model.fit_transform):
                # Check if the method supports timestamps
                import inspect
                sig = inspect.signature(self.model.fit_transform)
                if 'timestamps' in sig.parameters:
                    topics, probs = self.model.fit_transform(
                        documents=documents,
                        embeddings=embeddings,
                        timestamps=timestamps,
                        global_tuning=global_tuning,
                        **kwargs
                    )
                    
                    # Update our wrapper's topic information
                    topic_info = self.model.get_topic_info()
                    self.topics = {
                        row['Topic']: row.get('Name', f"Topic {row['Topic']}") if row['Topic'] != -1 else "Other"
                        for _, row in topic_info.iterrows()
                    }
                    self.topic_sizes = {
                        row['Topic']: row['Count'] for _, row in topic_info.iterrows()
                    }
                    
                    # Recompute topic embeddings
                    self._compute_topic_embeddings()
                    
                    # Set as fitted
                    self.is_fitted = True
                    
                    return topics, probs, np.array(timestamps)
                else:
                    raise ValueError("The underlying BERTopic model does not support timestamps")
            else:
                raise NotImplementedError("The underlying BERTopic model does not support fit_transform with timestamps")
                
        except Exception as e:
            logger.error(f"Error fitting model with timestamps: {e}")
            raise ValueError(f"Failed to fit model with timestamps: {e}")
    
    def visualize_topics_over_time(
        self,
        topics_over_time: pd.DataFrame,
        top_n_topics: int = 10,
        width: int = 1000,
        height: int = 600
    ) -> Any:
        """Visualize how topics evolve over time.
        
        Parameters
        ----------
        topics_over_time : pd.DataFrame
            DataFrame with topics over time data
        top_n_topics : int, optional
            Number of topics to display, by default 10
        width : int, optional
            Width of the plot, by default 1000
        height : int, optional
            Height of the plot, by default 600
            
        Returns
        -------
        Any
            Visualization object (plotly figure)
            
        Raises
        ------
        ValueError
            If the model is not fitted or topics_over_time is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualizing topics over time")
            
        try:
            # Use BERTopic's visualize_topics_over_time method
            if hasattr(self.model, 'visualize_topics_over_time') and callable(self.model.visualize_topics_over_time):
                return self.model.visualize_topics_over_time(
                    topics_over_time=topics_over_time,
                    top_n_topics=top_n_topics,
                    width=width,
                    height=height
                )
            else:
                # Fall back to a custom implementation or error message
                from meno.visualization.bertopic_viz import visualize_topics_over_time
                return visualize_topics_over_time(
                    self,
                    topics_over_time=topics_over_time,
                    top_n_topics=top_n_topics,
                    width=width,
                    height=height
                )
                
        except Exception as e:
            logger.error(f"Error visualizing topics over time: {e}")
            raise ValueError(f"Failed to visualize topics over time: {e}")
    
    def fit_with_seed_topics(
        self,
        documents: Union[List[str], pd.Series],
        seed_topic_list: List[List[str]],
        seed_topics: Optional[List[int]] = None,
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BERTopicModel":
        """Fit the model using seed topics for semi-supervised topic modeling.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            The documents to fit the model on
        seed_topic_list : List[List[str]]
            List of seed topics, where each topic is a list of keywords
        seed_topics : Optional[List[int]], optional
            List of seed topic assignments, by default None
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional arguments to pass to the model
            
        Returns
        -------
        BERTopicModel
            The fitted model
            
        Raises
        ------
        ValueError
            If the model cannot be fitted with seed topics
        """
        try:
            # Convert documents to list if needed
            if isinstance(documents, pd.Series):
                documents = documents.tolist()
                
            # Generate embeddings if not provided
            if embeddings is None:
                embeddings = self.embedding_model.embed_documents(documents)
                
            # Check if the model supports seed topics
            if hasattr(self.model, 'seed_topic_list'):
                # Set seed topics
                self.model.seed_topic_list = seed_topic_list
                
                # Fit the model with seed topics
                topics, probs = self.model.fit_transform(
                    documents=documents,
                    embeddings=embeddings,
                    y=seed_topics,  # Supervised labels when available
                    **kwargs
                )
                
                # Update our wrapper's topic information
                topic_info = self.model.get_topic_info()
                self.topics = {
                    row['Topic']: row.get('Name', f"Topic {row['Topic']}") if row['Topic'] != -1 else "Other"
                    for _, row in topic_info.iterrows()
                }
                self.topic_sizes = {
                    row['Topic']: row['Count'] for _, row in topic_info.iterrows()
                }
                
                # Recompute topic embeddings
                self._compute_topic_embeddings()
                
                # Set as fitted
                self.is_fitted = True
                
                return self
            else:
                raise NotImplementedError("The underlying BERTopic model does not support seed topics")
                
        except Exception as e:
            logger.error(f"Error fitting model with seed topics: {e}")
            raise ValueError(f"Failed to fit model with seed topics: {e}")
    
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
            # Use our more meaningful topic names from self.topics
            topic_info['Name'] = [self.topics.get(i, f"Topic {i}" if i != -1 else "Other") for i in topic_info['Topic']]
            
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
        
    def calculate_coherence(
        self,
        texts: List[List[str]],
        coherence: str = "c_v",
        top_n: int = 10,
    ) -> Union[float, Dict[str, float]]:
        """Calculate coherence metrics for the model.
        
        Parameters
        ----------
        texts : List[List[str]]
            Tokenized texts (list of token lists)
        coherence : str, optional
            Coherence metric to use, by default "c_v"
            Options: "c_v", "c_uci", "c_npmi", "u_mass", "all"
            If "all", returns all available metrics
        top_n : int, optional
            Number of top words per topic to use, by default 10
            
        Returns
        -------
        Union[float, Dict[str, float]]
            Coherence score or dictionary of scores if coherence="all"
            
        Raises
        ------
        ImportError
            If gensim is not available
        ValueError
            If model is not fitted
        """
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "Gensim is required for coherence calculation. "
                "Install with 'pip install gensim>=4.0.0'"
            )
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating coherence")
            
        return calculate_bertopic_coherence(
            model=self.model,
            texts=texts,
            coherence=coherence,
            top_n=top_n,
        )
    
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
        local_files_only: bool = False,
    ) -> "BERTopicModel":
        """Load a model from disk.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
        embedding_model : Optional[DocumentEmbedding], optional
            Document embedding model to use, by default None
        local_files_only : bool, optional
            Whether to use only local files and not download from Hugging Face, by default False
            
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
        
        # Create embedding model with local_files_only setting if not provided
        if embedding_model is None:
            model_name = "all-MiniLM-L6-v2"  # Default model
            embedding_model = DocumentEmbedding(
                model_name=model_name,
                use_gpu=False,
                local_files_only=local_files_only
            )
            
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
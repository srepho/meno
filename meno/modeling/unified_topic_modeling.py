"""Unified topic modeling interface for different topic modeling approaches."""

from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os
import pickle

from .base import BaseTopicModel
from .bertopic_model import BERTopicModel
from .top2vec_model import Top2VecModel
from .embeddings import DocumentEmbedding
from .coherence import calculate_generic_coherence, GENSIM_AVAILABLE
from ..utils.config import MenoConfig, load_config

logger = logging.getLogger(__name__)


class UnifiedTopicModeler(BaseTopicModel):
    """Unified topic modeling interface for different topic modeling approaches.
    
    This class provides a standardized interface to interact with various topic modeling 
    techniques available in the Meno library, including BERTopic and Top2Vec.
    
    Parameters
    ----------
    method : str
        The topic modeling method to use. Options include:
        - "bertopic": BERTopic model
        - "top2vec": Top2Vec model  
        - "embedding_cluster": Embedding-based clustering
    num_topics : int, optional
        The number of topics to discover, by default 10
    config_overrides : Dict[str, Any], optional
        Configuration overrides for the model, by default None
    embedding_model : Union[str, DocumentEmbedding], optional
        The embedding model to use, by default None
    """
    
    def __init__(
        self,
        method: str = "embedding_cluster",
        num_topics: Optional[int] = 10,
        config_overrides: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
        auto_detect_topics: bool = False,
        use_llm_labeling: bool = False,
        llm_model_type: str = "local",
        llm_model_name: Optional[str] = None
    ):
        self.method = method
        self.auto_detect_topics = auto_detect_topics
        
        # Handle automatic topic detection
        if auto_detect_topics:
            num_topics = None  # Force auto-detection
            
            # Add auto-detection configuration to overrides
            if config_overrides is None:
                config_overrides = {}
            config_overrides['auto_detect_topics'] = True
        
        self.num_topics = num_topics
        self.config_overrides = config_overrides or {}
        self.embedding_model = embedding_model
        
        # Store LLM parameters
        self.use_llm_labeling = use_llm_labeling
        self.llm_model_type = llm_model_type
        self.llm_model_name = llm_model_name
        
        self.model = self._create_model()
        self.is_fitted = False
        self.topics = {}
        self.topic_sizes = {}
        
    def _create_model(self) -> BaseTopicModel:
        """Create the appropriate topic model based on the specified method.
        
        Returns
        -------
        BaseTopicModel
            The instantiated topic model
        """
        # Create embedding model if specified as string
        if isinstance(self.embedding_model, str):
            self.embedding_model = DocumentEmbedding(model_name=self.embedding_model)
        
        # Create model based on method
        if self.method == "bertopic":
            return BERTopicModel(
                n_topics=self.num_topics,
                embedding_model=self.embedding_model,
                use_llm_labeling=self.use_llm_labeling,
                llm_model_type=self.llm_model_type,
                llm_model_name=self.llm_model_name,
                **self.config_overrides
            )
        elif self.method == "top2vec":
            return Top2VecModel(
                n_topics=self.num_topics,
                embedding_model=self.embedding_model,
                **self.config_overrides
            )
        elif self.method == "embedding_cluster":
            # Default to BERTopic with embedding_model for embedding-based clustering
            return BERTopicModel(
                n_topics=self.num_topics,
                embedding_model=self.embedding_model,
                use_llm_labeling=self.use_llm_labeling,
                llm_model_type=self.llm_model_type,
                llm_model_name=self.llm_model_name,
                **self.config_overrides
            )
        else:
            raise ValueError(f"Unknown topic modeling method: {self.method}")
    
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "UnifiedTopicModeler":
        """Fit the topic model to a set of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to model
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments passed to the underlying model
        
        Returns
        -------
        UnifiedTopicModeler
            The fitted topic modeler for method chaining
        """
        # Set n_topics parameter appropriately for the underlying model
        # Ensure consistency across different model implementations
        if 'num_topics' in kwargs and 'n_topics' not in kwargs:
            kwargs['n_topics'] = kwargs.pop('num_topics')
        elif self.num_topics is not None and 'n_topics' not in kwargs:
            kwargs['n_topics'] = self.num_topics
        
        # Fit the underlying model
        self.model.fit(documents, embeddings, **kwargs)
        
        # Copy important attributes from the underlying model
        self.topics = getattr(self.model, 'topics', {})
        self.topic_sizes = getattr(self.model, 'topic_sizes', {})
        self.is_fitted = True
        
        return self
    
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Transform documents to topic assignments.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to assign to topics
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments passed to the underlying model
        
        Returns
        -------
        Tuple[Any, Any]
            A tuple containing (topic_assignments, topic_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform can be called.")
        
        # Standardize parameter names for consistency
        if 'top_n' in kwargs:
            kwargs['top_n'] = kwargs['top_n']
        
        # Call the underlying model's transform method
        return self.model.transform(documents, embeddings, **kwargs)
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Fit the model and transform documents in one step.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to model and assign to topics
        embeddings : Optional[np.ndarray], optional
            Pre-computed document embeddings, by default None
        **kwargs : Any
            Additional keyword arguments passed to the underlying model
        
        Returns
        -------
        Tuple[Any, Any]
            A tuple containing (topic_assignments, topic_probabilities)
        """
        self.fit(documents, embeddings, **kwargs)
        return self.transform(documents, embeddings, **kwargs)
    
    def visualize_topics(self, **kwargs) -> Any:
        """Visualize discovered topics.
        
        Parameters
        ----------
        **kwargs : Any
            Visualization parameters passed to the underlying model's visualization method
        
        Returns
        -------
        Any
            The visualization object (typically a plotly Figure)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be visualized.")
        
        if hasattr(self.model, 'visualize_topics'):
            return self.model.visualize_topics(**kwargs)
        else:
            raise NotImplementedError(f"Visualization not implemented for {self.method}")
            
    def apply_llm_labeling(
        self,
        documents: Union[List[str], pd.Series],
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        detailed: bool = True
    ) -> "UnifiedTopicModeler":
        """Apply LLM-based topic labeling to improve topic names.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            The original documents used for topic modeling
        model_type : Optional[str], optional
            Type of LLM to use, by default None
            If None, uses the model_type specified during initialization
            Options: "local", "openai", "auto"
        model_name : Optional[str], optional
            Name of the model to use, by default None
            If None, uses the model_name specified during initialization
        detailed : bool, optional
            Whether to generate detailed topic descriptions, by default True
            
        Returns
        -------
        UnifiedTopicModeler
            The modeler with updated topic names
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet
        NotImplementedError
            If the underlying model doesn't support LLM labeling
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before applying LLM labeling")
            
        # Use instance parameters if not specified
        if model_type is None:
            model_type = self.llm_model_type
            
        if model_name is None:
            model_name = self.llm_model_name
            
        # Check if the underlying model supports LLM labeling
        if hasattr(self.model, 'apply_llm_labeling'):
            self.model.apply_llm_labeling(
                documents=documents,
                model_type=model_type,
                model_name=model_name,
                detailed=detailed
            )
            
            # Update our topic dictionaries
            self.topics = getattr(self.model, 'topics', {})
            
            return self
        else:
            raise NotImplementedError(
                f"LLM topic labeling is not implemented for {self.method}"
            )
            
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
            
        # If model has its own coherence calculation method, use that
        if hasattr(self.model, 'calculate_coherence'):
            return self.model.calculate_coherence(
                texts=texts,
                coherence=coherence,
                top_n=top_n
            )
        
        # Otherwise calculate coherence ourselves using the topic words
        topic_words = {}
        
        # Extract top words for each topic
        topic_info = self.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                # Extract words from representation 
                if isinstance(row['Representation'], str):
                    # Try to parse comma-separated words
                    try:
                        words = [word.strip() for word in row['Representation'].split(',')]
                    except:
                        # If parsing fails, use the whole string
                        words = [row['Representation']]
                elif isinstance(row['Representation'], list):
                    words = row['Representation']
                else:
                    words = [str(row['Representation'])]
                    
                topic_words[topic_id] = words
        
        # Calculate coherence using the generic function
        return calculate_generic_coherence(
            topic_words=topic_words,
            texts=texts,
            coherence=coherence,
            top_n=top_n
        )
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topic info can be retrieved.")
        
        if hasattr(self.model, 'get_topic_info'):
            return self.model.get_topic_info()
        else:
            # Create a standardized topic info dataframe
            data = []
            for topic_id, topic_words in self.topics.items():
                data.append({
                    'Topic': topic_id,
                    'Count': self.topic_sizes.get(topic_id, 0),
                    'Name': f"Topic {topic_id}",
                    'Representation': str(topic_words)
                })
            return pd.DataFrame(data)
    
    def get_document_topics(self, documents: Union[List[str], pd.Series]) -> pd.DataFrame:
        """Get topic assignments for documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List of document texts to assign to topics
        
        Returns
        -------
        pd.DataFrame
            DataFrame with document-topic assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before topics can be assigned.")
        
        topic_assignments, probabilities = self.transform(documents)
        
        # Create a standardized document-topic dataframe
        result = pd.DataFrame({
            'document_id': range(len(documents)),
            'topic': topic_assignments,
            'probability': np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else probabilities
        })
        
        return result
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save attributes
        model_data = {
            'method': self.method,
            'num_topics': self.num_topics,
            'config_overrides': self.config_overrides,
            'is_fitted': self.is_fitted,
            'topics': self.topics,
            'topic_sizes': self.topic_sizes
        }
        
        # Save underlying model if it has a save method
        model_path = f"{path}_underlying_model"
        if hasattr(self.model, 'save'):
            self.model.save(model_path)
            model_data['underlying_model_path'] = model_path
        else:
            # Fallback to pickle if no save method
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            model_data['underlying_model_pickle'] = f"{model_path}.pkl"
        
        # Save model data
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> "UnifiedTopicModeler":
        """Load a model from disk.
        
        Parameters
        ----------
        path : str
            Path to load the model from
        
        Returns
        -------
        UnifiedTopicModeler
            Loaded model
        """
        # Load model data
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            method=model_data['method'],
            num_topics=model_data['num_topics'],
            config_overrides=model_data['config_overrides']
        )
        
        # Load underlying model
        if 'underlying_model_path' in model_data:
            if instance.method == 'bertopic':
                instance.model = BERTopicModel.load(model_data['underlying_model_path'])
            elif instance.method == 'top2vec':
                instance.model = Top2VecModel.load(model_data['underlying_model_path'])
            else:
                # Fallback to using the appropriate class's load method
                model_class = instance.model.__class__
                instance.model = model_class.load(model_data['underlying_model_path'])
        elif 'underlying_model_pickle' in model_data:
            with open(model_data['underlying_model_pickle'], 'rb') as f:
                instance.model = pickle.load(f)
        
        # Set attributes
        instance.is_fitted = model_data['is_fitted']
        instance.topics = model_data['topics']
        instance.topic_sizes = model_data['topic_sizes']
        
        return instance


def create_topic_modeler(
    method: str = "embedding_cluster",
    num_topics: Optional[int] = 10,
    config_overrides: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[Union[str, DocumentEmbedding]] = None,
    auto_detect_topics: bool = False,
    offline_mode: bool = False,
    use_llm_labeling: bool = False,
    llm_model_type: str = "local",
    llm_model_name: Optional[str] = None
) -> BaseTopicModel:
    """Create a topic modeler with the specified method.
    
    This factory function creates and returns an appropriate topic modeler
    based on the specified method and configuration.
    
    Parameters
    ----------
    method : str, optional
        The topic modeling method to use, by default "embedding_cluster"
        Options: "bertopic", "top2vec", "embedding_cluster"
    num_topics : Optional[int], optional
        The number of topics to discover, by default 10
        If None or if auto_detect_topics=True, the model will automatically 
        determine the optimal number of topics
    config_overrides : Optional[Dict[str, Any]], optional
        Configuration overrides for the model, by default None
    embedding_model : Optional[Union[str, DocumentEmbedding]], optional
        The embedding model to use, by default None
    auto_detect_topics : bool, optional
        Whether to automatically detect the optimal number of topics, by default False
        If True, num_topics will be ignored and the model will determine the best
        number of topics based on the data
    offline_mode : bool, optional
        Whether to bypass module availability checks and assume modules are available,
        by default False. This is useful for environments with manually installed
        packages or restricted import capabilities.
    use_llm_labeling : bool, optional
        Whether to use LLM-based topic labeling, by default False
    llm_model_type : str, optional
        Type of LLM to use for topic labeling, by default "local"
        Options: "local", "openai", "auto"
    llm_model_name : Optional[str], optional
        Name of the LLM to use for topic labeling, by default None
        If None, defaults to appropriate model based on type
    
    Returns
    -------
    BaseTopicModel
        An instance of the appropriate topic model
    """
    config = load_config()
    config_overrides = config_overrides or {}
    
    # Handle auto-detection configuration
    if auto_detect_topics:
        if config_overrides is None:
            config_overrides = {}
        config_overrides['auto_detect_topics'] = True
        
        # Force num_topics to None for auto-detection
        num_topics = None
    
    # Special case for directly creating a specific model type
    if method == "bertopic":
        # Remove auto_detect_topics from config_overrides if it's already passed as a parameter
        model_config = config_overrides.copy()
        if 'auto_detect_topics' in model_config:
            del model_config['auto_detect_topics']
        
        # Handle offline mode for BERTopic
        if offline_mode:
            # In offline mode, we override the import check result
            import sys
            from importlib.util import find_spec
            
            # Check if bertopic might be available but import check failed
            bertopic_spec = find_spec("bertopic")
            if bertopic_spec is not None:
                # Force module availability flag to True if in offline mode
                import meno.modeling.bertopic_model
                meno.modeling.bertopic_model.BERTOPIC_AVAILABLE = True
                logger.info("Offline mode enabled: Using BERTopic module regardless of import check result")
            else:
                logger.warning("Offline mode enabled but bertopic module not found in sys.path")
            
        return BERTopicModel(
            num_topics=num_topics,  # Use standardized parameter name
            embedding_model=embedding_model,
            auto_detect_topics=auto_detect_topics,
            use_llm_labeling=use_llm_labeling,
            llm_model_type=llm_model_type,
            llm_model_name=llm_model_name,
            **model_config
        )
    elif method == "top2vec":
        # Remove auto_detect_topics from config_overrides if it's already passed as a parameter
        model_config = config_overrides.copy()
        if 'auto_detect_topics' in model_config:
            del model_config['auto_detect_topics']
        
        # Handle offline mode for Top2Vec
        if offline_mode:
            # In offline mode, we override the import check result
            import sys
            from importlib.util import find_spec
            
            # Check if top2vec might be available but import check failed
            top2vec_spec = find_spec("top2vec")
            if top2vec_spec is not None:
                # Force module availability flag to True if in offline mode
                import meno.modeling.top2vec_model
                meno.modeling.top2vec_model.TOP2VEC_AVAILABLE = True
                logger.info("Offline mode enabled: Using Top2Vec module regardless of import check result")
            else:
                logger.warning("Offline mode enabled but top2vec module not found in sys.path")
            
        return Top2VecModel(
            num_topics=num_topics,  # Use standardized parameter name
            embedding_model=embedding_model,
            auto_detect_topics=auto_detect_topics,
            **model_config
        )
    
    # Use the unified interface for other methods
    
    # Add offline mode to config overrides to pass it on to underlying models
    if offline_mode and config_overrides is not None:
        config_overrides['offline_mode'] = offline_mode
    elif offline_mode:
        config_overrides = {'offline_mode': offline_mode}
    
    return UnifiedTopicModeler(
        method=method,
        num_topics=num_topics,
        config_overrides=config_overrides,
        embedding_model=embedding_model,
        auto_detect_topics=auto_detect_topics,
        use_llm_labeling=use_llm_labeling,
        llm_model_type=llm_model_type,
        llm_model_name=llm_model_name
    )
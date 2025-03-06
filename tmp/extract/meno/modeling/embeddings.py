"""Document embedding module using transformer models."""

from typing import List, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch


class DocumentEmbedding:
    """Generate and manage document embeddings using transformer models.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the transformer model to use, by default "sentence-transformers/all-MiniLM-L6-v2"
        Supports any model compatible with sentence-transformers or HuggingFace transformers.
    device : str, optional
        Device to run the model on, by default "cuda" if available else "cpu"
    batch_size : int, optional
        Batch size for embedding generation, by default 32
    
    Attributes
    ----------
    model : SentenceTransformer
        Loaded transformer model
    embedding_dim : int
        Dimension of the embeddings
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize the document embedding model."""
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(
        self, 
        documents: Union[List[str], pd.Series],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series]
            List or Series of document texts to embed
        show_progress_bar : bool, optional
            Whether to show a progress bar, by default True
        
        Returns
        -------
        np.ndarray
            Document embeddings with shape (len(documents), embedding_dim)
        """
        # Convert pandas Series to list if needed
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def embed_topics(
        self,
        topics: Union[List[str], pd.Series],
        descriptions: Optional[Union[List[str], pd.Series]] = None,
    ) -> np.ndarray:
        """Generate embeddings for a list of topics, optionally with descriptions.
        
        Parameters
        ----------
        topics : Union[List[str], pd.Series]
            List or Series of topic names
        descriptions : Optional[Union[List[str], pd.Series]], optional
            List or Series of topic descriptions, by default None
            If provided, topics and descriptions are combined
        
        Returns
        -------
        np.ndarray
            Topic embeddings with shape (len(topics), embedding_dim)
        """
        # Convert to lists if needed
        if isinstance(topics, pd.Series):
            topics = topics.tolist()
        
        if descriptions is not None:
            if isinstance(descriptions, pd.Series):
                descriptions = descriptions.tolist()
            
            # Combine topics and descriptions
            texts_to_embed = [
                f"{topic}: {desc}" for topic, desc in zip(topics, descriptions)
            ]
        else:
            texts_to_embed = topics
        
        # Generate embeddings
        return self.embed_documents(texts_to_embed, show_progress_bar=False)
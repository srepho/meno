"""Document embedding module using transformer models (CPU-optimized by default)."""

from typing import List, Dict, Optional, Union, Any, Callable, Generator, Iterable
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class DocumentEmbedding:
    """Generate and manage document embeddings using transformer models.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the transformer model to use, by default "answerdotai/ModernBERT-base"
        Supports any model compatible with sentence-transformers or HuggingFace transformers.
    device : str, optional
        Device to run the model on, by default determined by use_gpu setting
    batch_size : int, optional
        Batch size for embedding generation, by default 32
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
        Setting to False (default) ensures CPU-only operation and avoids CUDA dependencies
    local_model_path : str, optional
        Path to locally downloaded model, by default None (will download from HuggingFace)
    
    Attributes
    ----------
    model : SentenceTransformer
        Loaded transformer model
    embedding_dim : int
        Dimension of the embeddings
    device : str
        Device being used for inference (either "cpu" or "cuda")
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_gpu: bool = False,
        local_model_path: Optional[str] = None,
    ):
        """Initialize the document embedding model."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.local_model_path = local_model_path
        
        # Set device - default to CPU
        if device is None:
            # Only use CUDA if explicitly requested and available
            self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        else:
            self.device = device
            
        # Load model
        if local_model_path and os.path.exists(local_model_path):
            self.model = SentenceTransformer(local_model_path, device=self.device)
        else:
            self.model = SentenceTransformer(model_name, device=self.device)
            
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def _create_batches(self, documents: List[str], batch_size: int) -> Generator[List[str], None, None]:
        """Split documents into batches for processing.
        
        Parameters
        ----------
        documents : List[str]
            List of document texts to split into batches
        batch_size : int
            Size of each batch
            
        Yields
        ------
        List[str]
            Batch of documents
        """
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]
    
    def embed_documents(
        self, 
        documents: Union[List[str], pd.Series, "pl.Series", "pl.DataFrame"],
        show_progress_bar: bool = True,
        text_column: Optional[str] = None,
    ) -> np.ndarray:
        """Generate embeddings for a list of documents.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series, pl.Series, pl.DataFrame]
            List, Series, or DataFrame of document texts to embed.
            If DataFrame, must provide text_column.
        show_progress_bar : bool, optional
            Whether to show a progress bar, by default True
        text_column : Optional[str], optional
            If documents is a DataFrame, the name of the column containing text, by default None
        
        Returns
        -------
        np.ndarray
            Document embeddings with shape (len(documents), embedding_dim)
        """
        # Handle different input types, converting to list
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        elif POLARS_AVAILABLE and isinstance(documents, pl.Series):
            documents = documents.to_list()
        elif POLARS_AVAILABLE and isinstance(documents, pl.DataFrame):
            if text_column is None:
                raise ValueError("text_column must be provided when documents is a DataFrame")
            documents = documents[text_column].to_list()
        elif isinstance(documents, pd.DataFrame):
            if text_column is None:
                raise ValueError("text_column must be provided when documents is a DataFrame")
            documents = documents[text_column].tolist()
        
        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def embed_documents_stream(
        self,
        documents_iterator: Iterable[str],
        batch_size: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Generate embeddings for a stream of documents, yielding batches of embeddings.
        
        This is useful for large datasets that don't fit in memory.
        
        Parameters
        ----------
        documents_iterator : Iterable[str]
            Iterator yielding document texts
        batch_size : Optional[int], optional
            Batch size for processing, by default None (uses self.batch_size)
        
        Yields
        ------
        np.ndarray
            Batches of document embeddings with shape (batch_size, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Process documents in batches
        batch = []
        for doc in documents_iterator:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield self.embed_documents(batch, show_progress_bar=False)
                batch = []
                
        # Process final batch if there are any remaining documents
        if batch:
            yield self.embed_documents(batch, show_progress_bar=False)
    
    def embed_polars_streaming(
        self,
        data_path: str,
        text_column: str,
        batch_size: Optional[int] = None,
        filter_condition: Optional[Callable] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Stream a large dataset using Polars for memory efficiency.
        
        Parameters
        ----------
        data_path : str
            Path to the data file (CSV, Parquet, etc.)
        text_column : str
            Name of the column containing text to embed
        batch_size : Optional[int], optional
            Batch size for processing, by default None (uses self.batch_size)
        filter_condition : Optional[Callable], optional
            Function taking a Polars DataFrame and returning a filtered DataFrame,
            by default None
            
        Yields
        ------
        np.ndarray
            Batches of document embeddings
            
        Raises
        ------
        ImportError
            If Polars is not available
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is required for streaming large datasets. "
                "Install with 'pip install \"meno[optimization]\"'"
            )
            
        if batch_size is None:
            batch_size = self.batch_size
        
        # Determine file format from extension
        file_format = os.path.splitext(data_path)[1].lower()
        
        # Use Polars lazy loading and streaming
        if file_format == ".csv":
            reader = pl.scan_csv(data_path)
        elif file_format in (".parquet", ".pq"):
            reader = pl.scan_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Apply filter if provided
        if filter_condition is not None:
            reader = filter_condition(reader)
        
        # Set up streaming with batches
        streaming_df = reader.select(text_column).collect(streaming=True)
        
        # Process in batches using the streaming iterator
        for batch_df in streaming_df.iter_batches(batch_size=batch_size):
            texts = batch_df[text_column].to_list()
            yield self.embed_documents(texts, show_progress_bar=False)
    
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
"""Document embedding module using transformer models (CPU-optimized by default)."""

from typing import List, Dict, Optional, Union, Any, Callable, Generator, Iterable, Tuple
import os
import numpy as np
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import json
from pathlib import Path
import shutil
import tempfile
import time
import hashlib

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


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
    cache_dir : str, optional
        Directory to cache embeddings, by default uses system temp directory
    use_mmap : bool, optional
        Whether to use memory-mapped storage for large embedding matrices, by default True
    precision : str, optional
        Precision for storing embeddings, by default "float32". Options: "float32", "float16"
    
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
        cache_dir: Optional[str] = None,
        use_mmap: bool = True,
        precision: str = "float32",
    ):
        """Initialize the document embedding model."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.local_model_path = local_model_path
        self.use_mmap = use_mmap
        self.precision = precision
        
        # Validate precision
        if precision not in ["float32", "float16"]:
            raise ValueError("precision must be 'float32' or 'float16'")
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "meno_embeddings"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device - default to CPU
        if device is None:
            # Only use CUDA if explicitly requested and available
            self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        else:
            self.device = device
            
        # Load model
        try:
            if local_model_path and os.path.exists(local_model_path):
                logger.info(f"Loading model from local path: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=self.device)
            else:
                # First try with local model cache handling
                # This ensures models can be loaded when behind firewalls
                local_cache_dir = os.path.expanduser("~/.cache/meno/models")
                os.makedirs(local_cache_dir, exist_ok=True)
                model_cache_path = os.path.join(local_cache_dir, model_name.replace("/", "_"))
                
                if os.path.exists(model_cache_path):
                    logger.info(f"Loading model from local cache: {model_cache_path}")
                    self.model = SentenceTransformer(model_cache_path, device=self.device)
                else:
                    logger.info(f"Downloading model: {model_name}")
                    self.model = SentenceTransformer(model_name, device=self.device)
                    # Save for future use
                    try:
                        self.model.save(model_cache_path)
                        logger.info(f"Saved model to local cache: {model_cache_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save model to local cache: {e}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
            
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Create a model hash for cache lookups
        self.model_hash = hashlib.md5(
            f"{self.model_name}_{self.embedding_dim}".encode()
        ).hexdigest()
        
        # Cache tracking
        self._embedding_cache = {}
    
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
    
    def _get_numpy_dtype(self) -> np.dtype:
        """Get NumPy dtype based on precision setting."""
        return np.float16 if self.precision == "float16" else np.float32
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute a hash for a text string for cache lookups."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _compute_corpus_hash(self, documents: List[str]) -> str:
        """Compute a hash for a collection of documents."""
        # Hash the concatenation of the first 100 characters of each document
        # along with the total count for a fast but reasonably unique hash
        sample = "".join([doc[:100] for doc in documents[:10]])
        return hashlib.md5(
            f"{sample}_{len(documents)}_{self.model_hash}".encode()
        ).hexdigest()
    
    def embed_documents(
        self, 
        documents: Union[List[str], pd.Series, "pl.Series", "pl.DataFrame"],
        show_progress_bar: bool = True,
        text_column: Optional[str] = None,
        cache: bool = True,
        cache_id: Optional[str] = None,
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
        cache : bool, optional
            Whether to cache the embeddings, by default True
        cache_id : Optional[str], optional
            Optional identifier for the cache, by default None
            
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
        
        # Check cache first if enabled
        if cache:
            # Use provided cache_id or compute hash of documents
            if cache_id is None:
                cache_id = self._compute_corpus_hash(documents)
            
            # Check if we have this in memory cache
            if cache_id in self._embedding_cache:
                return self._embedding_cache[cache_id]
                
            # Check if we have this in disk cache
            cache_path = self.cache_dir / f"{cache_id}.npy"
            if cache_path.exists():
                try:
                    # Load with memory mapping if enabled
                    if self.use_mmap:
                        embeddings = np.load(cache_path, mmap_mode='r')
                    else:
                        embeddings = np.load(cache_path)
                    
                    # Store in memory cache for faster future access
                    self._embedding_cache[cache_id] = embeddings
                    
                    logger.info(f"Loaded embeddings from cache: {cache_path}")
                    return embeddings
                except Exception as e:
                    logger.warning(f"Failed to load embeddings from cache: {e}")
        
        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        
        # Convert to requested precision
        if self.precision == "float16" and embeddings.dtype != np.float16:
            embeddings = embeddings.astype(np.float16)
        
        # Cache if enabled
        if cache:
            # Save to disk cache
            cache_path = self.cache_dir / f"{cache_id}.npy"
            
            # Create a memory-mapped array for disk storage if use_mmap is True
            if self.use_mmap:
                # First save to a temporary file to avoid partial writes
                temp_path = self.cache_dir / f"{cache_id}_temp.npy"
                
                # Create new memory-mapped array
                mmap_array = np.memmap(
                    temp_path, 
                    dtype=self._get_numpy_dtype(),
                    mode='w+', 
                    shape=embeddings.shape
                )
                
                # Copy data to memory-mapped array
                mmap_array[:] = embeddings[:]
                mmap_array.flush()
                
                # Move to final location
                if temp_path.exists():
                    shutil.move(temp_path, cache_path)
            else:
                # Regular save
                np.save(cache_path, embeddings)
            
            # Store metadata
            meta_path = self.cache_dir / f"{cache_id}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    "model": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "doc_count": len(documents),
                    "created": time.time(),
                    "precision": self.precision,
                }, f)
            
            # Add to memory cache
            self._embedding_cache[cache_id] = embeddings
        
        return embeddings
    
    def embed_documents_stream(
        self,
        documents_iterator: Iterable[str],
        batch_size: Optional[int] = None,
        cache: bool = False,
        cache_prefix: Optional[str] = None,
    ) -> Generator[Tuple[np.ndarray, List[str]], None, None]:
        """Generate embeddings for a stream of documents, yielding batches of embeddings.
        
        This is useful for large datasets that don't fit in memory.
        
        Parameters
        ----------
        documents_iterator : Iterable[str]
            Iterator yielding document texts
        batch_size : Optional[int], optional
            Batch size for processing, by default None (uses self.batch_size)
        cache : bool, optional
            Whether to cache embeddings, by default False
        cache_prefix : Optional[str], optional
            Prefix for cache identifiers, by default None
        
        Yields
        ------
        Tuple[np.ndarray, List[str]]
            Tuple of (batch of embeddings, batch of documents)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Process documents in batches
        batch = []
        batch_count = 0
        
        for doc in documents_iterator:
            batch.append(doc)
            if len(batch) >= batch_size:
                batch_count += 1
                cache_id = None
                if cache and cache_prefix:
                    cache_id = f"{cache_prefix}_batch_{batch_count}"
                
                embeddings = self.embed_documents(
                    batch, 
                    show_progress_bar=False,
                    cache=cache,
                    cache_id=cache_id
                )
                
                yield embeddings, batch
                batch = []
                
        # Process final batch if there are any remaining documents
        if batch:
            batch_count += 1
            cache_id = None
            if cache and cache_prefix:
                cache_id = f"{cache_prefix}_batch_{batch_count}"
                
            embeddings = self.embed_documents(
                batch, 
                show_progress_bar=False,
                cache=cache,
                cache_id=cache_id
            )
            
            yield embeddings, batch
    
    def embed_polars_streaming(
        self,
        data_path: str,
        text_column: str,
        batch_size: Optional[int] = None,
        filter_condition: Optional[Callable] = None,
        cache: bool = True,
    ) -> Generator[Tuple[np.ndarray, List[str], List[int]], None, None]:
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
        cache : bool, optional
            Whether to cache embeddings, by default True
            
        Yields
        ------
        Tuple[np.ndarray, List[str], List[int]]
            Tuple of (batch of embeddings, batch of documents, batch indices)
            
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
        
        # Create cache prefix from file path and column
        cache_prefix = None
        if cache:
            file_hash = hashlib.md5(data_path.encode()).hexdigest()[:8]
            cache_prefix = f"{file_hash}_{text_column}"
        
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
        # Polars API changed between versions; try compatible approaches
        batch_idx = 0
        try:
            # Newer polars versions use iter_slices(n_rows)
            for batch_df in streaming_df.iter_slices(batch_size):
                pass  # Just to test the API
            # If we got here, we can use iter_slices with n_rows
            batch_iterator = streaming_df.iter_slices(batch_size)
        except (TypeError, AttributeError):
            try:
                # Try older iter_batches API
                for batch_df in streaming_df.iter_batches(batch_size=batch_size):
                    pass  # Just to test the API
                # If we got here, we can use iter_batches
                batch_iterator = streaming_df.iter_batches(batch_size=batch_size)
            except (TypeError, AttributeError):
                # Fall back to chunk-wise processing
                logger.warning("Polars streaming API incompatible; falling back to chunked processing")
                batch_iterator = [streaming_df]
                
        # Process each batch
        for batch_df in batch_iterator:
            texts = batch_df[text_column].to_list()
            
            # Create cache ID for this batch
            cache_id = None
            if cache and cache_prefix:
                cache_id = f"{cache_prefix}_batch_{batch_idx}"
            
            # Generate embeddings
            embeddings = self.embed_documents(
                texts, 
                show_progress_bar=False,
                cache=cache,
                cache_id=cache_id
            )
            
            # Yield with batch index information
            indices = list(range(batch_idx * batch_size, batch_idx * batch_size + len(texts)))
            yield embeddings, texts, indices
            
            batch_idx += 1
    
    def embed_topics(
        self,
        topics: Union[List[str], pd.Series],
        descriptions: Optional[Union[List[str], pd.Series]] = None,
        cache: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of topics, optionally with descriptions.
        
        Parameters
        ----------
        topics : Union[List[str], pd.Series]
            List or Series of topic names
        descriptions : Optional[Union[List[str], pd.Series]], optional
            List or Series of topic descriptions, by default None
            If provided, topics and descriptions are combined
        cache : bool, optional
            Whether to cache the embeddings, by default True
        
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
        
        # Create cache ID for topics
        cache_id = None
        if cache:
            topic_hash = hashlib.md5(
                "".join(texts_to_embed).encode()
            ).hexdigest()[:12]
            cache_id = f"topics_{topic_hash}_{self.model_hash}"
        
        # Generate embeddings
        return self.embed_documents(
            texts_to_embed, 
            show_progress_bar=False,
            cache=cache,
            cache_id=cache_id
        )
    
    def get_cached_embeddings(self, cache_id: str) -> Optional[np.ndarray]:
        """Retrieve embeddings from cache if available.
        
        Parameters
        ----------
        cache_id : str
            Cache identifier
            
        Returns
        -------
        Optional[np.ndarray]
            Cached embeddings if available, otherwise None
        """
        # Check memory cache first
        if cache_id in self._embedding_cache:
            return self._embedding_cache[cache_id]
        
        # Check disk cache
        cache_path = self.cache_dir / f"{cache_id}.npy"
        if cache_path.exists():
            try:
                # Load with memory mapping if enabled
                if self.use_mmap:
                    embeddings = np.load(cache_path, mmap_mode='r')
                else:
                    embeddings = np.load(cache_path)
                
                # Store in memory cache for faster future access
                self._embedding_cache[cache_id] = embeddings
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings from cache: {e}")
        
        return None
    
    def clear_cache(self, cache_id: Optional[str] = None) -> None:
        """Clear embedding cache.
        
        Parameters
        ----------
        cache_id : Optional[str], optional
            Specific cache ID to clear, by default None (clears all)
        """
        if cache_id is not None:
            # Clear specific cache entry
            if cache_id in self._embedding_cache:
                del self._embedding_cache[cache_id]
                
            # Remove from disk cache
            cache_path = self.cache_dir / f"{cache_id}.npy"
            meta_path = self.cache_dir / f"{cache_id}_meta.json"
            
            if cache_path.exists():
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
                
            logger.info(f"Cleared cache for {cache_id}")
        else:
            # Clear all cache
            self._embedding_cache = {}
            
            # Clear disk cache
            for f in self.cache_dir.glob("*.npy"):
                f.unlink()
            for f in self.cache_dir.glob("*_meta.json"):
                f.unlink()
                
            logger.info("Cleared all embedding caches")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the embedding cache.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics and information
        """
        # Count cache files
        npy_files = list(self.cache_dir.glob("*.npy"))
        meta_files = list(self.cache_dir.glob("*_meta.json"))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in npy_files)
        
        # Get metadata for each cache entry
        cache_entries = []
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                # Add file size info
                cache_id = meta_file.stem.replace("_meta", "")
                npy_file = self.cache_dir / f"{cache_id}.npy"
                if npy_file.exists():
                    meta["file_size"] = npy_file.stat().st_size
                    meta["cache_id"] = cache_id
                    cache_entries.append(meta)
            except Exception:
                continue
        
        return {
            "cache_dir": str(self.cache_dir),
            "entry_count": len(npy_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "in_memory_cache_count": len(self._embedding_cache),
            "entries": cache_entries,
        }
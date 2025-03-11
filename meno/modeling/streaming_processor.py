"""
Large-scale streaming processor for topic modeling with Meno.

This module provides utilities for processing and modeling large text datasets
that might not fit in memory, using streaming and batched processing with
optimized memory usage and disk I/O.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterator
import logging
from pathlib import Path
import time
import os
import gc
import json
import shutil  # Add missing import
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Local imports
from ..modeling.embeddings import DocumentEmbedding
from ..modeling.unified_topic_modeling import UnifiedTopicModeler

# Set up logging
logger = logging.getLogger(__name__)


class StreamingProcessor:
    """
    Processor for handling large datasets with streaming and batched processing.
    
    This class provides utilities for processing and modeling large text datasets
    with efficient memory usage through streaming and batched processing.
    
    Attributes:
        embedding_model: Model for generating document embeddings
        topic_model: Topic model instance
        batch_size: Size of batches for processing
        temp_dir: Directory for temporary files
        use_quantization: Whether to use quantization for embeddings
        verbose: Whether to show progress information
    """
    
    def __init__(
        self,
        embedding_model: Union[str, DocumentEmbedding] = "all-MiniLM-L6-v2",
        topic_model: Optional[Union[str, UnifiedTopicModeler]] = "bertopic",
        batch_size: int = 1000,
        temp_dir: Optional[Union[str, Path]] = None,
        use_quantization: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the streaming processor.
        
        Args:
            embedding_model: Model name or instance for document embeddings
            topic_model: Topic model name or instance
            batch_size: Size of batches for processing
            temp_dir: Directory for temporary files
            use_quantization: Whether to use quantization for embeddings
            verbose: Whether to show progress information
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_quantization = use_quantization
        
        # Set up temporary directory
        if temp_dir is None:
            self.temp_dir = Path("./meno_temp")
        else:
            self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = DocumentEmbedding(
                model_name=embedding_model,
                device="cpu"  # Always start with CPU for better compatibility
            )
        else:
            self.embedding_model = embedding_model
        
        # Set up topic model
        if isinstance(topic_model, str):
            self.topic_model = UnifiedTopicModeler(
                method=topic_model,
                embedding_model=self.embedding_model
            )
        else:
            self.topic_model = topic_model
            
        # State tracking
        self.is_fitted = False
        self.total_documents = 0
        self.total_batches = 0
        self.embedding_files = []
        
        # Performance tracking
        self.timing = {
            "embedding": 0.0,
            "modeling": 0.0,
            "total": 0.0
        }
        
    def stream_from_file(
        self,
        file_path: Union[str, Path],
        text_column: str,
        id_column: Optional[str] = None,
        filter_condition: Optional[str] = None,
        file_format: Optional[str] = None,
        low_memory: bool = True,
    ) -> Iterator[Tuple[List[str], List]]:
        """
        Stream documents from a file in batches.
        
        Args:
            file_path: Path to the data file
            text_column: Name of the column containing document text
            id_column: Name of the column containing document IDs
            filter_condition: Optional filter expression for Polars
            file_format: Format of the file ('csv', 'parquet', etc.)
            low_memory: Whether to use low memory mode
        
        Yields:
            Tuple of (batch of documents, batch of document IDs)
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required for streaming from files. "
                            "Install with 'pip install polars'")
        
        file_path = Path(file_path)
        
        # Determine file format if not provided
        if file_format is None:
            file_format = file_path.suffix.lower().lstrip('.')
            if file_format == '':
                raise ValueError("Could not determine file format. Please specify explicitly.")
        
        # Set up lazy loading
        if file_format == 'csv':
            df_lazy = pl.scan_csv(file_path, low_memory=low_memory)
        elif file_format == 'parquet':
            df_lazy = pl.scan_parquet(file_path)
        elif file_format == 'json':
            df_lazy = pl.scan_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Apply filter if provided
        if filter_condition:
            df_lazy = df_lazy.filter(pl.expr(filter_condition))
        
        # Choose columns to select
        if id_column:
            df_lazy = df_lazy.select([id_column, text_column])
        else:
            df_lazy = df_lazy.select([text_column])
        
        # Stream in batches
        total_count = None
        if self.verbose:
            # Try to count rows (note: this may be slow for large files)
            try:
                total_count = df_lazy.collect().height
                logger.info(f"Streaming from file with {total_count} documents")
            except Exception:
                logger.info("Streaming from file (unknown document count)")
        
        progress_bar = None
        if self.verbose and total_count:
            progress_bar = tqdm(total=total_count, desc="Streaming documents")
        
        # Stream batches
        batch_count = 0
        try:
            for batch in df_lazy.collect_batches(batch_size=self.batch_size):
                batch_df = batch.to_pandas()
                docs = batch_df[text_column].tolist()
                
                if id_column:
                    ids = batch_df[id_column].tolist()
                else:
                    # Generate sequential IDs if not provided
                    start_idx = batch_count * self.batch_size
                    ids = list(range(start_idx, start_idx + len(docs)))
                
                if progress_bar:
                    progress_bar.update(len(docs))
                
                batch_count += 1
                yield docs, ids
        finally:
            if progress_bar:
                progress_bar.close()
    
    def create_embeddings_stream(
        self,
        documents_stream: Iterator[Tuple[List[str], List]],
        save_embeddings: bool = False,
        embedding_file: Optional[Union[str, Path]] = None,
        cache_embeddings: bool = True,
        use_mmap: bool = True,
    ) -> Iterator[Tuple[np.ndarray, List]]:
        """
        Create embeddings from a stream of documents.
        
        Args:
            documents_stream: Iterator yielding batches of documents
            save_embeddings: Whether to save embeddings to disk
            embedding_file: Path to save embeddings
            cache_embeddings: Whether to cache embeddings for faster reuse
            use_mmap: Whether to use memory mapping for large files
        
        Yields:
            Tuple of (batch of embeddings, batch of document IDs)
        """
        start_time = time.time()
        
        # Prepare embedding file if saving
        if save_embeddings:
            if embedding_file is None:
                # Generate a timestamped filename
                timestamp = int(time.time())
                embedding_file = self.temp_dir / f"embeddings_{timestamp}.npy"
            else:
                embedding_file = Path(embedding_file)
            
            self.embedding_files.append(embedding_file)
            
            # Create file metadata for tracking
            embedding_meta = {
                "created": time.time(),
                "doc_count": 0,
                "batch_count": 0,
                "embedding_dim": self.embedding_model.embedding_dimension,
                "model": str(self.embedding_model),
                "precision": "float16" if self.use_quantization else "float32",
            }
            
            # Check if we're appending to existing data
            if embedding_file.exists() and use_mmap:
                try:
                    # Get the existing shape to determine where to start appending
                    existing_embeddings = np.load(embedding_file, mmap_mode='r')
                    doc_count = existing_embeddings.shape[0]
                    embedding_meta["doc_count"] = doc_count
                    
                    # Load existing IDs
                    ids_file = embedding_file.parent / f"{embedding_file.stem}_ids.json"
                    if ids_file.exists():
                        with open(ids_file, 'r') as f:
                            ids_all = json.load(f)
                    else:
                        ids_all = []
                    
                    # Close the memory map
                    del existing_embeddings
                except Exception as e:
                    logger.warning(f"Error accessing existing embeddings file, creating new: {e}")
                    doc_count = 0
                    ids_all = []
            else:
                doc_count = 0
                ids_all = []
        
        # Generate a unique cache prefix for this batch processing job
        cache_prefix = None
        if cache_embeddings:
            timestamp = int(time.time())
            cache_prefix = f"stream_{timestamp}"
        
        # Stream batches
        batch_num = 0
        total_doc_count = 0
        
        for docs, ids in documents_stream:
            if not docs:
                continue
                
            # Generate embeddings with caching if enabled
            cache_id = f"{cache_prefix}_batch_{batch_num}" if cache_prefix else None
            embeddings = self.embedding_model.embed_documents(
                docs, 
                show_progress_bar=False,
                cache=cache_embeddings,
                cache_id=cache_id
            )
            
            # Apply quantization if requested
            if self.use_quantization and embeddings.dtype != np.float16:
                embeddings = embeddings.astype(np.float16)
            
            # Save if requested
            if save_embeddings:
                if use_mmap and doc_count > 0:
                    # Append to existing memory-mapped file
                    current_size = doc_count
                    new_size = current_size + len(docs)
                    
                    # Get current array shape and prepare new shape
                    embed_dim = embeddings.shape[1]
                    
                    # Create new memory-mapped array with increased size
                    temp_file = embedding_file.parent / f"{embedding_file.stem}_temp.npy"
                    
                    # Step 1: Create a new memory-mapped file with the combined size
                    mmap_dtype = np.float16 if self.use_quantization else np.float32
                    mm_array = np.memmap(
                        temp_file, 
                        dtype=mmap_dtype,
                        mode='w+', 
                        shape=(new_size, embed_dim)
                    )
                    
                    # Step 2: Copy existing data
                    if current_size > 0:
                        existing = np.load(embedding_file, mmap_mode='r')
                        mm_array[:current_size] = existing[:]
                        del existing  # Close the memory map
                    
                    # Step 3: Add new data
                    mm_array[current_size:new_size] = embeddings
                    mm_array.flush()
                    
                    # Step 4: Replace the original file
                    del mm_array  # Close the memory map
                    
                    # Move to final location
                    shutil.move(temp_file, embedding_file)
                    
                    # Update IDs
                    ids_all.extend(ids)
                    with open(embedding_file.parent / f"{embedding_file.stem}_ids.json", 'w') as f:
                        json.dump(ids_all, f)
                    
                    # Update doc count for next iteration
                    doc_count = new_size
                else:
                    # Legacy approach for compatibility: load all and append
                    if batch_num == 0 and not embedding_file.exists():
                        # First batch, create new file
                        if self.use_quantization and embeddings.dtype != np.float16:
                            embeddings_all = embeddings.astype(np.float16)
                        else:
                            embeddings_all = embeddings
                        ids_all = ids
                    else:
                        # Load existing, append, save
                        if embedding_file.exists():
                            embeddings_all = np.load(embedding_file)
                            # Load existing IDs
                            ids_path = embedding_file.parent / f"{embedding_file.stem}_ids.json"
                            if ids_path.exists():
                                with open(ids_path, 'r') as f:
                                    ids_all = json.load(f)
                            else:
                                ids_all = []
                        else:
                            embeddings_all = np.empty((0, embeddings.shape[1]))
                            ids_all = []
                            
                        # Append new data
                        embeddings_all = np.vstack([embeddings_all, embeddings])
                        ids_all.extend(ids)
                    
                    # Save to disk
                    np.save(embedding_file, embeddings_all)
                    with open(embedding_file.parent / f"{embedding_file.stem}_ids.json", 'w') as f:
                        json.dump(ids_all, f)
                
                # Update metadata
                embedding_meta["doc_count"] += len(docs)
                embedding_meta["batch_count"] += 1
                
                # Save metadata occasionally
                if batch_num % 10 == 0:
                    with open(embedding_file.parent / f"{embedding_file.stem}_meta.json", 'w') as f:
                        json.dump(embedding_meta, f)
            
            total_doc_count += len(docs)
            batch_num += 1
            
            yield embeddings, ids
        
        # Final metadata save if needed
        if save_embeddings and batch_num > 0:
            with open(embedding_file.parent / f"{embedding_file.stem}_meta.json", 'w') as f:
                json.dump(embedding_meta, f)
        
        # Update timing and stats
        self.timing["embedding"] += time.time() - start_time
        self.total_documents += total_doc_count
        self.total_batches += batch_num
        
        if self.verbose:
            logger.info(f"Processed {total_doc_count} documents in {batch_num} batches")
            logger.info(f"Embedding time: {self.timing['embedding']:.2f} seconds")
    
    def fit_topic_model_stream(
        self,
        embeddings_stream: Iterator[Tuple[np.ndarray, List]],
        n_topics: Optional[int] = None,
        min_topic_size: int = 20,
        method: Optional[str] = None,
        save_model: bool = True,
        model_path: Optional[Union[str, Path]] = None,
    ) -> UnifiedTopicModeler:
        """
        Fit a topic model using streaming embeddings.
        
        Args:
            embeddings_stream: Iterator yielding batches of embeddings
            n_topics: Number of topics to extract
            min_topic_size: Minimum size of topics
            method: Topic modeling method to use
            save_model: Whether to save the model
            model_path: Path to save the model
        
        Returns:
            Fitted topic model
        """
        start_time = time.time()
        
        # Collect all embeddings first (we need them all for clustering)
        all_embeddings = []
        all_ids = []
        
        if self.verbose:
            logger.info("Collecting embeddings for topic modeling...")
        
        for embeddings, ids in embeddings_stream:
            all_embeddings.append(embeddings)
            all_ids.extend(ids)
        
        # Stack all embeddings
        if len(all_embeddings) > 0:
            embeddings_matrix = np.vstack(all_embeddings)
        else:
            raise ValueError("No embeddings were provided")
        
        # Free memory
        del all_embeddings
        gc.collect()
        
        # If method is specified, update the topic model
        if method and hasattr(self.topic_model, 'method') and method != self.topic_model.method:
            if self.verbose:
                logger.info(f"Switching topic model method from {self.topic_model.method} to {method}")
            
            # Create a new topic model with the specified method
            self.topic_model = UnifiedTopicModeler(
                method=method,
                embedding_model=self.embedding_model,
                verbose=self.verbose
            )
        
        # Set topic model parameters if provided
        params = {}
        if n_topics is not None:
            params['n_topics'] = n_topics
        if min_topic_size is not None:
            params['min_topic_size'] = min_topic_size
        
        # Fit the topic model
        if self.verbose:
            logger.info(f"Fitting topic model on {embeddings_matrix.shape[0]} documents...")
        
        # Fake documents for BERTopic (it expects documents but actually only uses embeddings)
        # This is more efficient than passing actual documents
        fake_docs = [f"doc_{i}" for i in range(len(embeddings_matrix))]
        
        try:
            self.topic_model.fit(fake_docs, embeddings=embeddings_matrix, **params)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            raise
        
        # Save model if requested
        if save_model and self.is_fitted:
            if model_path is None:
                timestamp = int(time.time())
                model_path = self.temp_dir / f"topic_model_{timestamp}"
            
            self.topic_model.save(model_path)
            
            if self.verbose:
                logger.info(f"Saved topic model to {model_path}")
        
        # Update timing
        self.timing["modeling"] = time.time() - start_time
        self.timing["total"] = self.timing["embedding"] + self.timing["modeling"]
        
        if self.verbose:
            logger.info(f"Topic modeling time: {self.timing['modeling']:.2f} seconds")
            logger.info(f"Total processing time: {self.timing['total']:.2f} seconds")
            
            # Log topic model information
            topic_info = self.topic_model.get_topic_info()
            num_topics = len(topic_info[topic_info["Topic"] >= 0])
            logger.info(f"Found {num_topics} topics")
        
        return self.topic_model
    
    def process_file(
        self,
        file_path: Union[str, Path],
        text_column: str,
        id_column: Optional[str] = None,
        filter_condition: Optional[str] = None,
        n_topics: Optional[int] = None,
        min_topic_size: int = 20,
        topic_method: str = "bertopic",
        save_results: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Process a file end-to-end: streaming, embedding, and topic modeling.
        
        Args:
            file_path: Path to the data file
            text_column: Name of the column containing document text
            id_column: Name of the column containing document IDs
            filter_condition: Optional filter expression for Polars
            n_topics: Number of topics to extract
            min_topic_size: Minimum size of topics
            topic_method: Topic modeling method to use
            save_results: Whether to save results
            output_dir: Directory to save results
        
        Returns:
            Dictionary with processing results and metadata
        """
        total_start_time = time.time()
        
        # Set up output directory
        if save_results:
            if output_dir is None:
                output_dir = self.temp_dir / "results"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Stream documents from file
        doc_stream = self.stream_from_file(
            file_path=file_path,
            text_column=text_column,
            id_column=id_column,
            filter_condition=filter_condition
        )
        
        # Step 2: Create embeddings
        embedding_file = None
        if save_results:
            timestamp = int(time.time())
            embedding_file = output_dir / f"embeddings_{timestamp}.npy"
        
        embeddings_stream = self.create_embeddings_stream(
            documents_stream=doc_stream,
            save_embeddings=save_results,
            embedding_file=embedding_file
        )
        
        # Step 3: Fit topic model
        model_path = None
        if save_results:
            model_path = output_dir / f"topic_model_{int(time.time())}"
        
        self.fit_topic_model_stream(
            embeddings_stream=embeddings_stream,
            n_topics=n_topics,
            min_topic_size=min_topic_size,
            method=topic_method,
            save_model=save_results,
            model_path=model_path
        )
        
        # Collect results
        results = {
            "file_processed": str(file_path),
            "document_count": self.total_documents,
            "batch_count": self.total_batches,
            "timing": self.timing.copy(),
            "total_time": time.time() - total_start_time,
            "topic_count": len(self.topic_model.get_topic_info()[self.topic_model.get_topic_info()["Topic"] >= 0]),
        }
        
        if save_results:
            results["embedding_file"] = str(embedding_file) if embedding_file else None
            results["model_path"] = str(model_path) if model_path else None
            
            # Save results metadata
            with open(output_dir / f"metadata_{int(time.time())}.json", 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def update_model_with_stream(
        self,
        documents_stream: Iterator[Tuple[List[str], List]],
        update_type: str = "online",
        update_interval: int = 5,
    ) -> None:
        """
        Update the topic model with streaming documents.
        
        Args:
            documents_stream: Iterator yielding batches of documents
            update_type: Type of update ('online' or 'batch')
            update_interval: Number of batches between updates
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Will fit with the first batch.")
        
        batch_count = 0
        doc_buffer = []
        id_buffer = []
        
        for docs, ids in documents_stream:
            # Add to buffer
            doc_buffer.extend(docs)
            id_buffer.extend(ids)
            batch_count += 1
            
            # Check if it's time to update
            if (update_type == 'online' and batch_count % update_interval == 0) or \
               (update_type == 'batch' and len(doc_buffer) >= self.batch_size * update_interval):
                
                # Create embeddings
                embeddings = self.embedding_model.embed_documents(doc_buffer)
                
                # Update or fit model
                if not self.is_fitted:
                    # Initial fit
                    fake_docs = [f"doc_{i}" for i in range(len(embeddings))]
                    self.topic_model.fit(fake_docs, embeddings=embeddings)
                    self.is_fitted = True
                    
                    if self.verbose:
                        logger.info(f"Fitted initial model with {len(doc_buffer)} documents")
                else:
                    # Update existing model
                    if hasattr(self.topic_model, 'update_topics'):
                        # Get document-topic mapping
                        doc_topics, _ = self.topic_model.transform(doc_buffer, embeddings=embeddings)
                        primary_topics = np.argmax(doc_topics, axis=1)
                        
                        self.topic_model.update_topics(doc_buffer, primary_topics)
                        
                        if self.verbose:
                            logger.info(f"Updated model with {len(doc_buffer)} documents")
                    else:
                        logger.warning("Model does not support online updates.")
                
                # Clear buffer
                doc_buffer = []
                id_buffer = []
        
        # Process any remaining documents
        if doc_buffer:
            embeddings = self.embedding_model.embed_documents(doc_buffer)
            
            if not self.is_fitted:
                fake_docs = [f"doc_{i}" for i in range(len(embeddings))]
                self.topic_model.fit(fake_docs, embeddings=embeddings)
                self.is_fitted = True
            else:
                if hasattr(self.topic_model, 'update_topics'):
                    doc_topics, _ = self.topic_model.transform(doc_buffer, embeddings=embeddings)
                    primary_topics = np.argmax(doc_topics, axis=1)
                    self.topic_model.update_topics(doc_buffer, primary_topics)
    
    def clean_temp_files(self) -> None:
        """Clean temporary files created during processing."""
        # Remove embedding files
        for file in self.embedding_files:
            if Path(file).exists():
                Path(file).unlink()
                
                # Also remove ID files
                id_file = Path(file).parent / f"{Path(file).stem}_ids.json"
                if id_file.exists():
                    id_file.unlink()
        
        self.embedding_files = []
        
        if self.verbose:
            logger.info("Cleaned temporary files")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processing."""
        return {
            "document_count": self.total_documents,
            "batch_count": self.total_batches,
            "timing": self.timing.copy(),
            "embedding_dimension": self.embedding_model.embedding_dimension,
            "model_fitted": self.is_fitted,
        }
    
    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_column: str = "topic",
        output_probs_column: Optional[str] = "topic_probs",
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Apply the fitted topic model to a DataFrame.
        
        Args:
            df: DataFrame with documents
            text_column: Name of the column containing document text
            output_column: Name of the column to store topic assignments
            output_probs_column: Name of the column to store topic probabilities
            batch_size: Batch size (defaults to instance batch_size)
        
        Returns:
            DataFrame with topic assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before applying to data")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            docs = batch[text_column].tolist()
            
            # Get embeddings
            embeddings = self.embedding_model.embed_documents(docs)
            
            # Transform with topic model
            doc_topics, probs = self.topic_model.transform(docs, embeddings=embeddings)
            
            # Store results
            for j, (row_idx, row) in enumerate(batch.iterrows()):
                result_row = row.to_dict()
                result_row[output_column] = doc_topics[j][0]  # Primary topic
                
                if output_probs_column:
                    result_row[output_probs_column] = probs[j][0]  # Probability of primary topic
                
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def generate_embeddings_file(
        self,
        file_path: Union[str, Path],
        text_column: str,
        id_column: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        filter_condition: Optional[str] = None,
    ) -> Path:
        """
        Generate embeddings file from a data file.
        
        Args:
            file_path: Path to the data file
            text_column: Name of the column containing document text
            id_column: Name of the column containing document IDs
            output_file: Path to save embeddings
            filter_condition: Optional filter expression for Polars
        
        Returns:
            Path to the saved embeddings file
        """
        # Set up output file
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.temp_dir / f"embeddings_{timestamp}.npy"
        else:
            output_file = Path(output_file)
        
        # Stream documents
        doc_stream = self.stream_from_file(
            file_path=file_path,
            text_column=text_column,
            id_column=id_column,
            filter_condition=filter_condition
        )
        
        # Create embeddings
        list(self.create_embeddings_stream(
            documents_stream=doc_stream,
            save_embeddings=True,
            embedding_file=output_file
        ))
        
        if self.verbose:
            logger.info(f"Generated embeddings file at {output_file}")
            logger.info(f"Document count: {self.total_documents}")
        
        return output_file
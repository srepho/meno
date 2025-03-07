"""UMAP dimensionality reduction for visualization with memory-mapped caching."""

from typing import Optional, Union, Tuple, Dict, Any, List, Callable
import numpy as np
import pandas as pd
import umap
import os
from pathlib import Path
import tempfile
import logging
import hashlib
import json
import time
import shutil

# Setup logging
logger = logging.getLogger(__name__)


class UMAPCache:
    """Cache for UMAP projections to avoid recomputation."""
    
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        use_mmap: bool = True,
    ):
        """Initialize UMAP cache.
        
        Parameters
        ----------
        cache_dir : Optional[str], optional
            Directory to store cached projections, by default None
            If None, uses system temp directory
        use_mmap : bool, optional
            Whether to use memory mapping for cached projections, by default True
        """
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "meno_umap_cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_mmap = use_mmap
        
        # In-memory cache for faster lookups
        self._cache = {}
    
    def _compute_cache_key(
        self, 
        embeddings: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        n_components: int,
        metric: str,
    ) -> str:
        """Compute a cache key for UMAP parameters and data.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to project
        n_neighbors : int
            UMAP n_neighbors parameter
        min_dist : float
            UMAP min_dist parameter
        n_components : int
            UMAP n_components parameter
        metric : str
            UMAP distance metric
            
        Returns
        -------
        str
            Cache key
        """
        # Compute a hash of the embeddings
        # Use shape, mean, and std for a fast approximation
        # This avoids full data hashing for large matrices
        embedding_shape = embeddings.shape
        embedding_stats = (
            np.mean(embeddings),
            np.std(embeddings),
            np.min(embeddings),
            np.max(embeddings),
        )
        
        # Create a string with all parameters and data characteristics
        param_str = (
            f"shape:{embedding_shape}_"
            f"stats:{embedding_stats}_"
            f"n_neighbors:{n_neighbors}_"
            f"min_dist:{min_dist}_"
            f"n_components:{n_components}_"
            f"metric:{metric}"
        )
        
        # Hash the parameter string
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(
        self,
        embeddings: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        n_components: int,
        metric: str,
    ) -> Optional[np.ndarray]:
        """Get cached UMAP projection if available.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to project
        n_neighbors : int
            UMAP n_neighbors parameter
        min_dist : float
            UMAP min_dist parameter
        n_components : int
            UMAP n_components parameter
        metric : str
            UMAP distance metric
            
        Returns
        -------
        Optional[np.ndarray]
            Cached projection if available, otherwise None
        """
        cache_key = self._compute_cache_key(
            embeddings, n_neighbors, min_dist, n_components, metric
        )
        
        # Check memory cache first
        if cache_key in self._cache:
            logger.info(f"Found UMAP projection in memory cache: {cache_key}")
            return self._cache[cache_key]
        
        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            try:
                # Load with memory mapping if enabled
                if self.use_mmap:
                    projection = np.load(cache_path, mmap_mode='r')
                else:
                    projection = np.load(cache_path)
                
                # Add to memory cache
                self._cache[cache_key] = projection
                logger.info(f"Loaded UMAP projection from disk: {cache_key}")
                return projection
            except Exception as e:
                logger.warning(f"Failed to load cached UMAP projection: {e}")
        
        return None
    
    def save(
        self,
        projection: np.ndarray,
        embeddings: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        n_components: int,
        metric: str,
    ) -> None:
        """Save UMAP projection to cache.
        
        Parameters
        ----------
        projection : np.ndarray
            UMAP projection to cache
        embeddings : np.ndarray
            Original embeddings
        n_neighbors : int
            UMAP n_neighbors parameter
        min_dist : float
            UMAP min_dist parameter
        n_components : int
            UMAP n_components parameter
        metric : str
            UMAP distance metric
        """
        cache_key = self._compute_cache_key(
            embeddings, n_neighbors, min_dist, n_components, metric
        )
        
        # Add to memory cache
        self._cache[cache_key] = projection
        
        # Save to disk
        cache_path = self.cache_dir / f"{cache_key}.npy"
        meta_path = self.cache_dir / f"{cache_key}_meta.json"
        
        try:
            # Save projection
            np.save(cache_path, projection)
            
            # Save metadata
            with open(meta_path, 'w') as f:
                json.dump({
                    "shape": embeddings.shape,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "n_components": n_components,
                    "metric": metric,
                    "created": time.time(),
                }, f)
                
            logger.info(f"Saved UMAP projection to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save UMAP projection to cache: {e}")
    
    def clear(self, cache_key: Optional[str] = None) -> None:
        """Clear cache.
        
        Parameters
        ----------
        cache_key : Optional[str], optional
            Specific cache key to clear, by default None (clears all)
        """
        if cache_key is not None:
            # Clear specific entry
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            cache_path = self.cache_dir / f"{cache_key}.npy"
            meta_path = self.cache_dir / f"{cache_key}_meta.json"
            
            if cache_path.exists():
                os.remove(cache_path)
            if meta_path.exists():
                os.remove(meta_path)
                
            logger.info(f"Cleared UMAP cache for {cache_key}")
        else:
            # Clear all cache
            self._cache = {}
            
            for f in self.cache_dir.glob("*.npy"):
                os.remove(f)
            for f in self.cache_dir.glob("*_meta.json"):
                os.remove(f)
                
            logger.info("Cleared all UMAP cache")


# Global cache instance
_umap_cache = UMAPCache()


def get_umap_cache() -> UMAPCache:
    """Get the global UMAP cache instance.
    
    Returns
    -------
    UMAPCache
        Global UMAP cache
    """
    return _umap_cache


def set_umap_cache_dir(cache_dir: str) -> None:
    """Set the directory for UMAP cache.
    
    Parameters
    ----------
    cache_dir : str
        Directory for UMAP cache
    """
    global _umap_cache
    _umap_cache = UMAPCache(cache_dir=cache_dir)


def create_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """Create a UMAP projection of document embeddings for visualization.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings with shape (n_documents, embedding_dim)
    n_neighbors : int, optional
        Number of neighbors for UMAP, by default 15
    min_dist : float, optional
        Minimum distance for UMAP, by default 0.1
    n_components : int, optional
        Number of components for projection, by default 2
    metric : str, optional
        Distance metric for UMAP, by default "cosine"
    random_state : int, optional
        Random state for reproducibility, by default 42
    use_cache : bool, optional
        Whether to use caching for UMAP projections, by default True
    progress_callback : Optional[Callable[[float], None]], optional
        Callback for progress updates, by default None
    
    Returns
    -------
    np.ndarray
        UMAP projection with shape (n_documents, n_components)
    """
    if use_cache:
        # Check cache
        cache = get_umap_cache()
        cached_projection = cache.get(
            embeddings, n_neighbors, min_dist, n_components, metric
        )
        
        if cached_projection is not None:
            return cached_projection
    
    # No cache hit, compute projection
    if progress_callback:
        progress_callback(0.1)  # Starting
    
    # Create UMAP instance
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    
    # Fit and transform embeddings
    projection = reducer.fit_transform(embeddings)
    
    if progress_callback:
        progress_callback(1.0)  # Complete
    
    # Cache the result if caching is enabled
    if use_cache:
        cache = get_umap_cache()
        cache.save(
            projection, 
            embeddings,
            n_neighbors,
            min_dist,
            n_components,
            metric,
        )
    
    return projection
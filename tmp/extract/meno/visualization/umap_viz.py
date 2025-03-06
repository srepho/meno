"""UMAP dimensionality reduction for visualization."""

from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
import umap


def create_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
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
    
    Returns
    -------
    np.ndarray
        UMAP projection with shape (n_documents, n_components)
    """
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
    
    return projection
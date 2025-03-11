"""Lightweight topic modeling approaches that don't require heavy dependencies.

This module provides several simplified topic modeling approaches:

1. SimpleTopicModel: K-Means clustering on document embeddings
2. TFIDFTopicModel: TF-IDF vectorization with K-Means clustering
3. NMFTopicModel: Non-negative matrix factorization for topic discovery
4. LSATopicModel: Latent Semantic Analysis (LSA) using TruncatedSVD

These models are designed to be lightweight alternatives to more complex 
approaches like BERTopic and Top2Vec, without requiring additional dependencies
beyond scikit-learn.
"""

from .lightweight_models import SimpleTopicModel, TFIDFTopicModel, NMFTopicModel, LSATopicModel

__all__ = ["SimpleTopicModel", "TFIDFTopicModel", "NMFTopicModel", "LSATopicModel"]
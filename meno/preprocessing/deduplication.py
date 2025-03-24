"""Text deduplication utilities for preprocessing text data."""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


class TextDeduplicator:
    """Class for deduplicating text data.
    
    This class handles both exact and fuzzy deduplication of text data,
    preserving the mapping between original and deduplicated documents.
    
    Parameters
    ----------
    similarity_threshold : float, optional
        Threshold for fuzzy matching, by default 0.85
    hash_method : str, optional
        Method to use for hashing text for exact matching, by default "text"
        Options: "text" (hash of text string), "tokens" (hash of normalized tokens)
    
    Attributes
    ----------
    similarity_threshold : float
        Threshold for fuzzy matching
    hash_method : str
        Method used for hashing text
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        hash_method: str = "text",
    ):
        """Initialize the text deduplicator with specified options."""
        self.similarity_threshold = similarity_threshold
        self.hash_method = hash_method
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio between two strings.
        
        Parameters
        ----------
        text1 : str
            First text string
        text2 : str
            Second text string
            
        Returns
        -------
        float
            Similarity ratio (0-1) where 1 is identical
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def exact_deduplicate(
        self, 
        data: pd.DataFrame,
        text_column: str,
        id_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, int], List[List[int]]]:
        """Deduplicate a dataset based on exact text matching.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing text data
        text_column : str
            Name of the column containing text to deduplicate
        id_column : Optional[str], optional
            Name of a column to use as a unique identifier, by default None
            If None, the DataFrame index will be used
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, int], List[List[int]]]
            - Deduplicated DataFrame
            - Mapping from duplicate indices to their representative indices
            - Groups of duplicate indices
        """
        # Make a copy of the input dataframe
        df = data.copy()
        
        # Use index as ID if not specified
        if id_column is None:
            df['_temp_id'] = df.index
            id_column = '_temp_id'
        
        # Create a hash of each document text for deduplication
        if self.hash_method == "text":
            df['_doc_hash'] = df[text_column].apply(lambda x: hash(str(x) if x is not None else ""))
        else:
            # For token method, we'd implement token normalization and hashing
            df['_doc_hash'] = df[text_column].apply(lambda x: hash(str(x) if x is not None else ""))
        
        # Find duplicate groups
        duplicate_groups = []
        hash_to_indices = {}
        
        # Group by hash
        for idx, row in df.iterrows():
            doc_hash = row['_doc_hash']
            if doc_hash in hash_to_indices:
                hash_to_indices[doc_hash].append(idx)
            else:
                hash_to_indices[doc_hash] = [idx]
        
        # Create duplicate groups
        for hash_val, indices in hash_to_indices.items():
            if len(indices) > 1:  # Only include groups with duplicates
                duplicate_groups.append(indices)
        
        # Create mapping from duplicate to representative
        duplicate_map = {}
        for group in duplicate_groups:
            representative_idx = group[0]
            for idx in group[1:]:
                duplicate_map[idx] = representative_idx
        
        # Create deduplicated dataset by keeping only unique hash values
        deduplicated = df.drop_duplicates(subset=['_doc_hash']).copy()
        
        # Add original index for reference
        df['original_index'] = range(len(df))
        deduplicated['original_index'] = deduplicated.index
        
        # Clean up temporary columns if needed
        if '_temp_id' in deduplicated.columns:
            deduplicated = deduplicated.drop('_temp_id', axis=1)
        if '_doc_hash' in deduplicated.columns:
            deduplicated = deduplicated.drop('_doc_hash', axis=1)
        
        return deduplicated, duplicate_map, duplicate_groups
    
    def fuzzy_deduplicate(
        self,
        data: pd.DataFrame,
        text_column: str,
        id_column: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, int], List[pd.DataFrame]]:
        """Deduplicate a dataset based on fuzzy text matching.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing text data
        text_column : str
            Name of the column containing text to deduplicate
        id_column : Optional[str], optional
            Name of a column to use as a unique identifier, by default None
            If None, the DataFrame index will be used
        threshold : Optional[float], optional
            Similarity threshold, by default None
            If None, uses the value set in the constructor
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, int], List[pd.DataFrame]]
            - Deduplicated DataFrame
            - Mapping from duplicate indices to their representative indices
            - Groups of similar documents as DataFrames
        """
        # Set threshold
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Make a copy of the input dataframe
        df = data.copy()
        
        # Initialize results
        similarity_groups = []
        processed = set()
        
        # Process each document
        for i, row1 in df.iterrows():
            if i in processed:
                continue
                
            text1 = row1[text_column]
            if pd.isna(text1):
                text1 = ""
                
            group = [i]
            processed.add(i)
            
            # Compare with remaining documents
            for j, row2 in df.iloc[i+1:].iterrows():
                if j in processed:
                    continue
                    
                text2 = row2[text_column]
                if pd.isna(text2):
                    text2 = ""
                
                # Calculate similarity
                similarity = self.calculate_similarity(text1, text2)
                
                # If similar enough, add to group
                if similarity >= threshold:
                    group.append(j)
                    processed.add(j)
            
            # Only record groups with duplicates
            if len(group) > 1:
                similarity_groups.append(group)
        
        # Convert groups to DataFrames
        fuzzy_groups = []
        for group_indices in similarity_groups:
            group_docs = df.iloc[group_indices].copy()
            fuzzy_groups.append(group_docs)
        
        # Create a mapping to track which rows to keep
        keep_indices = set(range(len(df)))
        duplicate_map = {}  # Maps removed indices to their representative
        
        # For each group, keep only the first document and map others to it
        for group in similarity_groups:
            representative_idx = group[0]
            
            for idx in group[1:]:
                keep_indices.discard(idx)
                duplicate_map[idx] = representative_idx
        
        # Create deduplicated dataset
        deduplicated = df.iloc[list(keep_indices)].copy()
        
        # Add original index for reference
        df['original_index'] = range(len(df))
        deduplicated['original_index'] = deduplicated.index
        
        return deduplicated, duplicate_map, fuzzy_groups

    def deduplicate(
        self,
        data: pd.DataFrame,
        text_column: str,
        method: str = "exact",
        threshold: Optional[float] = None,
        id_column: Optional[str] = None,
        preserve_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, int], Union[List[List[int]], List[pd.DataFrame]]]:
        """Deduplicate a dataset using either exact or fuzzy matching.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing text data
        text_column : str
            Name of the column containing text to deduplicate
        method : str, optional
            Deduplication method to use, by default "exact"
            Options: "exact", "fuzzy"
        threshold : Optional[float], optional
            Similarity threshold for fuzzy matching, by default None
            If None, uses the value set in the constructor
        id_column : Optional[str], optional
            Name of a column to use as a unique identifier, by default None
            If None, the DataFrame index will be used
        preserve_columns : Optional[List[str]], optional
            Columns to preserve in the deduplicated DataFrame, by default None
            If None, all columns are preserved
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, int], Union[List[List[int]], List[pd.DataFrame]]]
            - Deduplicated DataFrame
            - Mapping from duplicate indices to their representative indices
            - Groups of similar documents (format depends on method)
        """
        # Validate parameters
        if method not in ["exact", "fuzzy"]:
            raise ValueError(f"Invalid method: {method}. Must be one of ['exact', 'fuzzy']")
        
        # Make sure text column exists
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not in DataFrame")
        
        # If preserve columns specified, validate they exist
        if preserve_columns:
            missing_cols = [col for col in preserve_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not in DataFrame")
        
        # Call appropriate method
        if method == "exact":
            deduplicated_df, duplicate_map, groups = self.exact_deduplicate(
                data, text_column, id_column
            )
        else:  # fuzzy
            deduplicated_df, duplicate_map, groups = self.fuzzy_deduplicate(
                data, text_column, id_column, threshold
            )
        
        # Filter columns if specified
        if preserve_columns:
            keep_cols = list(preserve_columns)
            if 'original_index' not in keep_cols:
                keep_cols.append('original_index')
            deduplicated_df = deduplicated_df[keep_cols]
        
        return deduplicated_df, duplicate_map, groups
    
    def map_results_to_full_dataset(
        self,
        original_df: pd.DataFrame,
        deduplicated_results: pd.DataFrame,
        duplicate_map: Dict[int, int],
        result_columns: List[str],
    ) -> pd.DataFrame:
        """Map results from deduplicated data back to the full dataset.
        
        Parameters
        ----------
        original_df : pd.DataFrame
            Original DataFrame before deduplication
        deduplicated_results : pd.DataFrame
            DataFrame with results from processing deduplicated data
        duplicate_map : Dict[int, int]
            Mapping from duplicate indices to their representative indices
        result_columns : List[str]
            Names of columns to map from deduplicated results to full dataset
            
        Returns
        -------
        pd.DataFrame
            Original DataFrame with results mapped back
        """
        # Make a copy of the original DataFrame
        result_df = original_df.copy()
        
        # Make sure all result columns exist in deduplicated_results
        missing_cols = [col for col in result_columns if col not in deduplicated_results.columns]
        if missing_cols:
            raise ValueError(f"Result columns {missing_cols} not in deduplicated results DataFrame")
        
        # Create a mapping from index to results
        index_to_result = {}
        for idx, row in deduplicated_results.iterrows():
            result_dict = {col: row[col] for col in result_columns}
            index_to_result[idx] = result_dict
        
        # Apply mapping to original dataset
        for col in result_columns:
            result_df[col] = None
        
        # Map results from deduplicated data back to full dataset
        for idx in result_df.index:
            if idx in index_to_result:
                # This document was kept in the deduplicated set
                for col in result_columns:
                    result_df.at[idx, col] = index_to_result[idx][col]
            elif idx in duplicate_map:
                # This document was removed as a duplicate
                rep_idx = duplicate_map[idx]
                for col in result_columns:
                    result_df.at[idx, col] = index_to_result[rep_idx][col]
        
        return result_df


def deduplicate_text(
    data: pd.DataFrame,
    text_column: str,
    method: str = "exact",
    threshold: float = 0.85,
    return_mapping: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[int, int], List]]:
    """Simple function for one-off text deduplication without creating a TextDeduplicator instance.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing text data
    text_column : str
        Name of the column containing text to deduplicate
    method : str, optional
        Deduplication method to use, by default "exact"
        Options: "exact", "fuzzy"
    threshold : float, optional
        Similarity threshold for fuzzy matching, by default 0.85
    return_mapping : bool, optional
        Whether to return mapping and groups information, by default False
        
    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[int, int], List]]
        If return_mapping is False, returns only the deduplicated DataFrame
        If return_mapping is True, returns a tuple with deduplicated DataFrame,
        mapping from duplicate indices to their representative indices,
        and groups of similar documents
    """
    # Create deduplicator
    deduplicator = TextDeduplicator(similarity_threshold=threshold)
    
    # Call deduplicate
    deduplicated_df, duplicate_map, groups = deduplicator.deduplicate(
        data, text_column, method=method, threshold=threshold
    )
    
    # Return based on return_mapping
    if return_mapping:
        return deduplicated_df, duplicate_map, groups
    else:
        return deduplicated_df
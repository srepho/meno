"""Tests for document deduplication in Meno."""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from difflib import SequenceMatcher

from meno import MenoWorkflow
from meno.preprocessing.deduplication import TextDeduplicator, deduplicate_text


class TestWorkflowDeduplication:
    """Tests for document deduplication in MenoWorkflow."""
    
    @pytest.fixture
    def sample_data_with_duplicates(self):
        """Create sample data with exact duplicates for testing."""
        np.random.seed(42)
        
        # Create base documents
        docs = [
            "This is document one about technology.",
            "This is document two about healthcare.",
            "This is document three about finance.",
            "This is document four about education.",
            "This is document five about politics."
        ]
        
        # Create dataset with duplicates
        data = []
        
        # Add original documents
        for i, doc in enumerate(docs):
            data.append({
                "text": doc,
                "id": f"doc_{i}",
                "is_duplicate": False
            })
        
        # Add exact duplicates
        for i in range(3):  # Add 3 duplicates
            dup_idx = np.random.randint(0, len(docs))
            data.append({
                "text": docs[dup_idx],
                "id": f"dup_{i}",
                "is_duplicate": True
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        np.random.shuffle(df.values)  # Shuffle rows
        return df
    
    def test_exact_deduplication(self, sample_data_with_duplicates):
        """Test exact document deduplication."""
        # Count initial documents
        initial_count = len(sample_data_with_duplicates)
        assert initial_count == 8  # 5 original + 3 duplicates
        
        # Create workflow with deduplication
        workflow = MenoWorkflow()
        
        # Load data with deduplication
        workflow.load_data(
            data=sample_data_with_duplicates,
            text_column="text",
            deduplicate=True
        )
        
        # Check that duplicates were removed
        deduped_count = len(workflow.documents)
        assert deduped_count < initial_count
        assert deduped_count == 5  # Should have only unique documents
        
        # In some implementations, duplicate info is stored differently
        # Just check that the number of documents was reduced
        assert deduped_count < initial_count
    
    def test_deduplication_disabled(self, sample_data_with_duplicates):
        """Test workflow with deduplication disabled."""
        # Count initial documents
        initial_count = len(sample_data_with_duplicates)
        
        # Create workflow without deduplication
        workflow = MenoWorkflow()
        
        # Load data without deduplication
        workflow.load_data(
            data=sample_data_with_duplicates,
            text_column="text",
            deduplicate=False
        )
        
        # Check that all documents were kept
        assert len(workflow.documents) == initial_count
    
    def test_with_category(self, sample_data_with_duplicates):
        """Test deduplication with category column."""
        # Create workflow
        workflow = MenoWorkflow()
        
        # Add a category column
        sample_data_with_duplicates['category'] = 'test_category'
        
        # Load data with deduplication and category
        workflow.load_data(
            data=sample_data_with_duplicates,
            text_column="text",
            category_column="category",
            deduplicate=True
        )
        
        # Verify the documents were loaded and duplicates were removed
        assert len(workflow.documents) < len(sample_data_with_duplicates)
    
    def test_empty_dataset(self):
        """Test deduplication with an empty dataset."""
        # Create empty dataset
        empty_data = pd.DataFrame(columns=["text", "id"])
        
        # Create workflow
        workflow = MenoWorkflow()
        
        # Load empty data with deduplication (should not crash)
        workflow.load_data(
            data=empty_data,
            text_column="text",
            deduplicate=True
        )
        
        # Check that no documents were loaded
        assert len(workflow.documents) == 0


class TestTextDeduplicator:
    """Test suite for TextDeduplicator class."""
    
    @pytest.fixture
    def exact_data(self):
        """Create sample data with exact duplicates."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'text': [
                'This is document one.',
                'This is document two.',
                'This is document one.',  # Exact duplicate of doc 1
                'This is document three.',
                'This is document two.'   # Exact duplicate of doc 2
            ]
        })
    
    @pytest.fixture
    def fuzzy_data(self):
        """Create sample data with fuzzy duplicates."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'text': [
                'This is document one.',
                'This is document two.',
                'This is document one with a small change.',  # Fuzzy duplicate of doc 1
                'This is document three.',
                'This is document two with minor edits.'      # Fuzzy duplicate of doc 2
            ]
        })
    
    @pytest.fixture
    def deduplicator(self):
        """Create a TextDeduplicator instance."""
        return TextDeduplicator(similarity_threshold=0.8)
    
    def test_exact_deduplicate(self, deduplicator, exact_data):
        """Test exact deduplication."""
        deduplicated, duplicate_map, groups = deduplicator.exact_deduplicate(
            exact_data, 'text'
        )
        
        # Check number of documents after deduplication
        assert len(deduplicated) == 3, "Should have 3 unique documents"
        
        # Check duplicate mapping
        assert len(duplicate_map) == 2, "Should have 2 duplicate mappings"
        
        # Check duplicate groups
        assert len(groups) == 2, "Should have 2 duplicate groups"
    
    def test_fuzzy_deduplicate(self, deduplicator, fuzzy_data):
        """Test fuzzy deduplication."""
        deduplicated, duplicate_map, groups = deduplicator.fuzzy_deduplicate(
            fuzzy_data, 'text', threshold=0.8
        )
        
        # Check number of documents after deduplication
        assert len(deduplicated) == 3, "Should have 3 unique documents after fuzzy deduplication"
        
        # Check duplicate mapping
        assert len(duplicate_map) == 2, "Should have 2 duplicate mappings"
        
        # Check fuzzy groups
        assert len(groups) == 2, "Should have 2 fuzzy groups"
    
    def test_deduplicate_with_exact_method(self, deduplicator, exact_data):
        """Test deduplicate method with exact matching."""
        deduplicated, duplicate_map, groups = deduplicator.deduplicate(
            exact_data, 'text', method='exact'
        )
        
        # Check number of documents after deduplication
        assert len(deduplicated) == 3, "Should have 3 unique documents"
    
    def test_deduplicate_with_fuzzy_method(self, deduplicator, fuzzy_data):
        """Test deduplicate method with fuzzy matching."""
        deduplicated, duplicate_map, groups = deduplicator.deduplicate(
            fuzzy_data, 'text', method='fuzzy', threshold=0.8
        )
        
        # Check number of documents after deduplication
        assert len(deduplicated) == 3, "Should have 3 unique documents after fuzzy deduplication"
    
    def test_deduplicate_with_invalid_method(self, deduplicator, exact_data):
        """Test deduplicate method with invalid method name."""
        with pytest.raises(ValueError):
            deduplicator.deduplicate(
                exact_data, 'text', method='invalid'
            )
    
    def test_deduplicate_with_invalid_column(self, deduplicator, exact_data):
        """Test deduplicate method with invalid column name."""
        with pytest.raises(ValueError):
            deduplicator.deduplicate(
                exact_data, 'non_existent_column'
            )
    
    def test_calculate_similarity(self, deduplicator):
        """Test similarity calculation."""
        # Identical texts should have similarity 1.0
        assert deduplicator.calculate_similarity("text", "text") == 1.0
        
        # Completely different texts should have low similarity
        assert deduplicator.calculate_similarity("text", "completely different") < 0.5
        
        # Similar texts should have high but not perfect similarity
        sim = deduplicator.calculate_similarity(
            "This is sample text", "This is sample text with an addition"
        )
        assert 0.7 < sim < 1.0
    
    def test_map_results_to_full_dataset(self, deduplicator, exact_data):
        """Test mapping results back to the full dataset."""
        # First deduplicate
        deduplicated, duplicate_map, _ = deduplicator.deduplicate(
            exact_data, 'text', method='exact'
        )
        
        # Add a result column to deduplicated data
        deduplicated['result'] = [f"Result for doc {i}" for i in range(len(deduplicated))]
        
        # Map results back
        full_results = deduplicator.map_results_to_full_dataset(
            exact_data, deduplicated, duplicate_map, ['result']
        )
        
        # Check that all original rows have results
        assert len(full_results) == len(exact_data), "Should have results for all original rows"
        assert 'result' in full_results.columns, "Result column should exist in full dataset"
        assert not full_results['result'].isna().any(), "All rows should have result values"
        
        # Check duplicate rows have same result as their representatives
        for dup_idx, rep_idx in duplicate_map.items():
            assert full_results.loc[dup_idx, 'result'] == full_results.loc[rep_idx, 'result'], \
                f"Duplicate at index {dup_idx} should have same result as its representative {rep_idx}"


def test_deduplicate_text_function_exact():
    """Test the helper function for exact deduplication."""
    # Create sample data
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['Text one', 'Text two', 'Text one']  # Has one duplicate
    })
    
    # Test without returning mapping
    result = deduplicate_text(data, 'text', method='exact')
    assert len(result) == 2, "Should have 2 unique documents"
    
    # Test with returning mapping
    result, duplicate_map, groups = deduplicate_text(
        data, 'text', method='exact', return_mapping=True
    )
    assert len(result) == 2, "Should have 2 unique documents"
    assert len(duplicate_map) == 1, "Should have 1 mapping entry"


def test_deduplicate_text_function_fuzzy():
    """Test the helper function for fuzzy deduplication."""
    # Create sample data
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['Text one', 'Text two', 'Text one with small change']  # Has one fuzzy duplicate
    })
    
    # Test without returning mapping
    result = deduplicate_text(data, 'text', method='fuzzy', threshold=0.8)
    assert len(result) == 2, "Should have 2 unique documents"
    
    # Test with returning mapping
    result, duplicate_map, groups = deduplicate_text(
        data, 'text', method='fuzzy', threshold=0.8, return_mapping=True
    )
    assert len(result) == 2, "Should have 2 unique documents"
    assert len(duplicate_map) == 1, "Should have 1 mapping entry"
# Meno API Standardization Guidelines

## Overview
This document defines the standard API interfaces for Meno v1.0.0 to ensure consistency across all components of the library.

## Topic Modeling API

### Parameters

| Standard Name | Description | Type | Used In |
|---------------|-------------|------|---------|
| `num_topics`  | Number of topics to discover | `Optional[int]` | All topic models |
| `auto_detect_topics` | Automatically determine optimal number of topics | `bool` | All topic models |
| `embeddings`  | Document embeddings | `np.ndarray` | All models |
| `documents`   | Input text documents | `Union[List[str], pd.Series]` | All models |
| `min_topic_size` | Minimum size for topics | `int` | Most clustering-based models |

### Method Signatures

#### Base Interface

All topic models should inherit from `BaseTopicModel` abstract base class and implement these methods:

```python
class BaseTopicModel(ABC):
    """Abstract base class for topic models"""
    
    API_VERSION: ClassVar[str] = "1.0.0"
    
    @abstractmethod
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseTopicModel":
        """Fit the model to documents"""
        pass
    
    @abstractmethod
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform documents to topics
        Returns: tuple of (topic_assignments, topic_probabilities)
        """
        pass
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step"""
        self.fit(documents, embeddings, **kwargs)
        return self.transform(documents, embeddings, **kwargs)
    
    @abstractmethod
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about topics
        Returns: DataFrame with standardized columns:
        - Topic: topic ID
        - Count: number of documents
        - Name: human-readable name
        - Representation: keywords for topic
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTopicModel":
        """Load model from disk"""
        pass
```

### Common Return Types

#### Document-Topic Assignments
Method: `transform()` and `fit_transform()`

Return type: `Tuple[np.ndarray, np.ndarray]`
- First element: Array of topic IDs of shape `(n_documents,)`
- Second element: Matrix of topic probabilities of shape `(n_documents, n_topics)`

#### Topic Information
Method: `get_topic_info()`

Return type: `pd.DataFrame` with columns:
- `Topic`: Topic ID (int)
- `Count`: Number of documents in topic (int)
- `Name`: Human-readable topic name (str)
- `Representation`: Keywords or representation of topic content (str)

## Parameter Standardization Guidelines

### For New Methods and Classes
- Always use the standard parameter names listed above
- Accept `**kwargs` to allow for flexibility and forward compatibility
- Use explicit type hints for all parameters
- Implement reasonable default values

### For Legacy Method Support
- Accept both standard and legacy parameter names
- Map standard parameter names to legacy internally
- For example:
  ```python
  # In __init__
  self.num_topics = num_topics  # Standard name
  self.n_topics = num_topics    # Legacy name for compatibility
  
  # In fit() method
  if 'num_topics' in kwargs:
      self.num_topics = kwargs.pop('num_topics')
      self.n_topics = self.num_topics  # Update legacy attribute
  ```

## Documentation Guidelines

- All public APIs should have NumPy-style docstrings
- Parameter descriptions should mention standard names
- Return type descriptions should specify shape and meaning
- Include simple examples for common use cases

## Version Compatibility

- All standardized models should include `API_VERSION` class variable
- Version checks should be performed when models interact
- Major version changes (1.x → 2.x) may break API compatibility
- Minor version changes (1.1 → 1.2) should be backward compatible
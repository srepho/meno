# API Standardization Guidelines for Meno v1.0.0

This document outlines the standardized API patterns for Meno v1.0.0, which should be followed by all components to ensure consistency and ease of use.

## Core Topic Modeling Interface

All topic modeling classes should inherit from the `BaseTopicModel` abstract base class, which defines the following core methods:

```python
class BaseTopicModel(ABC):
    """Abstract base class for topic models in Meno."""
    
    @abstractmethod
    def fit(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> "BaseTopicModel":
        """Fit the model to documents."""
        pass
    
    @abstractmethod
    def transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any]:
        """Transform documents to topics."""
        pass
    
    def fit_transform(
        self,
        documents: Union[List[str], pd.Series],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any]:
        """Fit and transform in one step."""
        self.fit(documents, embeddings)
        return self.transform(documents, embeddings)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTopicModel":
        """Load a model from disk."""
        pass
```

## Parameter Standardization

### Topic Model Parameters

All topic model classes should use consistent parameter names:

| Standard Name | Description | Used In |
|---------------|-------------|---------|
| `num_topics` | Number of topics to discover | High-level APIs, factory functions |
| `n_topics` | Number of topics to discover | Direct model initialization |
| `min_topic_size` | Minimum size of topics | All topic models |
| `embedding_model` | Embedding model to use | All topic models |
| `documents` | Input documents | All methods |
| `embeddings` | Pre-computed embeddings | All methods |

### Visualization Parameters

All visualization methods should use consistent parameter names:

| Standard Name | Description | Used In |
|---------------|-------------|---------|
| `width` | Width of the plot | All visualizations |
| `height` | Height of the plot | All visualizations |
| `title` | Title of the plot | All visualizations |
| `return_fig` | Whether to return the figure | All visualizations |
| `colorscale` | Color scale to use | All visualizations |
| `marker_size` | Size of markers | Scatter plots |

## Return Value Standardization

### Topic Model Return Values

All topic model methods should use consistent return values:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `fit` | `self` | Fitted model for method chaining |
| `transform` | `Tuple[np.ndarray, np.ndarray]` | (topic_assignments, topic_probabilities) |
| `fit_transform` | `Tuple[np.ndarray, np.ndarray]` | (topic_assignments, topic_probabilities) |
| `get_topic_info` | `pd.DataFrame` | DataFrame with topic information |
| `visualize_topics` | `plotly.graph_objects.Figure` | Interactive visualization |

### Embedding Model Return Values

All embedding model methods should use consistent return values:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `embed_documents` | `np.ndarray` | Document embeddings |
| `embed_words` | `np.ndarray` | Word embeddings |

## Method Naming Standardization

Use consistent method names across different implementations:

| Standard Name | Description | Alternative Names to Avoid |
|---------------|-------------|---------------------------|
| `fit` | Fit model to data | `train`, `learn` |
| `transform` | Transform documents to topics | `predict`, `infer` |
| `fit_transform` | Fit and transform in one step | `train_predict` |
| `save` | Save model to disk | `export`, `write` |
| `load` | Load model from disk | `import`, `read` |
| `get_topic_info` | Get information about topics | `topic_info`, `get_topics` |
| `visualize_topics` | Visualize topics | `plot_topics`, `show_topics` |
| `search_topics` | Search for topics | `find_topics`, `query_topics` |

## Configuration Standardization

All components should use the same configuration hierarchy:

```python
# Configuration override example
config_overrides = {
    "modeling": {
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 64
        },
        "topic_modeling": {
            "num_topics": 10,
            "min_topic_size": 5
        },
        "performance": {
            "use_mmap": True,
            "precision": "float16"
        }
    }
}
```

## Factory Functions

Use factory functions with consistent naming and parameters for creating objects:

```python
def create_topic_modeler(
    method: str = "embedding_cluster",
    num_topics: int = 10,
    config_overrides: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[Union[str, DocumentEmbedding]] = None
) -> BaseTopicModel:
    """Create a topic modeler with the specified method."""
    pass

def create_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    config_overrides: Optional[Dict[str, Any]] = None
) -> DocumentEmbedding:
    """Create an embedding model with the specified name."""
    pass
```

## Type Hints

All methods should use consistent type hints:

```python
from typing import List, Dict, Optional, Union, Any, Tuple, Callable

def method(
    param1: Union[List[str], pd.Series],
    param2: Optional[np.ndarray] = None,
    param3: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Method with proper type hints."""
    pass
```

## Docstrings

All methods should use NumPy-style docstrings:

```python
def method(param1, param2=None):
    """Short description of the method.
    
    Longer description of the method if needed,
    spanning multiple lines.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2, by default None
    
    Returns
    -------
    type
        Description of return value
    
    Raises
    ------
    ExceptionType
        When the exception is raised
    
    Examples
    --------
    >>> method("example", param2=42)
    expected_result
    """
    pass
```

## Error Handling

All components should use consistent error handling:

```python
if not self.is_fitted:
    raise ValueError("Model must be fitted before transform can be called.")

if not HAVE_DEPS:
    raise ImportError(
        "Package dependencies not installed. "
        "To use this feature, install with: pip install meno[feature]"
    )
```

## Warning Messages

Use consistent warning format:

```python
import warnings

warnings.warn(
    "Feature X is deprecated and will be removed in v1.0. "
    "Use Feature Y instead.",
    DeprecationWarning
)

warnings.warn(
    "Optional dependency not installed. "
    "To use this feature, install with: pip install package"
)
```

## Workflow API

The workflow API should use consistent method names and parameters:

```python
workflow = MenoWorkflow()

# Data loading
workflow.load_data(
    data=df,
    text_column="text",
    id_column="id",
    category_column="category"
)

# Preprocessing
workflow.preprocess_documents(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True
)

# Topic discovery
topics_df = workflow.discover_topics(
    method="embedding_cluster",
    num_topics=10
)

# Report generation
report_path = workflow.generate_report(
    output_path="report.html",
    title="Topic Analysis",
    open_browser=True
)
```
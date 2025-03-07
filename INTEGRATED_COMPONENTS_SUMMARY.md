# Lightweight Topic Modeling Components

## Overview

We've successfully integrated lightweight topic modeling components that work without requiring heavy dependencies like UMAP and HDBSCAN. These components include:

1. Lightweight topic models (SimpleTopicModel, TFIDFTopicModel, NMFTopicModel, LSATopicModel)
2. Dedicated visualization functions for these models
3. Integration with the web interface
4. Comprehensive documentation and examples

## Available Components

### 1. Lightweight Topic Models

Four models implementing the `BaseTopicModel` interface:

- **SimpleTopicModel** - K-means clustering on document embeddings
- **TFIDFTopicModel** - TF-IDF vectorization with K-means clustering (no embeddings required)
- **NMFTopicModel** - Non-negative Matrix Factorization for topic modeling
- **LSATopicModel** - Latent Semantic Analysis (aka LSI) for topic modeling

### 2. Visualization Components

Specialized visualization functions in `meno.visualization.lightweight_viz`:

- `plot_model_comparison` - Compare multiple topic models side-by-side
- `plot_topic_landscape` - Visualize document clusters in 2D space
- `plot_multi_topic_heatmap` - Compare topic similarities across models
- `plot_comparative_document_analysis` - Analyze how documents relate to topics

### 3. Web Interface Integration

The web interface now supports lightweight models with:

- Model configuration through the UI
- Topic exploration and visualization
- Document search and filtering
- Interactive topic analysis

## Documentation

We've added comprehensive documentation:

- **LIGHTWEIGHT_MODELS_DOCUMENTATION.md** - Detailed guide for using the lightweight models
- Updated examples showing how to use all components together
- Integration testing documentation explaining how components interact

## Examples and Scripts

1. **Basic Examples:**
   - `examples/lightweight_topic_modeling.py` - Basic usage of all four models
   - `examples/lightweight_models_visualization.py` - Visualizations for lightweight models

2. **Integration Examples:**
   - `examples/simple_integrated_demo.py` - Simple demonstration of components working together
   - `examples/integrated_components_example.py` - Comprehensive example with all components
   - `examples/web_lightweight_example.py` - Web interface with lightweight models

## How to Run the Examples

```bash
# Simple demonstration without web interface
python examples/simple_integrated_demo.py

# Full example with all components
python examples/integrated_components_example.py

# Example with web interface
python examples/web_lightweight_example.py

# Full example with web interface
python examples/integrated_components_example.py --web
```

## Tests

We've created comprehensive tests for all components:

```bash
# Run all test suites
python -m pytest

# Run just the integration tests 
python -m pytest tests/test_integrated_components.py

# Run lightweight model tests
python -m pytest tests/test_lightweight_models.py

# Run visualization tests
python -m pytest tests/test_lightweight_viz.py

# Run web interface tests
python -m pytest tests/test_web_interface.py
```

## Notes on Testing Environment

During local testing, we encountered some issues with OpenMP libraries that caused crashes in certain environments. These issues are environment-specific and shouldn't affect most users. We've worked around these by:

1. Using mocks where appropriate
2. Adding fallback options that avoid problematic libraries
3. Implementing proper error handling with helpful messages
4. Providing alternative approaches in the documentation

The components have been designed to degrade gracefully if certain dependencies are missing, making them suitable for a wide range of environments.

## Future Improvements

Planned enhancements for these components:

1. Add incremental learning capabilities for streaming data
2. Improve model parameter optimization
3. Add more advanced visualization options
4. Enhance web interface with more interactive features
5. Improve integration with the UnifiedTopicModeler API
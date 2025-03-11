# Changelog

All notable changes to the Meno project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-11-03

### Added
- **Advanced BERTopic Features:**
  - Added `merge_models()` to combine multiple topic models
  - Added topic manipulation functions: `merge_topics()`, `reduce_topics()`, `update_topics()`
  - Added dynamic topic modeling with `fit_transform_with_timestamps()`
  - Added semi-supervised topic modeling with `fit_with_seed_topics()`
  - Comprehensive examples in `examples/advanced_bertopic_features.py`

- **LLM Topic Labeling:**
  - Added `LLMTopicLabeler` class for generating human-readable topic names
  - Support for both local HuggingFace models and OpenAI API
  - Integrated labeling into BERTopicModel with `use_llm_labeling` parameter
  - Added post-fit labeling with `apply_llm_labeling()` method
  - Example implementations in `examples/llm_topic_labeling_example.py`
  - End-to-end workflow example in `examples/workflow_with_llm_labeling.py`

### Improved
- Enhanced topic coherence evaluation with new metrics
- Added topic hierarchy visualization with `plot_topic_hierarchy()`
- Improved documentation for advanced BERTopic features
- Added comprehensive docstrings for all new methods
- Updated README with examples of advanced features
- Better error handling in model fitting and evaluation

## [1.1.2] - 2025-08-12

### Fixed
- Improved error handling in embedding model for minimal installations
- Added robust fallbacks for missing attributes in preprocessing
- Fixed model initialization error handling
- Enhanced cache management for embedding models
- Improved compatibility with various installation configurations
- Fixed error when working with non-string values in normalization

## [1.1.0] - 2025-08-05

### Added
- **Lightweight Topic Models:**
  - Added `SimpleTopicModel`: K-Means based model with document embeddings
  - Added `TFIDFTopicModel`: Extremely fast model with minimal dependencies
  - Added `NMFTopicModel`: Non-negative Matrix Factorization for interpretable topics
  - Added `LSATopicModel`: Latent Semantic Analysis for semantic relationship discovery
  - Comprehensive documentation for all models in `docs/lightweight_models.md`
  - Example script demonstrating lightweight models in `examples/lightweight_models_visualization.py`

- **Advanced Topic Visualizations:**
  - Added `plot_model_comparison` for comparing different topic model results
  - Added `plot_topic_landscape` for visualizing topic relationships in 2D
  - Added `plot_multi_topic_heatmap` for analyzing topic similarities across models
  - Added `plot_comparative_document_analysis` for document-topic analysis
  - Comprehensive visualization documentation with examples

- **Web Interface for No-Code Exploration:**
  - Added `MenoWebApp` class providing a Dash-based web interface
  - Interactive data upload and preprocessing
  - Model configuration and training through UI
  - Interactive topic exploration and visualization
  - Document search and filtering functionality
  - CLI tool `meno-web` for launching the web interface
  - Detailed documentation in `docs/web_interface.md`
  - Example script in `examples/web_interface_example.py`

### Improved
- Enhanced README with new examples and feature descriptions
- Added comprehensive tests for all new components
- Updated installation options with `meno[web]` for web interface dependencies
- Streamlined package structure with clear organization of new components
- Improved API consistency across all model implementations

## [1.0.3] - 2025-03-08

### Fixed
- Fixed API inconsistency in MenoWorkflow.preprocess_documents() where it was passing parameters that MenoTopicModeler.preprocess() doesn't accept
- Fixed parameter handling for text normalization options in workflow.py
- Ensured preprocessing parameters are applied correctly to the TextNormalizer instance

## [1.0.2] - 2025-03-08

### Added
- Enhanced support for offline/air-gapped environments
- Added automatic fallback in spaCy model loading for offline environments
- Added more detailed error messages for network issues
- Added comprehensive examples for offline deployment

### Improved
- Strengthened error handling when working with missing network connectivity
- Refined documentation for installation in restricted environments
- Updated default embedding model implementation details
- Enhanced test coverage for offline usage scenarios

### Fixed
- Improved error handling in spaCy model downloader
- Added fallback paths for model loading in offline mode
- Fixed edge cases in local model path resolution

## [1.0.1] - 2024-08-03

### Added
- Added support for offline/air-gapped environments
- Added `local_files_only` parameter to `DocumentEmbedding` class
- Enhanced model loading to detect models in HuggingFace cache
- Added example script for using locally downloaded models
- Included wordcloud, gensim, and spacy in meno[minimal] installation
- Added automatic spaCy model download on package import
- Added support for custom embedding models in MenoTopicModeler
- Added support for local model paths in MenoWorkflow
- Changed default embedding model to sentence-transformers/all-MiniLM-L6-v2

### Improved
- Better detection of local model files in standard HuggingFace locations
- Updated README with offline installation instructions
- Added support for loading models from custom paths
- Improved error handling for offline environments
- Added detailed documentation for offline usage scenarios
- Added spaCy model as an optional dependency
- Improved test coverage for local model loading

### Fixed
- Improved error handling when models cannot be downloaded

## [1.0.0] - 2025-08-01

### Added
- **Standardized API:**
  - Comprehensive API standardization across all models
  - Automatic topic number detection capability
  - Consistent parameter naming across all components
  - Standardized visualization interface with uniform parameters
  - Memory-mapped storage for large embedding matrices
  - Common interface for working with document embeddings

### Changed
- **Breaking Changes:**
  - Renamed `n_topics` to `num_topics` throughout the codebase
  - Changed parameter types to be more consistent (e.g., Path parameters)
  - Standardized return types for all methods
  - Updated visualization functions to use consistent parameters
  - Updated save/load methods for consistent behavior

### Fixed
- Fixed inconsistent method signatures and return types
- Fixed memory issues with large datasets
- Improved type annotations throughout the codebase
- Standardized error handling and informative error messages

### Removed
- Removed legacy parameter names and compatibility layers
- Removed deprecated visualization functions
- Removed redundant code and unused imports

### Documentation
- Added comprehensive API reference documentation
- Created migration guide from 0.x to 1.0.0
- Added examples for all common use cases
- Improved docstrings with consistent formatting

## [0.9.0] - 2025-06-05

### Added
- **Team Configuration System:**
  - Comprehensive system for creating and sharing domain-specific configurations
  - Support for exporting/importing acronyms and spelling dictionaries
  - Command-line interface for team configuration management
  - Configuration comparison tools to identify differences between team configs
  - Versioning and attribution for domain knowledge sources
- **Performance Optimizations:**
  - CPU-optimized embedding with quantization support
  - Memory-efficient processing for large datasets
  - Polars integration for faster data processing
  - Streaming processing for larger-than-memory datasets
  - Graceful fallbacks when optional dependencies are missing
- **Example Implementations:**
  - `workflow_with_optimizations.py` demonstrating performance improvements
  - Team configuration sharing examples and utilities
  - Resource usage benchmarks and optimization guidelines

### Changed
- Improved caching of embeddings to reduce model load times
- Enhanced documentation with optimization recommendations
- Better error messages when optional dependencies are missing
- New visualizations for team-specific terminology

### Fixed
- Memory leak in repeated embedding generation
- Thread safety issues in parallel processing
- Compatibility issues with latest pandas/numpy versions

## [0.8.0] - 2025-06-03

### Added
- New `MenoWorkflow` class that provides an interactive, guided workflow for topic modeling
- Interactive acronym detection and expansion with HTML reports
- Interactive spelling correction with HTML reports
- Complete workflow that guides users from data loading to visualization
- Example script demonstrating the interactive workflow
- Re-exports of key functions in the package's top-level namespace

### Changed
- Updated package docstrings
- Improved metadata in `__init__.py`

## [0.7.0] - 2025-06-03

### Added
- Time series visualization module with functions for:
  - Line plots of topic trends over time
  - Heatmaps of topic intensity over time
  - Stacked area charts of topic composition
  - Ridge plots for topic distribution comparison
  - Calendar heatmaps for specific topics
- Geospatial visualization module with functions for:
  - Interactive point maps with topic coloring
  - Region choropleth maps for geographic distributions
  - Density heatmaps showing topic concentrations
  - Postcode-based mapping (with Australia example)
- Time-space visualization module with functions for:
  - Animated maps showing topic evolution over time
  - Space-time heatmaps for regional topic trends
  - Category-time plots for comparing topic trends across categories
- New MenoTopicModeler methods:
  - `visualize_topic_trends()` for time series visualization
  - `visualize_geospatial_topics()` for location-based maps
  - `visualize_timespace_topics()` for combined time-space analysis
- Comprehensive unit tests for all new visualization capabilities
- Example script demonstrating visualization with Australian insurance data

## [0.6.0] - 2025-06-03

### Added
- Enhanced HTML report generation with modern card-based design
- Interactive tabbed visualizations in HTML reports
- Topic similarity heatmap visualization
- Interactive word cloud visualization
- CSV export functionality for data tables
- Enhanced example reports showcasing all new features
- Comprehensive examples in README for all usage patterns

### Fixed
- Resolved compatibility issue between html_generator.py and HTMLReportConfig
- Improved handling of Pydantic models in configuration

### Changed
- Expanded MenoTopicModeler.generate_report with additional parameters
- Enhanced HTML template with responsive layout and improved typography
- Improved organization of example code and sample reports

## [0.5.0] - 2025-06-03

### Changed
- Updated Python version support to >=3.8,<3.13 (dropped Python 3.13 due to pyarrow compatibility)
- Reduced minimum pyarrow version to 11.0.0 for better compatibility
- Restructured README with installation at the top and expanded examples
- Added sample report generation scripts and visualizations

### Fixed
- Fixed import issues with fuzzywuzzy by fully transitioning to thefuzz

## [0.4.5] - 2025-06-03

### Fixed
- Updated code to import process from thefuzz instead of fuzzywuzzy 

## [0.4.4] - 2025-06-03

### Changed
- Replaced fuzzywuzzy and python-Levenshtein with thefuzz to resolve dependency warnings
- Simplified dependency structure by removing redundant dependencies

## [0.4.3] - 2025-06-03

### Fixed
- Fixed configuration file loading to properly handle installed packages
- Added built-in fallback to default configuration values
- Improved package distribution to include config files

## [0.4.2] - 2025-06-03

### Fixed
- Added python-Levenshtein to core dependencies to resolve warning from fuzzywuzzy

## [0.4.1] - 2025-06-03

### Changed
- Enhanced CPU-first design across all model components
- Modified DocumentEmbedding to default to CPU operation
- Updated BERTopicModel and Top2VecModel to explicitly use CPU by default
- Fixed dependency specifications for CPU-only PyTorch installation
- Improved documentation with clearer CPU-only installation instructions
- Added examples demonstrating CPU-only workflows

### Added
- New tests for CPU-only embedding functionality
- Installation instructions for CPU-only PyTorch configuration

## [0.2.0] - 2024-06-03

### Added
- Australian Insurance PII dataset example
- Configuration file for insurance dataset analysis
- Download utility for offline dataset access
- Example scripts for both online and local dataset usage
- Requirements file for example dependencies
- Instructions in README for working with the insurance dataset

### Changed
- Updated README with new examples section
- Improved documentation for example workflows

## [0.1.0] - 2024-05-30

### Added
- Initial release of Meno
- Core functionality for topic modeling
- Preprocessing module with text normalization
- Modeling module with embeddings and clustering
- Visualization tools for UMAP projections
- Reporting module for HTML report generation
- Configuration system with YAML support
- Basic project structure and documentation
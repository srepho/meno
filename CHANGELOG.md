# Changelog

All notable changes to the Meno project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
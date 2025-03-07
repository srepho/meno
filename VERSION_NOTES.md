# Meno Version Notes

## v1.0.0 Release Candidate (August 2025) - PLANNED

The upcoming 1.0.0 release will mark our first stable API release with long-term support:

### Final API Stabilization
- Complete API standardization across all components
- Finalize public interface with backward compatibility guarantees
- Add deprecation warnings for any legacy interfaces

### Performance Optimizations
- Finalize memory-mapped formats for all model components
- Complete streaming processor implementation
- Add final optimizations for CPU-only environments

### Documentation and Testing
- Complete comprehensive API documentation
- Add migration guides from earlier versions
- Expand test coverage to 90%+ for all core components
- Add performance benchmarks across different hardware configs

### Installation and Deployment
- Finalize dependency structure with minimal core requirements
- Add specialized installation profiles for different use cases
- Include containerized deployment examples

## v0.9.1 - Standardized API and Memory Optimizations

This release focuses on API standardization and continued performance improvements:

### API Standardization
- Made `UnifiedTopicModeler` properly inherit from `BaseTopicModel` abstract base class
- Standardized parameter naming across all topic modeling implementations
- Ensured consistent return types for `transform()` and other methods
- Created comprehensive API standardization guidelines
- Added consistent type hints throughout the codebase
- Added proper docstrings with standardized format

### Performance Improvements
- Added additional memory-mapping capabilities for large datasets
- Improved caching system for embeddings and UMAP projections
- Standardized performance configuration options
- Enhanced streaming processor with better memory management

### Memory Cleanup
- Removed unused virtual environments
- Eliminated redundant cache files
- Streamlined dependencies

### Documentation
- Added `api_standardization.md` with comprehensive guidelines
- Updated code examples to demonstrate standardized APIs
- Enhanced docstrings with proper NumPy-style formatting
- Added GitHub workflow for automated API compatibility checks

## v0.9.0 - Team Configuration and Performance Improvements

This release introduces a comprehensive team configuration system and significant performance improvements:

### Team Configuration System
- Added team configuration management for domain-specific knowledge
- Implemented export/import of acronyms and spelling dictionaries
- Added configuration comparison and merging utilities
- Created CLI for team configuration management

### Performance Optimizations
- Added memory-mapped storage for large embedding matrices
- Created content-based caching system for embeddings
- Added support for half-precision (float16) storage
- Implemented UMAP projection caching
- Added Polars integration for faster data processing
- Created streaming processor for datasets that don't fit in memory

### Documentation
- Added examples demonstrating team configuration usage
- Added comprehensive documentation for performance optimization
- Updated usage examples and configuration guides

## v0.8.0 - BERTopic Integration

This release adds comprehensive integration with BERTopic for advanced topic modeling:

### Key Features
- Added BERTopic model integration with Meno's preprocessing
- Implemented topic visualization enhancements
- Created hierarchical topic modeling support
- Added dynamic topic modeling for time series analysis
- Integrated BERTopic's zero-shot classification

### Examples
- Added simple BERTopic example using string model name
- Created examples with custom components
- Added workflow integration examples

## v0.7.0 - Enhanced Visualizations

This release focuses on visualization enhancements:

### Key Features
- Added interactive topic visualization with Plotly
- Created comparative visualization for topic differences
- Added geospatial visualization support
- Enhanced time series visualization
- Added comprehensive HTML report generation

## v0.6.0 - Workflow API

The initial public release of Meno:

### Key Features
- Created initial workflow API for easy topic discovery
- Added basic preprocessing for text normalization
- Implemented embeddings with sentence transformers
- Created simple topic visualization
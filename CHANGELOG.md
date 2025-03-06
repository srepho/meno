# Changelog

All notable changes to the Meno project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
# Changelog

All notable changes to this project will be documented in this file.

## [1.3.4] - 2023-03-21

### Added
- Multi-provider LLM integration support for:
  - Google Gemini models
  - Anthropic Claude models
  - Hugging Face Inference API
  - AWS Bedrock managed services
- New `generate_text_with_llm_multi` function with unified interface for all providers
- Support for both SDK and direct requests implementations for all providers
- Comprehensive caching system for all providers to minimize API costs
- Detailed documentation and examples for multi-provider integration
- New optional dependency group `llm_multi` for easy installation

### Changed
- Enhanced error handling for all LLM implementations
- Improved type hints for better IDE support
- Updated documentation for clarity and consistency

### Fixed
- Issue with caching in the requests implementation
- Error handling for Azure OpenAI deployments
- Compatibility with latest OpenAI SDK versions

## [1.3.3] - 2023-03-15

### Added
- Enhanced OpenAI Integration with both SDK and requests-based implementations
- Response caching for the requests implementation to avoid redundant API calls
- Added support for direct API usage with `generate_text_with_llm` utility function
- Added comprehensive error handling to improve robustness

### Changed
- Optimized threading model for batch processing to improve performance
- Enhanced Azure OpenAI support with proper parameter handling

### Fixed
- Fixed issue with rate limiting during batch processing
- Resolved token counting inaccuracies that could lead to truncated prompts

## [1.3.2] - 2023-03-01

### Added
- Enhanced OpenAI integration with caching and optimized deduplication
- Support for direct API usage without needing the full LLMTopicLabeler
- Added functions for text deduplication to reduce API costs
- Added concurrent.futures ThreadPoolExecutor for batch processing

### Fixed
- Fixed Azure OpenAI integration to use deployment_id instead of model parameter
- Resolved issues with rate limiting exceeding API quotas
- Improved error handling during batch processing

## [1.3.1] - 2023-02-20

### Added
- Added `generate_text_with_llm` utility function for direct API usage
- Enhanced support for various OpenAI models
- Added more flexible prompt templates

### Fixed
- Fixed model loading with non-default parameters
- Resolved thread safety issues in concurrent operations
- Improved error messages for API failures

## [1.3.0] - 2023-02-10

### Added
- Support for LLM-based topic labeling using HuggingFace and OpenAI models
- Added deduplication system to reduce redundant API calls
- Integrated with BERTopic for enhanced topic modeling
- Added support for LLM-based topic description generation
- Added cache directory for persistent API response storage

### Changed
- Improved topic modeling workflow with LLM integration
- Enhanced topic quality scoring with entropy-based metrics
- Updated visualization tools to include LLM-generated labels

### Fixed
- Fixed memory issues with large document sets
- Resolved race conditions in concurrent API calls

## [1.2.0] - 2023-01-20

### Added
- Added lightweight topic modeling module with minimal dependencies
- Enhanced embedding visualization with interactive 3D plots
- Support for time-series topic drift visualization
- Support for geospatial topic distribution visualization
- New documentation on memory optimization strategies

### Changed
- Improved performance for CPU-only deployments
- Enhanced incremental topic updating for streaming data

### Fixed
- Fixed compatibility issues with older Python versions
- Resolved memory leaks during embedding generation
- Improved error messaging for missing dependencies

## [1.1.0] - 2023-01-05

### Added
- Added incremental topic update capabilities for streaming data
- Support for fuzzy deduplication to reduce data redundancy
- Team configuration system for shared project settings
- Web interface for interactive topic exploration
- Memory optimization for large document collections

### Changed
- Improved reporting interface with more interactive elements
- Enhanced visualization suite with temporal analysis tools
- Updated documentation with advanced usage patterns

### Fixed
- Fixed compatibility issues with newer versions of dependencies
- Resolved performance bottlenecks in embedding generation
- Improved error handling in preprocessing pipeline

## [1.0.0] - 2022-12-15

### Added
- Initial release of meno topic modeling toolkit
- Support for traditional LDA and modern embedding-based approaches
- Interactive visualization and reporting capabilities
- Preprocessing utilities for messy text data
- Workflow system for end-to-end topic modeling
- Documentation and examples for common use cases
- CPU-optimized variants for deployment without GPU
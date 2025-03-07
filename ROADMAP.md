# Meno v1.0.0 Roadmap

This document outlines the planned changes and features for the Meno v1.0.0 release. The primary goals for this release are to reduce dependencies, optimize performance, and provide a more streamlined user experience.

## Dependency Optimization

### Core Objectives
- Reduce the total number of required dependencies
- Minimize version conflicts with other packages
- Make as many dependencies optional as possible
- Provide clear installation paths for different use cases

### Planned Changes
1. **Modular Dependency Structure**
   - Move more dependencies to optional dependency groups
   - Create specialized installation profiles (e.g., "minimal", "nlp", "viz")
   - Split large dependencies from the core package

2. **Alternative Implementations**
   - Provide fallbacks for functionality when optional dependencies are missing
   - Implement dependency-free alternatives for common operations
   - Add runtime checks with helpful error messages when dependencies are missing

3. **Embedding Optimization**
   - Make embedding models entirely optional
   - Support loading from local files to avoid network dependencies
   - Use smaller, more efficient models as defaults
   - Support for quantized models to reduce memory usage

4. **Visualization Standalone Mode**
   - Separate visualization dependencies from core functionality
   - Support for generating visualizations with minimal dependencies
   - Export options for different visualization libraries

## Performance Improvements

### Core Objectives
- Improve performance on large datasets
- Reduce memory usage across all operations
- Provide streaming/chunked processing for larger-than-memory data
- Optimize for CPU-only environments

### Planned Changes
1. **Memory Efficiency**
   - Add streaming data loading and processing
   - Implement memory-mapped storage for intermediate results
   - Support partial model loading for large embedding models
   - Add memory usage monitoring and recommendations

2. **Parallel Processing**
   - Add multi-processing support for CPU-intensive operations
   - Support for batch processing of documents
   - Implement parallel preprocessing pipeline
   - Add caching for repeated operations

3. **Polars Integration**
   - Native support for Polars DataFrame operations
   - Use Arrow memory format for efficient data transfer
   - Streaming CSV and parquet support

4. **Quantization and Model Optimization**
   - 8-bit quantization for all embedding models
   - Pruned model options for faster inference
   - Support for ONNX runtime optimization
   - Lazy loading for model components

## API Improvements

### Core Objectives
- Create a stable, consistent API for v1.0
- Improve documentation and examples
- Make interfaces more intuitive and Pythonic
- Add more customization options

### Planned Changes
1. **API Standardization**
   - Consistent parameter naming across the library
   - Unified return types and structures
   - Simplified class hierarchy with clear inheritance
   - Type hints throughout with better IDE integration

2. **Workflow Enhancements**
   - More customization points in the workflow
   - Additional hooks for custom processing steps
   - Plugin architecture for extensions
   - Improved progress tracking and logging

3. **Configuration System**
   - Simplified configuration validation
   - More sensible defaults
   - Environment variable support
   - Profile-based configuration management

4. **CLI Enhancement**
   - Expanded CLI functionality for common operations
   - Pipeline construction through CLI
   - Report generation directly from command line
   - Interactive mode for exploration

## Target Dependencies Reduction

Our goal for v1.0.0 is to significantly reduce the default dependency footprint:

### Current Dependencies (v0.9.0)
- **Core Dependencies:** 7
- **Full Installation Dependencies:** 25+
- **Total Size (disk):** ~2GB with full installation

### Target for v1.0.0
- **Core Dependencies:** 5 or fewer
- **Minimal Installation Size:** <50MB
- **Full Installation Dependencies:** Modularized for selection

## Timeline and Implementation Strategy

1. **Phase 1: Dependency Audit (1-2 weeks)**
   - Analyze all dependencies for usage patterns
   - Identify candidates for removal or replacement
   - Measure impact on functionality
   
2. **Phase 2: Modularization (2-3 weeks)**
   - Refactor code to move functionality to optional modules
   - Implement fallbacks for missing dependencies
   - Create installation profiles

3. **Phase 3: Performance Optimization (2-3 weeks)**
   - Implement memory-efficient processing
   - Add streaming data handling
   - Optimize core algorithms

4. **Phase 4: API Stabilization (1-2 weeks)**
   - Finalize public API design
   - Deprecate legacy interfaces
   - Complete documentation updates

5. **Phase 5: Testing and Release (1-2 weeks)**
   - Comprehensive testing on all platforms
   - Performance benchmarking
   - Documentation review
   - Final release preparation

## Contributing

We welcome contributions to help implement this roadmap! If you're interested in helping with any aspect of the v1.0.0 release, please check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines and open an issue to discuss your proposed changes.
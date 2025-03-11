# Meno v1.2.3: External Model Integration & Workflow Extensions

This release enhances Meno's flexibility by adding seamless integration with external topic modeling algorithms, including improved BERTopic integration and comprehensive test coverage.

## New Features

### Workflow Extensions
- **`get_preprocessed_data` Method**: Extract preprocessed documents from the workflow pipeline
  - Allows users to apply custom algorithms to Meno's preprocessed data
  - Returns a DataFrame with all metadata and preprocessing results
  - Maintains document alignment for easy integration back into Meno

- **`set_topic_assignments` Method**: Import topic assignments from external models
  - Enables direct integration with BERTopic, LDA, NMF, and other topic modeling libraries
  - Supports probabilistic topic assignments with confidence scores
  - Maintains full compatibility with Meno's visualization and reporting tools

### Example Scripts
- **External Topic Model Integration**: New example that demonstrates:
  - Using Meno for preprocessing
  - Exporting data to external algorithms (scikit-learn NMF and BERTopic)
  - Importing topic assignments back into Meno
  - Generating reports and visualizations with externally-derived topics

## Benefits for Users

- **Framework Flexibility**: Use your preferred topic modeling algorithm while leveraging Meno's powerful preprocessing, visualization, and reporting capabilities
- **Research Compatibility**: Easily compare results from different algorithms using the same preprocessing pipeline and visualization tools
- **Pipeline Integration**: Insert custom processing steps between Meno's preprocessing and topic visualization
- **Simplified BERTopic Integration**: More straightforward integration with BERTopic's advanced features

## Technical Improvements

- **Comprehensive Test Coverage**: Added extensive test suite for the new workflow extension methods
  - Unit tests for `get_preprocessed_data` functionality
  - Unit tests for `set_topic_assignments` with various inputs
  - Integration tests for end-to-end external model workflows
  - Validation of index alignment and error handling

- **Robust Index Handling**: Careful validation ensures document indexes remain aligned between Meno and external tools
- **Proper Error Messages**: Clear, descriptive error messages when data doesn't meet requirements
- **Minimal Dependencies**: New functionality requires no additional dependencies

## Documentation

- Added detailed docstrings with examples for new methods
- New example script with step-by-step comments
- Updated type hints for better IDE integration

This release makes Meno more flexible and interoperable with the broader topic modeling ecosystem while maintaining its user-friendly design and comprehensive visualization capabilities.
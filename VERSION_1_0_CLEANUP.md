# Meno v1.0.0 Final Cleanup Plan

This document outlines the necessary changes to finalize Meno v1.0.0, addressing API inconsistencies, breaking changes, and release preparation tasks.

## 1. API Standardization Fixes

### Parameter naming standardization
- [x] Replace all `n_topics` parameters with `num_topics` throughout codebase
- [x] Add `auto_detect_topics` to all model classes
- [ ] Standardize visualization parameters: use `width`/`height` consistently
- [ ] Make all Path parameters use `Union[str, Path]` with Path object conversion
- [ ] Remove all redundant parameter mapping code and legacy parameter handling
- [ ] Standardize embedding_model parameter typing

### Method signature standardization
- [ ] Rename `search_topics` to `find_similar_topics` across all implementations
- [ ] Add common interface methods to BaseTopicModel for model-specific capabilities:
  - [ ] Add `visualize_topics` to BaseTopicModel
  - [ ] Add `add_documents` to BaseTopicModel (with optional implementation)
  - [ ] Add `get_document_embeddings` to BaseTopicModel

### Return type standardization
- [ ] Ensure all transform() methods return consistent tuple types: `Tuple[np.ndarray, np.ndarray]`
- [ ] Standardize visualization return types to always return Plotly figures
- [ ] Document return types consistently in all docstrings
- [ ] Add runtime type checking for critical methods' returns

## 2. Breaking Changes

The following changes will break backward compatibility but are necessary for a clean v1.0.0 release:

- [ ] Remove all legacy attribute names (e.g., remove `n_topics` keeping only `num_topics`)
- [ ] Remove deprecated visualization functions replaced by enhanced implementations
- [ ] Completely remove redundant code like `kwargs['top_n'] = kwargs['top_n']`
- [ ] Enforce correct return types even if it changes behavior
- [ ] Remove any compatibility layers for pre-1.0 APIs

## 3. Documentation Updates

- [ ] Create comprehensive API reference documentation
- [ ] Add migration guide from 0.x to 1.0.0
- [ ] Update all docstrings to reflect standardized parameters and return types
- [ ] Add examples for all common use cases
- [ ] Create performance optimization guide
- [ ] Document all breaking changes

## 4. Version Updates

- [ ] Update version to 1.0.0 in all relevant files
  - [ ] pyproject.toml
  - [ ] __init__.py
  - [ ] VERSION_NOTES.md

## 5. Testing Requirements

- [ ] Create comprehensive test suite for standardized API
- [ ] Add specific tests for breaking changes
- [ ] Test on Python 3.10 (primary target)
- [ ] Test installation profiles
- [ ] Performance benchmark tests

## 6. Release Artifacts

- [ ] Final CHANGELOG.md update
- [ ] Release candidate tag
- [ ] PyPI package preparation
- [ ] Example notebook updates
- [ ] Release announcement draft

## Implementation Order

1. Fix API inconsistencies
2. Make breaking changes
3. Update documentation
4. Add tests for new standardized APIs
5. Run tests and fix any issues
6. Update version numbers
7. Create release candidate
8. Final testing
9. Release

## Notes

- The goal is a clean, consistent API even at the cost of backward compatibility
- Any legacy behavior should be documented in the migration guide
- Performance optimizations should be verified with benchmarks
- All changes should support the Python 3.10 target environment
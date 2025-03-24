# Meno 1.3.5 Release Notes

## New Features

### External LLM Deduplication

- Added comprehensive example for using deduplication with external LLM APIs
- New `external_llm_deduplication_example.py` demonstrates the full workflow:
  - Deduplicate documents (both exact and fuzzy) before LLM processing
  - Process only unique documents with external LLM APIs
  - Map results back to the full dataset
  - Analyze performance and cost savings
- Added detailed documentation in `docs/external_llm_deduplication.md`
- Provided code examples for OpenAI GPT and Anthropic Claude integration

## Improvements

- Enhanced `TextDeduplicator.map_results_to_full_dataset()` method documentation
- Added performance comparison tools for evaluating deduplication effectiveness
- Improved error handling in deduplication examples

## Bug Fixes

- Fixed parameter name inconsistency in documentation examples

## Documentation Updates

- Added new documentation file: `docs/external_llm_deduplication.md`
- Updated examples to show cost and token savings calculations
- Added visualizations for deduplication performance metrics
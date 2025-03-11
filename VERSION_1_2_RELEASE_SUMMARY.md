# Meno v1.2.0 Release Summary

## Overview

Version 1.2.0 of Meno enhances the topic modeling toolkit with powerful BERTopic integration and LLM-based topic labeling. These additions make topic models more user-friendly, flexible, and interpretable.

## Key Features 

### Advanced BERTopic Features

1. **Model Merging**
   - Combine multiple topic models trained on different datasets
   - Enable cross-domain knowledge transfer
   - Create unified models from specialized ones
   - Implemented via `merge_models()` method

2. **Topic Manipulation**
   - Merge similar topics with `merge_topics()`
   - Reduce topic count with `reduce_topics()`
   - Update topic metadata with `update_topics()`
   - Control topic granularity post-modeling

3. **Dynamic Topic Modeling**
   - Analyze how topics evolve over time
   - Track topic emergence, growth, and decline
   - Visualize temporal patterns using `visualize_topics_over_time()`
   - Implemented via `fit_transform_with_timestamps()`

4. **Semi-supervised Topic Modeling**
   - Guide topic discovery with seed topics
   - Combine domain expertise with data-driven discovery
   - Control topic focus areas while allowing for exploration
   - Implemented via `fit_with_seed_topics()`

### LLM-based Topic Labeling

1. **Multiple Model Options**
   - Local HuggingFace models (FLAN-T5, OPT, etc.)
   - OpenAI API integration (GPT-3.5/4)
   - Automatic fallback mechanisms for reliability

2. **Integration Options**
   - During model fitting with `use_llm_labeling=True`
   - Post-processing with `apply_llm_labeling()`
   - Standalone usage via `LLMTopicLabeler` class

3. **Customization**
   - Adjust detail level of topic descriptions
   - Control generation parameters
   - Provide example documents for context
   - Save and load labeling configurations

## Implementation Notes

The implementation prioritizes:

1. **User-friendly API** - Complex functionality through intuitive methods
2. **Backward compatibility** - Existing code continues to work
3. **Comprehensive examples** - See `examples/` directory:
   - `advanced_bertopic_features.py`
   - `llm_topic_labeling_example.py` 
   - `workflow_with_llm_labeling.py`

## Documentation

- Updated README with usage examples
- Comprehensive docstrings for all new methods
- Example scripts demonstrating all features

## Installation

Install with advanced features:

```bash
pip install "meno[llm]"           # For local LLM topic labeling
pip install "meno[llm_openai]"    # For OpenAI API integration
pip install "meno[full]"          # For all features
```

## What's Next

The roadmap for future development includes:

1. Incremental learning for streaming data
2. Enhanced multilingual support
3. Domain-specific fine-tuning options
4. More explainable AI features
5. Additional LLM integration options
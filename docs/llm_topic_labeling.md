# LLM-based Topic Labeling

The `LLMTopicLabeler` in Meno provides a way to generate human-readable topic names that are more intuitive and descriptive than traditional keyword-based labels. It leverages Language Models (LLMs) to create semantically meaningful topic names based on the top words and example documents from each topic.

## Features

- Support for both local HuggingFace models and OpenAI API
- Automatic fallback mechanisms if LLMs are not available
- Ability to generate both concise and detailed topic descriptions
- Integration with all topic models in Meno (BERTopic, SimpleTopicModel, NMFTopicModel, etc.)
- Customizable prompts and parameters for generating topic names

## Installation

The LLM-based topic labeling functionality has additional dependencies:

```bash
# For local model support (recommended for most users)
pip install "meno[llm]"

# For OpenAI API support (optional)
pip install "meno[openai]"

# For GPU acceleration with quantization (for 7B+ models)
pip install "meno[llm,gpu]"
```

## Basic Usage

Here's a simple example of how to use the LLMTopicLabeler:

```python
from meno.modeling import SimpleTopicModel, LLMTopicLabeler

# Train a topic model
model = SimpleTopicModel(num_topics=8)
model.fit(documents)

# Create LLM topic labeler with local model
labeler = LLMTopicLabeler(model_type="local", model_name="google/flan-t5-small")

# Generate improved topic names and update the model
updated_model = labeler.update_model_topic_names(model)

# Get the enhanced topic information
topic_info = updated_model.get_topic_info()
print(topic_info[["Topic", "Name"]])
```

## Using with OpenAI

If you have an OpenAI API key, you can use GPT models for higher quality topic names:

```python
import os
from meno.modeling import BERTopicModel, LLMTopicLabeler

# Set your API key
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY_PLACEHOLDER"

# Train a topic model
model = BERTopicModel(num_topics=12)
model.fit(documents)

# Create LLM topic labeler with OpenAI
labeler = LLMTopicLabeler(
    model_type="openai",
    model_name="gpt-3.5-turbo",
    temperature=0.5  # Lower temperature for more consistent names
)

# Generate improved topic names and update the model
updated_model = labeler.update_model_topic_names(model)
```

## Including Example Documents

For even better topic names, you can provide example documents for each topic:

```python
# Extract example documents for each topic
topic_assignments = model.transform(documents)[0]
example_docs_per_topic = {}

for topic_id in set(topic_assignments):
    if topic_id == -1:  # Skip outlier topic
        continue
    
    # Get documents for this topic
    doc_indices = [i for i, t in enumerate(topic_assignments) if t == topic_id]
    
    # Sample up to 5 documents
    sample_indices = doc_indices[:5]
    example_docs_per_topic[topic_id] = [documents[i] for i in sample_indices]

# Generate topic names with examples
updated_model = labeler.update_model_topic_names(
    model,
    example_docs_per_topic=example_docs_per_topic,
    detailed=True  # Generate more detailed descriptions
)
```

## Customizing the Labeler

The LLMTopicLabeler can be customized with several parameters:

```python
labeler = LLMTopicLabeler(
    model_type="local",               # "local", "openai", or "auto"
    model_name="google/flan-t5-base", # Model name or path
    max_new_tokens=50,                # Maximum tokens to generate
    temperature=0.7,                  # Temperature for generation (higher = more creative)
    enable_fallback=True,             # Whether to use rule-based fallback if LLM fails
    device="cuda",                    # Device to use: "cpu", "cuda", "mps", or "auto"
    verbose=True                      # Whether to show verbose output
)
```

## Using Offline Models

You can use locally downloaded models for offline environments:

```python
# Load from a local path containing model files
labeler = LLMTopicLabeler(
    model_type="local",
    model_name="/path/to/local/model",  # Path to directory with model files
    device="cpu"                        # Or "cuda" if GPU is available
)

# Update your existing model with offline LLM topic names
enhanced_model = labeler.update_model_topic_names(topic_model)
```

## Recommended Models

For topic labeling, we recommend these models based on your hardware constraints:

### Small Models (CPU-friendly)
- `google/flan-t5-small` (~80MB) - Very fast, basic quality
- `facebook/opt-125m` (~240MB) - Fast, reasonable quality
- `microsoft/phi-1` (~1.3GB) - Good balance of size and quality

### Mid-size Models (8GB+ GPU recommended)
- `stabilityai/stablelm-base-alpha-3b` (~5GB) - Good quality topic names
- `facebook/opt-2.7b` (~5GB) - Good descriptive capabilities
- `EleutherAI/pythia-2.8b` (~5GB) - Good semantic understanding

### 7-8B Range Models (16GB+ GPU recommended)
- `meta-llama/Llama-2-7b-hf` (~13GB) - Excellent quality topic names
- `bigscience/bloom-7b1` (~14GB) - Good multilingual capabilities
- `mistralai/Mistral-7B-v0.1` (~13GB) - Strong semantic reasoning

When using 7B+ models, we recommend installing with GPU support:

```bash
pip install "meno[llm,gpu]"
```

## Saving and Loading

You can save and load the LLMTopicLabeler configuration:

```python
# Save the configuration
labeler.save("path/to/labeler_config.json")

# Load the configuration
labeler = LLMTopicLabeler.load("path/to/labeler_config.json")
```

## Example with Topic Model Report

This example shows how to use LLM-generated topic names in HTML reports:

```python
from meno.reporting import generate_html_report
from meno.modeling import SimpleTopicModel, LLMTopicLabeler

# Train a topic model
model = SimpleTopicModel(num_topics=10)
model.fit(documents)

# Apply LLM topic labeling
labeler = LLMTopicLabeler()
enhanced_model = labeler.update_model_topic_names(model)

# Get topic assignments
topic_ids, topic_probs = enhanced_model.transform(documents)
topic_assignments = pd.DataFrame({
    "doc_id": range(len(documents)),
    "topic": topic_ids,
    "topic_probability": np.ones(len(topic_ids))
})

# Generate report with the enhanced topic names
generate_html_report(
    documents=pd.DataFrame({"text": documents}),
    topic_assignments=topic_assignments,
    output_path="topic_report.html",
    config={"title": "Topic Model with LLM-Generated Labels"}
)
```

## Performance Considerations

- Local models like `google/flan-t5-small` are faster but may produce less coherent topic names
- OpenAI models typically provide higher quality topic names but require an API key and incur usage costs
- For large numbers of topics, consider batching the requests to avoid rate limits or memory issues

## Compatibility

The LLMTopicLabeler is compatible with all topic models in Meno that implement the BaseTopicModel interface, including:

- BERTopicModel
- Top2VecModel
- SimpleTopicModel
- TFIDFTopicModel
- NMFTopicModel
- LSATopicModel
# Using Meno in Offline Environments

This guide explains how to use Meno in offline or air-gapped environments where direct access to HuggingFace model repositories may be restricted or unavailable.

## Overview

Meno supports several methods for using pre-trained models in offline environments:

1. Using manually downloaded model files 
2. Working with existing models in HuggingFace cache
3. Using custom local model directories
4. Setting the `local_files_only=True` parameter to prevent online lookups
5. Using the new `offline_mode=True` parameter to bypass module import checks

## Automatic SpaCy Model Installation

Starting from version 1.0.1, Meno will automatically attempt to download the required spaCy model (`en_core_web_sm`) when the package is imported. This is a one-time operation that ensures the necessary language model is available for text preprocessing.

To install the spaCy model explicitly without triggering the automatic download:

```bash
# Install the spaCy model directly (before or after installing Meno)
pip install en_core_web_sm

# Or install with the spacy_model extra
pip install "meno[spacy_model]"
```

## Preparing Models for Offline Use

Before moving to an offline environment, download the necessary models on a connected machine:

### Embedding Models

```python
# Download models to HuggingFace cache
from sentence_transformers import SentenceTransformer

# Download sentence-transformers model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"Model downloaded to: {model._model_card_vars['__path__']}")

# For BERTopic, you may want to download additional models
from bertopic import BERTopic
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

# You can also manually download models from HuggingFace Hub
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
```

### LLM Topic Labeling Models

For LLM-based topic labeling in offline mode:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Small models (use for CPU)
model_name = "google/flan-t5-small"

# Medium models (8GB+ GPU)
# model_name = "facebook/opt-2.7b"

# 7B-8B models (16GB+ GPU)
# model_name = "mistralai/Mistral-7B-v0.1"

# Download models and tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save to disk
output_path = f"./local_models/{model_name.split('/')[-1]}"
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)
print(f"Model saved to: {output_path}")
```

## Required Files

A complete model directory should contain:

- `config.json`: Model configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `tokenizer.json`: Tokenizer configuration
- `tokenizer_config.json`: Additional tokenizer settings
- `vocab.txt` or `spm.model`: Vocabulary or SentencePiece model
- `modules.json`: SentenceTransformers modules configuration (for sentence-transformers models)
- `sentence_bert_config.json`: SentenceTransformers configuration (for sentence-transformers models)

## Using Local Models

### Option 1: Direct Path to Models

Provide the exact path to a local model directory:

```python
from meno.modeling.embeddings import DocumentEmbedding
from meno import MenoTopicModeler

# Create embedding model with explicit path
embedding_model = DocumentEmbedding(
    local_model_path="/path/to/model/directory",
    use_gpu=False
)

# Use the embedding model with a topic modeler
modeler = MenoTopicModeler(embedding_model=embedding_model)
```

### Option 2: Using HuggingFace Cache with local_files_only

If you've downloaded models to the standard HuggingFace cache location:

```python
# Create embedding model with local_files_only
embedding_model = DocumentEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True,
    use_gpu=False
)
```

### Option 3: With BERTopic Models

For BERTopic integration with local models:

```python
from meno.modeling.bertopic_model import BERTopicModel

# Create embedding model
embedding_model = DocumentEmbedding(
    local_model_path="/path/to/embedding/model",
    use_gpu=False
)

# Create BERTopic model with the embedding model
bertopic_model = BERTopicModel(
    embedding_model=embedding_model
)

# When loading a saved model
loaded_model = BERTopicModel.load(
    path="/path/to/saved/bertopic/model",
    local_files_only=True
)
```

## Complete Offline Mode

For environments with strict dependency management or when working with manually installed packages:

```python
from meno import MenoWorkflow
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.unified_topic_modeling import create_topic_modeler

# 1. Create embedding model with offline mode
embedding_model = DocumentEmbedding(
    local_model_path="/path/to/model/directory",
    local_files_only=True
)

# 2. Create topic modeler with offline mode
# This will bypass module availability checks and assume components are available
topic_model = create_topic_modeler(
    method="bertopic",  # Will work even if bertopic is manually installed 
    embedding_model=embedding_model,
    offline_mode=True   # Bypass import checks and assume modules are available
)

# 3. Alternatively, use this mode with the workflow
workflow = MenoWorkflow(
    local_model_path="/path/to/model/directory",
    local_files_only=True,
    offline_mode=True   # Will use modules even if standard import checks fail
)
```

The `offline_mode=True` parameter tells Meno to bypass standard module import checks and attempt to use components even if they've been installed through non-standard methods (e.g., directly copying files, manual installation, or custom package loaders).

## Using LLM Topic Labeling Offline

For using LLM-based topic labeling in offline environments:

```python
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno.modeling.simple_models.lightweight_models import SimpleTopicModel

# 1. Create topic model
topic_model = SimpleTopicModel(num_topics=10)
topic_model.fit(documents)

# 2. Create LLM topic labeler with local model path
labeler = LLMTopicLabeler(
    model_type="local",
    model_name="/path/to/local/llm_model",  # Path to directory with saved model
    device="cpu",                           # or "cuda" if GPU is available
    enable_fallback=True                    # Fallback to rule-based if LLM fails
)

# 3. Apply improved topic names
updated_model = labeler.update_model_topic_names(topic_model)

# 4. Get improved topic info
topic_info = updated_model.get_topic_info()
print(topic_info[["Topic", "Name", "Count"]])
```

The LLM topic labeler works with models of different sizes:

1. Small models (CPU-friendly): Google's Flan-T5 Small, Facebook's OPT-125M
2. Medium models (8GB+ GPU recommended): StableLM 3B, Facebook OPT-2.7B
3. 7B-8B models (16GB+ GPU required): Llama 2 7B, Mistral 7B

## File Locations

Common cache locations where models may be found:

- HuggingFace Transformers models: `~/.cache/huggingface/hub/`
- Specific model path: `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`
- Meno's own cache: `~/.cache/meno/models/`

## Full Example

### Basic Offline Workflow

```python
import pandas as pd
from pathlib import Path
from meno import MenoWorkflow
from meno.modeling.embeddings import DocumentEmbedding

# 1. Set up paths
model_path = Path.home() / "offline_models" / "all-MiniLM-L6-v2"

# 2. Create embedding model
embedding_model = DocumentEmbedding(
    local_model_path=str(model_path),
    use_gpu=False
)

# 3. Create workflow (either way works)
# Option A: Pass the embedding model directly
workflow = MenoWorkflow(embedding_model=embedding_model)

# Option B: Use local_model_path parameter with offline mode
workflow2 = MenoWorkflow(
    local_model_path=str(model_path),
    local_files_only=True,
    offline_mode=True  # Use this for environments with manually installed packages
)

# 4. Load data
data = pd.DataFrame({
    "text": [
        "This is the first document.",
        "This is another document.",
        "This is the third document."
    ]
})
workflow.load_data(data=data, text_column="text")

# 5. Process data
workflow.preprocess_documents()
workflow.discover_topics(num_topics=2)

# 6. Generate report
workflow.generate_comprehensive_report("report.html")
```

### Advanced Offline Example with LLM Topic Labeling

```python
import pandas as pd
from pathlib import Path
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.simple_models.lightweight_models import NMFTopicModel
from meno.modeling.llm_topic_labeling import LLMTopicLabeler
from meno.reporting.html_generator import generate_html_report

# Set up paths to locally saved models
EMBEDDING_MODEL_PATH = "/path/to/offline_models/all-MiniLM-L6-v2"
LLM_MODEL_PATH = "/path/to/offline_models/flan-t5-small"

# 1. Load your data
documents = pd.read_csv("/path/to/offline_data.csv")["text"].tolist()

# 2. Create embedding model with local path
embedding_model = DocumentEmbedding(
    local_model_path=EMBEDDING_MODEL_PATH,
    use_gpu=False,
    local_files_only=True
)

# 3. Create and fit topic model
topic_model = NMFTopicModel(
    num_topics=10,
    max_features=2000,
    embedding_model=embedding_model
)
topic_model.fit(documents)

# 4. Get original topic info for comparison
original_topic_info = topic_model.get_topic_info().copy()
print("Original topics:")
print(original_topic_info[["Topic", "Name", "Count"]])

# 5. Use LLM for topic labeling with local model
try:
    labeler = LLMTopicLabeler(
        model_type="local",
        model_name=LLM_MODEL_PATH,
        device="cpu",
        enable_fallback=True
    )
    
    # Generate topic names and update the model
    updated_model = labeler.update_model_topic_names(topic_model)
    
    # Show improved topic names
    new_topic_info = updated_model.get_topic_info()
    print("\nLLM-generated topic names:")
    print(new_topic_info[["Topic", "Name", "Count"]])
    
except Exception as e:
    print(f"Failed to run LLM labeling: {e}")
    print("Continuing with original topic names")
    updated_model = topic_model

# 6. Generate the report
topic_assignments = pd.DataFrame({
    "doc_id": range(len(documents)),
    "topic": updated_model.transform(documents)[0],  # Get topic assignments
    "topic_probability": 0.9  # Placeholder for simplified example
})

# 7. Generate HTML report with the improved topic names
report_path = generate_html_report(
    documents=pd.DataFrame({"text": documents}),
    topic_assignments=topic_assignments,
    output_path="offline_report.html",
    config={
        "title": "Offline Topic Modeling Report",
        "include_interactive": True
    }
)

# 8. Save the model for future use
output_dir = Path("./output/offline_model")
updated_model.save(output_dir)
print(f"Model saved to {output_dir}")
print(f"Report generated at {report_path}")
```

## Troubleshooting

If you encounter issues:

1. Ensure all model files are present in the directory
2. Check paths for typos or incorrect structure
3. Try using absolute paths instead of relative paths
4. Verify file permissions on the model directory
5. Try using `offline_mode=True` if standard import checks are failing
6. Check if your model requires additional files beyond the standard ones

For detailed help on using local models, refer to `examples/local_model_example.py` in the Meno repository.
# Using Meno in Offline Environments

This guide explains how to use Meno in offline or air-gapped environments where direct access to HuggingFace model repositories may be restricted or unavailable.

## Overview

Meno supports several methods for using pre-trained models in offline environments:

1. Using manually downloaded model files 
2. Working with existing models in HuggingFace cache
3. Using custom local model directories
4. Setting the `local_files_only=True` parameter to prevent online lookups

## Preparing Models for Offline Use

Before moving to an offline environment, download the necessary models on a connected machine:

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

## File Locations

Common cache locations where models may be found:

- HuggingFace Transformers models: `~/.cache/huggingface/hub/`
- Specific model path: `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`
- Meno's own cache: `~/.cache/meno/models/`

## Full Example

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

# 3. Create workflow
workflow = MenoWorkflow(embedding_model=embedding_model)

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

## Troubleshooting

If you encounter issues:

1. Ensure all model files are present in the directory
2. Check paths for typos or incorrect structure
3. Try using absolute paths instead of relative paths
4. Verify file permissions on the model directory
5. Check if your model requires additional files beyond the standard ones

For detailed help on using local models, refer to `examples/local_model_example.py` in the Meno repository.
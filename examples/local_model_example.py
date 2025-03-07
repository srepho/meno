"""
Example showing how to use Meno with local models for offline environments.

This example demonstrates:
1. Using sentence-transformers with locally downloaded models
2. Loading BERTopic with local files only
3. Options for finding and loading models from local paths

Usage:
- First download a model manually (e.g., all-MiniLM-L6-v2)
- Place it in HuggingFace cache (~/.cache/huggingface/hub/) or a custom directory
- Run this example with local_files_only=True
"""

import pandas as pd
import os
from pathlib import Path
from meno import MenoTopicModeler
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.bertopic_model import BERTopicModel

# Sample data
data = pd.DataFrame({
    "text": [
        "The CEO and CFO met to discuss AI implementation in our CRM system.",
        "Customer submitted a claim for their vehicle accident on HWY 101.",
        "The CTO presented the ML strategy for improving customer retention.",
        "Policyholder received the EOB and was confused about the CPT codes."
    ]
})

# OPTION 1: Provide explicit path to a locally downloaded model
def using_explicit_local_path():
    print("\n=== Using explicitly provided local model path ===")
    
    # Point to your downloaded model (example path, adjust to your system)
    local_model_path = os.path.expanduser("~/models/all-MiniLM-L6-v2")
    
    # Create embedding model with explicit path
    embedding_model = DocumentEmbedding(
        local_model_path=local_model_path,
        use_gpu=False
    )
    
    # Create and run modeler with custom embedding model
    modeler = MenoTopicModeler(embedding_model=embedding_model)
    processed_docs = modeler.preprocess(data, text_column="text")
    topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=2)
    
    print(f"Discovered {len(topics_df['topic'].unique())} topics")
    print(topics_df[['text', 'topic']].head())


# OPTION 2: Using HuggingFace's standard cache location
def using_huggingface_cache():
    print("\n=== Using HuggingFace cache with local_files_only ===")
    
    # Create embedding model using HuggingFace cache with local_files_only
    embedding_model = DocumentEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True,
        use_gpu=False
    )
    
    # Create and run modeler with custom embedding model
    modeler = MenoTopicModeler(embedding_model=embedding_model)
    processed_docs = modeler.preprocess(data, text_column="text")
    topics_df = modeler.discover_topics(method="embedding_cluster", num_topics=2)
    
    print(f"Discovered {len(topics_df['topic'].unique())} topics")
    print(topics_df[['text', 'topic']].head())


# OPTION 3: Using BERTopic with local files only
def using_bertopic_with_local_files():
    print("\n=== Using BERTopic with local files only ===")
    
    # Create embedding model with local_files_only setting
    embedding_model = DocumentEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True,
        use_gpu=False
    )
    
    # Create BERTopic model with the embedding model
    bertopic_model = BERTopicModel(
        embedding_model=embedding_model,
        min_topic_size=1  # Small sample size
    )
    
    # Process documents
    bertopic_model.fit(data["text"].tolist())
    
    # Get topic info
    topic_info = bertopic_model.get_topic_info()
    print(topic_info.head())
    
    # Save model
    model_path = Path("./saved_model")
    bertopic_model.save(model_path)
    
    # Load model with local_files_only flag
    loaded_model = BERTopicModel.load(
        path=model_path,
        local_files_only=True
    )
    
    # Transform new documents with loaded model
    new_docs = [
        "Meeting about the AI strategy for next quarter",
        "Insurance claim processing for auto accident"
    ]
    topics, probs = loaded_model.transform(new_docs)
    print(f"Topics for new documents: {topics}")


# Run examples (comment out any you don't want to run)
if __name__ == "__main__":
    print("Meno Local Model Examples")
    print("=========================")
    print("These examples demonstrate using locally downloaded models.")
    print("To run successfully, you should have all-MiniLM-L6-v2 downloaded.")
    
    try:
        using_explicit_local_path()
    except Exception as e:
        print(f"Error with explicit path example: {e}")
        
    try:
        using_huggingface_cache()
    except Exception as e:
        print(f"Error with HuggingFace cache example: {e}")
        
    try:
        using_bertopic_with_local_files()
    except Exception as e:
        print(f"Error with BERTopic example: {e}")
        
    print("\nNote: If examples failed, ensure you have the models downloaded.")
    print("Paths to check:")
    print("1. Custom path provided in using_explicit_local_path()")
    print("2. ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/")
    print("3. ~/.cache/meno/models/sentence-transformers_all-MiniLM-L6-v2/")
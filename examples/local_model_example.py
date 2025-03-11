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
    modeler = MenoTopicModeler(
        embedding_model=embedding_model
    )
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
    modeler = MenoTopicModeler(
        embedding_model=embedding_model
    )
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
    
    # Check if BERTopic is available
    try:
        # Import the required modules to check availability
        from bertopic import BERTopic
        bertopic_available = True
    except ImportError:
        bertopic_available = False
    
    if not bertopic_available:
        raise ImportError("BERTopic is required for this model. Install with 'pip install bertopic>=0.15.0'")
        
    # Create BERTopic model with the embedding model
    # Force module availability flag to True to bypass import checks
    import meno.modeling.bertopic_model
    # Override import check result in offline environments
    meno.modeling.bertopic_model.BERTOPIC_AVAILABLE = True
    
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


# OPTION 4: Using the MenoWorkflow with local model
def using_workflow_with_local_model():
    print("\n=== Using MenoWorkflow with local model path ===")
    
    # Specify a local model path (adjust to your system)
    local_model_path = os.path.expanduser("~/models/all-MiniLM-L6-v2")
    
    # Create workflow with local model options and offline mode
    from meno.workflow import MenoWorkflow
    workflow = MenoWorkflow(
        local_model_path=local_model_path,
        local_files_only=True,
        offline_mode=True  # Enable offline mode to bypass import checks
    )
    
    # Load data and run workflow
    workflow.load_data(data, text_column="text")
    workflow.preprocess_documents()
    workflow.discover_topics(method="embedding_cluster", num_topics=2)
    
    # Print results
    topics_df = workflow.modeler.documents
    print(f"Discovered {len(topics_df['topic'].unique())} topics")
    print(topics_df[['text', 'topic']].head())


# Run examples (comment out any you don't want to run)
# OPTION 5: Using the complete offline mode
def using_complete_offline_mode():
    print("\n=== Using Complete Offline Mode ===")
    
    # Choose whichever local model path is available in your environment
    try:
        # Try standard HuggingFace cache
        cache_home = os.path.expanduser("~/.cache/huggingface/hub")
        model_files_dir = os.path.join(cache_home, "models--sentence-transformers--all-MiniLM-L6-v2")
        if os.path.exists(model_files_dir):
            # Find snapshots directory
            snapshots_dir = os.path.join(model_files_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    latest_snapshot = sorted(snapshot_dirs)[-1]
                    local_model_path = os.path.join(snapshots_dir, latest_snapshot)
                    print(f"Using model from HuggingFace cache: {local_model_path}")
        else:
            # Fallback to default in ~/models directory
            local_model_path = os.path.expanduser("~/models/all-MiniLM-L6-v2")
            print(f"Using model from user directory: {local_model_path}")
    except Exception:
        # Last resort fallback
        local_model_path = os.path.expanduser("~/models/all-MiniLM-L6-v2")
        print(f"Using fallback model path: {local_model_path}")
        
    # Using unified topic modeling interface with offline mode
    from meno.modeling.unified_topic_modeling import create_topic_modeler
    
    # Create embedding model with offline settings
    embedding_model = DocumentEmbedding(
        local_model_path=local_model_path,
        local_files_only=True
    )
    
    # Create topic modeler with offline mode
    topic_model = create_topic_modeler(
        method="bertopic",
        embedding_model=embedding_model,
        offline_mode=True,  # This bypasses import checks completely
        config_overrides={
            'min_topic_size': 1,  # Small sample size
        }
    )
    
    # Process documents
    topic_model.fit(data["text"].tolist())
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(topic_info.head())


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
        
    try:
        using_workflow_with_local_model()
    except Exception as e:
        print(f"Error with workflow example: {e}")
        
    try:
        using_complete_offline_mode()
    except Exception as e:
        print(f"Error with complete offline mode example: {e}")
        
    print("\nNote: If examples failed, ensure you have the models downloaded.")
    print("Paths to check:")
    print("1. Custom path provided in using_explicit_local_path()")
    print("2. ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/")
    print("3. ~/.cache/meno/models/sentence-transformers_all-MiniLM-L6-v2/")
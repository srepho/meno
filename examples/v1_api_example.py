"""
Meno v1.0.0 API Example

This example demonstrates the standardized v1.0.0 API for Meno's topic modeling capabilities,
showing all major features in one integrated example.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from meno.modeling.unified_topic_modeling import create_topic_modeler
from meno.modeling.embeddings import DocumentEmbedding
from meno.modeling.streaming_processor import StreamingProcessor
from meno.utils.team_config import get_team_config
from meno.visualization.static_plots import create_topic_visualization
from meno.workflow import create_workflow
from meno.preprocessing import normalize_text

# Sample data
documents = [
    "Machine learning algorithms require significant computational resources",
    "Deep neural networks have revolutionized computer vision tasks",
    "Natural language processing enables machines to understand text",
    "Reinforcement learning is used to train game-playing AI",
    "Data science relies on statistics and domain knowledge",
    "Feature engineering improves model performance significantly",
    "Transformers have improved natural language understanding",
    "Computer vision systems can now recognize objects in images",
    "Statistical models help understand patterns in data",
    "Unsupervised learning finds patterns without labeled data",
    "Cloud computing provides scalable resources for AI workloads",
    "GPUs accelerate deep learning model training times",
    "Transfer learning applies knowledge from one domain to another",
    "Convolutional neural networks excel at image recognition tasks",
    "Recurrent neural networks process sequential data effectively",
]

# Convert to pandas DataFrame
df = pd.DataFrame({"text": documents})

print("Meno v1.0.0 API Example")
print(f"Documents: {len(df)}")

# 1. Team Configuration
print("\n1. Team Configuration")
team_config = get_team_config("ai_research")

# Add domain-specific acronyms
team_config.add_acronyms({
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "NLP": "Natural Language Processing",
    "CV": "Computer Vision",
    "RL": "Reinforcement Learning", 
    "AI": "Artificial Intelligence",
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
})

# View the team configuration
print(f"Team configuration loaded: {team_config.team_name}")
print(f"Acronyms defined: {len(team_config.acronyms)}")

# Export configuration (optional)
config_path = team_config.export_config("ai_research_config.json")
print(f"Configuration exported to: {config_path}")

# 2. Preprocessing with team configuration
print("\n2. Preprocessing")
# Normalize text with team acronyms
df['normalized_text'] = df['text'].apply(lambda x: normalize_text(
    x,
    expand_acronyms=True,
    acronym_dict=team_config.acronyms,
))

# 3. Embedding using standardized API
print("\n3. Embedding generation")
embedding_model = DocumentEmbedding(model_name="all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(df['normalized_text'])
print(f"Generated embeddings shape: {embeddings.shape}")

# 4. Create topic model using unified API
print("\n4. Topic modeling with standardized API")
# Option 1: Using factory function
model = create_topic_modeler(
    method="bertopic",
    num_topics=5,                 # Standardized parameter name
    embedding_model=embedding_model,
    auto_detect_topics=False,     # Explicitly state we're using fixed number
)

# Fit model
model.fit(df['normalized_text'], embeddings=embeddings)

# Get topic information (standardized method)
topic_info = model.get_topic_info()
print("\nTopic Information:")
print(topic_info[['Topic', 'Count', 'Name', 'Representation']])

# 5. Document topic assignments
print("\n5. Document topic assignments")
# Transform documents to topic assignments (standardized return type)
topics, probs = model.transform(df['normalized_text'], embeddings)
print(f"Topic assignments shape: {topics.shape}")
print(f"Topic probabilities shape: {probs.shape}")

# 6. Topic similarity search (standardized method)
print("\n6. Topic similarity search")
similar_topics = model.find_similar_topics("artificial intelligence", n_topics=2)
print("Topics similar to 'artificial intelligence':")
for topic_id, topic_desc, score in similar_topics:
    print(f"  Topic {topic_id}: {topic_desc} (Score: {score:.4f})")

# 7. Visualization with standardized parameters
print("\n7. Visualization")
try:
    # Create visualization (standardized parameters)
    fig = model.visualize_topics(
        width=800,           # Standardized parameter
        height=600,          # Standardized parameter
    )
    print("Visualization created successfully")
    # fig.show()  # Uncomment to show in browser
except Exception as e:
    print(f"Visualization error: {e}")

# 8. Memory-efficient processing for large datasets
print("\n8. Memory-efficient processing (for large datasets)")
processor = StreamingProcessor(
    embedding_model=embedding_model,
    topic_model="bertopic",
    batch_size=5,           # Small batch size for demonstration
    use_quantization=True   # Use float16 to reduce memory
)

# Stream documents in batches
print("Streaming documents in batches:")
batch_count = 0
for batch_embeddings, batch_ids in processor.stream_documents(df['normalized_text']):
    batch_count += 1
    print(f"  Processed batch {batch_count}: {len(batch_ids)} documents")

# 9. Model persistence with standardized API
print("\n9. Model persistence")
# Save model with standardized path handling
model_path = Path("./meno_model")
model.save(model_path)
print(f"Model saved to: {model_path}")

# Load model
loaded_model = model.__class__.load(model_path)
print(f"Model loaded successfully: {type(loaded_model).__name__}")

# 10. Using workflow API
print("\n10. Complete workflow API")
# Create workflow with standardized configuration
workflow = create_workflow(
    num_topics=5,            # Standardized parameter
    embedding_model="all-MiniLM-L6-v2",
    topic_model="bertopic",
    auto_detect_topics=False
)

# Run workflow
result = workflow.run(df['text'])
print(f"Workflow completed with {len(result['topic_model'].topics)} topics")

print("\nv1.0.0 API example completed successfully.")
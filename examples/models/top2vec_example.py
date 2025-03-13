# Top2Vec Integration - Using Top2Vec for Topic Modeling
#
# This example demonstrates how to use Top2Vec for document embedding
# and topic discovery through the Meno framework.

import pandas as pd
from meno.modeling.top2vec_model import Top2VecModel

# Sample data
documents = [
    "Topic modeling is a type of statistical modeling for discovering abstract topics in text",
    "Machine learning algorithms build a model based on sample data to make predictions",
    "Natural language processing is a field of AI that helps computers understand human language",
    "Deep learning is part of machine learning methods based on artificial neural networks",
    "Clustering algorithms group similar objects into clusters without predefined categories",
    "Text mining is the process of deriving high-quality information from text",
    "Artificial intelligence is intelligence demonstrated by machines",
    "Neural networks are computing systems inspired by biological neural networks",
    "Unsupervised learning is a type of machine learning that looks for patterns in data",
    "Word embeddings represent words as vectors in a continuous vector space"
]

# Create a Top2Vec model with default parameters
model = Top2VecModel(
    min_count=1,             # Minimum word count (set low for small dataset)
    speed="learn",           # Can be "fast-learn", "learn", or "deep-learn"
    workers=4,               # Number of CPU cores to use
    use_embedding_model_tokenizer=True,
    embedding_model="all-MiniLM-L6-v2"  # Use a smaller SentenceTransformer model
)

# Fit the model on our documents
model.fit(documents)

# Get topic information
topic_info = model.get_topic_info()
print("Topic Information:")
print(topic_info)

# Get topic words for each topic
for topic_id in range(len(topic_info)):
    if topic_id >= 0:  # Skip outlier topic (-1)
        topic_words = model.get_topic_words(topic_id)
        print(f"\nTopic {topic_id} Words:")
        print(", ".join(topic_words[:10]))

# Get document-topic assignments
doc_info = model.get_document_info(documents)
print("\nDocument-Topic Assignments:")
print(doc_info[["Document", "Topic", "Score"]].head())

# Visualize the topics
try:
    topic_viz = model.visualize_topics()
    topic_viz.write_html("top2vec_topics.html")
    print("\nVisualization saved to top2vec_topics.html")
except Exception as e:
    print(f"\nVisualization could not be generated: {e}")

# Document search by keywords
search_results = model.search_documents_by_keywords(
    keywords=["machine learning", "neural networks"],
    num_docs=3
)
print("\nSearch Results for 'machine learning, neural networks':")
for doc, score in zip(search_results["documents"], search_results["scores"]):
    print(f"Score: {score:.4f} - {doc}")

print("\nNote: For a full-featured Top2Vec experience, use the complete model:")
print("from top2vec import Top2Vec")
print("model = Top2Vec(documents, speed='deep-learn', workers=8)")
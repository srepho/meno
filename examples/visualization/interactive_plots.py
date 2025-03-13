# Interactive Visualizations - Creating Plotly Visualizations
#
# This example demonstrates how to create interactive visualizations
# for topic modeling results using Plotly through the Meno framework.

import pandas as pd
from meno import MenoTopicModeler
from meno.visualization.interactive_plots import (
    plot_topic_distribution,
    plot_document_heatmap,
    create_topic_explorer
)

# Sample data (in a real application, load from a file)
documents = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Topic modeling is a type of statistical modeling for discovering abstract topics in document collections.",
    "Natural language processing (NLP) is a field of artificial intelligence focused on interactions between computers and human language.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Unsupervised learning is where the algorithm is given data without explicit instructions on what to do with it.",
    "Supervised learning algorithms build a model based on labeled training data.",
    "Clustering is the task of dividing data points into groups based on similarity.",
    "Classification in machine learning is the task of predicting which category an observation belongs to.",
    "Regression analysis is used to estimate the relationships among variables.",
    "Dimensionality reduction is the process of reducing the number of variables under consideration."
]

# Create a DataFrame
df = pd.DataFrame({"text": documents})

# Initialize modeler
modeler = MenoTopicModeler()
modeler.preprocess(df, text_column="text")
modeler.discover_topics(method="embedding_cluster", auto_detect_topics=True)

# Get document-topic information
doc_topics = modeler.get_document_topics()

# Create interactive topic distribution visualization
print("Creating topic distribution visualization...")
fig_dist = plot_topic_distribution(
    doc_topics=doc_topics,
    title="Topic Distribution",
    width=800,
    height=500
)
fig_dist.write_html("topic_distribution.html")
print("Saved to topic_distribution.html")

# Create document-topic heatmap
print("Creating document-topic heatmap...")
fig_heatmap = plot_document_heatmap(
    documents=documents[:10],  # Limit to first 10 for clarity
    doc_topics=doc_topics.iloc[:10],
    title="Document-Topic Distribution",
    width=900,
    height=600
)
fig_heatmap.write_html("document_topic_heatmap.html")
print("Saved to document_topic_heatmap.html")

# Create topic embedding visualization
print("Creating interactive embedding visualization...")
fig_embed = modeler.visualize_embeddings(
    plot_3d=True,
    include_topic_centers=True,
    width=1000,
    height=800
)
fig_embed.write_html("topic_embeddings_3d.html")
print("Saved to topic_embeddings_3d.html")

# Create topic explorer dashboard
print("Creating interactive topic explorer...")
try:
    explorer = create_topic_explorer(
        modeler,
        documents,
        title="Interactive Topic Explorer"
    )
    explorer.write_html("topic_explorer_dashboard.html")
    print("Saved to topic_explorer_dashboard.html")
except Exception as e:
    print(f"Could not create topic explorer: {e}")

# Create topic word distributions
print("Creating word distribution visualizations...")
for topic_id in modeler.get_topic_info()["Topic"].unique():
    if topic_id >= 0:  # Skip outlier topic (-1)
        try:
            fig_words = modeler.visualize_topic_words(
                topic_id=topic_id, 
                return_figure=True
            )
            fig_words.write_html(f"topic_{topic_id}_words.html")
            print(f"Saved topic {topic_id} word visualization")
        except Exception as e:
            print(f"Could not create visualization for topic {topic_id}: {e}")

print("\nAll visualizations have been created successfully!")
print("Open the HTML files in a web browser to explore the interactive visualizations.")
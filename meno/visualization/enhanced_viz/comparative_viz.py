"""
Comparative visualization module for comparing different topic models.

This module provides functions to compare topics across different models
and visualize the differences and similarities between them.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Union, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform


def compare_topic_models(
    models: List[Any],
    model_names: List[str],
    documents: List[str],
    n_topics: int = 10,
    topic_words: int = 10,
    comparison_method: str = "overlap",
    width: int = 1200,
    height: int = 800,
):
    """
    Compare topics from different topic models.

    Args:
        models: List of fitted topic models
        model_names: Names for each model
        documents: List of documents used for fitting
        n_topics: Number of top topics to compare
        topic_words: Number of words per topic to use
        comparison_method: Method to use for comparison ('overlap', 'embedding', or 'correlation')
        width: Width of the visualization in pixels, by default 1200
        height: Height of the visualization in pixels, by default 800

    Returns:
        fig: Plotly figure object
    """
    if len(models) != len(model_names):
        raise ValueError("Number of models must match number of model names")
        
    # Extract topic words from each model
    all_topics = []
    for i, model in enumerate(models):
        model_topics = []
        for topic_id in range(min(n_topics, len(model.get_topic_info()))):
            words = [word for word, _ in model.get_topic(topic_id)[:topic_words]]
            model_topics.append(words)
        all_topics.append(model_topics)
    
    # Create comparison visualization based on method
    if comparison_method == "overlap":
        return _create_overlap_visualization(all_topics, model_names, width, height)
    elif comparison_method == "embedding":
        return _create_embedding_visualization(all_topics, model_names, width, height)
    elif comparison_method == "correlation":
        # Get document-topic distributions
        doc_topics = []
        for model in models:
            doc_topics.append(model.transform(documents))
        return _create_correlation_visualization(doc_topics, model_names, width, height)
    else:
        raise ValueError("Invalid comparison method. Choose 'overlap', 'embedding', or 'correlation'")


def _create_overlap_visualization(all_topics, model_names, width, height):
    """Create visualization based on word overlap between topics."""
    n_models = len(model_names)
    n_topics = len(all_topics[0])
    
    # Calculate Jaccard similarity between topics
    similarity_matrices = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            sim_matrix = np.zeros((n_topics, n_topics))
            for t1 in range(n_topics):
                for t2 in range(n_topics):
                    set1 = set(all_topics[i][t1])
                    set2 = set(all_topics[j][t2])
                    if not (set1 and set2):
                        sim_matrix[t1, t2] = 0
                    else:
                        sim_matrix[t1, t2] = len(set1.intersection(set2)) / len(set1.union(set2))
            similarity_matrices.append((i, j, sim_matrix))
    
    # Create visualization
    fig = go.Figure()
    
    for i, j, sim_matrix in similarity_matrices:
        # Convert to long format for visualization
        model_i_name = model_names[i]
        model_j_name = model_names[j]
        
        sim_df = pd.DataFrame(sim_matrix)
        sim_df.index = [f"{model_i_name} Topic {t}" for t in range(n_topics)]
        sim_df.columns = [f"{model_j_name} Topic {t}" for t in range(n_topics)]
        
        sim_long = sim_df.reset_index().melt(id_vars="index", var_name="column", value_name="similarity")
        
        fig.add_trace(go.Heatmap(
            z=sim_matrix,
            x=[f"T{t}" for t in range(n_topics)],
            y=[f"T{t}" for t in range(n_topics)],
            colorscale="Viridis",
            name=f"{model_i_name} vs {model_j_name}",
            visible=(i == 0 and j == 1),
        ))
    
    # Create buttons for switching between comparisons
    buttons = []
    for idx, (i, j, _) in enumerate(similarity_matrices):
        visibility = [False] * len(similarity_matrices)
        visibility[idx] = True
        buttons.append({
            'method': 'update',
            'label': f"{model_names[i]} vs {model_names[j]}",
            'args': [{'visible': visibility}]
        })
    
    fig.update_layout(
        title="Topic Similarity Across Models (Jaccard Similarity)",
        width=width,
        height=height,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'y': 0.8
        }]
    )
    
    return fig


def _create_embedding_visualization(all_topics, model_names, width, height):
    """Create visualization based on embedding similarity between topics."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        raise ImportError("This visualization requires sentence-transformers. Install with 'pip install sentence-transformers'")
    
    n_models = len(model_names)
    n_topics = len(all_topics[0])
    
    # Convert topics to strings and get embeddings
    topic_texts = []
    topic_labels = []
    
    for i, model_topics in enumerate(all_topics):
        for t, words in enumerate(model_topics):
            topic_text = " ".join(words)
            topic_texts.append(topic_text)
            topic_labels.append(f"{model_names[i]} Topic {t}")
    
    # Get embeddings
    embeddings = model.encode(topic_texts)
    
    # Calculate 2D projection
    try:
        from umap import UMAP
        umap_model = UMAP(n_components=2, random_state=42)
        projection = umap_model.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        projection = pca.fit_transform(embeddings)
    
    # Create dataframe for visualization
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'topic': topic_labels,
        'model': [label.split()[0] for label in topic_labels],
        'topic_num': [int(label.split()[-1]) for label in topic_labels]
    })
    
    # Create visualization
    fig = px.scatter(
        df, x='x', y='y', color='model', symbol='model',
        hover_data=['topic', 'topic_num'],
        title="Topic Similarity Across Models (Embedding Space)",
        width=width,
        height=height,
    )
    
    # Add lines connecting most similar topics between models
    sim_matrix = cosine_similarity(embeddings)
    
    # Group embeddings by model
    model_indices = {}
    for i, model_name in enumerate(model_names):
        model_indices[model_name] = list(range(i * n_topics, (i + 1) * n_topics))
    
    # Find most similar topics between models
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:  # only do pairs once and not within same model
                continue
                
            for idx1 in model_indices[model1]:
                # Find most similar topic in model2
                similarities = [sim_matrix[idx1, idx2] for idx2 in model_indices[model2]]
                most_similar_idx = model_indices[model2][np.argmax(similarities)]
                max_sim = np.max(similarities)
                
                # Only draw lines for highly similar topics
                if max_sim > 0.7:  # Threshold for similarity
                    fig.add_trace(go.Scatter(
                        x=[df.loc[idx1, 'x'], df.loc[most_similar_idx, 'x']],
                        y=[df.loc[idx1, 'y'], df.loc[most_similar_idx, 'y']],
                        mode='lines',
                        line=dict(width=max_sim * 3, color='rgba(100,100,100,0.2)'),
                        showlegend=False
                    ))
    
    return fig


def _create_correlation_visualization(doc_topics, model_names, width, height):
    """Create visualization based on correlation of document-topic distributions."""
    n_models = len(model_names)
    
    # Create correlation matrix between all topics
    all_correlations = []
    
    for i in range(n_models):
        n_topics_i = doc_topics[i].shape[1]
        for j in range(i, n_models):
            n_topics_j = doc_topics[j].shape[1]
            
            # Calculate correlation between topic distributions
            corr_matrix = np.zeros((n_topics_i, n_topics_j))
            
            for t1 in range(n_topics_i):
                topic_i_dist = doc_topics[i][:, t1]
                for t2 in range(n_topics_j):
                    topic_j_dist = doc_topics[j][:, t2]
                    # Pearson correlation
                    if np.std(topic_i_dist) > 0 and np.std(topic_j_dist) > 0:
                        corr_matrix[t1, t2] = np.corrcoef(topic_i_dist, topic_j_dist)[0, 1]
                    else:
                        corr_matrix[t1, t2] = 0
            
            all_correlations.append((i, j, corr_matrix))
    
    # Create visualization
    fig = go.Figure()
    
    for i, j, corr_matrix in all_correlations:
        model_i_name = model_names[i]
        model_j_name = model_names[j]
        
        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=[f"{model_j_name} T{t}" for t in range(corr_matrix.shape[1])],
            y=[f"{model_i_name} T{t}" for t in range(corr_matrix.shape[0])],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            name=f"{model_i_name} vs {model_j_name}",
            visible=(i == 0 and j == 0),
        ))
    
    # Create buttons for switching between comparisons
    buttons = []
    for idx, (i, j, _) in enumerate(all_correlations):
        visibility = [False] * len(all_correlations)
        visibility[idx] = True
        buttons.append({
            'method': 'update',
            'label': f"{model_names[i]} vs {model_names[j]}",
            'args': [{'visible': visibility}]
        })
    
    fig.update_layout(
        title="Topic Correlation Across Models (Document Distribution)",
        width=width,
        height=height,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'y': 0.8
        }]
    )
    
    return fig


def visualize_topic_differences(
    model1: Any,
    model2: Any,
    model1_name: str,
    model2_name: str,
    documents: List[str],
    metric: str = "topic_word_difference",
    n_topics: int = 10,
    n_words: int = 20,
    width: int = 1200,
    height: int = 1000,
):
    """
    Visualize differences between topics in two different models.

    Args:
        model1: First topic model (already fitted)
        model2: Second topic model (already fitted)
        model1_name: Name for the first model
        model2_name: Name for the second model
        documents: List of documents used for fitting
        metric: Metric to use for comparison
               - 'topic_word_difference': Compare topic words
               - 'document_assignment': Compare document assignments
               - 'topic_distribution': Compare topic prevalence
        n_topics: Number of top topics to compare
        n_words: Number of words per topic to visualize
        width: Width of the visualization in pixels, by default 1200
        height: Height of the visualization in pixels, by default 1000

    Returns:
        fig: Plotly figure object
    """
    if metric == "topic_word_difference":
        return _visualize_word_differences(model1, model2, model1_name, model2_name, n_topics, n_words, width, height)
    elif metric == "document_assignment":
        return _visualize_document_assignment_differences(model1, model2, model1_name, model2_name, documents, width, height)
    elif metric == "topic_distribution":
        return _visualize_topic_distribution_differences(model1, model2, model1_name, model2_name, documents, width, height)
    else:
        raise ValueError("Invalid metric. Choose from 'topic_word_difference', 'document_assignment', or 'topic_distribution'")


def _visualize_word_differences(model1, model2, model1_name, model2_name, n_topics, n_words, width, height):
    """Visualize differences in top words for each topic between models."""
    # Get top words for each model
    model1_topics = {}
    for topic_id in range(min(n_topics, len(model1.get_topic_info()))):
        words_weights = model1.get_topic(topic_id)[:n_words]
        model1_topics[topic_id] = {word: weight for word, weight in words_weights}
    
    model2_topics = {}
    for topic_id in range(min(n_topics, len(model2.get_topic_info()))):
        words_weights = model2.get_topic(topic_id)[:n_words]
        model2_topics[topic_id] = {word: weight for word, weight in words_weights}
    
    # Match topics between models
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Create word vectors for all topics
    all_words = set()
    for topics in [model1_topics, model2_topics]:
        for topic_id, word_weights in topics.items():
            all_words.update(word_weights.keys())
    
    all_words = sorted(list(all_words))
    word_to_idx = {word: i for i, word in enumerate(all_words)}
    
    # Create vectors for each topic
    model1_vectors = np.zeros((len(model1_topics), len(all_words)))
    for topic_id, word_weights in model1_topics.items():
        for word, weight in word_weights.items():
            model1_vectors[topic_id, word_to_idx[word]] = weight
    
    model2_vectors = np.zeros((len(model2_topics), len(all_words)))
    for topic_id, word_weights in model2_topics.items():
        for word, weight in word_weights.items():
            model2_vectors[topic_id, word_to_idx[word]] = weight
    
    # Calculate similarity and match topics
    sim_matrix = cosine_similarity(model1_vectors, model2_vectors)
    matched_topics = []
    
    for i in range(len(model1_topics)):
        j = np.argmax(sim_matrix[i])
        matched_topics.append((i, j, sim_matrix[i, j]))
    
    # Sort by similarity
    matched_topics.sort(key=lambda x: x[2], reverse=True)
    
    # Create visualization
    fig = go.Figure()
    
    for i, (topic1, topic2, sim) in enumerate(matched_topics[:n_topics]):
        # Get words and weights
        words1 = []
        weights1 = []
        words1_set = set()
        
        words2 = []
        weights2 = []
        words2_set = set()
        
        # Add words from model 1
        for word, weight in sorted(model1_topics[topic1].items(), key=lambda x: x[1], reverse=True):
            words1.append(word)
            weights1.append(weight)
            words1_set.add(word)
        
        # Add words from model 2
        for word, weight in sorted(model2_topics[topic2].items(), key=lambda x: x[1], reverse=True):
            words2.append(word)
            weights2.append(weight)
            words2_set.add(word)
        
        # Create a subplot for this topic pair
        is_visible = (i == 0)
        
        # Words in both models
        common_words = words1_set.intersection(words2_set)
        model1_unique = words1_set - words2_set
        model2_unique = words2_set - words1_set
        
        # Assign colors based on word presence
        colors1 = ['rgba(31, 119, 180, 0.8)' if word in common_words else 'rgba(255, 127, 14, 0.8)' 
                   for word in words1]
        colors2 = ['rgba(31, 119, 180, 0.8)' if word in common_words else 'rgba(44, 160, 44, 0.8)' 
                   for word in words2]
        
        # Model 1 words
        fig.add_trace(go.Bar(
            x=words1,
            y=weights1,
            marker_color=colors1,
            name=f"{model1_name} (Topic {topic1})",
            visible=is_visible
        ))
        
        # Model 2 words
        fig.add_trace(go.Bar(
            x=words2,
            y=weights2,
            marker_color=colors2,
            name=f"{model2_name} (Topic {topic2})",
            visible=is_visible
        ))
        
        # Legend for colors
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(31, 119, 180, 0.8)'),
            name='Common words',
            visible=is_visible
        ))
        
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(255, 127, 14, 0.8)'),
            name=f'Unique to {model1_name}',
            visible=is_visible
        ))
        
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(44, 160, 44, 0.8)'),
            name=f'Unique to {model2_name}',
            visible=is_visible
        ))
    
    # Create buttons for switching between topic pairs
    buttons = []
    for i in range(min(n_topics, len(matched_topics))):
        topic1, topic2, sim = matched_topics[i]
        
        visibility = []
        for j in range(len(matched_topics[:n_topics])):
            # Each topic pair has 5 traces (2 bar charts + 3 legend items)
            visible = [j == i for _ in range(5)]
            visibility.extend(visible)
        
        buttons.append({
            'method': 'update',
            'label': f"Topics {topic1} & {topic2} (Sim: {sim:.2f})",
            'args': [{'visible': visibility}]
        })
    
    # Update layout
    fig.update_layout(
        title=f"Topic Word Comparison: {model1_name} vs {model2_name}",
        xaxis_title="Words",
        yaxis_title="Weight",
        barmode='group',
        width=width,
        height=height,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.1,
            'y': 0.8
        }]
    )
    
    return fig


def _visualize_document_assignment_differences(model1, model2, model1_name, model2_name, documents, width, height):
    """Visualize differences in document topic assignments between models."""
    # Get document-topic distributions
    doc_topics1 = model1.transform(documents)
    doc_topics2 = model2.transform(documents)
    
    # Get primary topic for each document in each model
    primary_topics1 = np.argmax(doc_topics1, axis=1)
    primary_topics2 = np.argmax(doc_topics2, axis=1)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    
    n_topics1 = doc_topics1.shape[1]
    n_topics2 = doc_topics2.shape[1]
    
    cm = confusion_matrix(primary_topics1, primary_topics2, 
                        labels=range(max(n_topics1, n_topics2)))
    
    # Normalize by row
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Create heatmap visualization
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=cm_norm[:n_topics1, :n_topics2],
        x=[f"T{i}" for i in range(n_topics2)],
        y=[f"T{i}" for i in range(n_topics1)],
        colorscale="Viridis",
        colorbar=dict(title="Proportion"),
        hovertemplate="Model 1 Topic %{y}<br>Model 2 Topic %{x}<br>Proportion: %{z:.2f}<extra></extra>"
    ))
    
    # Compute how many documents are assigned to each topic pair
    total_docs = len(documents)
    doc_counts = go.Figure()
    
    doc_counts.add_trace(go.Heatmap(
        z=cm[:n_topics1, :n_topics2],
        x=[f"T{i}" for i in range(n_topics2)],
        y=[f"T{i}" for i in range(n_topics1)],
        colorscale="Viridis",
        colorbar=dict(title="Count"),
        hovertemplate="Model 1 Topic %{y}<br>Model 2 Topic %{x}<br>Count: %{z}<extra></extra>"
    ))
    
    doc_counts.update_layout(
        title=f"Document Assignment Confusion Matrix (Counts): {model1_name} vs {model2_name}",
        xaxis_title=f"{model2_name} Topics",
        yaxis_title=f"{model1_name} Topics",
        width=width,
        height=height,
    )
    
    # Update layout
    fig.update_layout(
        title=f"Document Assignment Confusion Matrix: {model1_name} vs {model2_name}",
        xaxis_title=f"{model2_name} Topics",
        yaxis_title=f"{model1_name} Topics",
        width=width,
        height=height,
    )
    
    # Create button to toggle between proportion and count
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'method': 'update',
                    'label': 'Proportions',
                    'args': [
                        {'z': [cm_norm[:n_topics1, :n_topics2]]},
                        {'title': f"Document Assignment Confusion Matrix (Proportions): {model1_name} vs {model2_name}",
                         'colorbar': {'title': 'Proportion'}}
                    ]
                },
                {
                    'method': 'update',
                    'label': 'Counts',
                    'args': [
                        {'z': [cm[:n_topics1, :n_topics2]]},
                        {'title': f"Document Assignment Confusion Matrix (Counts): {model1_name} vs {model2_name}",
                         'colorbar': {'title': 'Count'}}
                    ]
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 1.1,
            'y': 0.8
        }]
    )
    
    return fig


def _visualize_topic_distribution_differences(model1, model2, model1_name, model2_name, documents, width, height):
    """Visualize differences in overall topic distributions between models."""
    # Get document-topic distributions
    doc_topics1 = model1.transform(documents)
    doc_topics2 = model2.transform(documents)
    
    # Calculate overall topic prevalence
    topic_prevalence1 = doc_topics1.mean(axis=0)
    topic_prevalence2 = doc_topics2.mean(axis=0)
    
    # Sort topics by prevalence
    sorted_topics1 = np.argsort(-topic_prevalence1)
    sorted_topics2 = np.argsort(-topic_prevalence2)
    
    # Get top words for each topic
    get_top_words = lambda model, topic_id: ", ".join([word for word, _ in model.get_topic(topic_id)[:5]])
    
    # Create dataframe for model 1
    df1 = pd.DataFrame({
        'Model': model1_name,
        'Topic': [f"Topic {i}" for i in sorted_topics1],
        'Topic_ID': sorted_topics1,
        'Prevalence': topic_prevalence1[sorted_topics1],
        'Top Words': [get_top_words(model1, i) for i in sorted_topics1]
    })
    
    # Create dataframe for model 2
    df2 = pd.DataFrame({
        'Model': model2_name,
        'Topic': [f"Topic {i}" for i in sorted_topics2],
        'Topic_ID': sorted_topics2,
        'Prevalence': topic_prevalence2[sorted_topics2],
        'Top Words': [get_top_words(model2, i) for i in sorted_topics2]
    })
    
    # Combine dataframes
    df = pd.concat([df1, df2])
    
    # Create visualization
    fig = px.bar(
        df, 
        x='Topic', 
        y='Prevalence',
        color='Model',
        barmode='group',
        hover_data=['Top Words'],
        title=f"Topic Prevalence Comparison: {model1_name} vs {model2_name}",
        labels={'Prevalence': 'Topic Prevalence (mean document-topic weight)'},
        width=width,
        height=height,
    )
    
    # Update layout
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig
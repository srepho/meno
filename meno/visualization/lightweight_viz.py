"""Visualization module optimized for lightweight topic models.

This module provides specialized visualization functions that work efficiently with
the lightweight topic models, with a focus on performance and minimal dependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def plot_model_comparison(
    document_lists: List[List[str]],
    model_names: List[str],
    models: List[Any],
    sample_size: int = 1000,
    random_seed: int = 42,
    width: int = 1000,
    height: int = 800,
    title: str = "Topic Model Comparison"
) -> go.Figure:
    """Create a visualization comparing different topic models on the same dataset.
    
    Parameters
    ----------
    document_lists : List[List[str]]
        List of document lists for each model (can be the same list multiple times)
    model_names : List[str]
        Names of the models to compare
    models : List[Any]
        List of fitted model instances
    sample_size : int, optional
        Number of documents to sample for visualization, by default 1000
    random_seed : int, optional
        Random seed for sampling, by default 42
    width : int, optional
        Plot width, by default 1000
    height : int, optional
        Plot height, by default 800
    title : str, optional
        Plot title, by default "Topic Model Comparison"
    
    Returns
    -------
    go.Figure
        Plotly figure with model comparison
    
    Raises
    ------
    ValueError
        If lengths of inputs don't match or models aren't fitted
    """
    # Validate inputs
    if not (len(document_lists) == len(model_names) == len(models)):
        raise ValueError("document_lists, model_names, and models must have the same length")
    
    # Check if models are fitted
    for i, model in enumerate(models):
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"Model {model_names[i]} is not fitted")
    
    # Create subplots: one row per model
    fig = make_subplots(
        rows=len(models),
        cols=2,
        subplot_titles=[f"{name}: Distribution" for name in model_names] + 
                       [f"{name}: Topics" for name in model_names],
        specs=[[{"type": "xy"}, {"type": "xy"}] for _ in range(len(models))],
        vertical_spacing=0.1
    )
    
    # Process each model
    for i, (docs, name, model) in enumerate(zip(document_lists, model_names, models)):
        row = i + 1  # Plotly uses 1-based indexing
        
        # Sample documents if needed
        if len(docs) > sample_size:
            np.random.seed(random_seed)
            sampled_indices = np.random.choice(len(docs), sample_size, replace=False)
            sampled_docs = [docs[j] for j in sampled_indices]
        else:
            sampled_docs = docs
            
        # Get topic info
        topic_info = model.get_topic_info()
        
        # Left column: Topic distribution
        topic_counts = topic_info['Size' if 'Size' in topic_info.columns else 'Count']
        topic_names = [f"Topic {topic}" for topic in topic_info['Topic']]
        
        fig.add_trace(
            go.Bar(
                x=topic_names,
                y=topic_counts,
                name=f"{name} Distribution",
                marker_color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
            ),
            row=row, col=1
        )
        
        # Right column: Topic coherence / keywords
        # Get top words for each topic
        all_words = []
        all_weights = []
        all_topics = []
        
        for _, topic_row in topic_info.iterrows():
            topic_id = topic_row['Topic']
            
            # Get top words - check which method is available
            if hasattr(model, 'get_topic'):
                topic_words = model.get_topic(topic_id)
                words = [word for word, _ in topic_words[:5]]
                weights = [weight for _, weight in topic_words[:5]]
            else:
                words = topic_row.get('Words', [])[:5]
                weights = [0.1 * (5 - i) for i in range(len(words))]  # Dummy weights
            
            all_words.extend(words)
            all_weights.extend(weights)
            all_topics.extend([f"Topic {topic_id}"] * len(words))
        
        # Create DataFrame for visualization
        word_df = pd.DataFrame({
            'Word': all_words,
            'Weight': all_weights,
            'Topic': all_topics
        })
        
        fig.add_trace(
            go.Bar(
                x=word_df['Topic'],
                y=word_df['Weight'],
                text=word_df['Word'],
                name=f"{name} Keywords",
                marker_color=px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)]
            ),
            row=row, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Topic", row=row, col=1)
        fig.update_yaxes(title_text="Document Count", row=row, col=1)
        fig.update_xaxes(title_text="Topic", row=row, col=2)
        fig.update_yaxes(title_text="Word Weight", row=row, col=2)
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height * len(models) // 2,
        showlegend=False
    )
    
    return fig


def plot_topic_landscape(
    model: Any,
    documents: List[str],
    method: str = "pca",
    width: int = 1000,
    height: int = 800,
    title: str = "Topic Landscape",
    include_documents: bool = True,
    sample_size: int = 500
) -> go.Figure:
    """Create a visualization of the topic landscape, showing relationships between topics.
    
    Parameters
    ----------
    model : Any
        Fitted topic model
    documents : List[str]
        Documents used to fit the model
    method : str, optional
        Dimensionality reduction method ('pca' or 'direct'), by default "pca"
    width : int, optional
        Plot width, by default 1000
    height : int, optional
        Plot height, by default 800
    title : str, optional
        Plot title, by default "Topic Landscape"
    include_documents : bool, optional
        Whether to include document points, by default True
    sample_size : int, optional
        Number of documents to sample if include_documents=True, by default 500
    
    Returns
    -------
    go.Figure
        Plotly figure with topic landscape
    
    Raises
    ------
    ValueError
        If model is not fitted or method is invalid
    """
    # Check if model is fitted
    if not hasattr(model, 'is_fitted') or not model.is_fitted:
        raise ValueError("Model is not fitted")
    
    # Get topic info
    topic_info = model.get_topic_info()
    
    # Create topic vectors
    topic_words = {}
    
    for topic_id in topic_info['Topic'].unique():
        # Skip -1 (outlier) topic if present
        if topic_id < 0:
            continue
            
        # Get topic words - check which method is available
        if hasattr(model, 'get_topic'):
            words_with_weights = model.get_topic(topic_id)
            topic_words[topic_id] = {word: weight for word, weight in words_with_weights}
        elif 'Words' in topic_info.columns:
            # Get from topic_info DataFrame
            words = topic_info[topic_info['Topic'] == topic_id]['Words'].iloc[0]
            # Assume equal weights or decreasing weights
            weights = [1.0 - 0.02 * i for i in range(len(words))]
            topic_words[topic_id] = {word: weight for word, weight in zip(words, weights)}
    
    # Create document-topic matrix if documents are provided
    if include_documents and documents:
        # Sample documents if needed
        if len(documents) > sample_size:
            np.random.seed(42)
            sampled_indices = np.random.choice(len(documents), sample_size, replace=False)
            sampled_docs = [documents[j] for j in sampled_indices]
        else:
            sampled_docs = documents
            sampled_indices = list(range(len(documents)))
            
        # Get document topics
        doc_topic_matrix = model.transform(sampled_docs)
    
    # Create topic embedding matrix
    topic_ids = sorted([tid for tid in topic_words.keys() if tid >= 0])
    
    if method == "pca":
        # Create a feature matrix where each row is a topic
        # Use word weights as features
        all_words = set()
        for word_dict in topic_words.values():
            all_words.update(word_dict.keys())
        
        word_list = sorted(list(all_words))
        word_to_idx = {word: i for i, word in enumerate(word_list)}
        
        # Create feature matrix
        feature_matrix = np.zeros((len(topic_ids), len(word_list)))
        
        for i, topic_id in enumerate(topic_ids):
            for word, weight in topic_words[topic_id].items():
                feature_matrix[i, word_to_idx[word]] = weight
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        topic_embedding = pca.fit_transform(feature_matrix)
        
    elif method == "direct":
        # Just use first two dimensions of topic vectors directly
        # This is a fallback when we can't create a proper embedding
        topic_embedding = np.zeros((len(topic_ids), 2))
        
        for i, topic_id in enumerate(topic_ids):
            word_weights = list(topic_words[topic_id].values())
            # Use the first two weights, or pad with zeros
            if len(word_weights) >= 2:
                topic_embedding[i] = word_weights[:2]
            elif len(word_weights) == 1:
                topic_embedding[i] = [word_weights[0], 0]
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'pca' or 'direct'.")
    
    # Create visualization
    fig = go.Figure()
    
    # Add document points if requested
    if include_documents and documents:
        # Project documents into the same space as topics
        if method == "pca":
            # Need to transform documents into the word-feature space
            doc_feature_matrix = np.zeros((len(sampled_docs), len(word_list)))
            
            # For simplicity, just use the document-topic weights
            # and multiply by the topic feature matrix to get approximate positions
            doc_embedding = doc_topic_matrix @ feature_matrix
            
            # Project using the same PCA transformation
            doc_embedding = pca.transform(doc_embedding)
        else:
            # For direct method, use the first two dimensions of topic weights
            doc_embedding = doc_topic_matrix[:, :2]
            
            # If we have fewer than 2 topics, pad with zeros
            if doc_embedding.shape[1] < 2:
                padding = np.zeros((doc_embedding.shape[0], 2 - doc_embedding.shape[1]))
                doc_embedding = np.hstack([doc_embedding, padding])
        
        # Get dominant topic for each document
        dominant_topics = np.argmax(doc_topic_matrix, axis=1)
        
        # Create scatter plot of documents
        for topic_id in topic_ids:
            mask = dominant_topics == topic_id
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=doc_embedding[mask, 0],
                        y=doc_embedding[mask, 1],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.5,
                        ),
                        name=f"Docs: Topic {topic_id}",
                        hovertext=[sampled_docs[i][:100] + "..." if len(sampled_docs[i]) > 100 else sampled_docs[i] 
                                  for i in range(len(sampled_docs)) if mask[i]],
                        hoverinfo='text'
                    )
                )
    
    # Add topic points
    for i, topic_id in enumerate(topic_ids):
        # Get top words for hover info
        top_words = ", ".join(list(topic_words[topic_id].keys())[:5])
        
        # Get topic size
        size = topic_info[topic_info['Topic'] == topic_id]['Size' if 'Size' in topic_info.columns else 'Count'].iloc[0]
        
        # Scale size between 15 and 50 based on document count
        scaled_size = 15 + min(35, size / 5)
        
        fig.add_trace(
            go.Scatter(
                x=[topic_embedding[i, 0]],
                y=[topic_embedding[i, 1]],
                mode='markers+text',
                marker=dict(
                    size=scaled_size,
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                text=[f"T{topic_id}"],
                textposition="middle center",
                name=f"Topic {topic_id}",
                hovertext=f"Topic {topic_id}<br>Size: {size}<br>Words: {top_words}",
                hoverinfo='text'
            )
        )
    
    # Add connections between topics based on similarity
    if len(topic_ids) > 1:
        # Calculate similarity between topics
        sim_matrix = np.zeros((len(topic_ids), len(topic_ids)))
        
        for i, tid1 in enumerate(topic_ids):
            for j, tid2 in enumerate(topic_ids):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    # Calculate Jaccard similarity between word sets
                    words1 = set(topic_words[tid1].keys())
                    words2 = set(topic_words[tid2].keys())
                    
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    
                    sim_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Add edges for topics with similarity above threshold
        threshold = 0.1
        
        for i, tid1 in enumerate(topic_ids):
            for j, tid2 in enumerate(topic_ids):
                if i < j and sim_matrix[i, j] > threshold:
                    # Add a line between topics
                    opacity = min(1.0, sim_matrix[i, j] * 2)  # Scale opacity with similarity
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[topic_embedding[i, 0], topic_embedding[j, 0]],
                            y=[topic_embedding[i, 1], topic_embedding[j, 1]],
                            mode='lines',
                            line=dict(width=1, color=f'rgba(100,100,100,{opacity})'),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        hovermode='closest',
        xaxis=dict(title='Dimension 1', showgrid=False, zeroline=False),
        yaxis=dict(title='Dimension 2', showgrid=False, zeroline=False)
    )
    
    return fig


def plot_multi_topic_heatmap(
    models: List[Any],
    model_names: List[str],
    document_lists: List[List[str]],
    show_alignment: bool = True,
    width: int = 900,
    height: int = 700,
    title: str = "Multi-Model Topic Comparison"
) -> go.Figure:
    """Create a heatmap comparing topics across multiple models.
    
    Parameters
    ----------
    models : List[Any]
        List of fitted model instances
    model_names : List[str]
        Names of the models to compare
    document_lists : List[List[str]]
        List of document lists for each model (can be the same list multiple times)
    show_alignment : bool, optional
        Whether to try to align similar topics, by default True
    width : int, optional
        Plot width, by default 900
    height : int, optional
        Plot height, by default 700
    title : str, optional
        Plot title, by default "Multi-Model Topic Comparison"
    
    Returns
    -------
    go.Figure
        Plotly figure with topic comparison heatmap
    
    Raises
    ------
    ValueError
        If lengths of inputs don't match or models aren't fitted
    """
    # Validate inputs
    if not (len(document_lists) == len(model_names) == len(models)):
        raise ValueError("document_lists, model_names, and models must have the same length")
    
    # Check if models are fitted
    for i, model in enumerate(models):
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            raise ValueError(f"Model {model_names[i]} is not fitted")
    
    # Get topic information from all models
    all_topic_words = []
    
    for model, docs in zip(models, document_lists):
        topic_info = model.get_topic_info()
        
        model_topic_words = {}
        
        for topic_id in topic_info['Topic'].unique():
            # Skip -1 (outlier) topic if present
            if topic_id < 0:
                continue
                
            # Get topic words - check which method is available
            if hasattr(model, 'get_topic'):
                words_with_weights = model.get_topic(topic_id)
                model_topic_words[topic_id] = [word for word, _ in words_with_weights[:10]]
            elif 'Words' in topic_info.columns:
                # Get from topic_info DataFrame
                words = topic_info[topic_info['Topic'] == topic_id]['Words'].iloc[0]
                model_topic_words[topic_id] = words[:10]  # Use top 10 words
        
        all_topic_words.append(model_topic_words)
    
    # Calculate similarity matrix between all topics
    all_topics_flat = []
    model_indices = []
    topic_indices = []
    
    for i, topic_dict in enumerate(all_topic_words):
        for topic_id, words in topic_dict.items():
            all_topics_flat.append(words)
            model_indices.append(i)
            topic_indices.append(topic_id)
    
    # Calculate Jaccard similarity between all topic pairs
    n_topics = len(all_topics_flat)
    similarity_matrix = np.zeros((n_topics, n_topics))
    
    for i in range(n_topics):
        for j in range(n_topics):
            words1 = set(all_topics_flat[i])
            words2 = set(all_topics_flat[j])
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Create model-to-model similarity matrices
    model_pair_matrices = {}
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            # Get topics from both models
            model_i_indices = [k for k in range(n_topics) if model_indices[k] == i]
            model_j_indices = [k for k in range(n_topics) if model_indices[k] == j]
            
            # Create similarity matrix for this model pair
            pair_matrix = np.zeros((len(model_i_indices), len(model_j_indices)))
            
            for idx_i, i_flat_idx in enumerate(model_i_indices):
                for idx_j, j_flat_idx in enumerate(model_j_indices):
                    pair_matrix[idx_i, idx_j] = similarity_matrix[i_flat_idx, j_flat_idx]
            
            model_pair_matrices[(i, j)] = pair_matrix
    
    # Create visualization
    if len(models) == 2:
        # For two models, create a direct heatmap
        i, j = 0, 1
        sim_matrix = model_pair_matrices[(i, j)]
        
        # Get topic IDs for both models
        model_i_topics = [topic_indices[k] for k in range(n_topics) if model_indices[k] == i]
        model_j_topics = [topic_indices[k] for k in range(n_topics) if model_indices[k] == j]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=[f"Topic {t}" for t in model_j_topics],
            y=[f"Topic {t}" for t in model_i_topics],
            colorscale='Viridis',
            hoverongaps=False,
            text=[[f"Similarity: {sim_matrix[i][j]:.2f}" for j in range(sim_matrix.shape[1])] 
                  for i in range(sim_matrix.shape[0])],
            hoverinfo="text+x+y"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Topic Similarity: {model_names[i]} vs {model_names[j]}",
            xaxis_title=model_names[j],
            yaxis_title=model_names[i],
            width=width,
            height=height
        )
        
    else:
        # For multiple models, create a subplot with heatmaps
        n_models = len(models)
        subplot_titles = [f"{model_names[i]} vs {model_names[j]}" 
                          for i in range(n_models) for j in range(i+1, n_models)]
        
        # Calculate subplot dimensions
        n_comparisons = n_models * (n_models - 1) // 2
        n_cols = min(3, n_comparisons)
        n_rows = (n_comparisons + n_cols - 1) // n_cols  # Ceiling division
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Add heatmaps
        subplot_idx = 0
        for i in range(n_models):
            for j in range(i+1, n_models):
                sim_matrix = model_pair_matrices[(i, j)]
                
                # Get topic IDs for both models
                model_i_topics = [topic_indices[k] for k in range(n_topics) if model_indices[k] == i]
                model_j_topics = [topic_indices[k] for k in range(n_topics) if model_indices[k] == j]
                
                # Calculate subplot position
                row = subplot_idx // n_cols + 1
                col = subplot_idx % n_cols + 1
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=sim_matrix,
                        x=[f"T{t}" for t in model_j_topics],
                        y=[f"T{t}" for t in model_i_topics],
                        colorscale='Viridis',
                        hoverongaps=False,
                        text=[[f"Similarity: {sim_matrix[i][j]:.2f}" for j in range(sim_matrix.shape[1])] 
                              for i in range(sim_matrix.shape[0])],
                        hoverinfo="text+x+y",
                        showscale=(subplot_idx == 0)  # Only show colorbar for first heatmap
                    ),
                    row=row, col=col
                )
                
                # Update axes
                fig.update_xaxes(title_text=f"{model_names[j]}", row=row, col=col)
                fig.update_yaxes(title_text=f"{model_names[i]}", row=row, col=col)
                
                subplot_idx += 1
        
        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=height * n_rows // 2
        )
    
    return fig


def plot_comparative_document_analysis(
    model: Any,
    documents: List[str],
    document_labels: Optional[List[str]] = None,
    highlight_docs: Optional[List[int]] = None,
    width: int = 1000,
    height: int = 800,
    title: str = "Document-Topic Analysis"
) -> go.Figure:
    """Create a visualization for analyzing document-topic relationships.
    
    Parameters
    ----------
    model : Any
        Fitted topic model
    documents : List[str]
        Documents to analyze
    document_labels : Optional[List[str]], optional
        Labels for documents, by default None (uses indices)
    highlight_docs : Optional[List[int]], optional
        Indices of documents to highlight, by default None
    width : int, optional
        Plot width, by default 1000
    height : int, optional
        Plot height, by default 800
    title : str, optional
        Plot title, by default "Document-Topic Analysis"
    
    Returns
    -------
    go.Figure
        Plotly figure with document analysis
    
    Raises
    ------
    ValueError
        If model is not fitted or inputs don't match
    """
    # Check if model is fitted
    if not hasattr(model, 'is_fitted') or not model.is_fitted:
        raise ValueError("Model is not fitted")
    
    # Use document indices if no labels provided
    if document_labels is None:
        document_labels = [f"Doc {i}" for i in range(len(documents))]
    
    if len(documents) != len(document_labels):
        raise ValueError("documents and document_labels must have the same length")
    
    # Get document-topic distribution
    doc_topic_matrix = model.transform(documents)
    
    # Get topic info
    topic_info = model.get_topic_info()
    
    # Create topic labels
    topic_labels = [f"Topic {topic_id}" for topic_id in range(doc_topic_matrix.shape[1])]
    
    # Create document topic heatmap
    fig = go.Figure(data=go.Heatmap(
        z=doc_topic_matrix,
        x=topic_labels,
        y=document_labels,
        colorscale='Viridis',
        hoverongaps=False,
        text=[[f"{doc_topic_matrix[i][j]:.2f}" for j in range(doc_topic_matrix.shape[1])] 
              for i in range(doc_topic_matrix.shape[0])],
        hoverinfo="text+x+y"
    ))
    
    # Highlight specific documents if requested
    if highlight_docs:
        # Add rectangles around highlighted documents
        for doc_idx in highlight_docs:
            if 0 <= doc_idx < len(documents):
                fig.add_shape(
                    type="rect",
                    x0=-0.5,
                    x1=doc_topic_matrix.shape[1] - 0.5,
                    y0=doc_idx - 0.5,
                    y1=doc_idx + 0.5,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(0,0,0,0)"
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title="Topics",
        yaxis_title="Documents"
    )
    
    return fig
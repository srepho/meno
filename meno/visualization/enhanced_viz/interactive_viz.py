"""
Interactive visualization module for topic models.

This module provides functions to create interactive visualizations and dashboards
for exploring topic models and their results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import plotly.subplots as sp
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc


def create_interactive_topic_explorer(
    model: Any,
    documents: List[str],
    doc_metadata: Optional[pd.DataFrame] = None,
    embedding_method: str = "umap",
    color_by: str = "topic",
    return_app: bool = False,
):
    """
    Create an interactive topic explorer dashboard with Dash.

    Args:
        model: Fitted topic model
        documents: List of documents used for fitting
        doc_metadata: Optional metadata for each document (must have same length as documents)
        embedding_method: Method for dimensionality reduction ('umap', 'tsne', or 'pca')
        color_by: Attribute to color points by ('topic' or a column name in doc_metadata)
        return_app: If True, returns the Dash app object instead of running it

    Returns:
        If return_app is True, returns the Dash app object; otherwise runs the app on a local server
    """
    try:
        import dash
        from dash import html, dcc
        import dash_bootstrap_components as dbc
    except ImportError:
        raise ImportError("This visualization requires dash and dash_bootstrap_components. Install with 'pip install dash dash-bootstrap-components'")

    # Get document-topic distributions
    doc_topics = model.transform(documents)
    
    # Get 2D embeddings for visualization
    embeddings = model.get_document_embeddings()
    
    if embedding_method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
    elif embedding_method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif embedding_method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid embedding method. Choose 'umap', 'tsne', or 'pca'")
    
    # Fit and transform embeddings
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Get topic assignments for each document
    primary_topics = np.argmax(doc_topics, axis=1)
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'document': documents,
        'topic': [f"Topic {t}" for t in primary_topics],
        'topic_id': primary_topics,
        'topic_score': [doc_topics[i, primary_topics[i]] for i in range(len(documents))]
    })
    
    # Add metadata if provided
    if doc_metadata is not None:
        if len(doc_metadata) != len(documents):
            raise ValueError("doc_metadata must have the same length as documents")
        
        for col in doc_metadata.columns:
            df[col] = doc_metadata[col].values
    
    # Get topics info
    topic_info = model.get_topic_info()
    
    # Create Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Interactive Topic Explorer", className="text-center my-4"),
                html.P("Explore document-topic relationships and search for related documents.", 
                       className="text-center mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            # Topic selector and embeddings plot
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Document Embeddings"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Color by:"),
                                dcc.Dropdown(
                                    id='color-dropdown',
                                    options=[{'label': 'Topic', 'value': 'topic'}] + 
                                            ([{'label': col, 'value': col} for col in doc_metadata.columns] 
                                             if doc_metadata is not None else []),
                                    value=color_by,
                                    clearable=False
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Filter by Topic:"),
                                dcc.Dropdown(
                                    id='topic-filter-dropdown',
                                    options=[{'label': 'All Topics', 'value': 'all'}] + 
                                            [{'label': f"Topic {row['Topic']}", 'value': row['Topic']} 
                                             for _, row in topic_info.iterrows() if row['Topic'] != -1],
                                    value='all',
                                    clearable=False
                                )
                            ], width=6)
                        ]),
                        dcc.Graph(id='embedding-plot')
                    ])
                ], className="mb-4"),
                
                # Document viewer
                dbc.Card([
                    dbc.CardHeader("Document Viewer"),
                    dbc.CardBody([
                        html.Pre(id='document-content', style={'whiteSpace': 'pre-wrap', 'height': '300px', 'overflowY': 'scroll'})
                    ])
                ])
            ], width=8),
            
            # Topic info and search
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Topic Information"),
                    dbc.CardBody([
                        html.Label("Select Topic:"),
                        dcc.Dropdown(
                            id='topic-dropdown',
                            options=[{'label': f"Topic {row['Topic']}", 'value': row['Topic']} 
                                     for _, row in topic_info.iterrows() if row['Topic'] != -1],
                            value=0,
                            clearable=False
                        ),
                        html.Div(id='topic-words', className="mt-3"),
                        dcc.Graph(id='topic-barplot')
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Search Documents"),
                    dbc.CardBody([
                        dbc.InputGroup([
                            dbc.Input(id='search-input', placeholder='Enter search terms...', type='text'),
                            dbc.Button('Search', id='search-button', color='primary')
                        ]),
                        html.Div(id='search-results', className="mt-3", style={'height': '300px', 'overflowY': 'scroll'})
                    ])
                ])
            ], width=4)
        ])
    ], fluid=True)
    
    # Callbacks
    @app.callback(
        Output('embedding-plot', 'figure'),
        [Input('color-dropdown', 'value'),
         Input('topic-filter-dropdown', 'value')]
    )
    def update_embedding_plot(color_by, topic_filter):
        if topic_filter == 'all':
            filtered_df = df
        else:
            filtered_df = df[df['topic_id'] == int(topic_filter)]
        
        if color_by == 'topic':
            fig = px.scatter(
                filtered_df, x='x', y='y', color='topic',
                hover_data=['document', 'topic_score'],
                title="Document Embedding Space",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
        else:
            fig = px.scatter(
                filtered_df, x='x', y='y', color=color_by,
                hover_data=['document', 'topic', 'topic_score'],
                title="Document Embedding Space"
            )
        
        fig.update_layout(
            clickmode='event+select',
            legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
            margin=dict(l=20, r=20, t=40, b=30)
        )
        
        return fig
    
    @app.callback(
        Output('document-content', 'children'),
        [Input('embedding-plot', 'clickData')]
    )
    def display_document(clickData):
        if clickData is None:
            return "Click on a point to view document content."
        
        point_index = clickData['points'][0]['pointIndex']
        
        # Get metadata if available
        metadata_str = ""
        if doc_metadata is not None:
            metadata_items = []
            for col in doc_metadata.columns:
                metadata_items.append(f"{col}: {df.iloc[point_index][col]}")
            metadata_str = "\n".join(metadata_items)
        
        return (f"Topic: {df.iloc[point_index]['topic']} (score: {df.iloc[point_index]['topic_score']:.2f})\n" +
                (f"{metadata_str}\n\n" if metadata_str else "\n") +
                f"Document:\n{df.iloc[point_index]['document']}")
    
    @app.callback(
        [Output('topic-words', 'children'),
         Output('topic-barplot', 'figure')],
        [Input('topic-dropdown', 'value')]
    )
    def update_topic_info(topic_id):
        # Get top words for the selected topic
        topic_words = model.get_topic(topic_id)
        
        # Create word cloud HTML
        words_html = html.Div([
            html.H5(f"Top Words for Topic {topic_id}"),
            html.Div([
                html.Span(word, style={
                    'fontSize': f"{max(1.0, 1.0 + weight * 2)}em",
                    'margin': '5px',
                    'padding': '3px',
                    'display': 'inline-block'
                }) for word, weight in topic_words[:20]
            ])
        ])
        
        # Create bar plot of top words
        words = [word for word, _ in topic_words[:10]]
        weights = [weight for _, weight in topic_words[:10]]
        
        fig = px.bar(
            x=words, y=weights,
            labels={'x': 'Word', 'y': 'Weight'},
            title=f"Top 10 Words in Topic {topic_id}"
        )
        
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
        
        return words_html, fig
    
    @app.callback(
        Output('search-results', 'children'),
        [Input('search-button', 'click')],
        [State('search-input', 'value')]
    )
    def search_documents(n_clicks, search_terms):
        if n_clicks is None or not search_terms:
            return "Enter search terms to find related documents."
        
        # Use model to find similar documents
        query_embeddings = model.embedding_model.embed_documents([search_terms])
        doc_embeddings = model.get_document_embeddings()
        
        similarities = cosine_similarity(query_embeddings, doc_embeddings)[0]
        
        # Get top 10 most similar documents
        top_indices = np.argsort(-similarities)[:10]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append(html.Div([
                html.H6(f"Result {i+1} - Topic {df.iloc[idx]['topic']} (Similarity: {similarities[idx]:.2f})"),
                html.P(df.iloc[idx]['document'][:200] + "..." if len(df.iloc[idx]['document']) > 200 else df.iloc[idx]['document']),
                html.Hr()
            ]))
        
        return results
    
    if return_app:
        return app
    else:
        app.run_server(debug=True)


def create_topic_dashboard(
    model: Any,
    documents: List[str],
    time_data: Optional[pd.DataFrame] = None,
    geo_data: Optional[pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
    num_topics: int = 10,
    return_app: bool = False,
):
    """
    Create an advanced topic model dashboard with time series, geospatial, and metadata visualizations.

    Args:
        model: Fitted topic model
        documents: List of documents used for fitting
        time_data: Optional DataFrame with document index and time/date information
        geo_data: Optional DataFrame with document index and geospatial information (lat/lon)
        metadata: Optional DataFrame with additional document metadata
        num_topics: Number of top topics to display
        return_app: If True, returns the Dash app object instead of running it

    Returns:
        If return_app is True, returns the Dash app object; otherwise runs the app
    """
    try:
        import dash
        from dash import html, dcc
        import dash_bootstrap_components as dbc
    except ImportError:
        raise ImportError("This visualization requires dash and dash_bootstrap_components. Install with 'pip install dash dash-bootstrap-components'")
    
    # Get document-topic distributions
    doc_topics = model.transform(documents)
    
    # Get topic information
    topic_info = model.get_topic_info().sort_values('Count', ascending=False)
    top_topics = topic_info['Topic'].values[:num_topics]
    
    # Create DataFrame with document-topic information
    df = pd.DataFrame({
        'document': documents,
        'primary_topic': np.argmax(doc_topics, axis=1),
        'primary_topic_score': np.max(doc_topics, axis=1)
    })
    
    # Add topic probability columns
    for topic_id in top_topics:
        if topic_id >= 0:  # Exclude -1 if present
            df[f'topic_{topic_id}'] = doc_topics[:, topic_id]
    
    # Add time data if provided
    has_time = False
    if time_data is not None:
        if len(time_data) != len(documents):
            raise ValueError("time_data must have the same length as documents")
        
        for col in time_data.columns:
            df[col] = time_data[col].values
        
        # Identify time column
        time_col = next((col for col in time_data.columns if pd.api.types.is_datetime64_any_dtype(time_data[col])), None)
        if time_col:
            has_time = True
    
    # Add geo data if provided
    has_geo = False
    if geo_data is not None:
        if len(geo_data) != len(documents):
            raise ValueError("geo_data must have the same length as documents")
        
        # Check for latitude and longitude columns
        lat_col = next((col for col in geo_data.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in geo_data.columns if 'lon' in col.lower() or 'lng' in col.lower()), None)
        
        if lat_col and lon_col:
            df['latitude'] = geo_data[lat_col].values
            df['longitude'] = geo_data[lon_col].values
            has_geo = True
    
    # Add metadata if provided
    if metadata is not None:
        if len(metadata) != len(documents):
            raise ValueError("metadata must have the same length as documents")
        
        for col in metadata.columns:
            df[col] = metadata[col].values
    
    # Create Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Topic Model Dashboard", className="text-center my-4"),
                html.P("Explore topics, distributions, temporal trends, and spatial patterns.", 
                       className="text-center mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            # Topic selection sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Topics"),
                    dbc.CardBody([
                        html.Label("Select Topic:"),
                        dcc.Dropdown(
                            id='topic-selector',
                            options=[{'label': f"Topic {row['Topic']} ({row['Count']} docs)", 'value': row['Topic']} 
                                     for _, row in topic_info.iterrows() if row['Topic'] != -1],
                            value=top_topics[0] if len(top_topics) > 0 else 0,
                            clearable=False
                        ),
                        html.Div(id='selected-topic-info', className="mt-3"),
                        html.Hr(),
                        html.Label("Topics Overview:"),
                        dcc.Graph(id='topic-distribution-plot')
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Topic Relationships"),
                    dbc.CardBody([
                        dcc.Graph(id='topic-similarity-plot'),
                        html.Div(id='topic-similarity-info', className="mt-2")
                    ])
                ])
            ], width=3),
            
            # Main content area
            dbc.Col([
                dbc.Tabs([
                    # Overview tab
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Topic Word Cloud"),
                                    dbc.CardBody([
                                        dcc.Graph(id='topic-wordcloud')
                                    ])
                                ])
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Document Preview"),
                                    dbc.CardBody([
                                        html.Div(id='topic-documents', style={'height': '300px', 'overflowY': 'scroll'})
                                    ])
                                ])
                            ], width=6)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Similar Topics"),
                                    dbc.CardBody([
                                        dcc.Graph(id='similar-topics-plot')
                                    ])
                                ])
                            ], width=12)
                        ])
                    ], label="Overview"),
                    
                    # Time series tab (only if time data is available)
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Topic Trends Over Time"),
                                    dbc.CardBody([
                                        html.Label("Time Aggregation:"),
                                        dcc.Dropdown(
                                            id='time-aggregation',
                                            options=[
                                                {'label': 'Day', 'value': 'D'},
                                                {'label': 'Week', 'value': 'W'},
                                                {'label': 'Month', 'value': 'M'},
                                                {'label': 'Quarter', 'value': 'Q'},
                                                {'label': 'Year', 'value': 'Y'}
                                            ],
                                            value='M',
                                            clearable=False
                                        ),
                                        dcc.Graph(id='topic-time-trend')
                                    ])
                                ])
                            ], width=12)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Topic Composition Over Time"),
                                    dbc.CardBody([
                                        dcc.Graph(id='topic-time-composition')
                                    ])
                                ])
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Topic Heatmap Over Time"),
                                    dbc.CardBody([
                                        dcc.Graph(id='topic-time-heatmap')
                                    ])
                                ])
                            ], width=6)
                        ])
                    ], label="Time Series", disabled=not has_time),
                    
                    # Geospatial tab (only if geo data is available)
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Topic Geographic Distribution"),
                                    dbc.CardBody([
                                        dcc.Graph(id='topic-geo-map', style={'height': '500px'})
                                    ])
                                ])
                            ], width=12)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Topic Geographic Density"),
                                    dbc.CardBody([
                                        dcc.Graph(id='topic-geo-density')
                                    ])
                                ])
                            ], width=12)
                        ])
                    ], label="Geospatial", disabled=not has_geo),
                    
                    # Document explorer tab
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Document Explorer"),
                                    dbc.CardBody([
                                        html.Label("Search Documents:"),
                                        dbc.InputGroup([
                                            dbc.Input(id='document-search', placeholder='Enter search terms...', type='text'),
                                            dbc.Button('Search', id='search-button', color='primary')
                                        ], className="mb-3"),
                                        html.Label("Filter by Topic:"),
                                        dcc.Dropdown(
                                            id='document-topic-filter',
                                            options=[{'label': 'All Topics', 'value': 'all'}] + 
                                                    [{'label': f"Topic {t}", 'value': t} for t in top_topics if t >= 0],
                                            value='all',
                                            clearable=False,
                                            className="mb-3"
                                        ),
                                        html.Div(id='document-list', style={'height': '500px', 'overflowY': 'scroll'})
                                    ])
                                ])
                            ], width=12)
                        ])
                    ], label="Documents")
                ])
            ], width=9)
        ])
    ], fluid=True)
    
    # Define callbacks
    @app.callback(
        [Output('selected-topic-info', 'children'),
         Output('topic-wordcloud', 'figure'),
         Output('topic-documents', 'children')],
        [Input('topic-selector', 'value')]
    )
    def update_topic_info(topic_id):
        # Get topic information
        topic_words = model.get_topic(topic_id)
        
        # Topic info
        info_html = html.Div([
            html.H5(f"Topic {topic_id}"),
            html.P(f"Documents: {topic_info[topic_info['Topic'] == topic_id]['Count'].values[0]}"),
            html.P("Top Words: " + ", ".join([word for word, _ in topic_words[:5]]))
        ])
        
        # Word cloud visualization using bar chart
        words = [word for word, _ in topic_words[:15]]
        weights = [weight for _, weight in topic_words[:15]]
        
        wordcloud_fig = px.bar(
            x=words, y=weights,
            labels={'x': 'Word', 'y': 'Weight'},
            title=f"Top Words in Topic {topic_id}"
        )
        
        wordcloud_fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
        
        # Get sample documents for this topic
        topic_docs = df[df['primary_topic'] == topic_id].sort_values('primary_topic_score', ascending=False).head(5)
        
        doc_html = []
        for i, (_, row) in enumerate(topic_docs.iterrows()):
            doc_html.append(html.Div([
                html.H6(f"Document {i+1} (Score: {row['primary_topic_score']:.2f})"),
                html.P(row['document'][:200] + "..." if len(row['document']) > 200 else row['document']),
                html.Hr()
            ]))
        
        return info_html, wordcloud_fig, doc_html
    
    @app.callback(
        Output('topic-distribution-plot', 'figure'),
        [Input('topic-selector', 'value')]
    )
    def update_topic_distribution(selected_topic):
        # Create dataframe with topic counts
        topic_counts = topic_info[topic_info['Topic'] >= 0].sort_values('Count', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            topic_counts, 
            x='Topic', 
            y='Count', 
            title="Top 10 Topics by Document Count",
            color=['Selected' if t == selected_topic else 'Other' for t in topic_counts['Topic']],
            color_discrete_map={'Selected': '#FAA', 'Other': '#AAF'}
        )
        
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
        
        return fig
    
    @app.callback(
        Output('topic-similarity-plot', 'figure'),
        [Input('topic-selector', 'value')]
    )
    def update_topic_similarity(selected_topic):
        # Get topic embeddings
        topic_vectors = []
        topic_ids = []
        
        for topic_id in top_topics:
            if topic_id >= 0:
                # Get topic embedding by averaging word embeddings with weights
                topic_words = model.get_topic(topic_id)
                topic_vector = None
                
                # Attempt to get topic embedding
                if hasattr(model, 'get_topic_embedding'):
                    try:
                        topic_vector = model.get_topic_embedding(topic_id)
                    except (AttributeError, NotImplementedError):
                        pass
                
                if topic_vector is None:
                    # Fallback: use a vector of word weights
                    words = [word for word, _ in topic_words[:20]]
                    word_vectors = np.zeros(len(words))
                    for i, (_, weight) in enumerate(topic_words[:20]):
                        word_vectors[i] = weight
                    
                    topic_vector = word_vectors
                
                topic_vectors.append(topic_vector)
                topic_ids.append(topic_id)
        
        # Calculate similarity between selected topic and all others
        if not topic_vectors:
            # Empty plot if no topic vectors
            fig = px.bar(
                x=[], y=[],
                title="No topic similarity data available"
            )
            return fig
        
        # Convert to numpy array
        topic_vectors = np.vstack(topic_vectors)
        
        # Calculate cosine similarity
        selected_idx = topic_ids.index(selected_topic) if selected_topic in topic_ids else 0
        similarities = cosine_similarity([topic_vectors[selected_idx]], topic_vectors)[0]
        
        # Create dataframe for visualization
        sim_df = pd.DataFrame({
            'Topic': [f"Topic {topic_id}" for topic_id in topic_ids],
            'Similarity': similarities
        })
        
        # Sort by similarity (excluding the selected topic)
        sim_df = sim_df.sort_values('Similarity', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            sim_df, 
            x='Topic', 
            y='Similarity', 
            title=f"Topics Similar to Topic {selected_topic}",
            color=['Selected' if t == f"Topic {selected_topic}" else 'Other' for t in sim_df['Topic']],
            color_discrete_map={'Selected': '#FAA', 'Other': '#AAF'}
        )
        
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
        
        return fig
    
    @app.callback(
        Output('similar-topics-plot', 'figure'),
        [Input('topic-selector', 'value')]
    )
    def update_similar_topics_plot(selected_topic):
        # Create detailed visualization of the selected topic and its most similar topics
        
        # Get top words for selected topic
        selected_words = model.get_topic(selected_topic)
        
        # Get most similar topics (from similarity calculation)
        topic_vectors = []
        topic_ids = []
        
        for topic_id in top_topics:
            if topic_id >= 0:
                # Get topic embedding
                topic_words = model.get_topic(topic_id)
                words = [word for word, _ in topic_words[:20]]
                word_vectors = np.zeros(len(words))
                for i, (_, weight) in enumerate(topic_words[:20]):
                    word_vectors[i] = weight
                
                topic_vectors.append(word_vectors)
                topic_ids.append(topic_id)
        
        # Calculate similarity between selected topic and all others
        if not topic_vectors or selected_topic not in topic_ids:
            # Empty plot if no topic vectors or selected topic not found
            fig = px.bar(
                x=[], y=[],
                title="No similar topics data available"
            )
            return fig
        
        # Convert to numpy array and calculate similarity
        topic_vectors = np.vstack(topic_vectors)
        selected_idx = topic_ids.index(selected_topic)
        similarities = cosine_similarity([topic_vectors[selected_idx]], topic_vectors)[0]
        
        # Get indices of top 3 most similar topics (excluding self)
        similarities[selected_idx] = 0  # Exclude self
        top_similar_indices = np.argsort(-similarities)[:3]
        
        # Create subplots for visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"Topic {selected_topic} (Selected)"] + 
                           [f"Topic {topic_ids[idx]} (Sim: {similarities[idx]:.2f})" for idx in top_similar_indices]
        )
        
        # Add bar for selected topic
        words = [word for word, _ in selected_words[:10]]
        weights = [weight for _, weight in selected_words[:10]]
        
        fig.add_trace(
            go.Bar(x=words, y=weights, marker_color='#FAA', name=f"Topic {selected_topic}"),
            row=1, col=1
        )
        
        # Add bars for similar topics
        for i, idx in enumerate(top_similar_indices):
            topic_id = topic_ids[idx]
            topic_words = model.get_topic(topic_id)
            
            words = [word for word, _ in topic_words[:10]]
            weights = [weight for _, weight in topic_words[:10]]
            
            row, col = (1, 2) if i == 0 else (2, 1) if i == 1 else (2, 2)
            
            fig.add_trace(
                go.Bar(x=words, y=weights, marker_color='#AAF', name=f"Topic {topic_id}"),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Selected Topic and Most Similar Topics",
            showlegend=False,
            height=600
        )
        
        return fig
    
    # Callbacks for time series tab (if time data is available)
    if has_time:
        @app.callback(
            [Output('topic-time-trend', 'figure'),
             Output('topic-time-composition', 'figure'),
             Output('topic-time-heatmap', 'figure')],
            [Input('topic-selector', 'value'),
             Input('time-aggregation', 'value')]
        )
        def update_time_visuals(selected_topic, time_agg):
            # Time column must be present
            time_col = next((col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])), None)
            if not time_col:
                empty_fig = px.line(title="No time data available")
                return empty_fig, empty_fig, empty_fig
            
            # Create time series for topics
            df_time = df.copy()
            
            # Ensure time column is datetime
            df_time[time_col] = pd.to_datetime(df_time[time_col])
            
            # Group by time and calculate topic prevalence
            topic_columns = [col for col in df.columns if col.startswith('topic_')]
            
            # Resample by time aggregation
            time_groups = df_time.set_index(time_col).resample(time_agg)
            
            # Create dataframe with topic prevalence over time
            topic_time_df = time_groups[topic_columns].mean().reset_index()
            
            # Line chart for selected topic over time
            selected_col = f'topic_{selected_topic}'
            if selected_col not in topic_time_df.columns:
                empty_fig = px.line(title="Selected topic not found in time data")
                return empty_fig, empty_fig, empty_fig
            
            # Time trend for selected topic
            trend_fig = px.line(
                topic_time_df, 
                x=time_col, 
                y=selected_col,
                title=f"Topic {selected_topic} Trend Over Time",
                labels={selected_col: 'Topic Prevalence'}
            )
            
            # Stacked area chart for topic composition
            # Top 5 topics
            top_cols = topic_time_df[topic_columns].mean().sort_values(ascending=False).head(5).index
            
            composition_fig = px.area(
                topic_time_df, 
                x=time_col, 
                y=top_cols,
                title="Topic Composition Over Time",
                labels={col: f"Topic {col.split('_')[1]}" for col in top_cols}
            )
            
            # Heatmap of topic intensity over time
            pivot_data = []
            
            for _, row in topic_time_df.iterrows():
                time_val = row[time_col]
                for col in topic_columns:
                    topic_id = int(col.split('_')[1])
                    pivot_data.append({
                        'Time': time_val,
                        'Topic': f"Topic {topic_id}",
                        'Intensity': row[col]
                    })
            
            pivot_df = pd.DataFrame(pivot_data)
            
            # Create heatmap
            heatmap_fig = px.density_heatmap(
                pivot_df, 
                x='Time', 
                y='Topic', 
                z='Intensity',
                histfunc='avg',
                title="Topic Intensity Heatmap Over Time",
                labels={'Intensity': 'Topic Prevalence'}
            )
            
            return trend_fig, composition_fig, heatmap_fig
    
    # Callbacks for geospatial tab (if geo data is available)
    if has_geo:
        @app.callback(
            [Output('topic-geo-map', 'figure'),
             Output('topic-geo-density', 'figure')],
            [Input('topic-selector', 'value')]
        )
        def update_geo_visuals(selected_topic):
            # Check if latitude and longitude are present
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                empty_fig = px.scatter_mapbox(title="No geospatial data available")
                return empty_fig, empty_fig
            
            # Filter by selected topic
            geo_df = df[df['primary_topic'] == selected_topic].copy()
            
            # Skip if no documents for this topic with geo data
            if len(geo_df) == 0:
                empty_fig = px.scatter_mapbox(title="No geospatial data for selected topic")
                return empty_fig, empty_fig
            
            # Create scatter map
            map_fig = px.scatter_mapbox(
                geo_df,
                lat='latitude',
                lon='longitude',
                size='primary_topic_score',
                color_discrete_sequence=['red'],
                zoom=3,
                title=f"Geographic Distribution of Topic {selected_topic}"
            )
            
            map_fig.update_layout(mapbox_style='open-street-map')
            
            # Create density map
            density_fig = px.density_mapbox(
                geo_df,
                lat='latitude',
                lon='longitude',
                z='primary_topic_score',
                radius=10,
                zoom=3,
                title=f"Density of Topic {selected_topic}"
            )
            
            density_fig.update_layout(mapbox_style='open-street-map')
            
            return map_fig, density_fig
    
    # Callback for document explorer
    @app.callback(
        Output('document-list', 'children'),
        [Input('search-button', 'click'),
         Input('document-topic-filter', 'value')],
        [State('document-search', 'value')]
    )
    def update_document_list(n_clicks, topic_filter, search_term):
        filtered_df = df.copy()
        
        # Filter by topic if not 'all'
        if topic_filter != 'all':
            filtered_df = filtered_df[filtered_df['primary_topic'] == int(topic_filter)]
        
        # Search in documents if search term provided
        if search_term and n_clicks:
            filtered_df = filtered_df[filtered_df['document'].str.contains(search_term, case=False)]
        
        # Sort by topic score and get top 20
        filtered_df = filtered_df.sort_values('primary_topic_score', ascending=False).head(20)
        
        # Create document list
        doc_items = []
        
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            doc_items.append(html.Div([
                html.H6(f"Document {i+1} - Topic {row['primary_topic']} (Score: {row['primary_topic_score']:.2f})"),
                html.P(row['document'][:300] + "..." if len(row['document']) > 300 else row['document']),
                html.Hr()
            ]))
        
        if not doc_items:
            return html.P("No documents found matching the criteria.")
        
        return doc_items
    
    if return_app:
        return app
    else:
        app.run_server(debug=True)
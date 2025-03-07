"""Web interface for Meno topic modeling.

This module provides a web-based UI for using Meno's topic modeling capabilities,
with a focus on the lightweight models for better performance.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from meno.modeling.simple_models import (
    SimpleTopicModel,
    TFIDFTopicModel,
    NMFTopicModel,
    LSATopicModel
)
from meno.modeling.unified_topic_modeling import UnifiedTopicModeler
from meno.visualization.lightweight_viz import (
    plot_topic_landscape,
    plot_comparative_document_analysis
)


class MenoWebApp:
    """Web app for Meno topic modeling.
    
    A Dash-based web interface that allows users to upload data, configure models,
    and explore topic modeling results interactively.
    """
    
    def __init__(self, port: int = 8050, debug: bool = False):
        """Initialize the web app.
        
        Parameters
        ----------
        port : int, optional
            Port to run the server on, by default 8050
        debug : bool, optional
            Whether to run in debug mode, by default False
        """
        self.port = port
        self.debug = debug
        self.temp_dir = tempfile.mkdtemp(prefix="meno_webapp_")
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Meno Topic Modeling"
        )
        self.init_layout()
        self.init_callbacks()
        
    def init_layout(self):
        """Initialize the app layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Meno Topic Modeling", className="display-4 text-center my-4"),
                    html.P(
                        "Interactive topic modeling with lightweight models",
                        className="lead text-center mb-4"
                    )
                ], width=12)
            ]),
            
            # Tabs for different stages
            dbc.Tabs([
                # Data Upload Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Upload Data"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Upload(
                                                id="upload-data",
                                                children=html.Div([
                                                    "Drag and Drop or ",
                                                    html.A("Select a CSV or TXT File")
                                                ]),
                                                style={
                                                    "width": "100%",
                                                    "height": "60px",
                                                    "lineHeight": "60px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center",
                                                    "margin": "10px"
                                                },
                                                multiple=False
                                            )
                                        ], width=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(id="upload-status")
                                        ], width=12)
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Or Enter Sample Text"),
                                            dcc.Textarea(
                                                id="sample-text",
                                                placeholder="Enter multiple documents separated by new lines...",
                                                style={"width": "100%", "height": "150px"}
                                            ),
                                            dbc.Button(
                                                "Use Sample Text",
                                                id="use-sample-text",
                                                color="primary",
                                                className="mt-2"
                                            )
                                        ], width=12)
                                    ])
                                ])
                            ]),
                            html.Div(id="sample-text-status", className="mt-3")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Data Preview"),
                                dbc.CardBody([
                                    html.Div(id="data-preview")
                                ])
                            ]),
                            html.Div(id="data-stats", className="mt-3")
                        ], width=6)
                    ])
                ], label="1. Data Upload", tab_id="tab-data"),
                
                # Model Configuration Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Model Selection"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Select Model Type:"),
                                            dcc.Dropdown(
                                                id="model-type",
                                                options=[
                                                    {"label": "Simple K-Means", "value": "simple_kmeans"},
                                                    {"label": "TF-IDF K-Means", "value": "tfidf"},
                                                    {"label": "NMF Topic Model", "value": "nmf"},
                                                    {"label": "LSA Topic Model", "value": "lsa"}
                                                ],
                                                value="simple_kmeans",
                                                clearable=False
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Number of Topics:"),
                                            dcc.Slider(
                                                id="num-topics",
                                                min=2,
                                                max=20,
                                                step=1,
                                                value=5,
                                                marks={i: str(i) for i in range(2, 21, 2)}
                                            )
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Model Description:"),
                                            html.Div(id="model-description", className="p-2 bg-light border rounded")
                                        ], width=12)
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button(
                                                "Train Model",
                                                id="train-model",
                                                color="success",
                                                className="mt-2",
                                                size="lg"
                                            )
                                        ], width=12, className="text-center")
                                    ])
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Advanced Options"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Random Seed:"),
                                            dbc.Input(
                                                id="random-seed",
                                                type="number",
                                                value=42,
                                                min=0,
                                                step=1
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Max Features (for TF-IDF/NMF/LSA):"),
                                            dbc.Input(
                                                id="max-features",
                                                type="number",
                                                value=1000,
                                                min=100,
                                                step=100
                                            )
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Preprocessing Options:"),
                                            dbc.Checklist(
                                                id="preprocessing-options",
                                                options=[
                                                    {"label": "Remove stopwords", "value": "stop"},
                                                    {"label": "Convert to lowercase", "value": "lower"},
                                                    {"label": "Apply lemmatization", "value": "lemma"}
                                                ],
                                                value=["stop", "lower"]
                                            )
                                        ], width=12)
                                    ])
                                ])
                            ]),
                            html.Div(id="model-status", className="mt-3")
                        ], width=6)
                    ])
                ], label="2. Configure Model", tab_id="tab-model", disabled=True),
                
                # Results Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Topics Overview"),
                                dbc.CardBody([
                                    html.Div(id="topics-overview")
                                ])
                            ])
                        ], width=12)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Topic Landscape"),
                                dbc.CardBody([
                                    dcc.Graph(id="topic-landscape-viz")
                                ])
                            ])
                        ], width=7),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Topic Details"),
                                dbc.CardBody([
                                    html.Label("Select Topic:"),
                                    dcc.Dropdown(
                                        id="topic-selector",
                                        options=[],
                                        clearable=False
                                    ),
                                    html.Div(id="topic-details", className="mt-3")
                                ])
                            ])
                        ], width=5)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Document-Topic Analysis"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Number of documents to display:"),
                                            dcc.Slider(
                                                id="num-docs-display",
                                                min=5,
                                                max=50,
                                                step=5,
                                                value=20,
                                                marks={i: str(i) for i in range(5, 51, 5)}
                                            )
                                        ], width=12)
                                    ]),
                                    dcc.Graph(id="doc-topic-viz")
                                ])
                            ])
                        ], width=12)
                    ])
                ], label="3. Explore Results", tab_id="tab-results", disabled=True),
                
                # Document Search Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Search Documents"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.InputGroup([
                                                dbc.Input(
                                                    id="search-query",
                                                    placeholder="Enter search terms..."
                                                ),
                                                dbc.Button(
                                                    "Search",
                                                    id="search-button",
                                                    color="primary"
                                                )
                                            ])
                                        ], width=12)
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Filter by Topic:"),
                                            dcc.Dropdown(
                                                id="topic-filter",
                                                options=[],
                                                value="all",
                                                clearable=False
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Sort by:"),
                                            dcc.Dropdown(
                                                id="sort-by",
                                                options=[
                                                    {"label": "Relevance", "value": "relevance"},
                                                    {"label": "Topic Score", "value": "score"}
                                                ],
                                                value="relevance",
                                                clearable=False
                                            )
                                        ], width=6)
                                    ])
                                ])
                            ])
                        ], width=12)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Search Results"),
                                dbc.CardBody([
                                    html.Div(id="search-results")
                                ])
                            ])
                        ], width=12)
                    ])
                ], label="4. Search", tab_id="tab-search", disabled=True)
            ], id="main-tabs", active_tab="tab-data"),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        "Meno Topic Modeling - Lightweight Edition",
                        className="text-center text-muted"
                    )
                ], width=12)
            ])
        ], fluid=True)
    
    def init_callbacks(self):
        """Initialize the callbacks for the app."""
        # State storage using dcc.Store
        self.app.layout.children.append(dcc.Store(id="data-store"))
        self.app.layout.children.append(dcc.Store(id="model-store"))
        self.app.layout.children.append(dcc.Store(id="results-store"))
        
        # Model description callback
        @self.app.callback(
            Output("model-description", "children"),
            Input("model-type", "value")
        )
        def update_model_description(model_type):
            descriptions = {
                "simple_kmeans": """
                    **Simple K-Means Topic Model**
                    
                    This model uses document embeddings combined with K-Means clustering to identify topics.
                    It's suitable for:
                    - Medium to large datasets
                    - When document similarity is important
                    - Finding topics based on semantic meaning
                """,
                "tfidf": """
                    **TF-IDF K-Means Topic Model**
                    
                    This model uses TF-IDF vectorization with K-Means clustering.
                    It's suitable for:
                    - Very large datasets
                    - When word frequency is important
                    - Extremely lightweight processing with minimal dependencies
                """,
                "nmf": """
                    **Non-negative Matrix Factorization (NMF) Topic Model**
                    
                    Uses NMF on TF-IDF matrices to discover topics.
                    It's suitable for:
                    - Finding patterns of word co-occurrence
                    - When you need interpretable topics
                    - More granular topic control than clustering
                """,
                "lsa": """
                    **Latent Semantic Analysis (LSA) Topic Model**
                    
                    Uses truncated SVD on TF-IDF matrices to discover topics.
                    It's suitable for:
                    - Capturing semantic structure in text
                    - Handling synonymy and polysemy
                    - Very fast processing of large document sets
                """
            }
            return dcc.Markdown(descriptions.get(model_type, ""))
        
        # Data upload callback
        @self.app.callback(
            [Output("upload-status", "children"),
             Output("data-preview", "children"),
             Output("data-stats", "children"),
             Output("data-store", "data"),
             Output("tab-model", "disabled")],
            [Input("upload-data", "contents"),
             Input("use-sample-text", "n_clicks")],
            [State("upload-data", "filename"),
             State("sample-text", "value")]
        )
        def process_data(contents, n_clicks, filename, sample_text):
            trigger = dash.callback_context.triggered[0]["prop_id"]
            
            documents = []
            
            if trigger == "upload-data.contents" and contents:
                try:
                    import base64
                    import io
                    
                    # Decode the file contents
                    content_type, content_string = contents.split(",")
                    decoded = base64.b64decode(content_string)
                    
                    if filename.endswith(".csv"):
                        # Read CSV file
                        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                        
                        # Get the text column (assume it's the first one or named 'text')
                        text_col = df.columns[0]
                        if "text" in df.columns:
                            text_col = "text"
                            
                        documents = df[text_col].tolist()
                        
                    elif filename.endswith(".txt"):
                        # Read text file
                        text = decoded.decode("utf-8")
                        documents = [doc.strip() for doc in text.split("\n") if doc.strip()]
                    
                    # Filter out empty documents
                    documents = [doc for doc in documents if doc.strip()]
                    
                    return [
                        html.Div([
                            html.H5(f"Successfully processed {filename}"),
                            html.P(f"Found {len(documents)} documents")
                        ], className="alert alert-success"),
                        html.Div([
                            html.H5("Data Preview:"),
                            html.Ul([
                                html.Li(doc[:100] + "..." if len(doc) > 100 else doc)
                                for doc in documents[:5]
                            ])
                        ]),
                        html.Div([
                            html.H5("Data Statistics:"),
                            html.P(f"Total documents: {len(documents)}"),
                            html.P(f"Average document length: {sum(len(doc.split()) for doc in documents) / len(documents):.1f} words")
                        ]),
                        documents,
                        False
                    ]
                except Exception as e:
                    return [
                        html.Div([
                            html.H5("Error processing file"),
                            html.P(str(e))
                        ], className="alert alert-danger"),
                        html.Div(),
                        html.Div(),
                        None,
                        True
                    ]
                
            elif trigger == "use-sample-text.n_clicks" and n_clicks and sample_text:
                documents = [doc.strip() for doc in sample_text.split("\n") if doc.strip()]
                
                return [
                    html.Div([
                        html.H5("Successfully processed sample text"),
                        html.P(f"Found {len(documents)} documents")
                    ], className="alert alert-success"),
                    html.Div([
                        html.H5("Data Preview:"),
                        html.Ul([
                            html.Li(doc[:100] + "..." if len(doc) > 100 else doc)
                            for doc in documents[:5]
                        ])
                    ]),
                    html.Div([
                        html.H5("Data Statistics:"),
                        html.P(f"Total documents: {len(documents)}"),
                        html.P(f"Average document length: {sum(len(doc.split()) for doc in documents) / len(documents):.1f} words")
                    ]),
                    documents,
                    False
                ]
            
            return [
                html.Div(),
                html.Div(),
                html.Div(),
                None,
                True
            ]
        
        # Model training callback
        @self.app.callback(
            [Output("model-status", "children"),
             Output("model-store", "data"),
             Output("tab-results", "disabled"),
             Output("tab-search", "disabled"),
             Output("main-tabs", "active_tab")],
            Input("train-model", "n_clicks"),
            [State("data-store", "data"),
             State("model-type", "value"),
             State("num-topics", "value"),
             State("random-seed", "value"),
             State("max-features", "value"),
             State("preprocessing-options", "value")]
        )
        def train_model(n_clicks, documents, model_type, num_topics, random_seed, max_features, preprocessing):
            if not n_clicks or not documents:
                return html.Div(), None, True, True, "tab-model"
            
            # Model parameters
            params = {
                "method": model_type,
                "num_topics": num_topics,
                "random_state": random_seed,
                "config_overrides": {}
            }
            
            # Add model-specific parameters
            if model_type in ["tfidf", "nmf", "lsa"]:
                params["config_overrides"]["max_features"] = max_features
            
            # Initialize the topic modeler
            # Note: For a real implementation, we would do preprocessing based on the options
            try:
                modeler = UnifiedTopicModeler(**params)
                
                # Train the model
                import time
                start_time = time.time()
                modeler.fit(documents)
                training_time = time.time() - start_time
                
                # Get topic information
                topic_info = modeler.get_topic_info()
                
                # Store model results
                results = {
                    "model_type": model_type,
                    "num_topics": num_topics,
                    "training_time": training_time,
                    "topic_counts": topic_info["Count" if "Count" in topic_info.columns else "Size"].tolist(),
                    "topic_names": topic_info["Name"].tolist(),
                    "topic_ids": topic_info["Topic"].tolist()
                }
                
                # Placeholder to indicate success - in a real implementation we'd
                # serialize the model or its results
                model_data = {
                    "params": params,
                    "is_trained": True,
                    "model_id": f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                return [
                    html.Div([
                        html.H5("Model trained successfully!"),
                        html.P(f"Training completed in {training_time:.2f} seconds"),
                        html.P(f"Found {len(topic_info)} topics")
                    ], className="alert alert-success"),
                    model_data,
                    False,
                    False,
                    "tab-results"
                ]
            except Exception as e:
                return [
                    html.Div([
                        html.H5("Error training model"),
                        html.P(str(e))
                    ], className="alert alert-danger"),
                    None,
                    True,
                    True,
                    "tab-model"
                ]
        
        # Topic selector options callback
        @self.app.callback(
            [Output("topic-selector", "options"),
             Output("topic-selector", "value"),
             Output("topic-filter", "options")],
            [Input("model-store", "data"),
             Input("results-store", "data")]
        )
        def update_topic_selector(model_data, results_data):
            if not model_data or not model_data.get("is_trained"):
                return [], None, []
            
            # In a real implementation, we'd get this data from the model
            # For now, let's use dummy data
            topics = [
                {"label": f"Topic {i}", "value": i}
                for i in range(5)  # Assuming 5 topics for this example
            ]
            
            # Options for topic filter (includes "All Topics")
            filter_options = [{"label": "All Topics", "value": "all"}] + topics
            
            return topics, 0, filter_options
        
        # Topics overview callback
        @self.app.callback(
            Output("topics-overview", "children"),
            [Input("model-store", "data")]
        )
        def update_topics_overview(model_data):
            if not model_data or not model_data.get("is_trained"):
                return html.Div()
            
            # For a real implementation, load the model and get topic info
            # For now, create a placeholder for the demo
            topic_data = {
                "Topic": [0, 1, 2, 3, 4],
                "Count": [25, 20, 18, 15, 10], 
                "Name": [
                    "Service: customer support response",
                    "Product: quality performance value",
                    "Software: interface user documentation",
                    "Hardware: battery device power",
                    "Training: materials documentation learning"
                ],
                "Words": [
                    ["service", "customer", "support", "response", "time"],
                    ["product", "quality", "performance", "value", "reliable"],
                    ["software", "interface", "user", "documentation", "examples"],
                    ["hardware", "battery", "device", "power", "life"],
                    ["training", "materials", "documentation", "learning", "comprehensive"]
                ]
            }
            
            # Create a bar chart figure
            fig = px.bar(
                topic_data, 
                x="Topic", 
                y="Count", 
                color="Count",
                labels={"Topic": "Topic ID", "Count": "Document Count"},
                title="Topics by Document Count"
            )
            
            # Create a table of topics
            topic_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Topic ID"),
                        html.Th("Name"),
                        html.Th("Count"),
                        html.Th("Top Words")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(topic),
                        html.Td(name),
                        html.Td(count),
                        html.Td(", ".join(words[:5]))
                    ]) for topic, name, count, words in zip(
                        topic_data["Topic"],
                        topic_data["Name"],
                        topic_data["Count"],
                        topic_data["Words"]
                    )
                ])
            ], bordered=True, hover=True)
            
            return html.Div([
                dcc.Graph(figure=fig),
                html.Hr(),
                topic_table
            ])
        
        # Topic details callback
        @self.app.callback(
            Output("topic-details", "children"),
            [Input("topic-selector", "value"),
             Input("model-store", "data")]
        )
        def update_topic_details(topic_id, model_data):
            if topic_id is None or not model_data or not model_data.get("is_trained"):
                return html.Div()
            
            # For a real implementation, load the model and get topic details
            # For now, create placeholder data
            topic_details = [
                {
                    "name": "Service: customer support response",
                    "count": 25,
                    "words": [
                        ("service", 0.85), ("customer", 0.78), ("support", 0.72),
                        ("response", 0.65), ("time", 0.59), ("helpful", 0.54),
                        ("issue", 0.49), ("team", 0.45), ("assistance", 0.42)
                    ],
                    "documents": [
                        "Customer service was excellent and the product arrived on time.",
                        "Technical support was helpful in resolving my installation issues.",
                        "Customer service response time could be improved."
                    ]
                },
                {
                    "name": "Product: quality performance value",
                    "count": 20,
                    "words": [
                        ("product", 0.87), ("quality", 0.81), ("performance", 0.75),
                        ("value", 0.68), ("reliable", 0.63), ("excellent", 0.58),
                        ("expectations", 0.53), ("price", 0.48), ("design", 0.44)
                    ],
                    "documents": [
                        "Product quality is outstanding and exceeded my expectations.",
                        "Performance is excellent even with large datasets.",
                        "The product is reliable and hasn't crashed in months of use."
                    ]
                },
                {
                    "name": "Software: interface user documentation",
                    "count": 18,
                    "words": [
                        ("software", 0.86), ("interface", 0.79), ("user", 0.73),
                        ("documentation", 0.66), ("examples", 0.61), ("features", 0.56),
                        ("intuitive", 0.52), ("navigate", 0.47), ("learning", 0.43)
                    ],
                    "documents": [
                        "The software has a steep learning curve but powerful features.",
                        "User interface is intuitive and easy to navigate.",
                        "The documentation lacks examples and could be improved."
                    ]
                },
                {
                    "name": "Hardware: battery device power",
                    "count": 15,
                    "words": [
                        ("hardware", 0.88), ("battery", 0.82), ("device", 0.76),
                        ("power", 0.69), ("life", 0.64), ("integration", 0.59),
                        ("systems", 0.54), ("impressive", 0.49), ("compared", 0.45)
                    ],
                    "documents": [
                        "The hardware integration works seamlessly with our existing systems.",
                        "Battery life is impressive compared to previous models.",
                        "Device power management has improved significantly."
                    ]
                },
                {
                    "name": "Training: materials documentation learning",
                    "count": 10,
                    "words": [
                        ("training", 0.89), ("materials", 0.83), ("documentation", 0.77),
                        ("learning", 0.70), ("comprehensive", 0.65), ("structured", 0.60),
                        ("resources", 0.55), ("guide", 0.50), ("tutorials", 0.46)
                    ],
                    "documents": [
                        "Training materials are comprehensive and well-structured.",
                        "The learning resources include helpful video tutorials.",
                        "Documentation provides clear step-by-step guides."
                    ]
                }
            ]
            
            # Get the selected topic's details
            if 0 <= topic_id < len(topic_details):
                details = topic_details[topic_id]
                
                # Create bar chart for word weights
                words = [word for word, _ in details["words"]]
                weights = [weight for _, weight in details["words"]]
                
                word_fig = px.bar(
                    x=words, 
                    y=weights,
                    labels={"x": "Word", "y": "Weight"},
                    title=f"Top Words for {details['name']}"
                )
                
                # Create document list
                document_list = html.Ul([
                    html.Li(doc) for doc in details["documents"]
                ])
                
                return html.Div([
                    html.H5(details["name"]),
                    html.P(f"Document count: {details['count']}"),
                    html.Hr(),
                    dcc.Graph(figure=word_fig),
                    html.Hr(),
                    html.H6("Sample Documents:"),
                    document_list
                ])
            
            return html.Div()
        
        # Topic landscape callback
        @self.app.callback(
            Output("topic-landscape-viz", "figure"),
            [Input("model-store", "data")]
        )
        def update_topic_landscape(model_data):
            if not model_data or not model_data.get("is_trained"):
                return go.Figure()
            
            # For a real implementation, load the model and generate visualization
            # For now, create a placeholder visualization
            
            # Create a placeholder topic landscape
            fig = go.Figure()
            
            # Topic nodes
            topics = [
                {"id": 0, "name": "Service", "x": 0.2, "y": 0.8, "size": 25},
                {"id": 1, "name": "Product", "x": 0.8, "y": 0.2, "size": 20},
                {"id": 2, "name": "Software", "x": 0.8, "y": 0.8, "size": 18},
                {"id": 3, "name": "Hardware", "x": 0.2, "y": 0.2, "size": 15},
                {"id": 4, "name": "Training", "x": 0.5, "y": 0.5, "size": 10}
            ]
            
            # Add topic nodes
            for topic in topics:
                fig.add_trace(
                    go.Scatter(
                        x=[topic["x"]],
                        y=[topic["y"]],
                        mode="markers+text",
                        marker=dict(
                            size=topic["size"] + 15,
                            symbol="diamond",
                            line=dict(width=2, color="black")
                        ),
                        text=[f"T{topic['id']}"],
                        textposition="middle center",
                        name=topic["name"],
                        hovertext=f"Topic {topic['id']}: {topic['name']}<br>Size: {topic['size']}",
                        hoverinfo="text"
                    )
                )
            
            # Add edges between topics
            edges = [
                (0, 2, 0.6),  # Service - Software
                (1, 3, 0.7),  # Product - Hardware
                (2, 4, 0.5),  # Software - Training
                (0, 4, 0.4),  # Service - Training
                (1, 2, 0.3)   # Product - Software
            ]
            
            for i, j, weight in edges:
                fig.add_trace(
                    go.Scatter(
                        x=[topics[i]["x"], topics[j]["x"]],
                        y=[topics[i]["y"], topics[j]["y"]],
                        mode="lines",
                        line=dict(width=weight * 5, color=f"rgba(100,100,100,{weight})"),
                        hoverinfo="none",
                        showlegend=False
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Topic Landscape",
                xaxis=dict(title="Dimension 1", showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(title="Dimension 2", showgrid=False, zeroline=False, showticklabels=False),
                hovermode="closest",
                showlegend=False
            )
            
            return fig
        
        # Document-topic visualization callback
        @self.app.callback(
            Output("doc-topic-viz", "figure"),
            [Input("model-store", "data"),
             Input("num-docs-display", "value")]
        )
        def update_doc_topic_viz(model_data, num_docs):
            if not model_data or not model_data.get("is_trained") or not num_docs:
                return go.Figure()
            
            # For a real implementation, load the model and analyze documents
            # For now, create a placeholder visualization
            
            # Create dummy document-topic matrix
            import numpy as np
            np.random.seed(42)
            
            doc_topic_matrix = np.zeros((num_docs, 5))
            
            # Assign each document a primary topic with high weight
            for i in range(num_docs):
                primary_topic = i % 5
                doc_topic_matrix[i, primary_topic] = 0.6 + 0.3 * np.random.random()
                
                # Distribute remaining probability
                remaining = 1.0 - doc_topic_matrix[i, primary_topic]
                secondary_topics = [j for j in range(5) if j != primary_topic]
                weights = np.random.dirichlet(np.ones(len(secondary_topics)))
                
                for j, topic in enumerate(secondary_topics):
                    doc_topic_matrix[i, topic] = remaining * weights[j]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=doc_topic_matrix,
                x=[f"Topic {i}" for i in range(5)],
                y=[f"Doc {i}" for i in range(num_docs)],
                colorscale="Viridis",
                hoverongaps=False,
                text=[[f"{doc_topic_matrix[i][j]:.2f}" for j in range(5)] 
                      for i in range(num_docs)],
                hoverinfo="text+x+y"
            ))
            
            # Update layout
            fig.update_layout(
                title="Document-Topic Distribution",
                xaxis_title="Topics",
                yaxis_title="Documents"
            )
            
            return fig
        
        # Search results callback
        @self.app.callback(
            Output("search-results", "children"),
            [Input("search-button", "n_clicks"),
             Input("topic-filter", "value"),
             Input("sort-by", "value")],
            [State("search-query", "value"),
             State("model-store", "data"),
             State("data-store", "data")]
        )
        def update_search_results(n_clicks, topic_filter, sort_by, query, model_data, documents):
            if not n_clicks or not query or not model_data or not model_data.get("is_trained") or not documents:
                return html.Div("Enter a search query to find related documents")
            
            # For a real implementation, use the model to search documents
            # For now, create placeholder results
            
            # Create some dummy results
            results = [
                {
                    "document": "Customer service was excellent and the product arrived on time.",
                    "score": 0.92,
                    "topic": 0,
                    "topic_name": "Service: customer support response"
                },
                {
                    "document": "The software has a steep learning curve but powerful features.",
                    "score": 0.85,
                    "topic": 2,
                    "topic_name": "Software: interface user documentation"
                },
                {
                    "document": "Product quality is outstanding and exceeded my expectations.",
                    "score": 0.78,
                    "topic": 1,
                    "topic_name": "Product: quality performance value"
                },
                {
                    "document": "Technical support was helpful in resolving my installation issues.",
                    "score": 0.76,
                    "topic": 0,
                    "topic_name": "Service: customer support response"
                },
                {
                    "document": "User interface is intuitive and easy to navigate.",
                    "score": 0.72,
                    "topic": 2,
                    "topic_name": "Software: interface user documentation"
                }
            ]
            
            # Filter by topic if needed
            if topic_filter != "all":
                topic_id = int(topic_filter)
                results = [r for r in results if r["topic"] == topic_id]
            
            # Sort by selected criterion
            if sort_by == "relevance":
                results = sorted(results, key=lambda x: x["score"], reverse=True)
            elif sort_by == "score":
                results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            if not results:
                return html.Div("No matching documents found")
            
            # Create result cards
            result_cards = []
            
            for i, result in enumerate(results):
                card = dbc.Card([
                    dbc.CardHeader(f"Result {i+1} - {result['topic_name']} (Score: {result['score']:.2f})"),
                    dbc.CardBody([
                        html.P(result["document"])
                    ])
                ], className="mb-3")
                
                result_cards.append(card)
            
            return html.Div(result_cards)
            
    def run(self):
        """Run the web app server."""
        self.app.run_server(debug=self.debug, port=self.port)
        
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def launch_web_interface(port: int = 8050, debug: bool = False):
    """Launch the Meno web interface.
    
    Parameters
    ----------
    port : int, optional
        Port to run the web server on, by default 8050
    debug : bool, optional
        Whether to run in debug mode, by default False
    """
    app = MenoWebApp(port=port, debug=debug)
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    launch_web_interface(debug=True)
"""Visualizations for displaying feedback impact on topic models.

This module provides specialized visualizations for showing how user feedback
affects topic models, including before/after comparisons and transition analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import time


def plot_feedback_impact(
    feedback_manager: Any,
    figsize: Tuple[int, int] = (14, 12),
    max_topics: int = 10,
    title: str = "Feedback Impact Analysis",
):
    """
    Generate a comprehensive visualization of feedback impact on topic modeling.
    
    Parameters
    ----------
    feedback_manager : Any
        The TopicFeedbackManager instance containing feedback history
    figsize : Tuple[int, int], optional
        Figure size, by default (14, 12)
    max_topics : int, optional
        Maximum number of topics to show in charts, by default 10
    title : str, optional
        Main figure title, by default "Feedback Impact Analysis"
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the visualizations
    """
    # Get impact data if available
    if not hasattr(feedback_manager, "feedback_sessions") or not feedback_manager.feedback_sessions:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No feedback sessions available", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14)
        ax.axis('off')
        return fig
        
    # Get impact data
    impact = {}
    if hasattr(feedback_manager, "evaluate_impact"):
        impact = feedback_manager.evaluate_impact(detailed=True)
    else:
        # Create a simplified impact structure if evaluate_impact isn't available
        impact = {
            "total_documents": len(feedback_manager.documents),
            "documents_changed": 0,
            "percent_changed": 0,
            "topic_changes": {},
            "original_topic_counts": {},
            "current_topic_counts": {},
            "topic_transitions": []
        }
        
        # Count topic changes
        original_topics = []
        current_topics = []
        
        # Get feedback data from all sessions
        for session in feedback_manager.feedback_sessions:
            for item in session["feedback"]:
                doc_idx = item["document_index"]
                orig = item["original_topic"]
                curr = item["corrected_topic"]
                
                if orig != curr:
                    impact["documents_changed"] += 1
                    
                    # Track transition
                    transition_key = (orig, curr)
                    found = False
                    for trans in impact["topic_transitions"]:
                        if trans["from_topic"] == orig and trans["to_topic"] == curr:
                            trans["count"] += 1
                            found = True
                            break
                    
                    if not found:
                        impact["topic_transitions"].append({
                            "from_topic": orig,
                            "to_topic": curr,
                            "count": 1
                        })
                
                # Add to lists for counting
                original_topics.append(orig)
                current_topics.append(curr)
        
        # Calculate percent changed
        if impact["total_documents"] > 0:
            impact["percent_changed"] = (impact["documents_changed"] / impact["total_documents"]) * 100
            
        # Count topic distributions
        for topic in set(original_topics + current_topics):
            impact["original_topic_counts"][topic] = original_topics.count(topic)
            impact["current_topic_counts"][topic] = current_topics.count(topic)
            impact["topic_changes"][topic] = current_topics.count(topic) - original_topics.count(topic)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Topic distribution changes
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get topic counts
    orig_counts = impact["original_topic_counts"]
    curr_counts = impact["current_topic_counts"]
    
    # Get all topics
    all_topics = sorted(list(set(list(orig_counts.keys()) + list(curr_counts.keys()))))
    
    # Filter to top N topics by total count
    topic_total_counts = {t: orig_counts.get(t, 0) + curr_counts.get(t, 0) for t in all_topics}
    top_topics = sorted(all_topics, key=lambda t: topic_total_counts[t], reverse=True)[:max_topics]
    
    # Prepare data
    x = np.arange(len(top_topics))
    width = 0.35
    
    # Plot bars
    orig_values = [orig_counts.get(t, 0) for t in top_topics]
    curr_values = [curr_counts.get(t, 0) for t in top_topics]
    
    ax1.bar(x - width/2, orig_values, width, label='Before Feedback', color='#1f77b4', alpha=0.7)
    ax1.bar(x + width/2, curr_values, width, label='After Feedback', color='#ff7f0e', alpha=0.7)
    
    # Add labels and title
    ax1.set_title('Topic Distribution Changes')
    ax1.set_ylabel('Document Count')
    ax1.set_xlabel('Topic')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in top_topics], rotation=45, ha='right')
    ax1.legend()
    
    # Add net change values above bars
    for i, topic in enumerate(top_topics):
        orig = orig_counts.get(topic, 0)
        curr = curr_counts.get(topic, 0)
        net_change = curr - orig
        
        if net_change != 0:
            color = 'green' if net_change > 0 else 'red'
            ax1.annotate(f"{net_change:+d}", 
                        xy=(i, max(orig, curr) + 1),
                        ha='center', va='bottom',
                        color=color, fontweight='bold')
    
    # 2. Topic transition flows (simplified Sankey)
    ax2 = fig.add_subplot(gs[0, 1])
    
    transitions = impact["topic_transitions"]
    
    if not transitions:
        ax2.text(0.5, 0.5, "No topic transitions found", 
                horizontalalignment='center', verticalalignment='center')
        ax2.axis('off')
    else:
        # Limit to top transitions by count
        transitions.sort(key=lambda x: x["count"], reverse=True)
        top_transitions = transitions[:min(15, len(transitions))]
        
        # Basic arrow plot
        y_pos = 0
        spacing = 1
        y_positions = {}
        
        # Track which topics we've seen
        seen_topics = set()
        
        # First pass: assign y-positions to all topics in transitions
        for t in top_transitions:
            from_topic = t["from_topic"]
            to_topic = t["to_topic"]
            
            if from_topic not in seen_topics:
                y_positions[from_topic] = y_pos
                y_pos += spacing
                seen_topics.add(from_topic)
            
            if to_topic not in seen_topics:
                y_positions[to_topic] = y_pos
                y_pos += spacing
                seen_topics.add(to_topic)
        
        # Draw arrows between topics
        max_count = max(t["count"] for t in top_transitions)
        
        for t in top_transitions:
            from_topic = t["from_topic"]
            to_topic = t["to_topic"]
            count = t["count"]
            
            # Arrow width proportional to count
            arrow_width = 0.1 + (count / max_count) * 0.9
            
            # Draw arrow from left (original) to right (new)
            ax2.annotate("",
                xy=(1, y_positions[to_topic]),    # end point
                xytext=(0, y_positions[from_topic]),  # start point
                arrowprops=dict(
                    arrowstyle="->",
                    color="blue",
                    alpha=0.6,
                    linewidth=1 + arrow_width * 5,
                    shrinkA=5,
                    shrinkB=5,
                ),
            )
            
            # Add count label near middle of arrow
            mid_y = (y_positions[from_topic] + y_positions[to_topic]) / 2
            ax2.text(0.5, mid_y, str(count), 
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=8)
        
        # Add topic labels
        for topic, y in y_positions.items():
            ax2.text(-0.1, y, f"Topic {topic}", horizontalalignment='right')
            ax2.text(1.1, y, f"Topic {topic}", horizontalalignment='left')
        
        # Configure axes
        ax2.set_title('Topic Transition Flows')
        ax2.set_xlim(-0.2, 1.2)
        ax2.set_ylim(-0.5, y_pos + 0.5)
        ax2.axis('off')
    
    # 3. Feedback session progression
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Extract session data
    sessions = []
    if hasattr(feedback_manager, "get_feedback_summary"):
        summary = feedback_manager.get_feedback_summary()
        if not isinstance(summary, str):
            sessions = summary
    
    if not sessions:
        # Create session data manually
        sessions = []
        for i, session in enumerate(feedback_manager.feedback_sessions):
            session_id = i + 1
            n_docs = session["n_documents"]
            n_changed = session["n_changed"]
            pct_changed = session["pct_changed"]
            
            sessions.append({
                "session_id": session_id,
                "documents_reviewed": n_docs,
                "topics_changed": n_changed,
                "percent_changed": pct_changed
            })
        
        sessions = pd.DataFrame(sessions)
    
    if len(sessions) == 0:
        ax3.text(0.5, 0.5, "No feedback session data available", 
                horizontalalignment='center', verticalalignment='center')
        ax3.axis('off')
    else:
        # Plot percentage changed over sessions
        ax3.plot(sessions['session_id'], sessions['percent_changed'], 
                 marker='o', linestyle='-', color='green')
        
        # Add labels
        for i, row in sessions.iterrows():
            ax3.annotate(f"{int(row['topics_changed'])}/{int(row['documents_reviewed'])}",
                        (row['session_id'], row['percent_changed']),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        ax3.set_title('Feedback Impact per Session')
        ax3.set_xlabel('Session ID')
        ax3.set_ylabel('% Documents Changed')
        ax3.set_xticks(sessions['session_id'])
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Cumulative feedback impact
    ax4 = fig.add_subplot(gs[1, 1])
    
    if len(sessions) == 0:
        ax4.text(0.5, 0.5, "No feedback sessions yet", horizontalalignment='center',
                 verticalalignment='center')
        ax4.axis('off')
    else:
        # Calculate cumulative metrics
        cumulative_reviewed = sessions['documents_reviewed'].cumsum()
        cumulative_changed = sessions['topics_changed'].cumsum()
        pct_cumulative = (cumulative_changed / impact['total_documents']) * 100
        
        # Plot cumulative percentage
        ax4.plot(sessions['session_id'], pct_cumulative, 
                marker='s', linestyle='-', color='purple')
        
        # Add total document count line
        ax4.axhline(y=100, color='red', linestyle='--', alpha=0.3)
        ax4.text(1, 102, f"Total: {impact['total_documents']} docs", color='red')
        
        # Add labels
        for i, row in sessions.iterrows():
            cumul_pct = pct_cumulative.iloc[i]
            cumul_changed = cumulative_changed.iloc[i]
            ax4.annotate(f"{int(cumul_changed)} docs",
                        (row['session_id'], cumul_pct),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        ax4.set_title('Cumulative Feedback Impact')
        ax4.set_xlabel('Session')
        ax4.set_ylabel('% Total Documents Changed')
        ax4.set_xticks(sessions['session_id'])
        ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall impact summary
    plt.figtext(0.5, 0.02, 
                f"Total Documents: {impact['total_documents']} | "
                f"Changed: {impact['documents_changed']} | "
                f"Impact Rate: {impact['percent_changed']:.1f}%",
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def create_feedback_comparison_dashboard(
    before_model: Any,
    after_model: Any,
    documents: List[str],
    changed_indices: Optional[List[int]] = None,
    title: str = "Feedback Impact Comparison Dashboard",
):
    """
    Create an interactive dashboard comparing model results before and after feedback.
    
    Parameters
    ----------
    before_model : Any
        Topic model before feedback was applied
    after_model : Any
        Topic model after feedback was applied
    documents : List[str]
        List of documents
    changed_indices : Optional[List[int]], optional
        List of document indices that had topic changes, by default None
    title : str, optional
        Dashboard title, by default "Feedback Impact Comparison Dashboard"
        
    Returns
    -------
    dash.Dash
        The Dash app for visualization
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError(
            "This visualization requires Dash to be installed. "
            "Please install it with: pip install dash"
        )
    
    # Extract topic data
    try:
        # Try to get document info
        before_topics = before_model.get_document_topics()
        after_topics = after_model.get_document_topics()
    except (AttributeError, TypeError):
        try:
            # Try alternate method
            before_topics = pd.DataFrame({"topic": before_model.doc_topics.topic})
            after_topics = pd.DataFrame({"topic": after_model.doc_topics.topic})
        except (AttributeError, TypeError):
            raise ValueError("Could not extract topic information from the models")
    
    # Extract embeddings if available
    try:
        embeddings = before_model.get_document_embeddings()
    except (AttributeError, TypeError):
        try:
            embeddings = before_model.doc_embeddings
        except (AttributeError, TypeError):
            embeddings = None
    
    # Determine changed documents
    if changed_indices is None:
        changed_indices = []
        for i, (before, after) in enumerate(zip(before_topics["topic"], after_topics["topic"])):
            if before != after:
                changed_indices.append(i)
    
    # Get reduced embeddings for visualization if available
    if embeddings is not None:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(embeddings)
        except:
            reduced_embeddings = None
    else:
        reduced_embeddings = None
    
    # Create Dash app
    app = dash.Dash(__name__, title=title)
    
    # Define layout
    app.layout = html.Div([
        html.H1(title, style={"textAlign": "center"}),
        
        html.Div([
            html.Div([
                html.H3("Topic Distribution Comparison"),
                dcc.Graph(id="topic-distribution-graph"),
            ], className="graph-container"),
            
            html.Div([
                html.H3("Document Embedding Space"),
                dcc.Graph(id="embedding-space-graph") if reduced_embeddings is not None else
                html.Div("Document embeddings not available for visualization"),
            ], className="graph-container"),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px"}),
        
        html.Div([
            html.H3("Document Comparison"),
            
            html.Div([
                html.Div([
                    html.Label("Select Document:"),
                    dcc.Dropdown(
                        id="document-selector",
                        options=[
                            {"label": f"Doc {i}: {doc[:50]}...", "value": i}
                            for i, doc in enumerate(documents)
                        ],
                        value=changed_indices[0] if changed_indices else 0,
                    ),
                ], style={"width": "50%"}),
                
                html.Div([
                    html.Label("Filter By:"),
                    dcc.RadioItems(
                        id="document-filter",
                        options=[
                            {"label": "All Documents", "value": "all"},
                            {"label": "Changed Documents Only", "value": "changed"},
                        ],
                        value="changed",
                        inline=True,
                    ),
                ], style={"width": "50%"}),
            ], style={"display": "flex", "gap": "20px"}),
            
            html.Div([
                html.Div([
                    html.H4("Before Feedback"),
                    html.Div(id="before-topic-info"),
                    html.Div(id="before-document-text", className="document-text"),
                ], className="document-panel"),
                
                html.Div([
                    html.H4("After Feedback"),
                    html.Div(id="after-topic-info"),
                    html.Div(id="after-document-text", className="document-text"),
                ], className="document-panel"),
            ], style={"display": "flex", "gap": "20px"}),
            
            html.Div([
                html.H3("Document Navigator"),
                html.Button("Previous", id="prev-doc-button"),
                html.Button("Next", id="next-doc-button"),
                html.Div(id="document-nav-info"),
            ], style={"marginTop": "20px", "textAlign": "center"}),
        ], className="document-container"),
        
        html.Footer([
            html.P("Generated with Meno Topic Modeling Toolkit"),
        ], style={"textAlign": "center", "marginTop": "30px", "color": "#666"}),
        
        # Include CSS
        html.Style("""
            .document-text {
                border: 1px solid #ddd;
                padding: 10px;
                background-color: #f9f9f9;
                height: 300px;
                overflow-y: auto;
                margin-top: 10px;
                white-space: pre-wrap;
            }
            
            .document-panel {
                width: 50%;
                padding: 10px;
                border: 1px solid #eee;
                border-radius: 5px;
            }
            
            .document-container {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f5f5f5;
            }
            
            .graph-container {
                flex: 1;
                min-width: 500px;
                border: 1px solid #eee;
                padding: 10px;
                border-radius: 5px;
            }
            
            button {
                padding: 5px 15px;
                margin: 0 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            
            button:hover {
                background-color: #45a049;
            }
            
            .changed {
                color: red;
                font-weight: bold;
            }
        """),
    ])
    
    # Callback for updating document info
    @app.callback(
        [
            Output("before-topic-info", "children"),
            Output("after-topic-info", "children"),
            Output("before-document-text", "children"),
            Output("after-document-text", "children"),
            Output("document-nav-info", "children"),
        ],
        [
            Input("document-selector", "value"),
        ]
    )
    def update_document_info(doc_idx):
        if doc_idx is None:
            return "No document selected", "No document selected", "", "", ""
        
        doc_idx = int(doc_idx)
        document = documents[doc_idx]
        
        before_topic = before_topics.iloc[doc_idx]["topic"] if doc_idx < len(before_topics) else "Unknown"
        after_topic = after_topics.iloc[doc_idx]["topic"] if doc_idx < len(after_topics) else "Unknown"
        
        changed = before_topic != after_topic
        change_class = " changed" if changed else ""
        
        before_info = html.Div([
            html.P([
                "Topic: ",
                html.Span(f"{before_topic}", className=change_class)
            ]),
        ])
        
        after_info = html.Div([
            html.P([
                "Topic: ",
                html.Span(f"{after_topic}", className=change_class)
            ]),
            html.P(f"Status: {'Changed' if changed else 'Unchanged'}", 
                   style={"color": "red" if changed else "green", "fontWeight": "bold"}),
        ])
        
        doc_text = html.Div(document)
        
        nav_info = f"Document {doc_idx + 1} of {len(documents)}"
        if changed:
            nav_info += " (Changed)"
        
        return before_info, after_info, doc_text, doc_text, nav_info
    
    # Callback for document navigation
    @app.callback(
        Output("document-selector", "value"),
        [
            Input("prev-doc-button", "n_clicks"),
            Input("next-doc-button", "n_clicks"),
            Input("document-filter", "value"),
        ],
        [dash.dependencies.State("document-selector", "value")]
    )
    def navigate_documents(prev_clicks, next_clicks, filter_type, current_doc_idx):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_doc_idx
        
        if current_doc_idx is None:
            return 0
        
        current_doc_idx = int(current_doc_idx)
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Filter documents based on selection
        if filter_type == "changed":
            available_indices = changed_indices
        else:
            available_indices = list(range(len(documents)))
        
        if not available_indices:
            return current_doc_idx
        
        # Find current position in filtered list
        try:
            current_position = available_indices.index(current_doc_idx)
        except ValueError:
            # If current document is not in the filtered list, return the first one
            return available_indices[0]
        
        if triggered_id == "prev-doc-button":
            new_position = (current_position - 1) % len(available_indices)
        elif triggered_id == "next-doc-button":
            new_position = (current_position + 1) % len(available_indices)
        else:
            return current_doc_idx
        
        return available_indices[new_position]
    
    # Callback for topic distribution graph
    @app.callback(
        Output("topic-distribution-graph", "figure"),
        [Input("document-filter", "value")]
    )
    def update_topic_distribution(filter_type):
        # Count topics before and after
        before_counts = before_topics["topic"].value_counts().to_dict()
        after_counts = after_topics["topic"].value_counts().to_dict()
        
        # Get all topics
        all_topics = sorted(list(set(list(before_counts.keys()) + list(after_counts.keys()))))
        
        # Filter to top N topics by total count
        topic_total_counts = {t: before_counts.get(t, 0) + after_counts.get(t, 0) for t in all_topics}
        top_topics = sorted(all_topics, key=lambda t: topic_total_counts[t], reverse=True)[:10]
        
        # Convert to strings for display
        top_topics_str = [str(t) for t in top_topics]
        
        # Prepare data
        before_values = [before_counts.get(t, 0) for t in top_topics]
        after_values = [after_counts.get(t, 0) for t in top_topics]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_topics_str,
            y=before_values,
            name="Before Feedback",
            marker_color="#1f77b4"
        ))
        
        fig.add_trace(go.Bar(
            x=top_topics_str,
            y=after_values,
            name="After Feedback",
            marker_color="#ff7f0e"
        ))
        
        # Update layout
        fig.update_layout(
            title="Topic Distribution Comparison",
            xaxis_title="Topic",
            yaxis_title="Document Count",
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    # Callback for embedding space graph (if embeddings are available)
    if reduced_embeddings is not None:
        @app.callback(
            Output("embedding-space-graph", "figure"),
            [
                Input("document-filter", "value"),
                Input("document-selector", "value")
            ]
        )
        def update_embedding_space(filter_type, selected_doc_idx):
            # Create a dataframe with embeddings and topic info
            df = pd.DataFrame({
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "before_topic": before_topics["topic"],
                "after_topic": after_topics["topic"],
                "changed": before_topics["topic"] != after_topics["topic"]
            })
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot for unchanged documents
            unchanged = df[~df["changed"]]
            fig.add_trace(go.Scatter(
                x=unchanged["x"],
                y=unchanged["y"],
                mode="markers",
                marker=dict(
                    size=8,
                    color="gray",
                    opacity=0.5
                ),
                name="Unchanged",
                hoverinfo="text",
                hovertext=[f"Doc {i}<br>Topic: {row['before_topic']}" 
                          for i, row in unchanged.iterrows()]
            ))
            
            # Add scatter plot for changed documents
            changed = df[df["changed"]]
            fig.add_trace(go.Scatter(
                x=changed["x"],
                y=changed["y"],
                mode="markers",
                marker=dict(
                    size=10,
                    color="red",
                    opacity=0.8
                ),
                name="Changed",
                hoverinfo="text",
                hovertext=[f"Doc {i}<br>Before: {row['before_topic']}<br>After: {row['after_topic']}" 
                          for i, row in changed.iterrows()]
            ))
            
            # Highlight selected document if available
            if selected_doc_idx is not None:
                selected_doc_idx = int(selected_doc_idx)
                if selected_doc_idx < len(df):
                    fig.add_trace(go.Scatter(
                        x=[df.iloc[selected_doc_idx]["x"]],
                        y=[df.iloc[selected_doc_idx]["y"]],
                        mode="markers",
                        marker=dict(
                            size=15,
                            color="yellow",
                            line=dict(
                                color="black",
                                width=2
                            )
                        ),
                        name="Selected",
                        hoverinfo="text",
                        hovertext=f"Selected Doc {selected_doc_idx}"
                    ))
            
            # Update layout
            fig.update_layout(
                title="Document Embedding Space",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
    
    return app


def plot_topic_feedback_distribution(
    topic_model: Any,
    documents: List[str],
    original_topics: List[Any],
    corrected_topics: List[Any],
    figsize: Tuple[int, int] = (12, 8),
    max_words: int = 10,
    show_wordclouds: bool = True,
):
    """
    Plot the distribution of topic feedback and show the most affected topics.
    
    Parameters
    ----------
    topic_model : Any
        Topic model instance
    documents : List[str]
        List of documents
    original_topics : List[Any]
        List of original topic assignments
    corrected_topics : List[Any]
        List of corrected topic assignments after feedback
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    max_words : int, optional
        Maximum number of words to show per topic, by default 10
    show_wordclouds : bool, optional
        Whether to include wordclouds, by default True
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualizations
    """
    # Convert topics to same type if needed
    original_topics = [str(t) for t in original_topics]
    corrected_topics = [str(t) for t in corrected_topics]
    
    # Find topics that have changed
    changed_indices = [i for i, (orig, corr) in enumerate(zip(original_topics, corrected_topics)) 
                      if orig != corr]
    
    # Calculate changes per topic
    topic_changes = {}
    for idx in changed_indices:
        orig = original_topics[idx]
        corr = corrected_topics[idx]
        
        # Decrement count for original topic
        if orig not in topic_changes:
            topic_changes[orig] = 0
        topic_changes[orig] -= 1
        
        # Increment count for corrected topic
        if corr not in topic_changes:
            topic_changes[corr] = 0
        topic_changes[corr] += 1
    
    # Sort topics by absolute change
    topics_by_impact = sorted(topic_changes.keys(), 
                             key=lambda t: abs(topic_changes[t]), 
                             reverse=True)
    
    # Get top words for each impacted topic
    topic_words = {}
    try:
        # Try to get top words from model
        for topic in topics_by_impact:
            try:
                words = topic_model.get_topic_words(topic, n_words=max_words)
                topic_words[topic] = words
            except:
                # Fall back to simple word counting for affected documents
                from collections import Counter
                import re
                
                # Get documents with this topic (either originally or corrected)
                relevant_docs = [documents[i] for i in range(len(documents)) 
                               if original_topics[i] == topic or corrected_topics[i] == topic]
                
                # Count words
                words = []
                for doc in relevant_docs:
                    words.extend(re.findall(r'\b\w+\b', doc.lower()))
                
                # Get top words
                word_counts = Counter(words)
                top_words = word_counts.most_common(max_words)
                topic_words[topic] = [w[0] for w in top_words]
    except:
        # If we can't get top words, just create empty lists
        topic_words = {t: [] for t in topics_by_impact}
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if show_wordclouds and len(topics_by_impact) > 0:
        # Use grid with wordclouds
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        # Use single plot
        ax1 = fig.add_subplot(111)
        ax2 = None
    
    # Plot topic changes
    if not topics_by_impact:
        ax1.text(0.5, 0.5, "No topic changes found", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14)
        ax1.axis('off')
    else:
        # Plot for top N most impacted topics
        top_n = min(10, len(topics_by_impact))
        top_topics = topics_by_impact[:top_n]
        
        # Create sorted bar chart
        y_pos = np.arange(len(top_topics))
        changes = [topic_changes[t] for t in top_topics]
        
        # Create horizontal bar chart with color based on value
        bars = ax1.barh(y_pos, changes, align='center')
        
        # Color bars based on values (green for positive, red for negative)
        for i, change in enumerate(changes):
            color = 'green' if change > 0 else 'red'
            bars[i].set_color(color)
        
        # Add topic labels
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"Topic {t}" for t in top_topics])
        
        # Add zero line
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels
        ax1.set_title('Topic Feedback Impact (Documents Added/Removed)')
        ax1.set_xlabel('Change in Document Count')
        
        # Add count labels
        for i, v in enumerate(changes):
            ax1.text(v + (1 if v >= 0 else -1), i, str(v), 
                    color='black', va='center')
    
    # Add wordclouds for impacted topics
    if show_wordclouds and ax2 is not None and len(topics_by_impact) > 0:
        top_n = min(6, len(topics_by_impact))
        top_topics = topics_by_impact[:top_n]
        
        # Create grid for wordclouds
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        
        gs2 = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=gs[1], wspace=0.3, hspace=0.4
        )
        
        # Create wordcloud for each topic
        for i, topic in enumerate(top_topics):
            row = i // n_cols
            col = i % n_cols
            
            ax = fig.add_subplot(gs2[row, col])
            
            # Get words and weights
            words = topic_words.get(topic, [])
            
            if not words:
                ax.text(0.5, 0.5, "No words available", 
                        horizontalalignment='center', verticalalignment='center')
                ax.axis('off')
                continue
            
            # Create simple bar chart of words
            y_pos = np.arange(len(words))
            ax.barh(y_pos, np.linspace(1, 0.1, len(words)), align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            
            # Remove x ticks
            ax.set_xticks([])
            
            # Add title with change info
            change = topic_changes[topic]
            change_text = f"+{change}" if change > 0 else f"{change}"
            ax.set_title(f"Topic {topic} ({change_text})")
    
    plt.tight_layout()
    return fig
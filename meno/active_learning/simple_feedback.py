"""Simple feedback system for topic modeling in Jupyter notebooks.

This module provides a simpler alternative to the CleanLab-based active learning,
designed to work well in Jupyter notebooks without requiring complex ipywidgets.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd
from IPython.display import display, HTML, clear_output


class SimpleFeedback:
    """Simple topic feedback system for Jupyter notebooks.
    
    This class provides an easy way to get user feedback on topic assignments
    directly in a Jupyter notebook with minimal dependencies. It uses basic
    HTML/CSS displays rather than complex ipywidgets.
    
    Parameters
    ----------
    documents : List[str]
        The text of the documents being analyzed
    topics : List[str]
        Current topic assignments for each document
    topic_names : List[str]
        List of all available topic names/labels
    topic_descriptions : Optional[List[str]]
        Optional descriptions for each topic to help users understand them
    topic_wordclouds : Optional[List[str]]
        Optional HTML for wordclouds or topic visualizations
    callback : Optional[Callable]
        Optional callback function to call after feedback is collected
    """
    
    def __init__(
        self,
        documents: List[str],
        topics: List[str],
        topic_names: List[str],
        topic_descriptions: Optional[List[str]] = None,
        topic_wordclouds: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
    ):
        """Initialize the simple feedback system."""
        self.documents = documents
        self.topics = topics
        self.topic_names = topic_names
        self.topic_descriptions = topic_descriptions if topic_descriptions else [""] * len(topic_names)
        self.topic_wordclouds = topic_wordclouds
        self.callback = callback
        
        # Initialize feedback collection
        self.feedback = {}  # document_idx -> new_topic
        self.current_document_idx = 0
        self.documents_to_review = list(range(len(documents)))
        
        # Create a dataframe for easier management
        self.df = pd.DataFrame({
            "text": self.documents,
            "original_topic": self.topics,
            "current_topic": self.topics.copy(),
        })
    

    def display_topics(self):
        """Display all available topics with descriptions."""
        html = """
        <style>
            .topic-table {
                margin-bottom: 20px;
                width: 100%;
                border-collapse: collapse;
            }
            .topic-table th, .topic-table td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .topic-table th {
                background-color: #f2f2f2;
            }
            .topic-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
        <h3>Available Topics</h3>
        <table class="topic-table">
            <tr>
                <th>ID</th>
                <th>Topic Name</th>
                <th>Description</th>
            </tr>
        """
        
        for i, (name, desc) in enumerate(zip(self.topic_names, self.topic_descriptions)):
            html += f"""
            <tr>
                <td>{i}</td>
                <td><b>{name}</b></td>
                <td>{desc}</td>
            </tr>
            """
        
        html += "</table>"
        display(HTML(html))
        
        # Display wordclouds if available
        if self.topic_wordclouds:
            wordcloud_html = """
            <style>
                .wordcloud-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .wordcloud-item {
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                    max-width: 300px;
                }
                .wordcloud-title {
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
            </style>
            <h3>Topic Visualizations</h3>
            <div class="wordcloud-container">
            """
            
            for i, (name, wordcloud) in enumerate(zip(self.topic_names, self.topic_wordclouds)):
                wordcloud_html += f"""
                <div class="wordcloud-item">
                    <div class="wordcloud-title">Topic {i}: {name}</div>
                    {wordcloud}
                </div>
                """
            
            wordcloud_html += "</div>"
            display(HTML(wordcloud_html))
    
    def get_uncertain_documents(
        self,
        topic_probs: np.ndarray,
        n_samples: int = 20,
        method: str = "entropy"
    ) -> List[int]:
        """Get indices of documents with highest uncertainty.
        
        Parameters
        ----------
        topic_probs : np.ndarray
            Topic probability matrix with shape (n_documents, n_topics)
        n_samples : int, optional
            Number of uncertain documents to return, by default 20
        method : str, optional
            Method to calculate uncertainty, by default "entropy"
            Options: "entropy", "margin", "least_confidence"
            
        Returns
        -------
        List[int]
            Indices of uncertain documents
        """
        if method == "entropy":
            # Compute entropy of probability distribution
            with np.errstate(divide="ignore", invalid="ignore"):
                log_probs = np.log(topic_probs)
                log_probs[~np.isfinite(log_probs)] = 0
                uncertainty = -np.sum(topic_probs * log_probs, axis=1)
        
        elif method == "margin":
            # Compute margin between top two probabilities
            sorted_probs = np.sort(topic_probs, axis=1)
            uncertainty = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        
        elif method == "least_confidence":
            # Use 1 - max probability
            uncertainty = 1 - np.max(topic_probs, axis=1)
        
        else:
            raise ValueError(
                f"Unknown uncertainty method: {method}. "
                f"Must be one of: entropy, margin, least_confidence"
            )
        
        # Sort by uncertainty (higher is more uncertain)
        sorted_indices = np.argsort(uncertainty)[::-1]
        
        # Return top n_samples
        return sorted_indices[:n_samples].tolist()
    
    def get_diverse_sample(
        self,
        embeddings: np.ndarray,
        n_samples: int = 20
    ) -> List[int]:
        """Get indices of diverse documents using K-means clustering.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings matrix with shape (n_documents, embedding_dim)
        n_samples : int, optional
            Number of diverse documents to return, by default 20
            
        Returns
        -------
        List[int]
            Indices of diverse documents
        """
        from sklearn.cluster import KMeans
        
        # Cluster documents
        kmeans = KMeans(
            n_clusters=min(n_samples, len(embeddings)),
            random_state=42,
            n_init=10 if 'n_init' in KMeans().get_params() else None,
        )
        kmeans.fit(embeddings)
        
        # Get closest document to each cluster center
        closest_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.sqrt(np.sum((embeddings - center) ** 2, axis=1))
            closest_idx = np.argmin(distances)
            closest_indices.append(closest_idx)
        
        return closest_indices
    
    def set_documents_to_review(self, indices: List[int]):
        """Set specific documents to review.
        
        Parameters
        ----------
        indices : List[int]
            List of document indices to review
        """
        self.documents_to_review = indices
        self.current_document_idx = 0
    
    def display_document(self, idx: int, display_dropdown: bool = True):
        """Display a single document with topic information."""
        doc = self.documents[idx]
        current_topic = self.df.at[idx, "current_topic"]
        original_topic = self.df.at[idx, "original_topic"]
        
        # Create CSS class for changed topic
        topic_class = "topic-value"
        if current_topic != original_topic:
            topic_class += " changed"
            
        # Create HTML for document display
        html = f"""
        <style>
            .document-container {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #f9f9f9;
            }}
            .document-header {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }}
            .document-text {{
                margin-bottom: 15px;
                max-height: 300px;
                overflow-y: auto;
                background-color: white;
                padding: 10px;
                border: 1px solid #eee;
                border-radius: 4px;
            }}
            .topic-info {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .topic-label {{
                font-weight: bold;
                margin-right: 10px;
                width: 150px;
            }}
            .topic-value {{
                padding: 5px;
                background-color: #e9ecef;
                border-radius: 4px;
            }}
            .changed {{
                color: #007bff;
                font-weight: bold;
            }}
        </style>
        <div class="document-container">
            <div class="document-header">
                <h3>Document {idx + 1} of {len(self.documents)}</h3>
                <div>Document Index: {idx}</div>
            </div>
            <div class="document-text">
                {doc.replace('\n', '<br>')}
            </div>
            <div class="topic-info">
                <div class="topic-label">Original Topic:</div>
                <div class="topic-value">{original_topic}</div>
            </div>
            <div class="topic-info">
                <div class="topic-label">Current Topic:</div>
                <div class="{topic_class}">
                    {current_topic}
                </div>
            </div>
        </div>
        """
        
        display(HTML(html))
        
        if display_dropdown:
            # Function to handle dropdown change
            from ipywidgets import Dropdown, Button, HBox, VBox, Output
            import ipywidgets as widgets
            
            # Create dropdown for topic selection
            dropdown = Dropdown(
                options=self.topic_names,
                value=current_topic,
                description='Select Topic:',
                style={'description_width': 'initial'},
                layout={'width': '400px'}
            )
            
            # Create save button
            save_button = Button(
                description='Save',
                button_style='primary',
                icon='check',
                layout={'width': '100px'}
            )
            
            # Create next button
            next_button = Button(
                description='Next',
                button_style='info',
                icon='arrow-right',
                layout={'width': '100px'}
            )
            
            # Create skip button
            skip_button = Button(
                description='Skip',
                button_style='warning',
                icon='fast-forward',
                layout={'width': '100px'}
            )
            
            # Create output widget for status messages
            output = Output()
            
            # Handle save button click
            def on_save_clicked(b):
                with output:
                    clear_output()
                    new_topic = dropdown.value
                    self.df.at[idx, "current_topic"] = new_topic
                    self.feedback[idx] = new_topic
                    print(f"Saved! Topic changed to: {new_topic}")
            
            # Handle next button click
            def on_next_clicked(b):
                self.current_document_idx += 1
                if self.current_document_idx >= len(self.documents_to_review):
                    self.current_document_idx = 0
                clear_output(wait=True)
                self.display_document(self.documents_to_review[self.current_document_idx])
            
            # Handle skip button click
            def on_skip_clicked(b):
                self.current_document_idx += 1
                if self.current_document_idx >= len(self.documents_to_review):
                    self.current_document_idx = 0
                clear_output(wait=True)
                self.display_document(self.documents_to_review[self.current_document_idx])
            
            # Connect buttons to functions
            save_button.on_click(on_save_clicked)
            next_button.on_click(on_next_clicked)
            skip_button.on_click(on_skip_clicked)
            
            # Display widgets
            display(VBox([
                dropdown,
                HBox([save_button, next_button, skip_button]),
                output
            ]))
    
    def start_review(self, n_samples: int = 20):
        """Start reviewing documents.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of documents to review, by default 20
        """
        if not self.documents_to_review:
            # If no documents specified, use the first n_samples
            self.documents_to_review = list(range(min(n_samples, len(self.documents))))
        
        self.current_document_idx = 0
        self.display_document(self.documents_to_review[self.current_document_idx])
    
    def display_summary(self):
        """Display a summary of the feedback collected."""
        changed_count = 0
        for idx in self.feedback:
            if self.df.at[idx, "original_topic"] != self.df.at[idx, "current_topic"]:
                changed_count += 1
        
        html = f"""
        <style>
            .summary-container {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #f2f2f2;
            }}
            .summary-header {{
                text-align: center;
                margin-bottom: 15px;
                font-size: 20px;
                font-weight: bold;
            }}
            .summary-stat {{
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
            }}
            .summary-label {{
                font-weight: bold;
            }}
            .summary-value {{
                text-align: right;
            }}
            .changed-topics {{
                margin-top: 20px;
            }}
            .topic-change-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .topic-change-table th, .topic-change-table td {{
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            .topic-change-table th {{
                background-color: #f2f2f2;
            }}
            .topic-change-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
        </style>
        <div class="summary-container">
            <div class="summary-header">Feedback Summary</div>
            <div class="summary-stat">
                <div class="summary-label">Total documents reviewed:</div>
                <div class="summary-value">{len(self.feedback)}</div>
            </div>
            <div class="summary-stat">
                <div class="summary-label">Documents with changed topics:</div>
                <div class="summary-value">{changed_count}</div>
            </div>
            <div class="summary-stat">
                <div class="summary-label">Change rate:</div>
                <div class="summary-value">{changed_count/len(self.feedback):.1%}</div>
            </div>
        """
        
        if changed_count > 0:
            html += """
            <div class="changed-topics">
                <h4>Topic Changes</h4>
                <table class="topic-change-table">
                    <tr>
                        <th>Document ID</th>
                        <th>Original Topic</th>
                        <th>New Topic</th>
                    </tr>
            """
            
            for idx in self.feedback:
                original = self.df.at[idx, "original_topic"]
                current = self.df.at[idx, "current_topic"]
                if original != current:
                    html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{original}</td>
                        <td>{current}</td>
                    </tr>
                    """
            
            html += """
                </table>
            </div>
            """
        
        html += "</div>"
        display(HTML(html))
    
    def get_updated_topics(self) -> List[str]:
        """Get the updated list of topics with user feedback incorporated.
        
        Returns
        -------
        List[str]
            Updated topic list
        """
        return self.df["current_topic"].tolist()
    
    def apply_updates(self):
        """Apply the updates to the original data and call the callback if provided."""
        if self.callback:
            self.callback(self.get_updated_topics())
        return self.get_updated_topics()
    
    def export_to_csv(self, filename: str):
        """Export the feedback to a CSV file.
        
        Parameters
        ----------
        filename : str
            Path to save the CSV file
        """
        export_df = self.df.copy()
        export_df["changed"] = export_df["original_topic"] != export_df["current_topic"]
        export_df.to_csv(filename, index=True)
        print(f"Feedback exported to {filename}")
    
    def import_from_csv(self, filename: str):
        """Import feedback from a CSV file.
        
        Parameters
        ----------
        filename : str
            Path to the CSV file
        """
        import_df = pd.read_csv(filename, index_col=0)
        # Update only the rows that are in the file
        for idx in import_df.index:
            if idx < len(self.df):
                self.df.at[idx, "current_topic"] = import_df.at[idx, "current_topic"]
                if import_df.at[idx, "original_topic"] != import_df.at[idx, "current_topic"]:
                    self.feedback[idx] = import_df.at[idx, "current_topic"]
        print(f"Imported feedback from {filename}")


class TopicFeedbackManager:
    """High-level manager for incorporating user feedback into topic models.
    
    This class provides integration with Meno's topic modeling pipeline
    and makes it easy to incorporate feedback into existing models.
    
    Parameters
    ----------
    modeler : Any
        The Meno topic modeler instance
    """
    
    def __init__(self, modeler):
        """Initialize the feedback manager with a modeler."""
        self.modeler = modeler
        self.feedback_system = None
    
    def setup_feedback(
        self,
        n_samples: int = 20,
        uncertainty_ratio: float = 0.7,
        uncertainty_method: str = "entropy",
        topic_descriptions: Optional[List[str]] = None,
        topic_wordclouds: Optional[List[str]] = None,
    ):
        """Set up the feedback system with the current model state.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of documents to review, by default 20
        uncertainty_ratio : float, optional
            Ratio of uncertain documents to diverse ones, by default 0.7
        uncertainty_method : str, optional
            Method to calculate uncertainty, by default "entropy"
        topic_descriptions : Optional[List[str]], optional
            Descriptions for each topic, by default None
        topic_wordclouds : Optional[List[str]], optional
            HTML or image data for topic wordclouds, by default None
            
        Returns
        -------
        SimpleFeedback
            The feedback system instance
        """
        # Get documents and current topics
        try:
            documents = self.modeler.get_preprocessed_documents()
        except (AttributeError, TypeError):
            try:
                documents = self.modeler.preprocessed_texts
            except (AttributeError, TypeError):
                raise ValueError("Could not find documents in the modeler. Make sure preprocessing has been run.")
        
        # Get topic assignments
        try:
            topic_info = self.modeler.get_document_info()
            topics = topic_info["Topic"].tolist()
        except (AttributeError, TypeError):
            try:
                topic_df = self.modeler.topic_df
                topics = topic_df["topic"].tolist()
            except (AttributeError, TypeError):
                raise ValueError("Could not find topic assignments in the modeler. Make sure topic discovery has been run.")
        
        # Get topic names
        try:
            topic_names = self.modeler.get_topic_names()
        except (AttributeError, TypeError):
            try:
                topic_names = self.modeler.topics
            except (AttributeError, TypeError):
                # Fall back to using unique topic values
                topic_names = sorted(list(set(topics)))
        
        # Create the feedback system
        self.feedback_system = SimpleFeedback(
            documents=documents,
            topics=topics,
            topic_names=topic_names,
            topic_descriptions=topic_descriptions,
            topic_wordclouds=topic_wordclouds,
            callback=self._update_model,
        )
        
        # Select documents to review
        if n_samples > 0:
            n_uncertainty = int(n_samples * uncertainty_ratio)
            n_diversity = n_samples - n_uncertainty
            
            uncertain_docs = []
            diverse_docs = []
            
            # Get uncertain documents if probabilities are available
            try:
                # Try to get probability matrix
                probs = self.modeler.model.transform(documents)
                uncertain_docs = self.feedback_system.get_uncertain_documents(
                    probs, n_samples=n_uncertainty, method=uncertainty_method
                )
            except (AttributeError, TypeError):
                # Fall back to random selection
                uncertain_docs = np.random.choice(
                    len(documents), size=n_uncertainty, replace=False
                ).tolist()
            
            # Get diverse documents if embeddings are available
            try:
                embeddings = self.modeler.get_document_embeddings()
                remaining_indices = [i for i in range(len(documents)) if i not in uncertain_docs]
                remaining_embeddings = embeddings[remaining_indices]
                diverse_indices = self.feedback_system.get_diverse_sample(
                    remaining_embeddings, n_samples=n_diversity
                )
                diverse_docs = [remaining_indices[i] for i in diverse_indices]
            except (AttributeError, TypeError):
                # Fall back to random selection
                all_indices = set(range(len(documents)))
                uncertain_set = set(uncertain_docs)
                remaining = list(all_indices - uncertain_set)
                diverse_docs = np.random.choice(
                    remaining, size=min(n_diversity, len(remaining)), replace=False
                ).tolist()
            
            # Combine and set the documents to review
            docs_to_review = uncertain_docs + diverse_docs
            self.feedback_system.set_documents_to_review(docs_to_review)
        
        return self.feedback_system
    
    def _update_model(self, updated_topics):
        """Update the model with the new topic assignments.
        
        Parameters
        ----------
        updated_topics : List[str]
            Updated topic assignments
        """
        # This method can be implemented based on how the specific
        # modeler should be updated with new topic assignments
        try:
            # Try to use a specific update method if available
            self.modeler.update_topic_assignments(updated_topics)
        except (AttributeError, TypeError):
            # Fall back to directly updating topic assignments
            try:
                self.modeler.topic_df["topic"] = updated_topics
            except (AttributeError, TypeError):
                try:
                    topic_info = self.modeler.get_document_info()
                    topic_info["Topic"] = updated_topics
                except (AttributeError, TypeError):
                    print("Warning: Could not automatically update the model. Manual update may be required.")
    
    def start_review(self):
        """Start the review process."""
        if self.feedback_system is None:
            raise ValueError("Feedback system not set up. Call setup_feedback() first.")
        
        # Display topics for reference
        self.feedback_system.display_topics()
        
        # Start the review process
        self.feedback_system.start_review()
    
    def get_updated_model(self):
        """Get the updated model with feedback incorporated.
        
        Returns
        -------
        Any
            The updated modeler instance
        """
        if self.feedback_system is None:
            raise ValueError("Feedback system not set up. Call setup_feedback() first.")
        
        self.feedback_system.apply_updates()
        return self.modeler
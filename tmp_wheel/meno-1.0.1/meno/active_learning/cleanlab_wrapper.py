"""Cleanlab integration for active learning in topic modeling."""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd
from IPython.display import display
import ipywidgets as widgets

try:
    import cleanlab
    from cleanlab.classification import CleanLearning
    from cleanlab.dataset import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False


class ActiveLearningManager:
    """Class for active learning with Cleanlab for topic model refinement.
    
    This class provides methods for finding documents with potentially incorrect
    topic assignments, presenting them for manual labeling, and updating the model.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings with shape (n_documents, embedding_dim)
    predicted_topics : Union[List[str], pd.Series]
        Current topic assignments for each document
    document_texts : Union[List[str], pd.Series]
        Original document texts
    topic_names : List[str]
        List of available topic names
    model_update_callback : Optional[Callable], optional
        Callback function to update the model with new labels, by default None
    
    Attributes
    ----------
    embeddings : np.ndarray
        Document embeddings
    texts : List[str]
        Document texts
    predicted_topics : List[str]
        Current topic assignments
    topic_names : List[str]
        Available topic names
    topic_to_idx : Dict[str, int]
        Mapping from topic names to indices
    idx_to_topic : Dict[int, str]
        Mapping from indices to topic names
    model_update_callback : Optional[Callable]
        Callback function to update the model
    label_issues : Optional[np.ndarray]
        Indices of documents with potential label issues
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        predicted_topics: Union[List[str], pd.Series],
        document_texts: Union[List[str], pd.Series],
        topic_names: List[str],
        model_update_callback: Optional[Callable] = None,
    ):
        """Initialize the active learning manager with document data."""
        if not CLEANLAB_AVAILABLE:
            raise ImportError(
                "Cleanlab is required for active learning. "
                "Install it with: pip install cleanlab"
            )
        
        # Convert to lists if needed
        if isinstance(predicted_topics, pd.Series):
            predicted_topics = predicted_topics.tolist()
        if isinstance(document_texts, pd.Series):
            document_texts = document_texts.tolist()
        
        # Store document data
        self.embeddings = embeddings
        self.texts = document_texts
        self.predicted_topics = predicted_topics
        
        # Create topic mappings
        self.topic_names = topic_names
        self.topic_to_idx = {topic: i for i, topic in enumerate(topic_names)}
        self.idx_to_topic = {i: topic for i, topic in enumerate(topic_names)}
        
        # Store callback
        self.model_update_callback = model_update_callback
        
        # Initialize label issues
        self.label_issues = None
        
        # Create dataframe for labeling
        self.df = pd.DataFrame({
            "text": self.texts,
            "predicted_topic": self.predicted_topics,
            "corrected_topic": self.predicted_topics.copy(),
        })
    
    def find_label_issues(
        self,
        probabilities: Optional[np.ndarray] = None,
        method: str = "confident_learning",
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Find documents with potential label issues.
        
        Parameters
        ----------
        probabilities : Optional[np.ndarray], optional
            Topic probability matrix with shape (n_documents, n_topics), by default None
            If None, uses KNN classifier to generate probabilities from embeddings
        method : str, optional
            Method for finding label issues, by default "confident_learning"
            Options: "confident_learning", "predicted_neq_given", "self_confidence"
        n_jobs : int, optional
            Number of processes for parallel processing, by default 1
        
        Returns
        -------
        np.ndarray
            Indices of documents with potential label issues
        """
        # Convert string labels to integer indices
        y = np.array([self.topic_to_idx.get(topic, -1) for topic in self.predicted_topics])
        
        # Filter out documents with unknown topics
        valid_mask = y >= 0
        valid_embeddings = self.embeddings[valid_mask]
        valid_y = y[valid_mask]
        
        # Generate probabilities if not provided
        if probabilities is None:
            from sklearn.neighbors import KNeighborsClassifier
            
            # Create and fit classifier
            classifier = KNeighborsClassifier(
                n_neighbors=min(20, len(valid_y) // len(set(valid_y))),
                weights="distance",
                n_jobs=n_jobs,
            )
            classifier.fit(valid_embeddings, valid_y)
            
            # Predict probabilities
            probabilities = classifier.predict_proba(self.embeddings)
        
        # Find label issues
        label_issues = find_label_issues(
            labels=y,
            pred_probs=probabilities,
            return_indices_ranked_by="self_confidence",
            n_jobs=n_jobs,
        )
        
        self.label_issues = label_issues
        return label_issues
    
    def create_labeling_interface(
        self,
        indices: Optional[np.ndarray] = None,
        max_examples: int = 100,
        include_confidence: bool = True,
    ) -> widgets.VBox:
        """Create an interactive labeling interface for Jupyter.
        
        Parameters
        ----------
        indices : Optional[np.ndarray], optional
            Indices of documents to label, by default None
            If None, uses label issues from find_label_issues
        max_examples : int, optional
            Maximum number of examples to display, by default 100
        include_confidence : bool, optional
            Whether to include confidence estimates, by default True
        
        Returns
        -------
        widgets.VBox
            Jupyter widget for interactive labeling
        """
        if indices is None:
            if self.label_issues is None:
                indices = np.arange(len(self.texts))
            else:
                indices = self.label_issues
        
        # Limit to max_examples
        indices = indices[:max_examples]
        
        # Create widgets for each document
        document_widgets = []
        
        for i, idx in enumerate(indices):
            # Get document text and predicted topic
            text = self.texts[idx]
            predicted_topic = self.predicted_topics[idx]
            
            # Create dropdown for topic selection
            dropdown = widgets.Dropdown(
                options=self.topic_names,
                value=predicted_topic,
                description="Topic:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="300px"),
            )
            
            # Create label for document text
            text_widget = widgets.HTML(
                value=f"<b>Document {i+1}/{len(indices)}</b> (Index: {idx}):<br><p>{text}</p>"
            )
            
            # Create container for this document
            container = widgets.VBox([
                text_widget,
                widgets.HBox([
                    dropdown,
                    widgets.Label(f"Predicted: {predicted_topic}"),
                ]),
                widgets.HTML("<hr>"),
            ])
            
            # Store index and dropdown for later access
            container.idx = idx
            container.dropdown = dropdown
            
            document_widgets.append(container)
        
        # Create save button
        save_button = widgets.Button(
            description="Save Labels",
            button_style="primary",
            icon="check",
        )
        
        # Create output widget for status messages
        output = widgets.Output()
        
        # Define save function
        def on_save_button_clicked(b):
            with output:
                output.clear_output()
                
                # Update corrected topics
                for container in document_widgets:
                    idx = container.idx
                    new_topic = container.dropdown.value
                    self.df.at[idx, "corrected_topic"] = new_topic
                
                # Count changes
                changes = sum(
                    1 for idx in indices
                    if self.df.at[idx, "predicted_topic"] != self.df.at[idx, "corrected_topic"]
                )
                
                print(f"Saved {changes} label corrections.")
                
                # Call update callback if provided
                if self.model_update_callback is not None:
                    self.model_update_callback(
                        self.df["corrected_topic"].tolist(),
                    )
                    print("Model updated with new labels.")
        
        # Connect save button to function
        save_button.on_click(on_save_button_clicked)
        
        # Create main widget
        main_widget = widgets.VBox([
            widgets.VBox(document_widgets),
            save_button,
            output,
        ])
        
        return main_widget
    
    def get_corrected_labels(self) -> List[str]:
        """Get the current corrected topic labels.
        
        Returns
        -------
        List[str]
            List of corrected topic labels
        """
        return self.df["corrected_topic"].tolist()
    
    def update_model(self) -> Any:
        """Update the model with corrected labels.
        
        Returns
        -------
        Any
            Result from the model update callback
        
        Raises
        ------
        ValueError
            If no model update callback is provided
        """
        if self.model_update_callback is None:
            raise ValueError(
                "No model update callback provided. "
                "Provide a callback function when initializing ActiveLearningManager."
            )
        
        return self.model_update_callback(
            self.get_corrected_labels(),
        )
    
    def get_uncertainty_sample(
        self,
        probabilities: np.ndarray,
        n_samples: int = 100,
        method: str = "entropy",
    ) -> np.ndarray:
        """Sample documents with high uncertainty for labeling.
        
        Parameters
        ----------
        probabilities : np.ndarray
            Topic probability matrix with shape (n_documents, n_topics)
        n_samples : int, optional
            Number of samples to return, by default 100
        method : str, optional
            Uncertainty sampling method, by default "entropy"
            Options: "entropy", "margin", "least_confidence"
        
        Returns
        -------
        np.ndarray
            Indices of documents with high uncertainty
        """
        if method == "entropy":
            # Compute entropy of probability distribution
            with np.errstate(divide="ignore", invalid="ignore"):
                log_probs = np.log(probabilities)
                log_probs[~np.isfinite(log_probs)] = 0
                uncertainty = -np.sum(probabilities * log_probs, axis=1)
        
        elif method == "margin":
            # Compute margin between top two probabilities
            sorted_probs = np.sort(probabilities, axis=1)
            uncertainty = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        
        elif method == "least_confidence":
            # Use 1 - max probability
            uncertainty = 1 - np.max(probabilities, axis=1)
        
        else:
            raise ValueError(
                f"Unknown uncertainty method: {method}. "
                f"Must be one of: entropy, margin, least_confidence"
            )
        
        # Sort by uncertainty (higher is more uncertain)
        sorted_indices = np.argsort(uncertainty)[::-1]
        
        # Return top n_samples
        return sorted_indices[:n_samples]
    
    def get_diversity_sample(
        self,
        n_samples: int = 100,
        method: str = "kmeans",
    ) -> np.ndarray:
        """Sample diverse documents for labeling.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to return, by default 100
        method : str, optional
            Diversity sampling method, by default "kmeans"
            Options: "kmeans", "random"
        
        Returns
        -------
        np.ndarray
            Indices of diverse documents
        """
        if method == "kmeans":
            from sklearn.cluster import KMeans
            
            # Cluster documents
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=42,
                n_init=10,
            )
            kmeans.fit(self.embeddings)
            
            # Get closest document to each cluster center
            closest_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.sqrt(np.sum((self.embeddings - center) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                closest_indices.append(closest_idx)
            
            return np.array(closest_indices)
        
        elif method == "random":
            # Random sampling
            return np.random.choice(
                len(self.texts),
                size=min(n_samples, len(self.texts)),
                replace=False,
            )
        
        else:
            raise ValueError(
                f"Unknown diversity method: {method}. "
                f"Must be one of: kmeans, random"
            )
    
    def get_combined_sample(
        self,
        probabilities: np.ndarray,
        n_samples: int = 100,
        uncertainty_ratio: float = 0.7,
        uncertainty_method: str = "entropy",
        diversity_method: str = "kmeans",
    ) -> np.ndarray:
        """Sample documents using combination of uncertainty and diversity.
        
        Parameters
        ----------
        probabilities : np.ndarray
            Topic probability matrix with shape (n_documents, n_topics)
        n_samples : int, optional
            Number of samples to return, by default 100
        uncertainty_ratio : float, optional
            Ratio of uncertainty samples to diversity samples, by default 0.7
        uncertainty_method : str, optional
            Uncertainty sampling method, by default "entropy"
        diversity_method : str, optional
            Diversity sampling method, by default "kmeans"
        
        Returns
        -------
        np.ndarray
            Indices of documents for labeling
        """
        # Calculate number of each type of sample
        n_uncertainty = int(n_samples * uncertainty_ratio)
        n_diversity = n_samples - n_uncertainty
        
        # Get uncertainty samples
        uncertainty_indices = self.get_uncertainty_sample(
            probabilities,
            n_samples=n_uncertainty,
            method=uncertainty_method,
        )
        
        # Get diversity samples, excluding uncertainty samples
        remaining_indices = np.array([
            i for i in range(len(self.texts))
            if i not in uncertainty_indices
        ])
        
        if len(remaining_indices) == 0:
            # If all indices are in uncertainty_indices, just return those
            return uncertainty_indices[:n_samples]
        
        # Filter embeddings for diversity sampling
        diversity_embeddings = self.embeddings[remaining_indices]
        
        # Get diversity samples on remaining documents
        if diversity_method == "kmeans":
            from sklearn.cluster import KMeans
            
            # Cluster remaining documents
            kmeans = KMeans(
                n_clusters=n_diversity,
                random_state=42,
                n_init=10,
            )
            kmeans.fit(diversity_embeddings)
            
            # Get closest document to each cluster center
            diversity_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.sqrt(np.sum((diversity_embeddings - center) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                diversity_indices.append(remaining_indices[closest_idx])
            
            diversity_indices = np.array(diversity_indices)
        
        elif diversity_method == "random":
            # Random sampling from remaining indices
            diversity_indices = np.random.choice(
                remaining_indices,
                size=min(n_diversity, len(remaining_indices)),
                replace=False,
            )
        
        else:
            raise ValueError(
                f"Unknown diversity method: {diversity_method}. "
                f"Must be one of: kmeans, random"
            )
        
        # Combine and return all samples
        return np.concatenate([uncertainty_indices, diversity_indices])
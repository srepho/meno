#!/usr/bin/env python
"""Example script demonstrating the Meno web interface.

This script shows how to launch and customize the Meno topic modeling
web interface for interactive exploration of text data.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from meno.web_interface import MenoWebApp, launch_web_interface


def basic_launch(port=8050, debug=False):
    """Launch the web interface with basic settings.
    
    Parameters
    ----------
    port : int, optional
        Port to run the web server on, by default 8050
    debug : bool, optional
        Whether to run in debug mode, by default False
    """
    print("=" * 80)
    print("Meno Topic Modeling Web Interface Example")
    print("=" * 80)
    print("\nThis example demonstrates Meno's topic modeling capabilities with a web interface.")
    print("The interface allows you to:")
    print("  - Upload text data or enter sample text directly")
    print("  - Choose from different lightweight topic models")
    print("  - Explore topics and their relationships")
    print("  - Visualize document-topic distributions")
    print("  - Search for documents related to specific topics or queries")
    print("\nModels available:")
    print("  - Simple K-Means: Uses document embeddings with K-Means clustering")
    print("  - TF-IDF K-Means: Extremely lightweight approach using TF-IDF with K-Means")
    print("  - NMF: Non-negative Matrix Factorization for topic modeling")
    print("  - LSA: Latent Semantic Analysis for topic modeling")
    print("\nOnce the server starts, open your browser at:")
    print(f"  http://localhost:{port}/")
    print("\nPress Ctrl+C to stop the server when you're done.")
    print("=" * 80)
    
    # Launch the web interface
    launch_web_interface(port=port, debug=debug)


def customized_web_app(port=8060, debug=True):
    """Create a customized web app with additional features.
    
    Parameters
    ----------
    port : int, optional
        Port to run the web server on, by default 8060
    debug : bool, optional
        Whether to run in debug mode, by default True
    """
    print("Creating customized web app...")
    
    # Create the web app
    app = MenoWebApp(port=port, debug=debug)
    
    # Access the underlying Dash app for customization
    dash_app = app.app
    
    # Example: Add a custom title to the app
    dash_app.title = "Meno Topic Explorer - Custom Edition"
    
    # Example: Modify the app to log events to console
    old_callback = app.app.callback_map['model-status.children']['callback']
    
    @dash_app.callback(*dash_app.callback_map['model-status.children']['callback_args'])
    def custom_train_model_callback(*args, **kwargs):
        print("Model training started with parameters:")
        if len(args) >= 3:
            print(f"  - Model type: {args[2]}")
            print(f"  - Number of topics: {args[3]}")
        
        # Call the original callback
        result = old_callback(*args, **kwargs)
        
        print("Model training completed!")
        return result
    
    # Run the app
    try:
        print(f"Running customized web app on port {app.port}...")
        print(f"Open your browser at: http://localhost:{port}/")
        print("Press Ctrl+C to stop the server when you're done.")
        app.run()
    finally:
        app.cleanup()


def integrated_workflow_demo(input_file=None, port=8070, debug=True):
    """Demonstrate programmatic data loading with the web interface.
    
    Parameters
    ----------
    input_file : str, optional
        Path to CSV file with text data
    port : int, optional
        Port to run the web server on, by default 8070
    debug : bool, optional
        Whether to run in debug mode, by default True
    """
    print("Starting integrated workflow demo...")
    
    # Use sample data if no input file provided
    temp_file = None
    if not input_file:
        # Create sample data
        sample_data = [
            "Customer service was excellent and the product arrived on time.",
            "The package was damaged during shipping but customer service resolved the issue.",
            "Product quality is outstanding and exceeded my expectations.",
            "The software has a steep learning curve but powerful features.",
            "Technical support was helpful in resolving my installation issues.",
            "User interface is intuitive and easy to navigate.",
            "The documentation lacks examples and could be improved.",
            "Performance is excellent even with large datasets.",
            "Pricing is competitive compared to similar products.",
            "Regular updates keep adding valuable new features."
        ]
        
        # Create a temporary CSV file
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / "sample_data.csv"
        pd.DataFrame({"text": sample_data}).to_csv(temp_file, index=False)
        
        input_file = str(temp_file)
        print(f"Created sample data file: {input_file}")
    
    # Create the web app
    app = MenoWebApp(port=port, debug=debug)
    
    # Add startup instructions for users
    original_callback = app.app.callback_map['upload-status.children']['callback']
    
    @app.app.callback(*app.app.callback_map['upload-status.children']['callback_args'])
    def modified_upload_status_callback(*args, **kwargs):
        """Add instructions to the upload status message."""
        result = original_callback(*args, **kwargs)
        
        # If no result yet, add instructions
        if not result or not isinstance(result[0], dict):
            from dash import html
            return [
                html.Div([
                    html.H5("Welcome to Meno Topic Explorer!"),
                    html.P([
                        "To get started: ",
                        html.Ol([
                            html.Li("Upload a CSV or TXT file with your documents"),
                            html.Li("Or use the sample text area below to enter text"),
                            html.Li("Then configure and train your topic model")
                        ])
                    ]),
                    html.P(f"Sample data file available at: {input_file}")
                ], className="alert alert-info")
            ] + list(result[1:])
        
        return result
    
    # Start the app
    try:
        print(f"Running integrated workflow demo on port {app.port}...")
        print(f"Sample data available at: {input_file}")
        print(f"Open your browser at: http://localhost:{port}/")
        print("Press Ctrl+C to stop the server when you're done.")
        app.run()
    finally:
        app.cleanup()
        # Clean up the temporary file if we created it
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            if os.path.exists(os.path.dirname(temp_file)):
                try:
                    os.rmdir(os.path.dirname(temp_file))
                except OSError:
                    pass  # Directory not empty


def main():
    """Parse command line arguments and run the selected example mode."""
    parser = argparse.ArgumentParser(description="Meno Web Interface Examples")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["basic", "custom", "integrated"],
        default="basic",
        help="Example mode to run (basic, custom, or integrated)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8050,
        help="Port to run the web server on (default: 8050)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="Input CSV file path (only used in integrated mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        basic_launch(port=args.port, debug=args.debug)
    elif args.mode == "custom":
        customized_web_app(port=args.port, debug=args.debug)
    elif args.mode == "integrated":
        integrated_workflow_demo(args.input, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
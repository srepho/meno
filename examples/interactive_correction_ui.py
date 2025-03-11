"""Interactive UI for text correction with domain adaptation."""

import pandas as pd
import os
import json
import argparse
from pathlib import Path
import tempfile
import webbrowser
import time
from datetime import datetime

from meno.preprocessing.acronyms import AcronymExpander
from meno.preprocessing.spelling import SpellingCorrector
from meno.nlp.domain_adapters import get_domain_adapter

# Optional dependencies
try:
    from IPython.display import HTML, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, callback, Input, Output, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


def process_text(
    text,
    domain="general",
    run_acronym_expansion=True,
    run_spelling_correction=True,
    ignore_case=True,
    learn_corrections=True,
):
    """Process text with domain-specific settings."""
    if not text:
        return "", "", []
    
    # Initialize processors
    acronym_expander = AcronymExpander(
        domain=domain if domain != "general" else None,
        ignore_case=ignore_case,
    )
    
    spelling_corrector = SpellingCorrector(
        domain=domain if domain != "general" else None,
        min_word_length=3,
        min_score=75,
        ignore_case=ignore_case,
        use_keyboard_proximity=True,
        learn_corrections=learn_corrections,
    )
    
    # Process text
    processed_text = text
    corrections = []
    
    # Apply spelling correction
    if run_spelling_correction:
        # Find misspelled words
        original_words = text.split()
        corrected_text = spelling_corrector.correct_text(text)
        corrected_words = corrected_text.split()
        
        # Track corrections
        for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
            if orig != corr:
                corrections.append({
                    "type": "spelling",
                    "original": orig,
                    "correction": corr,
                    "position": i,
                    "confidence": 0.8,  # Placeholder
                })
        
        processed_text = corrected_text
    
    # Apply acronym expansion
    if run_acronym_expansion:
        # Find acronyms
        acronyms = acronym_expander.extract_acronyms(processed_text)
        expanded_text = acronym_expander.expand_acronyms(processed_text)
        
        # Track expansions
        for acronym in acronyms:
            if f"{acronym} (" in expanded_text and acronym in acronym_expander.acronym_dict:
                expansion = acronym_expander.acronym_dict[acronym]
                corrections.append({
                    "type": "acronym",
                    "original": acronym,
                    "correction": f"{acronym} ({expansion})",
                    "position": -1,  # Not tracked
                    "confidence": 1.0,  # Dictionary-based
                })
        
        processed_text = expanded_text
    
    return processed_text, domain, corrections


def run_gradio_interface():
    """Run Gradio interface for text correction."""
    if not GRADIO_AVAILABLE:
        print("Gradio is not installed. Please install with 'pip install gradio'")
        return
    
    # Define domains
    domains = [
        "general",
        "medical",
        "technical",
        "financial",
        "legal",
    ]
    
    # Create interface
    with gr.Blocks(title="Interactive Text Correction") as app:
        gr.Markdown("# Interactive Text Correction")
        gr.Markdown("Correct spelling and expand acronyms with domain-specific knowledge")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to process...",
                    lines=5
                )
                
                with gr.Row():
                    domain_dropdown = gr.Dropdown(
                        choices=domains,
                        label="Domain",
                        value="general"
                    )
                    
                    acronym_checkbox = gr.Checkbox(
                        label="Expand Acronyms",
                        value=True
                    )
                    
                    spelling_checkbox = gr.Checkbox(
                        label="Correct Spelling",
                        value=True
                    )
                
                with gr.Row():
                    ignore_case_checkbox = gr.Checkbox(
                        label="Ignore Case",
                        value=True
                    )
                    
                    learn_checkbox = gr.Checkbox(
                        label="Learn Corrections",
                        value=True
                    )
                
                process_button = gr.Button("Process Text")
            
            with gr.Column():
                text_output = gr.Textbox(
                    label="Processed Text",
                    lines=5
                )
                
                corrections_json = gr.JSON(
                    label="Corrections Made"
                )
        
        # Define processing function
        def process_with_ui(
            text,
            domain,
            run_acronym_expansion,
            run_spelling_correction,
            ignore_case,
            learn_corrections
        ):
            processed_text, domain, corrections = process_text(
                text,
                domain=domain,
                run_acronym_expansion=run_acronym_expansion,
                run_spelling_correction=run_spelling_correction,
                ignore_case=ignore_case,
                learn_corrections=learn_corrections
            )
            
            return processed_text, corrections
        
        # Connect components
        process_button.click(
            process_with_ui,
            inputs=[
                text_input,
                domain_dropdown,
                acronym_checkbox,
                spelling_checkbox,
                ignore_case_checkbox,
                learn_checkbox
            ],
            outputs=[
                text_output,
                corrections_json
            ]
        )
        
        # Examples
        examples = [
            [
                "The CEO and CFO met to discuss the AI implementaiton in our CRM system.",
                "general",
                True,
                True,
                True,
                True
            ],
            [
                "The patient recieved the medication and was diagnosd with hypertension.",
                "medical",
                True,
                True,
                True,
                True
            ],
            [
                "The API documentaiton for the REST servis needs to be updated.",
                "technical",
                True,
                True,
                True,
                True
            ],
            [
                "The ROI of the investmant was higher than expeted according to the CFO.",
                "financial",
                True,
                True,
                True,
                True
            ],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[
                text_input,
                domain_dropdown,
                acronym_checkbox,
                spelling_checkbox,
                ignore_case_checkbox,
                learn_checkbox
            ]
        )
    
    # Launch interface
    app.launch(share=False)


def run_dash_interface():
    """Run Dash interface for text correction."""
    if not DASH_AVAILABLE:
        print("Dash is not installed. Please install with 'pip install dash dash-bootstrap-components'")
        return
    
    # Define domains
    domains = [
        {"label": "General", "value": "general"},
        {"label": "Medical/Healthcare", "value": "medical"},
        {"label": "Technical/IT", "value": "technical"},
        {"label": "Financial/Banking", "value": "financial"},
        {"label": "Legal/Compliance", "value": "legal"},
    ]
    
    # Create app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Interactive Text Correction"
    )
    
    # Define layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Interactive Text Correction"),
                html.P("Correct spelling and expand acronyms with domain-specific knowledge"),
                html.Hr(),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Input"),
                    dbc.CardBody([
                        dbc.Label("Enter text to process:"),
                        dcc.Textarea(
                            id="text-input",
                            style={"width": "100%", "height": 150},
                            placeholder="Enter text here..."
                        ),
                        html.Br(),
                        html.Br(),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Domain:"),
                                dcc.Dropdown(
                                    id="domain-dropdown",
                                    options=domains,
                                    value="general"
                                )
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Label("Options:"),
                                dbc.Checklist(
                                    id="options-checklist",
                                    options=[
                                        {"label": "Expand Acronyms", "value": "acronyms"},
                                        {"label": "Correct Spelling", "value": "spelling"},
                                        {"label": "Ignore Case", "value": "ignore_case"},
                                        {"label": "Learn Corrections", "value": "learn"}
                                    ],
                                    value=["acronyms", "spelling", "ignore_case", "learn"],
                                    inline=True
                                )
                            ], width=6)
                        ]),
                        
                        html.Br(),
                        dbc.Button("Process Text", id="process-button", color="primary")
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Output"),
                    dbc.CardBody([
                        dbc.Label("Processed Text:"),
                        dcc.Textarea(
                            id="text-output",
                            style={"width": "100%", "height": 150},
                            readOnly=True
                        ),
                        html.Br(),
                        html.Br(),
                        
                        dbc.Label("Corrections Made:"),
                        html.Div(id="corrections-output")
                    ])
                ])
            ], width=6)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Example Inputs"),
                    dbc.CardBody([
                        dbc.Button("Example 1: General", id="example-1", className="me-2"),
                        dbc.Button("Example 2: Medical", id="example-2", className="me-2"),
                        dbc.Button("Example 3: Technical", id="example-3", className="me-2"),
                        dbc.Button("Example 4: Financial", id="example-4", className="me-2")
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)
    
    # Define callbacks
    @app.callback(
        [
            Output("text-output", "value"),
            Output("corrections-output", "children")
        ],
        [
            Input("process-button", "n_clicks")
        ],
        [
            State("text-input", "value"),
            State("domain-dropdown", "value"),
            State("options-checklist", "value")
        ],
        prevent_initial_call=True
    )
    def process_callback(n_clicks, text, domain, options):
        if not text:
            return "", ""
        
        run_acronym_expansion = "acronyms" in options
        run_spelling_correction = "spelling" in options
        ignore_case = "ignore_case" in options
        learn_corrections = "learn" in options
        
        processed_text, domain, corrections = process_text(
            text,
            domain=domain,
            run_acronym_expansion=run_acronym_expansion,
            run_spelling_correction=run_spelling_correction,
            ignore_case=ignore_case,
            learn_corrections=learn_corrections
        )
        
        # Create corrections display
        if not corrections:
            corrections_display = html.P("No corrections made.")
        else:
            corrections_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Type"),
                        html.Th("Original"),
                        html.Th("Correction"),
                        html.Th("Confidence")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(c["type"].capitalize()),
                        html.Td(c["original"]),
                        html.Td(c["correction"]),
                        html.Td(f"{c['confidence']:.2f}")
                    ]) for c in corrections
                ])
            ], bordered=True, striped=True, size="sm")
            
            corrections_display = corrections_table
        
        return processed_text, corrections_display
    
    # Example callbacks
    @app.callback(
        [
            Output("text-input", "value"),
            Output("domain-dropdown", "value")
        ],
        [
            Input("example-1", "n_clicks"),
            Input("example-2", "n_clicks"),
            Input("example-3", "n_clicks"),
            Input("example-4", "n_clicks")
        ],
        prevent_initial_call=True
    )
    def set_example(ex1, ex2, ex3, ex4):
        ctx = dash.callback_context
        button_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        
        if button_id == "example-1":
            return "The CEO and CFO met to discuss the AI implementaiton in our CRM system.", "general"
        elif button_id == "example-2":
            return "The patient recieved the medication and was diagnosd with hypertension.", "medical"
        elif button_id == "example-3":
            return "The API documentaiton for the REST servis needs to be updated.", "technical"
        elif button_id == "example-4":
            return "The ROI of the investmant was higher than expeted according to the CFO.", "financial"
        else:
            return "", "general"
    
    # Run server
    app.run_server(debug=False, port=8050)


def generate_html_report(
    text,
    processed_text,
    corrections,
    domain,
    options,
    output_path=None
):
    """Generate an HTML report of the text correction."""
    # Create temp file if no output path provided
    if not output_path:
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(temp_dir, f"correction_report_{timestamp}.html")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Correction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
            .header {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
            .text-area {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; white-space: pre-wrap; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .correction {{ background-color: #e6f7ff; }}
            .acronym {{ background-color: #e6ffe6; }}
            .highlight {{ background-color: yellow; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Text Correction Report</h1>
                <p>Domain: <strong>{domain}</strong></p>
                <p>Options: {", ".join(options)}</p>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="card">
                <h2>Original Text</h2>
                <div class="text-area">{text}</div>
            </div>
            
            <div class="card">
                <h2>Processed Text</h2>
                <div class="text-area">{processed_text}</div>
            </div>
    """
    
    if corrections:
        html_content += """
            <div class="card">
                <h2>Corrections Made</h2>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Original</th>
                        <th>Correction</th>
                        <th>Confidence</th>
                    </tr>
        """
        
        for correction in corrections:
            html_content += f"""
                    <tr class="{correction['type']}">
                        <td>{correction['type'].capitalize()}</td>
                        <td>{correction['original']}</td>
                        <td>{correction['correction']}</td>
                        <td>{correction['confidence']:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return output_path


def main():
    """Main function for text correction UI."""
    parser = argparse.ArgumentParser(description="Interactive Text Correction")
    parser.add_argument("--ui", choices=["dash", "gradio", "cli"], default="cli",
                      help="UI framework to use (default: cli)")
    parser.add_argument("--text", type=str, help="Text to process (for CLI mode)")
    parser.add_argument("--domain", type=str, default="general",
                      help="Domain for processing (for CLI mode)")
    parser.add_argument("--output", type=str, help="Output path for report (for CLI mode)")
    
    args = parser.parse_args()
    
    # Run appropriate interface
    if args.ui == "dash":
        if not DASH_AVAILABLE:
            print("Dash is not installed. Falling back to CLI mode.")
            print("Install Dash with: pip install dash dash-bootstrap-components")
            args.ui = "cli"
        else:
            run_dash_interface()
            return
    
    elif args.ui == "gradio":
        if not GRADIO_AVAILABLE:
            print("Gradio is not installed. Falling back to CLI mode.")
            print("Install Gradio with: pip install gradio")
            args.ui = "cli"
        else:
            run_gradio_interface()
            return
    
    # CLI mode
    if args.ui == "cli":
        text = args.text
        
        if not text:
            text = input("Enter text to process: ")
        
        domain = args.domain
        print(f"\nDomain: {domain}")
        
        # Process with default options
        options = ["acronyms", "spelling", "ignore_case", "learn"]
        print(f"Options: {', '.join(options)}")
        
        run_acronym_expansion = "acronyms" in options
        run_spelling_correction = "spelling" in options
        ignore_case = "ignore_case" in options
        learn_corrections = "learn" in options
        
        # Process text
        processed_text, domain, corrections = process_text(
            text,
            domain=domain,
            run_acronym_expansion=run_acronym_expansion,
            run_spelling_correction=run_spelling_correction,
            ignore_case=ignore_case,
            learn_corrections=learn_corrections
        )
        
        # Print results
        print("\nOriginal Text:")
        print(text)
        
        print("\nProcessed Text:")
        print(processed_text)
        
        if corrections:
            print("\nCorrections Made:")
            print("{:<10} {:<15} {:<20} {:<10}".format("Type", "Original", "Correction", "Confidence"))
            print("-" * 60)
            
            for c in corrections:
                print("{:<10} {:<15} {:<20} {:<10.2f}".format(
                    c["type"].capitalize(),
                    c["original"],
                    c["correction"],
                    c["confidence"]
                ))
        else:
            print("\nNo corrections made.")
        
        # Generate HTML report
        output_path = args.output
        report_path = generate_html_report(
            text,
            processed_text,
            corrections,
            domain,
            options,
            output_path
        )
        
        print(f"\nReport generated at: {report_path}")
        
        # Open in browser
        print("Opening report in browser...")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
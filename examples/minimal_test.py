"""
Minimal test for Meno Workflow acronym detection and spelling correction
using a synthetic insurance claims dataset.

This example demonstrates:
1. Creating a synthetic insurance claims dataset with:
   - Acronyms that need expansion (e.g., P&C, LOB)
   - Common misspellings (e.g., policyhlder, cliam)
2. Using the MenoWorkflow class to:
   - Generate interactive HTML reports for acronyms and misspellings
   - Expand acronyms in the text
   - Correct spelling errors
3. Visualizing the results with minimal dependencies

This serves as a quick test of the core workflow functionality
without requiring the full suite of dependencies needed for topic modeling.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import random
import webbrowser
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the meno package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Try to import from the workflow module
    from meno.preprocessing.acronyms import AcronymExpander
    from meno.preprocessing.spelling import SpellingCorrector
    from meno.utils.config import load_config, WorkflowMenoConfig
    from meno.workflow import MenoWorkflow, load_workflow_config
    
    # Check if the workflow module is available
    workflow_available = True
    print("Meno workflow module is available! Using the full workflow implementation.")
except ImportError as e:
    # Fall back to a simplified implementation
    workflow_available = False
    print(f"Could not import Meno workflow: {e}")
    print("Using minimal implementation instead.")

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Create synthetic insurance claims dataset
print("\n🔍 Generating synthetic insurance claims dataset...")

# Sample business names for more realistic data
business_names = [
    "Sunshine Cafe", "Metro Auto Repair", "Lakeside Apartments", 
    "City Dental Care", "Golden Oak Furniture", "Mountain View Hotel",
    "Riverside Medical Center", "Harbor View Restaurant", "Green Valley Farm",
    "Blue Ocean Seafood", "Starlight Cinema", "Oakwood Apartments"
]

# Dates for random timestamps
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = (end_date - start_date).days

# Create some sample insurance claims text
claims_text = [
    "Customer's vehicle was damaged in a parking lot by a shopping cart. The policy covers this type of incident.",
    "Claimant's home flooded due to heavy rain. Water damage to first floor requiring extensive repairs.",
    "Vehicle collided with another car at an intersection. Front-end damage and possible injury to driver.",
    "Tree fell on roof during storm causing damage to shingles and gutters. No interior water damage reported.",
    "Insured slipped on ice in parking lot and broke wrist requiring treatment at hospital.",
    "Kitchen fire damaged cabinets and appliances. Smoke damage throughout home requiring professional cleaning.",
    "Policyholder's laptop was stolen from car while parked at shopping mall. Filed police report.",
    "Hail storm damaged roof and vehicle. Multiple claims being filed for same incident.",
    "Customer reports water leak from upstairs bathroom causing ceiling damage in living room below.",
    "Jewelry was stolen during home burglary. Items include engagement ring (insured) and watch.",
    "Rear-end collision at stop light. Customer not at fault. Other driver cited by police.",
    "Wind storm blew down fence on property line. Neighbors disputing responsibility for replacement.",
    "Theft of bicycle from garage. Door was left unlocked overnight according to insured.",
    "Customer backed into pole in parking garage. Rear bumper and taillight damage on vehicle.",
    "Dog bit visitor to home requiring stitches. Medical payments coverage being requested.",
    "Hail damage to roof of insured property. Multiple similar claims in neighborhood after storm."
]

# Create some synthetic insurance acronyms in some of the texts
claims_with_acronyms = [
    "The CTP insurance should cover the TPD claim according to our PDS.",
    "The ROI on this premium has been declining according to our KPI metrics.",
    "Our CEO instructed the CFO to review all P&C claims this quarter.",
    "The insured requested an EOB for their medical claim.",
    "The CSR processed the NCD for the customer's auto policy renewal.",
    "Please check the MOB and YOB figures for this LOB segment.",
    "The IT department is upgrading our CRM system for better customer management.",
    "The UW department rejected the application due to insufficient data.",
    "Our EBITDA has improved despite higher CAC in our insurance vertical.",
    "The MCR for this period exceeds our target according to the QA team.",
    "Please forward the COI to the client by EOD.",
    "The BI component of this claim exceeds the PD aspect.",
]

# Create some synthetic misspellings in some of the texts
misspelled_claims = [
    "The policyhlder recieved a premiume notice for their vehical insurance.",
    "The custmer filled a cliam for water damaage to their property.",
    "The agent forgott to proccess the covrage extension request.",
    "The adjstar calculated the decuction incorrectly on the claimform.",
    "The policey includes protection for personall property and liabiltiy.",
    "The insurence company denied the claim, stating the deductable was not met.",
    "The benefitt payment was delayed due to missing documention.",
    "The claim adjustor noted significnt damaage to the exterior.",
    "The policy holdr reported the theft immedietly to the authorities.",
    "The assesssment of total loses was completed last week."
]

# Combine all the texts and create a proper dataframe
all_texts = claims_text + claims_with_acronyms + misspelled_claims
claim_numbers = [f"CLM-{random.randint(10000, 99999)}" for _ in range(len(all_texts))]
policy_numbers = [f"POL-{random.randint(100000, 999999)}" for _ in range(len(all_texts))]
claim_types = ["Auto", "Property", "Liability", "Medical", "Business"] 
claim_statuses = ["Open", "In Review", "Pending", "Closed", "Denied"]
regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest"]

# Create random dates within range
claim_dates = [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(len(all_texts))]

# Create the dataframe
claims_df = pd.DataFrame({
    "claim_id": claim_numbers,
    "policy_id": policy_numbers,
    "claim_date": claim_dates,
    "claim_type": [random.choice(claim_types) for _ in range(len(all_texts))],
    "claim_status": [random.choice(claim_statuses) for _ in range(len(all_texts))],
    "region": [random.choice(regions) for _ in range(len(all_texts))],
    "business_name": [random.choice(business_names) if random.random() > 0.7 else None for _ in range(len(all_texts))],
    "claim_text": all_texts
})

# Shuffle the dataframe
claims_df = claims_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Created synthetic dataset with {len(claims_df)} claims")
print(claims_df[["claim_id", "claim_type", "claim_status", "region"]].head(3))

# Define custom acronyms and spelling corrections for insurance domain
custom_acronyms = {
    "P&C": "Property and Casualty",
    "LOB": "Line of Business",
    "UW": "Underwriting", 
    "CSR": "Customer Service Representative",
    "EOB": "Explanation of Benefits",
    "BI": "Bodily Injury",
    "PD": "Property Damage",
    "TPD": "Total and Permanent Disability",
    "CTP": "Compulsory Third Party",
    "PDS": "Product Disclosure Statement",
    "NCD": "No Claim Discount",
    "YOB": "Year of Birth",
    "MOB": "Month of Birth",
    "CAC": "Customer Acquisition Cost",
    "KPI": "Key Performance Indicator",
    "MCR": "Medical Cost Ratio",
    "COI": "Certificate of Insurance",
    "EOD": "End of Day",
    "QA": "Quality Assurance"
}

custom_spelling_corrections = {
    "policyhlder": "policyholder",
    "recieved": "received",
    "premiume": "premium",
    "vehical": "vehicle",
    "custmer": "customer",
    "cliam": "claim",
    "damaage": "damage",
    "forgott": "forgot",
    "proccess": "process", 
    "covrage": "coverage",
    "adjstar": "adjuster",
    "decuction": "deduction",
    "claimform": "claim form",
    "policey": "policy",
    "personall": "personal",
    "liabiltiy": "liability",
    "insurence": "insurance",
    "deductable": "deductible",
    "benefitt": "benefit",
    "documention": "documentation",
    "adjustor": "adjuster",
    "significnt": "significant",
    "holdr": "holder",
    "immedietly": "immediately",
    "assesssment": "assessment",
    "loses": "losses"
}

# Try to use the MenoWorkflow if available, otherwise use the minimal implementation
if workflow_available:
    print("\n🔄 Using Meno Workflow implementation...")
    
    try:
        # Initialize the workflow
        workflow = MenoWorkflow()
        
        # Load the data
        workflow.load_data(
            data=claims_df,
            text_column="claim_text",
            time_column="claim_date",
            category_column="claim_type",
            geo_column="region"
        )
        
        # Add custom acronyms and spelling corrections
        workflow.acronym_expander.add_acronyms(custom_acronyms)
        workflow.spelling_corrector.add_corrections(custom_spelling_corrections)
        
        # Generate acronym report
        print("\n📊 Generating acronym report...")
        acronym_report_path = workflow.generate_acronym_report(
            min_length=2,
            min_count=1,
            output_path=str(output_dir / "minimal_acronyms.html")
        )
        print(f"Acronym report generated at {acronym_report_path}")
        
        # Generate misspelling report
        print("\n📝 Generating misspelling report...")
        misspelling_report_path = workflow.generate_misspelling_report(
            min_length=4,
            min_count=1,
            output_path=str(output_dir / "minimal_misspellings.html")
        )
        print(f"Misspelling report generated at {misspelling_report_path}")
        
        # Apply corrections to the text
        print("\n🛠️ Applying corrections to the text...")
        workflow.expand_acronyms()
        workflow.correct_spelling()
        
        # Show sample of processed text
        print("\nSample of processed text:")
        for i, text in enumerate(workflow.documents["claim_text"].head(3)):
            print(f"[{i+1}] {text[:150]}..." if len(text) > 150 else f"[{i+1}] {text}")
            
        # Open reports in browser
        print("\nOpening reports in browser...")
        webbrowser.open(f'file://{os.path.abspath(acronym_report_path)}')
        webbrowser.open(f'file://{os.path.abspath(misspelling_report_path)}')
        
    except Exception as e:
        print(f"Error using workflow: {str(e)}")
        workflow_available = False
        print("Falling back to minimal implementation...")

# Minimal implementation (used if workflow_available is False)
if not workflow_available:
    print("\n🔄 Using minimal implementation...")
    
    # Create simple acronym and spelling detector functions
    def detect_acronyms(texts, min_length=2, min_count=1):
        """Detect potential acronyms in the texts."""
        # Simple regex for acronyms (all caps with at least min_length characters)
        acronym_pattern = re.compile(r'\b[A-Z][A-Z0-9&]{' + str(min_length - 1) + r',}\b')
        
        # Extract acronyms from all texts
        all_acronyms = []
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
            acronyms = acronym_pattern.findall(text)
            all_acronyms.extend(acronyms)
        
        # Count frequencies
        acronym_counts = {}
        for acronym in all_acronyms:
            if acronym in acronym_counts:
                acronym_counts[acronym] += 1
            else:
                acronym_counts[acronym] = 1
        
        # Filter by minimum count
        filtered_counts = {k: v for k, v in acronym_counts.items() if v >= min_count}
        
        # Sort by count (descending)
        sorted_counts = dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_counts
    
    def detect_potential_misspellings(texts, min_length=4, min_count=1):
        """Detect potential misspellings in the texts."""
        # This is a very simplified detection that just looks for words
        # that are not in a basic dictionary
        
        # Create a very simple "dictionary" of common words
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
            "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "vehicle", "damage", "insurance", "claim", "policy", "customer", "insured",
            "home", "property", "payment", "premium", "coverage", "auto", "accident", 
            "report", "injury", "medical", "liability", "adjuster", "deductible",
            "benefits", "theft", "fire", "water", "collision", "storm", "health",
            "life", "business", "agent", "company", "underwriting", "renewal",
            "department", "system", "management", "customer", "data", "application"
        ]
        
        # Extract all words from the text
        all_words = []
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Filter rare and short words
        filtered_words = {
            word: count for word, count in word_counts.items() 
            if count >= min_count and len(word) >= min_length and word not in common_words
        }
        
        # Convert to the expected format
        misspellings = {}
        for word, count in filtered_words.items():
            misspellings[word] = [("(unknown)", 0)]
        
        # Sort by count and limit
        sorted_misspellings = {
            k: v for k, v in sorted(
                misspellings.items(),
                key=lambda x: word_counts[x[0]],
                reverse=True
            )
        }
        
        return sorted_misspellings
    
    def generate_acronym_html_report(acronym_counts, texts):
        """Generate a simple HTML report for acronym review."""
        import html
        
        # Generate sample contexts
        acronym_contexts = {}
        for acronym in acronym_counts:
            # Get up to 3 sample texts containing the acronym
            samples = []
            pattern = r'\b' + acronym + r'\b'
            for text in texts:
                if pd.isna(text) or not isinstance(text, str):
                    continue
                if len(samples) >= 3:
                    break
                if re.search(pattern, text):
                    # Truncate long texts
                    if len(text) > 200:
                        text = text[:200] + "..."
                    samples.append(text)
            acronym_contexts[acronym] = samples
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Insurance Acronym Detection Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #3498db;
                    margin-top: 30px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 30px;
                }
                th, td {
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                    color: #2c3e50;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .context {
                    font-style: italic;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                    font-size: 0.9em;
                }
                .count {
                    font-weight: bold;
                    color: #e74c3c;
                }
                .known {
                    background-color: #e8f8f5;
                }
                .expansion {
                    font-weight: normal;
                    color: #27ae60;
                    margin-left: 10px;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <h1>Insurance Acronym Detection Report</h1>
            
            <h2>Detected Acronyms</h2>
            <p>Found """ + str(len(acronym_counts)) + """ potential acronyms in the insurance claims dataset.</p>
            
            <table>
                <tr>
                    <th>Acronym</th>
                    <th>Count</th>
                    <th>Sample Contexts</th>
                    <th>Suggested Expansion</th>
                </tr>
        """
        
        # Add rows for each acronym
        for acronym, count in acronym_counts.items():
            # Format contexts
            contexts_html = ""
            if acronym_contexts[acronym]:
                for context in acronym_contexts[acronym]:
                    contexts_html += f'<div class="context">{html.escape(context)}</div>'
            else:
                contexts_html = "<em>No context examples available</em>"
            
            # Is this a known acronym?
            is_known = acronym in custom_acronyms
            row_class = ' class="known"' if is_known else ''
            expansion_html = f'<span class="expansion">{custom_acronyms[acronym]}</span>' if is_known else '<em>Unknown</em>'
            
            # Add row
            html_content += f"""
                <tr{row_class}>
                    <td>{html.escape(acronym)}</td>
                    <td class="count">{count}</td>
                    <td>{contexts_html}</td>
                    <td>{expansion_html}</td>
                </tr>
            """
        
        # Close HTML
        html_content += """
            </table>
            
            <p>This report was generated using the Meno minimal implementation for acronym detection.</p>
        </body>
        </html>
        """
        
        return html_content
    
    def generate_misspelling_html_report(misspellings, texts):
        """Generate a simple HTML report for misspelling review."""
        import html
        
        # Generate sample contexts
        misspelling_contexts = {}
        for word in misspellings:
            # Get up to 3 sample texts containing the word
            samples = []
            pattern = r'\b' + word + r'\b'
            for text in texts:
                if pd.isna(text) or not isinstance(text, str):
                    continue
                if len(samples) >= 3:
                    break
                if re.search(pattern, text, re.IGNORECASE):
                    # Truncate long texts
                    if len(text) > 200:
                        text = text[:200] + "..."
                    samples.append(text)
            misspelling_contexts[word] = samples
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Insurance Spelling Correction Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #3498db;
                    margin-top: 30px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 30px;
                }
                th, td {
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                    color: #2c3e50;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .context {
                    font-style: italic;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                    font-size: 0.9em;
                }
                .misspelled {
                    font-weight: bold;
                    color: #e74c3c;
                }
                .known {
                    background-color: #e8f8f5;
                }
                .correction {
                    font-weight: normal;
                    color: #27ae60;
                    margin-left: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Insurance Spelling Correction Report</h1>
            
            <h2>Detected Potential Misspellings</h2>
            <p>Found """ + str(len(misspellings)) + """ potential misspellings in the insurance claims dataset.</p>
            
            <table>
                <tr>
                    <th>Potential Misspelling</th>
                    <th>Sample Contexts</th>
                    <th>Suggested Correction</th>
                </tr>
        """
        
        # Add rows for each misspelling
        for word in misspellings:
            # Format contexts
            contexts_html = ""
            if misspelling_contexts[word]:
                for context in misspelling_contexts[word]:
                    # Highlight the misspelled word
                    highlighted_context = re.sub(
                        r'\b' + re.escape(word) + r'\b',
                        f'<span class="misspelled">{word}</span>',
                        html.escape(context),
                        flags=re.IGNORECASE
                    )
                    contexts_html += f'<div class="context">{highlighted_context}</div>'
            else:
                contexts_html = "<em>No context examples available</em>"
            
            # Is this a known misspelling?
            is_known = word in custom_spelling_corrections
            row_class = ' class="known"' if is_known else ''
            correction_html = f'<span class="correction">{custom_spelling_corrections[word]}</span>' if is_known else '<em>Unknown</em>'
            
            # Add row
            html_content += f"""
                <tr{row_class}>
                    <td>{html.escape(word)}</td>
                    <td>{contexts_html}</td>
                    <td>{correction_html}</td>
                </tr>
            """
        
        # Close HTML
        html_content += """
            </table>
            
            <p>This report was generated using the Meno minimal implementation for spelling correction.</p>
        </body>
        </html>
        """
        
        return html_content
    
    # Detect acronyms
    print("\n📊 Detecting acronyms...")
    acronym_counts = detect_acronyms(claims_df["claim_text"], min_length=2, min_count=1)
    print(f"Detected {len(acronym_counts)} acronyms")
    
    # Generate acronym report
    print("Generating acronym report...")
    acronym_report = generate_acronym_html_report(acronym_counts, claims_df["claim_text"])
    acronym_report_path = output_dir / "minimal_acronyms.html"
    with open(acronym_report_path, 'w', encoding='utf-8') as f:
        f.write(acronym_report)
    print(f"Acronym report generated at {acronym_report_path}")
    
    # Detect misspellings
    print("\n📝 Detecting potential misspellings...")
    misspellings = detect_potential_misspellings(claims_df["claim_text"], min_length=3, min_count=1)
    print(f"Detected {len(misspellings)} potential misspellings")
    
    # Generate misspelling report
    print("Generating misspelling report...")
    misspelling_report = generate_misspelling_html_report(misspellings, claims_df["claim_text"])
    misspelling_report_path = output_dir / "minimal_misspellings.html"
    with open(misspelling_report_path, 'w', encoding='utf-8') as f:
        f.write(misspelling_report)
    print(f"Misspelling report generated at {misspelling_report_path}")
    
    # Open reports in browser
    print("\nOpening reports in browser...")
    webbrowser.open(f'file://{os.path.abspath(acronym_report_path)}')
    webbrowser.open(f'file://{os.path.abspath(misspelling_report_path)}')

print("\n✅ Minimal test complete!")
print("""
The generated reports demonstrate:
1. Interactive acronym detection with context examples
2. Interactive spelling correction with context examples
3. Graceful fallback to minimal implementation if dependencies are missing

These html reports can be shared with domain experts to:
- Improve organization-specific acronym dictionaries
- Fix common misspellings in domain text
- Document industry terminology for team knowledge sharing

The workflow can be extended with topic modeling, visualization, and more.
""")
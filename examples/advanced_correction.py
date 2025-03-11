"""Advanced text correction example with domain adaptation and metrics."""

import pandas as pd
import os
import json
from pathlib import Path

from meno.preprocessing.acronyms import AcronymExpander
from meno.preprocessing.spelling import SpellingCorrector
from meno.nlp.context_processor import ContextualProcessor, process_context
from meno.nlp.domain_adapters import get_domain_adapter
from meno.nlp.evaluation import CorrectionEvaluator, evaluate_spelling_correction

# Sample data
sample_data = [
    "The CEO and CFO met to discuss the AI implementaiton in our CRM system.",
    "Customer submited a clam for their vehical accident on HWY 101.",
    "The CTO presented the ML stategy for improving cust retention.",
    "Policyholder recieved the EOB and was confusd about the CPT codes.",
    "The FDA aproved the new drug after clinical trails showed improvment.",
    "The API documentaiton for the REST servis needs to be updated.",
    "The WHO reccommended new guidlines for healthcare workrs.",
    "The CRO discussed the ROI of the new marekting campain.",
]

# Create output directory
output_dir = Path("./output/advanced_correction")
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.DataFrame({"text": sample_data})
print(f"Loaded {len(df)} sample texts")

# Define domains for different examples
text_domains = {
    0: "general",     # CEO/CFO/AI/CRM
    1: "insurance",   # claim, vehicle, accident
    2: "tech",        # CTO, ML, customer retention
    3: "insurance",   # policyholder, EOB, CPT codes
    4: "healthcare",  # FDA, drug, clinical trials
    5: "tech",        # API, REST service, documentation
    6: "healthcare",  # WHO, guidelines, healthcare workers
    7: "marketing",   # CRO, ROI, marketing campaign
}

# Add domains column to DataFrame
df["domain"] = df.index.map(lambda i: text_domains.get(i, "general"))
print("Domains assigned:", df["domain"].value_counts().to_dict())

# Create domain-specific adapters
healthcare_adapter = get_domain_adapter("healthcare")
tech_adapter = get_domain_adapter("technical")

# Create a standard spelling corrector with common misspellings
standard_corrector = SpellingCorrector(
    min_word_length=3,
    min_score=75,
)

# Create a domain-enhanced spelling corrector
domain_corrector = SpellingCorrector(
    min_word_length=3,
    min_score=75,
    use_keyboard_proximity=True,
    learn_corrections=True,
)

# Create acronym expanders
standard_expander = AcronymExpander()
healthcare_expander = AcronymExpander(domain="healthcare")
tech_expander = AcronymExpander(domain="tech")

# Initialize contextual processor if dependencies are available
try:
    context_processor = ContextualProcessor(use_spacy=True, use_transformers=False)
    context_available = True
    print("Context processor initialized successfully")
except ImportError:
    context_available = False
    print("Context processor not available - SpaCy not installed")

# Process each text with domain-specific settings
results = []

for idx, row in df.iterrows():
    text = row["text"]
    domain = row["domain"]
    
    # Create result object
    result = {
        "original": text,
        "domain": domain,
        "standard_spelling": "",
        "domain_spelling": "",
        "standard_acronyms": "",
        "domain_acronyms": "",
        "context_enhanced": "",
    }
    
    # 1. Standard spelling correction
    result["standard_spelling"] = standard_corrector.correct_text(text)
    
    # 2. Domain-specific spelling correction
    if domain == "healthcare":
        domain_corrector = SpellingCorrector(
            domain="medical",
            min_word_length=3,
            use_keyboard_proximity=True,
        )
    elif domain == "tech":
        domain_corrector = SpellingCorrector(
            domain="technical",
            min_word_length=3,
            use_keyboard_proximity=True,
        )
    
    result["domain_spelling"] = domain_corrector.correct_text(text)
    
    # 3. Standard acronym expansion
    result["standard_acronyms"] = standard_expander.expand_acronyms(text)
    
    # 4. Domain-specific acronym expansion
    if domain == "healthcare":
        result["domain_acronyms"] = healthcare_expander.expand_acronyms(text)
    elif domain == "tech":
        result["domain_acronyms"] = tech_expander.expand_acronyms(text)
    else:
        result["domain_acronyms"] = standard_expander.expand_acronyms(text)
    
    # 5. Context-enhanced processing if available
    if context_available:
        try:
            # Apply context-aware correction
            context_info = process_context(text, use_spacy=True)
            
            # Extract acronyms from context
            extracted_acronyms = context_info.get("acronyms", {})
            
            # Use extracted acronyms to enhance expansion
            custom_expander = AcronymExpander(custom_mappings=extracted_acronyms)
            corrected_text = domain_corrector.correct_text(text)
            result["context_enhanced"] = custom_expander.expand_acronyms(corrected_text)
        except Exception as e:
            print(f"Error in context processing: {e}")
            result["context_enhanced"] = "Error in context processing"
    
    results.append(result)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_path = output_dir / "correction_results.csv"
results_df.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")

# Create evaluation dataset with "correct" versions of the texts
correct_texts = [
    "The CEO and CFO met to discuss the AI implementation in our CRM (Customer Relationship Management) system.",
    "Customer submitted a claim for their vehicle accident on HWY 101.",
    "The CTO (Chief Technology Officer) presented the ML (Machine Learning) strategy for improving customer retention.",
    "Policyholder received the EOB (Explanation of Benefits) and was confused about the CPT (Current Procedural Terminology) codes.",
    "The FDA (Food and Drug Administration) approved the new drug after clinical trials showed improvement.",
    "The API (Application Programming Interface) documentation for the REST service needs to be updated.",
    "The WHO (World Health Organization) recommended new guidelines for healthcare workers.",
    "The CRO (Chief Revenue Officer) discussed the ROI (Return on Investment) of the new marketing campaign.",
]

# Add to DataFrame
results_df["correct_text"] = correct_texts

# Save enhanced results
results_path = output_dir / "correction_results_with_reference.csv"
results_df.to_csv(results_path, index=False)

# Perform evaluation
print("\nPerforming evaluation...")

# Create evaluator
evaluator = CorrectionEvaluator()

# Evaluate standard spelling correction
standard_metrics = evaluate_spelling_correction(
    original_texts=results_df["original"],
    corrected_texts=results_df["standard_spelling"],
    reference_texts=results_df["correct_text"],
    output_path=output_dir / "standard_spelling_metrics.json"
)

# Evaluate domain spelling correction
domain_metrics = evaluate_spelling_correction(
    original_texts=results_df["original"],
    corrected_texts=results_df["domain_spelling"],
    reference_texts=results_df["correct_text"],
    output_path=output_dir / "domain_spelling_metrics.json"
)

# Print metrics summary
print("\nStandard Spelling Correction Metrics:")
for key, value in standard_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

print("\nDomain-Specific Spelling Correction Metrics:")
for key, value in domain_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

if context_available:
    # Evaluate context-enhanced correction
    context_metrics = evaluate_spelling_correction(
        original_texts=results_df["original"],
        corrected_texts=results_df["context_enhanced"],
        reference_texts=results_df["correct_text"],
        output_path=output_dir / "context_metrics.json"
    )
    
    print("\nContext-Enhanced Correction Metrics:")
    for key, value in context_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Generate comparison of all methods
methods_comparison = {
    "standard": results_df["standard_spelling"].tolist(),
    "domain": results_df["domain_spelling"].tolist(),
    "acronym": results_df["standard_acronyms"].tolist(),
    "domain_acronym": results_df["domain_acronyms"].tolist(),
}

if context_available:
    methods_comparison["context"] = results_df["context_enhanced"].tolist()

# Save comparison data
with open(output_dir / "methods_comparison.json", "w") as f:
    # Convert to list of dictionaries for JSON serialization
    comparison_data = []
    for i, original in enumerate(results_df["original"].tolist()):
        item = {"id": i, "original": original, "reference": correct_texts[i]}
        for method, texts in methods_comparison.items():
            item[method] = texts[i]
        comparison_data.append(item)
    
    json.dump(comparison_data, f, indent=2)

print(f"\nComparison data saved to {output_dir / 'methods_comparison.json'}")
print("\nAdvanced correction example completed successfully!")
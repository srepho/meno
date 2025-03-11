"""Cross-document learning for improved preprocessing."""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

from meno.preprocessing.acronyms import AcronymExpander
from meno.preprocessing.spelling import SpellingCorrector
from meno.nlp.context_processor import ContextualProcessor
from meno.nlp.evaluation import evaluate_acronym_expansion, evaluate_spelling_correction

# Check for optional dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("SpaCy not available. Installing is recommended for better results.")
    print("Install with: pip install spacy")
    print("Download model with: python -m spacy download en_core_web_sm")

# Create output directory
output_dir = Path("./output/cross_document")
os.makedirs(output_dir, exist_ok=True)

# Sample corpus with different domains
sample_corpus = [
    # Medical documents
    "The patient was diagnosed with HTN and DM. Their BP was 140/90 mmHg.",
    "She recieved antiobiotics for a UTI and was shecduled for a follow-up.",
    "The MD ordered CBC and BMP labs. Pt was tachycardic post-op.",
    "CHF patient presented with SOB and was admited to the CCU.",
    "The patient under went abdominal surgery and develop a post-op infection.",
    
    # Technical documents
    "The API integration with the DB failed due to a DNS issue.",
    "Our ML algorithm achieved 92% acuracy on the test dataset.",
    "The front-end devs implemented the UI based on the UX team's design.",
    "The sysadmin updated the VM's OS to fix the securty vulnerability.",
    "We need to refactr the codebase to improve maintenability.",
    
    # Financial documents
    "The ROI on the investment exceeded our projections for Q2.",
    "The CFO presented the YoY growth figures to the board.",
    "The bank lowered its APR for mortgages due to the decreasing interest rates.",
    "The company's EBITDA improved by 15% after the acquistion.",
    "Investors recieved dividends as per the company's policy."
]

# Create DataFrame with domain labels
domains = ["medical"] * 5 + ["technical"] * 5 + ["financial"] * 5
corpus_df = pd.DataFrame({"text": sample_corpus, "domain": domains})

print(f"Sample corpus created with {len(corpus_df)} documents")
print("Domain distribution:")
print(corpus_df["domain"].value_counts())

# Initialize context processor for corpus learning
if SPACY_AVAILABLE:
    try:
        context_processor = ContextualProcessor(use_spacy=True)
        print("Context processor initialized successfully")
    except Exception as e:
        context_processor = None
        print(f"Error initializing context processor: {e}")
else:
    context_processor = None
    print("Context processor not available (SpaCy not installed)")

# Phase 1: Learn from corpus
learned_acronyms = {}
learned_misspellings = {}
domain_specific_terms = {}

print("\n--- Phase 1: Learning from corpus ---")

# Learn from each domain separately
for domain in corpus_df["domain"].unique():
    print(f"\nProcessing {domain} documents...")
    
    # Get domain documents
    domain_docs = corpus_df[corpus_df["domain"] == domain]["text"].tolist()
    
    # Learn domain-specific information from corpus
    if context_processor:
        print(f"Learning from {len(domain_docs)} {domain} documents with context processor...")
        
        learned_info = context_processor.learn_from_corpus(
            corpus=domain_docs,
            output_path=output_dir / f"{domain}_learned_info.json"
        )
        
        # Extract learned acronyms
        if "acronyms" in learned_info:
            domain_acronyms = {}
            for acronym, expansions in learned_info["acronyms"].items():
                # Take most frequent expansion
                if expansions:
                    best_expansion = max(expansions.items(), key=lambda x: x[1])[0]
                    domain_acronyms[acronym] = best_expansion
            
            if domain_acronyms:
                learned_acronyms[domain] = domain_acronyms
                print(f"Learned {len(domain_acronyms)} acronyms for {domain} domain")
        
        # Extract domain terminology
        if "terminology" in learned_info:
            domain_specific_terms[domain] = learned_info["terminology"]
            print(f"Learned {len(domain_specific_terms[domain])} terms for {domain} domain")
    
    # Basic learning without context processor
    else:
        print("Basic learning without context processor...")
        
        # Create standard acronym expander
        acronym_expander = AcronymExpander()
        
        # Extract acronyms
        all_acronyms = {}
        for doc in domain_docs:
            acronyms = acronym_expander.extract_acronyms_batch([doc])
            for acronym, count in acronyms.items():
                if acronym not in all_acronyms:
                    all_acronyms[acronym] = 0
                all_acronyms[acronym] += count
        
        # Filter to frequent acronyms
        if all_acronyms:
            threshold = 1  # At least 2 occurrences
            domain_acronyms = {k: v for k, v in all_acronyms.items() if v > threshold}
            learned_acronyms[domain] = domain_acronyms
            print(f"Extracted {len(domain_acronyms)} potential acronyms for {domain} domain")

# Phase 2: Find misspellings
print("\n--- Phase 2: Finding potential misspellings ---")

# Initialize spelling corrector
spelling_corrector = SpellingCorrector(min_word_length=3, min_score=75)

# Find potential misspellings in each domain
for domain in corpus_df["domain"].unique():
    print(f"\nAnalyzing {domain} documents for misspellings...")
    
    # Get domain documents
    domain_docs = corpus_df[corpus_df["domain"] == domain]["text"].tolist()
    
    # Extract all words
    all_words = []
    for doc in domain_docs:
        words = [w for w in doc.split() if len(w) >= 3 and w.isalpha()]
        all_words.extend(words)
    
    # Find potential misspellings
    potential_misspellings = {}
    for word in set(all_words):
        corrected = spelling_corrector.correct_word(word)
        if word != corrected:
            potential_misspellings[word] = corrected
    
    if potential_misspellings:
        learned_misspellings[domain] = potential_misspellings
        print(f"Found {len(potential_misspellings)} potential misspellings in {domain} domain")

# Save learned information
learned_info = {
    "acronyms": learned_acronyms,
    "misspellings": learned_misspellings,
    "terminology": domain_specific_terms
}

learned_path = output_dir / "learned_information.json"
with open(learned_path, "w") as f:
    json.dump(learned_info, f, indent=2)

print(f"\nLearned information saved to {learned_path}")

# Phase 3: Apply learned information to new documents
print("\n--- Phase 3: Applying learned information to new documents ---")

# Create new test documents
test_documents = [
    # Medical test
    "The pt was dx with HTN and had elevated BP in the ER.",
    
    # Technical test
    "The dev team refactord the API to improve perfomance.",
    
    # Financial test
    "The ROI calculation showed incresed returns for Q3."
]

test_domains = ["medical", "technical", "financial"]
test_df = pd.DataFrame({"text": test_documents, "domain": test_domains})

print(f"Created {len(test_df)} test documents for evaluation")

# Process with and without learned information
results = []

for idx, row in test_df.iterrows():
    text = row["text"]
    domain = row["domain"]
    
    # Process without learned information (baseline)
    standard_expander = AcronymExpander()
    standard_corrector = SpellingCorrector()
    
    baseline_spelling = standard_corrector.correct_text(text)
    baseline_acronyms = standard_expander.expand_acronyms(baseline_spelling)
    
    # Process with learned information
    domain_acronyms = learned_acronyms.get(domain, {})
    domain_misspellings = learned_misspellings.get(domain, {})
    
    # Create enhanced processors
    enhanced_expander = AcronymExpander(custom_mappings=domain_acronyms)
    enhanced_corrector = SpellingCorrector(dictionary=domain_misspellings)
    
    enhanced_spelling = enhanced_corrector.correct_text(text)
    enhanced_text = enhanced_expander.expand_acronyms(enhanced_spelling)
    
    # Save results
    results.append({
        "domain": domain,
        "original": text,
        "baseline": baseline_acronyms,
        "enhanced": enhanced_text
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save results
results_path = output_dir / "processing_results.csv"
results_df.to_csv(results_path, index=False)

print(f"Results saved to {results_path}")

# Display comparison
print("\nComparison of baseline vs. enhanced processing:")
for idx, row in results_df.iterrows():
    print(f"\nDomain: {row['domain']}")
    print(f"Original: {row['original']}")
    print(f"Baseline: {row['baseline']}")
    print(f"Enhanced: {row['enhanced']}")

# Phase 4: Create reference data for evaluation
print("\n--- Phase 4: Creating reference data for evaluation ---")

# Create reference corrections
reference_corrections = {
    # Medical corrections
    "pt": "patient",
    "dx": "diagnosed",
    "HTN": "hypertension",
    "BP": "blood pressure",
    "ER": "emergency room",
    
    # Technical corrections
    "dev": "developer",
    "refactord": "refactored",
    "API": "Application Programming Interface",
    "perfomance": "performance",
    
    # Financial corrections
    "ROI": "Return on Investment",
    "incresed": "increased",
    "Q3": "third quarter"
}

# Create reference texts
reference_texts = [
    "The patient was diagnosed with hypertension (HTN) and had elevated blood pressure (BP) in the emergency room (ER).",
    "The developer team refactored the Application Programming Interface (API) to improve performance.",
    "The Return on Investment (ROI) calculation showed increased returns for third quarter (Q3)."
]

# Add reference data to results
results_df["reference"] = reference_texts

# Save enhanced results
reference_path = output_dir / "processing_results_with_reference.csv"
results_df.to_csv(reference_path, index=False)

print(f"Reference data created and saved to {reference_path}")

# Phase 5: Evaluate results
print("\n--- Phase 5: Evaluating results ---")

# Evaluate baseline vs. enhanced
baseline_metrics = evaluate_spelling_correction(
    original_texts=results_df["original"],
    corrected_texts=results_df["baseline"],
    reference_texts=results_df["reference"],
    reference_corrections=reference_corrections,
    output_path=output_dir / "baseline_metrics.json"
)

enhanced_metrics = evaluate_spelling_correction(
    original_texts=results_df["original"],
    corrected_texts=results_df["enhanced"],
    reference_texts=results_df["reference"],
    reference_corrections=reference_corrections,
    output_path=output_dir / "enhanced_metrics.json"
)

# Print comparison
print("\nBaseline Metrics:")
for key, value in baseline_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

print("\nEnhanced Metrics (with cross-document learning):")
for key, value in enhanced_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Phase 6: Create Meno-compatible dictionaries
print("\n--- Phase 6: Creating Meno-compatible dictionaries ---")

# Create dictionary directory
dict_dir = output_dir / "dictionaries"
os.makedirs(dict_dir, exist_ok=True)

# Save domain-specific dictionaries
for domain, acronyms in learned_acronyms.items():
    # Save acronyms
    acr_path = dict_dir / f"{domain}_acronyms.json"
    with open(acr_path, "w") as f:
        json.dump(acronyms, f, indent=2)
    
    # Save misspellings if available
    if domain in learned_misspellings:
        mis_path = dict_dir / f"{domain}_misspellings.json"
        with open(mis_path, "w") as f:
            json.dump(learned_misspellings[domain], f, indent=2)
    
    print(f"Created {domain} dictionaries")

print(f"\nDictionaries saved to {dict_dir}")
print("\nCross-document learning example completed successfully!")
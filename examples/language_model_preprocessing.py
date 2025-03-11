"""Example for using the language model enhanced preprocessing components.

This example demonstrates how to use the LMPreprocessor class to perform
context-aware acronym expansion and spelling correction using language models.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path to allow running this script independently
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from meno.preprocessing.lm_enhanced import LMPreprocessor, LMAcronymExpander, LMSpellingCorrector


def simple_example():
    """Simple example of using LMPreprocessor for a single text."""
    # Initialize preprocessor with tiny model for fast inference
    preprocessor = LMPreprocessor(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Small fast model
        use_gpu=False,  # Use CPU for example
        domain="tech",  # Include tech-specific dictionaries
    )
    
    # Sample text with acronyms and misspellings
    text = """
    Our company CTO recntly approved the new ML project that uses API 
    for accessing customer data. The CEO believes that this AI initiative
    will improv the companys profitibility and eficiency.
    """
    
    # Process text with all features
    processed_text = preprocessor.preprocess_text(
        text,
        correct_spelling=True,
        expand_acronyms=True,
        normalize=False,  # Keep original casing and punctuation
        context_aware=True,
    )
    
    # Print the results
    print("Original text:")
    print(text)
    print("\nProcessed text:")
    print(processed_text)


def batch_processing_example():
    """Example of batch processing with LMPreprocessor."""
    # Initialize preprocessor
    preprocessor = LMPreprocessor(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        custom_acronyms={
            "ROI": "Return on Investment",
            "CAC": "Customer Acquisition Cost",
            "LTV": "Lifetime Value",
        },
    )
    
    # Sample batch of texts
    texts = [
        "The ROI on this campain has been excelent, well above industry avg.",
        "Our CAC is decreesing while LTV is incresing.",
        "The CEO and CTO will present the AI strategy at the upcomming meeting.",
        "The API documentaton is incomplte and has typos.",
    ]
    
    # Process in batch
    processed_texts = preprocessor.preprocess_batch(
        texts,
        show_progress=True,  # Show progress bar
    )
    
    # Create DataFrame to display results
    df = pd.DataFrame({
        "Original": texts,
        "Processed": processed_texts,
    })
    
    print("\nBatch Processing Results:")
    pd.set_option('display.max_colwidth', None)
    print(df)


def acronym_analysis_example():
    """Example of analyzing acronyms in a corpus."""
    # Initialize preprocessor
    preprocessor = LMPreprocessor(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        domain="finance",  # Include finance-specific dictionaries
    )
    
    # Sample financial texts with acronyms
    texts = [
        "The ROI on this project exceeded expectations.",
        "Our company's EBITDA has improved by 15% this quarter.",
        "The CEO and CFO will present the financial results.",
        "The P&L statement shows improved margins.",
        "We need to review the ROI and CAC for our new product line.",
        "The IPO is scheduled for next quarter, according to the CEO.",
        "Our CAGR of 12% is above industry average.",
        "The CFO presented the EPS forecast to the board.",
    ]
    
    # Analyze acronyms in the corpus
    acronyms_df = preprocessor.find_document_acronyms(texts)
    
    print("\nAcronym Analysis Results:")
    print(acronyms_df)


def spelling_correction_example():
    """Example of using context-aware spelling correction."""
    # Initialize spelling corrector directly
    spelling_corrector = LMSpellingCorrector(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        domain="tech",
        learn_corrections=True,
    )
    
    # Sample texts with misspellings
    texts = [
        "The progam runs efficiently on the servor.",
        "We need to optmize the algorthm for better performance.",
        "The developr implemented the API corectly.",
        "The databse query is causing the bottlneck.",
    ]
    
    # Analyze misspellings based on context
    for text in texts:
        print(f"\nOriginal: {text}")
        corrected = spelling_corrector.correct_text(text)
        print(f"Corrected: {corrected}")
        
        # Find likely misspellings with confidence scores
        misspellings = spelling_corrector.find_likely_misspellings(text)
        if misspellings:
            print("Detected misspellings:")
            for word, correction, confidence in misspellings:
                print(f"  {word} -> {correction} (confidence: {confidence:.2f})")


def custom_domain_example():
    """Example with custom domain-specific vocabulary."""
    # Custom insurance acronyms and spelling corrections
    insurance_acronyms = {
        "UW": "Underwriting",
        "DOL": "Date of Loss",
        "POL": "Policy",
        "PH": "Policyholder",
        "BI": "Bodily Injury",
        "PD": "Property Damage",
        "PIP": "Personal Injury Protection",
    }
    
    insurance_spelling = {
        "policyhlder": "policyholder",
        "premiume": "premium",
        "deductable": "deductible",
        "endorsment": "endorsement",
        "adjestment": "adjustment",
        "clame": "claim",
        "covrage": "coverage",
    }
    
    # Initialize preprocessor with custom dictionaries
    preprocessor = LMPreprocessor(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        custom_acronyms=insurance_acronyms,
        custom_spelling=insurance_spelling,
    )
    
    # Sample insurance text
    text = """
    The PH filed a clame after the car accidnt. The UW department reviewed the 
    POL and confirmed BI and PD covrage. The deductable was applied according to 
    the endorsment added on the DOL.
    """
    
    # Process text
    processed_text = preprocessor.preprocess_text(text)
    
    print("\nCustom Domain Example (Insurance):")
    print("Original text:")
    print(text)
    print("\nProcessed text:")
    print(processed_text)


def spacy_integration_example():
    """Example of using a spaCy model through sentence-transformers."""
    try:
        import spacy
        import spacy_sentence_bert
    except ImportError:
        print("\nSpaCy integration example requires spacy and spacy_sentence_bert packages.")
        print("Install with: pip install spacy spacy_sentence_bert")
        return
    
    try:
        # Load spaCy model with SBERT component
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentence_bert", config={"model_name": "paraphrase-MiniLM-L3-v2"})
        
        # Create a custom preprocessor using the spaCy model
        preprocessor = LMPreprocessor(
            model=nlp.get_pipe("sentence_bert").model,  # Pass the model directly
            domain="tech",
        )
        
        # Sample text
        text = """
        The IT deparment requiers an upgarde of the databse servor. 
        The CTO approved the ML project for predicting user bhavior.
        """
        
        # Process text
        processed_text = preprocessor.preprocess_text(text)
        
        print("\nSpaCy Integration Example:")
        print("Original text:")
        print(text)
        print("\nProcessed text:")
        print(processed_text)
        
    except Exception as e:
        print(f"\nSpaCy integration example failed: {e}")
        print("Make sure to run: python -m spacy download en_core_web_sm")


def save_load_example():
    """Example of saving and loading learned corrections."""
    # Initialize preprocessor
    preprocessor = LMPreprocessor(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    )
    
    # Process text to learn new corrections
    texts = [
        "The teh comany's stratgy for AI has been updated.",
        "The DNN model outperfroms other ML algorithms.",
        "The API docmentation should be impoved.",
    ]
    
    # Process texts to learn corrections
    for text in texts:
        preprocessor.preprocess_text(text)
    
    # Save learned corrections
    save_path = "learned_corrections.json"
    preprocessor.save_learned_corrections(save_path)
    
    print(f"\nSaved learned corrections to {save_path}")
    
    # Create a new preprocessor and load corrections
    new_preprocessor = LMPreprocessor(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    )
    
    success = new_preprocessor.load_learned_corrections(save_path)
    
    if success:
        print(f"Successfully loaded learned corrections from {save_path}")
        
        # Test with new text
        test_text = "The API docmentation has been updated by the comany."
        corrected = new_preprocessor.preprocess_text(test_text)
        
        print("\nCorrected with loaded corrections:")
        print(f"Original: {test_text}")
        print(f"Corrected: {corrected}")
    
    # Clean up file
    if os.path.exists(save_path):
        os.remove(save_path)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Language Model Enhanced Preprocessing Examples")
    print("=" * 50)
    
    # Run examples
    simple_example()
    batch_processing_example()
    acronym_analysis_example()
    spelling_correction_example()
    custom_domain_example()
    spacy_integration_example()
    save_load_example()
    
    print("\n" + "=" * 50)
    print("Examples completed")
    print("=" * 50)
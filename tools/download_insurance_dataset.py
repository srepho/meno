"""
Utility script to download and prepare the Australian Insurance PII Dataset for offline use.
This is useful for distributing examples without requiring users to download from Hugging Face.
"""

import os
import sys
import pandas as pd
from datasets import load_dataset
import json
from pathlib import Path

def main():
    """Download and save the Australian Insurance PII Dataset locally."""
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Create insurance dataset directory
    insurance_dir = output_dir / "australian_insurance"
    insurance_dir.mkdir(exist_ok=True)
    
    try:
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("soates/australian-insurance-pii-dataset-corrected")
        
        # Save each split
        for split in dataset.keys():
            print(f"Processing '{split}' split...")
            split_data = dataset[split]
            
            # Convert to pandas DataFrame
            df = pd.DataFrame({
                "text": split_data["original_text"],
                "redacted_text": split_data["redacted_text"],
                "id": split_data["id"],
                "annotations": split_data["annotations"]
            })
            
            # Save to CSV
            output_path = insurance_dir / f"{split}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} documents to {output_path}")
            
            # Save sample JSON for reference
            if split == "train":
                sample = split_data[0]
                sample_path = insurance_dir / "sample.json"
                with open(sample_path, "w") as f:
                    json.dump(sample, f, indent=2)
                print(f"Saved sample document to {sample_path}")
        
        # Create README file with dataset information
        readme_path = insurance_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("""# Australian Insurance PII Dataset

This dataset contains insurance complaint letters with various types of insurance issues, making it ideal for testing topic modeling capabilities.

## Dataset Structure

- **train.csv**: Training set with 1243 documents
- **validation.csv**: Validation set with 155 documents
- **test.csv**: Test set with 156 documents

## Columns

- **text**: Original insurance complaint letter
- **redacted_text**: Version with personally identifiable information redacted
- **annotations**: JSON string containing annotated PII entities
- **id**: Unique identifier for each document

## Source

This dataset is originally from the Hugging Face Datasets repository:
https://huggingface.co/datasets/soates/australian-insurance-pii-dataset-corrected

## License

Please refer to the original dataset for licensing information.
""")
        print(f"Created README at {readme_path}")
        
        print("\nDataset download complete!")
        print(f"Dataset saved to {insurance_dir}")
        print("You can now use this local copy by modifying the example script to load from these files.")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
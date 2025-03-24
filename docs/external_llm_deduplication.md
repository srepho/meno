# Using Deduplication with External LLMs in Meno

This document explains how to use Meno's deduplication capabilities to optimize external LLM processing. By deduplicating text data before sending it to external LLM APIs (like OpenAI GPT or Anthropic Claude), you can significantly reduce costs, processing time, and token usage.

## Overview

When processing large datasets with external LLMs, deduplication offers several advantages:

1. **Cost Savings**: Process only unique documents, reducing API costs
2. **Faster Processing**: Fewer API calls means faster overall processing
3. **Token Efficiency**: Reduce token usage by not processing duplicates
4. **Consistent Results**: Ensure identical documents receive identical LLM outputs

The general workflow is:

1. Deduplicate your dataset (exact or fuzzy matching)
2. Process only the unique documents with an external LLM API
3. Map the results back to the full dataset, including duplicates

## Step 1: Deduplicate Your Dataset

```python
from meno.preprocessing.deduplication import TextDeduplicator
import pandas as pd

# Load your dataset
data = pd.read_csv("your_dataset.csv")

# Create a TextDeduplicator - adjust threshold as needed
deduplicator = TextDeduplicator(similarity_threshold=0.85)

# Choose your deduplication method
deduplicated_data, duplicate_map, groups = deduplicator.deduplicate(
    data=data,
    text_column="text",  # Column containing text to analyze
    method="fuzzy",      # Use "exact" for exact matching only
    threshold=0.85       # Only used for fuzzy matching
)

print(f"Original dataset: {len(data)} documents")
print(f"After deduplication: {len(deduplicated_data)} unique documents")
print(f"Removed {len(data) - len(deduplicated_data)} duplicates/near-duplicates")
```

## Step 2: Process Unique Documents with External LLM

### OpenAI Example

```python
import openai
import os
from tqdm import tqdm

# Set up OpenAI API key (use environment variables for security)
openai.api_key = os.environ["OPENAI_API_KEY"]

# Process only the deduplicated documents
responses = []
for idx, row in tqdm(deduplicated_data.iterrows(), total=len(deduplicated_data)):
    # Create your prompt
    prompt = f"Analyze this text and classify it into a category:\n\n{row['text']}"
    
    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful text classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    # Store results with document ID for mapping back later
    responses.append({
        "id": row["id"],  # Assuming 'id' is your unique identifier
        "category": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    })

# Convert responses to DataFrame
responses_df = pd.DataFrame(responses)

# Add responses to the deduplicated data
for col in responses_df.columns:
    if col != "id":  # Skip the ID column
        deduplicated_data[col] = responses_df[col].values
```

### Anthropic Claude Example

```python
from anthropic import Anthropic
import os
import json
import re
from tqdm import tqdm

# Set up Anthropic API
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Process only the deduplicated documents
responses = []
for idx, row in tqdm(deduplicated_data.iterrows(), total=len(deduplicated_data)):
    # Create your prompt - using Anthropic's format
    prompt = f"""Human: Classify this text into a category and provide a brief summary in JSON format.

Text:
{row['text']}

Please return your analysis as valid JSON with keys: category, summary"""
    
    # Call Claude API
    try:
        response = anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0.3,
            system="You are a helpful text classifier.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response assuming JSON format
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', response.content[0].text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try parsing the whole response
                result = json.loads(response.content[0].text)
                
            # Add document ID and token usage
            result["id"] = row["id"]
            result["tokens"] = response.usage.input_tokens + response.usage.output_tokens
            
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            result = {
                "id": row["id"],
                "category": "Error",
                "summary": response.content[0].text[:100] + "...",
                "tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
    except Exception as e:
        # Handle API errors
        result = {
            "id": row["id"],
            "category": "Error",
            "summary": f"API Error: {str(e)}",
            "tokens": 0
        }
        
    responses.append(result)

# Convert to DataFrame
responses_df = pd.DataFrame(responses)

# Add responses to deduplicated data
for col in responses_df.columns:
    if col != "id":  # Skip the ID column
        deduplicated_data[col] = responses_df[col].values
```

## Step 3: Map Results Back to Full Dataset

Now that you've processed the unique documents with your LLM, you need to map the results back to the full dataset. This ensures all documents, including duplicates, have the appropriate LLM outputs.

```python
# Map LLM results back to full dataset
full_results = deduplicator.map_results_to_full_dataset(
    original_df=data,
    deduplicated_results=deduplicated_data,
    duplicate_map=duplicate_map,
    result_columns=["category", "summary", "tokens"]  # Columns to map back
)

print(f"Successfully mapped results to all {len(full_results)} documents")
```

## Step 4: Calculate Savings

You can quantify the benefits of using deduplication:

```python
# Calculate token and cost savings
total_tokens = sum(full_results["tokens"])
total_documents = len(full_results)
unique_documents = len(deduplicated_data)

# Estimate tokens without deduplication
estimated_tokens_without_dedup = total_tokens * total_documents / unique_documents
tokens_saved = estimated_tokens_without_dedup - total_tokens

# Estimate cost savings (adjust cost per token as needed)
cost_per_token = 0.0001  # Example rate
cost_saved = tokens_saved * cost_per_token

print(f"Total documents: {total_documents}")
print(f"Unique documents processed: {unique_documents}")
print(f"Documents skipped: {total_documents - unique_documents}")
print(f"Tokens used with deduplication: {total_tokens:,}")
print(f"Estimated tokens without deduplication: {estimated_tokens_without_dedup:,.0f}")
print(f"Tokens saved: {tokens_saved:,.0f} ({tokens_saved/estimated_tokens_without_dedup*100:.1f}%)")
print(f"Estimated cost saved: ${cost_saved:.2f}")
```

## Advanced Configuration 

### Adjusting Similarity Threshold

The similarity threshold determines how closely documents must match to be considered duplicates in fuzzy deduplication:

```python
# More strict matching (only very similar documents are considered duplicates)
deduplicator = TextDeduplicator(similarity_threshold=0.95)

# More lenient matching (catches more variation, but may group distinct documents)
deduplicator = TextDeduplicator(similarity_threshold=0.75)
```

### Batch Processing

For very large datasets, you may want to process in batches:

```python
# Process in batches to avoid memory issues
batch_size = 10000
total_batches = len(deduplicated_data) // batch_size + 1

all_responses = []
for batch in range(total_batches):
    start_idx = batch * batch_size
    end_idx = min((batch + 1) * batch_size, len(deduplicated_data))
    
    batch_data = deduplicated_data.iloc[start_idx:end_idx]
    print(f"Processing batch {batch+1}/{total_batches} ({len(batch_data)} documents)")
    
    # Process batch with LLM (code similar to previous examples)
    # ...
    
    all_responses.extend(batch_responses)
```

## Complete Example

For a comprehensive example showing all these steps together, including visualizations and performance comparisons, see the `external_llm_deduplication_example.py` file in the examples directory.

## Best Practices

1. **Choose the right deduplication method**:
   - Use exact deduplication for faster processing when documents are truly identical
   - Use fuzzy deduplication when dealing with slight variations or paraphrasing

2. **Tune the similarity threshold**:
   - Start with 0.85 and adjust based on your specific dataset
   - Lower thresholds catch more duplicates but may group related but distinct texts
   - Higher thresholds are more precise but may miss near-duplicates

3. **Preserve document IDs**:
   - Ensure your dataset has a unique identifier column for mapping
   - If no ID exists, the deduplicator will use DataFrame indices

4. **Monitor token usage**:
   - Track token consumption to quantify savings
   - Use these metrics to optimize threshold settings

5. **Handle LLM errors gracefully**:
   - Implement retry logic for API failures
   - Have fallback options for failed requests

6. **Review duplicate groups**:
   - Examine the fuzzy groups returned by the deduplicator
   - Verify that grouped documents are truly similar enough to receive the same LLM output

By following these practices, you can achieve significant cost and time savings when using external LLMs on large datasets.
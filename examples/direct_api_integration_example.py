"""
Example demonstrating how to integrate direct standard OpenAI API calls with Meno.

This example shows how to use the simple requests-based approach while
leveraging Meno's functionality.
"""

import requests
import time
from typing import List, Dict, Any, Optional

# Constants
API_KEY = "your_api_key" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"  # Can be changed to any model you prefer

def generate_call_from_text(text: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    Make a direct API call to OpenAI using the requests library.
    
    Args:
        text: The user input text to process
        system_prompt: The system prompt to use
        
    Returns:
        The generated response text or an error message
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        response_data = response.json()
        
        if not response_data.get('choices') or len(response_data['choices']) == 0:
            return "[No response generated.]"
            
        return response_data['choices'][0]['message']['content'].strip()
        
    except Exception as e:
        return f"[Error: {str(e)}]"

# Integration with Meno
def categorize_texts_with_direct_api(texts: List[str], categories: List[str]) -> List[str]:
    """
    Categorize a list of texts using direct OpenAI API calls.
    Returns the category for each text.
    
    Args:
        texts: List of texts to categorize
        categories: List of possible categories
        
    Returns:
        List of assigned categories
    """
    results = []
    categories_str = ", ".join(categories)
    
    system_prompt = f"""You are a text categorization assistant. 
    Categorize each text into exactly one of these categories: {categories_str}.
    Respond with ONLY the category name, nothing else."""
    
    print(f"Categorizing {len(texts)} texts...")
    
    for i, text in enumerate(texts):
        prompt = f"Categorize this text into one of these categories ({categories_str}):\n\n{text}"
        
        print(f"Processing text {i+1}/{len(texts)}...")
        start_time = time.time()
        
        result = generate_call_from_text(
            text=prompt,
            system_prompt=system_prompt
        )
        
        end_time = time.time()
        
        # Ensure the result is one of the categories
        if result in categories:
            results.append(result)
        else:
            # Try to find the category in the response
            for category in categories:
                if category.lower() in result.lower():
                    results.append(category)
                    break
            else:
                # If no category is found, use a default
                results.append("uncategorized")
                print(f"Warning: Could not determine category from response: {result}")
        
        print(f"  Assigned category: {results[-1]} (took {end_time - start_time:.2f}s)")
    
    return results

# Example function for topic labeling
def generate_topic_labels(keyword_groups: List[List[str]]) -> List[str]:
    """
    Generate topic labels for groups of keywords using direct API calls.
    
    Args:
        keyword_groups: List of keyword groups to generate labels for
        
    Returns:
        List of topic labels
    """
    results = []
    
    system_prompt = """You are a topic labeling expert. 
    Create a concise, descriptive label (1-3 words) for each set of keywords.
    The label should capture the core theme represented by the keywords."""
    
    print(f"Generating labels for {len(keyword_groups)} keyword groups...")
    
    for i, keywords in enumerate(keyword_groups):
        keywords_str = ", ".join(keywords)
        prompt = f"Create a concise topic label (1-3 words) for these keywords: {keywords_str}"
        
        print(f"Processing keyword group {i+1}/{len(keyword_groups)}...")
        
        result = generate_call_from_text(
            text=prompt,
            system_prompt=system_prompt
        )
        
        results.append(result)
        print(f"  Generated label: {result}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("MENO DIRECT API INTEGRATION EXAMPLE")
    print("Note: This example requires a valid OpenAI API key.")
    
    if API_KEY == "your_api_key":
        print("Please update the API_KEY variable at the top of this file to run the example.")
        exit()
    
    # Example 1: Simple categorization
    print("\n=== Text Categorization Example ===")
    categories = ["business", "technology", "health", "entertainment", "sports"]
    
    texts = [
        "Apple announced its newest iPhone with improved camera and battery life.",
        "The basketball team won the championship after an intense overtime period.",
        "Researchers discovered a new treatment that could help patients with diabetes.",
        "The streaming service released a new comedy series that critics are praising."
    ]
    
    categories_result = categorize_texts_with_direct_api(texts, categories)
    
    print("\nResults:")
    for i, (text, category) in enumerate(zip(texts, categories_result)):
        print(f"{i+1}. Text: {text[:50]}...")
        print(f"   Category: {category}")
    
    # Example 2: Topic labeling
    print("\n=== Topic Labeling Example ===")
    keyword_groups = [
        ["finance", "banking", "investment", "stocks", "money"],
        ["machine learning", "artificial intelligence", "neural networks", "deep learning"],
        ["climate", "environment", "sustainability", "renewable", "green"],
        ["smartphone", "mobile", "app", "android", "ios"]
    ]
    
    labels = generate_topic_labels(keyword_groups)
    
    print("\nResults:")
    for i, (keywords, label) in enumerate(zip(keyword_groups, labels)):
        print(f"{i+1}. Keywords: {', '.join(keywords)}")
        print(f"   Label: {label}")
    
    print("\nAll examples completed!")
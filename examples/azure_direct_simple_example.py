import requests

# Replace these with your Azure OpenAI details
api_key = "your_azure_api_key"
resource_name = "your-resource-name"  # The first part of your endpoint URL
deployment_id = "your-deployment-id"  # Your model deployment name in Azure
api_version = "2023-12-01-preview"  # Update as needed

# Construct Azure endpoint URL
api_endpoint = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"

def generate_call_from_text(text):
    # Define your API endpoint, API key, and model
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key  # Azure uses 'api-key' instead of 'Authorization'
    }
    
    # Note: For Azure, don't include 'model' in the payload - it's specified via deployment_id in the URL
    payload = {
        # No 'model' field for Azure
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    }
    
    response = requests.post(api_endpoint, headers=headers, json=payload)
    
    if response.status_code != 200:
        return f"[Error: {response.status_code} - {response.text}]"
    
    response_data = response.json()
    if not response_data.get('choices') or len(response_data['choices']) == 0:
        return "[No response generated.]"
    
    return response_data['choices'][0]['message']['content'].strip()

# Example usage
text = "Hello!"
result = generate_call_from_text(text)
print(result)
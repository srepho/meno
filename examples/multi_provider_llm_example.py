"""
Example demonstrating how to use the multi-provider LLM integration in Meno.

This example shows how to use the generate_text_with_llm_multi function with:
1. OpenAI (original implementation)
2. Google's Gemini AI
3. Anthropic's Claude
4. Hugging Face Inference API
5. AWS Bedrock
"""

import os
import time
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the extended LLM function
try:
    from meno.modeling.llm_topic_labeling_extended import generate_text_with_llm_multi
except ImportError:
    logger.error("Could not import generate_text_with_llm_multi. Make sure Meno is installed.")
    raise

# API keys - replace with your own or set environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")


def demo_openai():
    """Demonstrate OpenAI integration."""
    if not OPENAI_API_KEY:
        logger.warning("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
        return
    
    logger.info("\n=== Testing OpenAI Integration ===")
    prompt = "What are the key differences between transformers and RNNs in deep learning?"
    
    # Test with SDK
    logger.info("\n1. Using OpenAI SDK:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=OPENAI_API_KEY,
            provider="openai",
            library="sdk",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (SDK): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with OpenAI SDK: {e}")
    
    # Test with direct requests
    logger.info("\n2. Using OpenAI with direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=OPENAI_API_KEY,
            provider="openai",
            library="requests",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=800,
            enable_cache=True
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (Requests): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with direct requests: {e}")


def demo_google_gemini():
    """Demonstrate Google Gemini integration."""
    if not GOOGLE_API_KEY:
        logger.warning("No Google API key found. Set the GOOGLE_API_KEY environment variable.")
        return
    
    logger.info("\n=== Testing Google Gemini Integration ===")
    prompt = "What are the key differences between transformers and RNNs in deep learning?"
    
    # Test with SDK
    logger.info("\n1. Using Google SDK:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=GOOGLE_API_KEY,
            provider="google",
            library="sdk",
            model_name="gemini-pro",
            temperature=0.7,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (SDK): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with Google SDK: {e}")
    
    # Test with direct requests
    logger.info("\n2. Using Google with direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=GOOGLE_API_KEY,
            provider="google",
            library="requests",
            model_name="gemini-pro",
            temperature=0.7,
            max_tokens=800,
            enable_cache=True
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (Requests): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with direct requests: {e}")


def demo_anthropic_claude():
    """Demonstrate Anthropic Claude integration."""
    if not ANTHROPIC_API_KEY:
        logger.warning("No Anthropic API key found. Set the ANTHROPIC_API_KEY environment variable.")
        return
    
    logger.info("\n=== Testing Anthropic Claude Integration ===")
    prompt = "Compare and contrast different approaches to prompt engineering for large language models."
    
    # Test with SDK
    logger.info("\n1. Using Anthropic SDK:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=ANTHROPIC_API_KEY,
            provider="anthropic",
            library="sdk",
            model_name="claude-3-haiku-20240307",  # Use smallest model to save costs
            temperature=0.7,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (SDK): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with Anthropic SDK: {e}")
    
    # Test with direct requests
    logger.info("\n2. Using Anthropic with direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=ANTHROPIC_API_KEY,
            provider="anthropic",
            library="requests",
            model_name="claude-3-haiku-20240307",  # Use smallest model to save costs
            temperature=0.7,
            max_tokens=800,
            enable_cache=True,
            api_version="2023-06-01"  # Specify API version
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (Requests): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with direct requests: {e}")


def demo_huggingface_inference():
    """Demonstrate Hugging Face Inference API integration."""
    if not HUGGINGFACE_API_KEY:
        logger.warning("No Hugging Face API key found. Set the HUGGINGFACE_API_KEY environment variable.")
        return
    
    logger.info("\n=== Testing Hugging Face Integration ===")
    prompt = "What are the most important considerations for deploying machine learning models in production?"
    
    # Test with SDK
    logger.info("\n1. Using Hugging Face SDK:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=HUGGINGFACE_API_KEY,
            provider="huggingface",
            library="sdk",
            model_name="mistralai/Mistral-7B-Instruct-v0.2",  # Use a smaller model
            temperature=0.7,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (SDK): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with Hugging Face SDK: {e}")
    
    # Test with direct requests
    logger.info("\n2. Using Hugging Face with direct requests:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=HUGGINGFACE_API_KEY,
            provider="huggingface",
            library="requests",
            model_name="mistralai/Mistral-7B-Instruct-v0.2",  # Use a smaller model
            temperature=0.7,
            max_tokens=800,
            enable_cache=True,
            additional_params={
                "parameters": {
                    "do_sample": True,
                    "top_p": 0.95
                }
            }
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (Requests): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with direct requests: {e}")


def demo_aws_bedrock():
    """Demonstrate AWS Bedrock integration."""
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        logger.warning("No AWS credentials found. Set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        return
    
    logger.info("\n=== Testing AWS Bedrock Integration ===")
    prompt = "What are the best practices for fine-tuning language models for specific tasks?"
    
    # Test with SDK (boto3)
    logger.info("\n1. Using AWS Bedrock SDK:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=AWS_ACCESS_KEY,
            api_secret=AWS_SECRET_KEY,
            provider="bedrock",
            library="sdk",
            model_name="anthropic.claude-instant-v1",  # Use a less expensive model
            temperature=0.7,
            max_tokens=800,
            region_name="us-east-1"
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (SDK): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with AWS Bedrock SDK: {e}")
    
    # For Bedrock, the requests implementation uses boto3 under the hood for AWS SigV4 signing
    logger.info("\n2. Using AWS Bedrock with requests wrapper:")
    try:
        start_time = time.time()
        response = generate_text_with_llm_multi(
            text=prompt,
            api_key=AWS_ACCESS_KEY,
            api_secret=AWS_SECRET_KEY,
            provider="bedrock",
            library="requests",
            model_name="amazon.titan-text-express-v1",  # Use Amazon's model
            temperature=0.7,
            max_tokens=800,
            region_name="us-east-1",
            enable_cache=True
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response (Requests): {response[:200]}...")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error with requests implementation: {e}")


def run_multi_provider_comparison(prompt: str):
    """Run the same prompt through all available providers for comparison."""
    logger.info("\n=== Multi-Provider Comparison ===")
    logger.info(f"Prompt: {prompt}")
    
    providers = []
    if OPENAI_API_KEY:
        providers.append(("openai", OPENAI_API_KEY, None))
    if GOOGLE_API_KEY:
        providers.append(("google", GOOGLE_API_KEY, None))
    if ANTHROPIC_API_KEY:
        providers.append(("anthropic", ANTHROPIC_API_KEY, None))
    if HUGGINGFACE_API_KEY:
        providers.append(("huggingface", HUGGINGFACE_API_KEY, None))
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        providers.append(("bedrock", AWS_ACCESS_KEY, AWS_SECRET_KEY))
    
    results = {}
    for provider, api_key, api_secret in providers:
        logger.info(f"\nTesting {provider.capitalize()}...")
        try:
            # Use SDK for consistency
            params = {
                "text": prompt,
                "api_key": api_key,
                "provider": provider,
                "library": "sdk",
                "temperature": 0.7,
                "max_tokens": 800,
                "enable_cache": True
            }
            
            # Add provider-specific parameters
            if provider == "openai":
                params["model_name"] = "gpt-3.5-turbo"
            elif provider == "google":
                params["model_name"] = "gemini-pro"
            elif provider == "anthropic":
                params["model_name"] = "claude-3-haiku-20240307"
            elif provider == "huggingface":
                params["model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
            elif provider == "bedrock":
                params["model_name"] = "anthropic.claude-instant-v1"
                params["api_secret"] = api_secret
                params["region_name"] = "us-east-1"
            
            start_time = time.time()
            response = generate_text_with_llm_multi(**params)
            elapsed = time.time() - start_time
            
            results[provider] = {
                "response": response,
                "time": elapsed
            }
            
            logger.info(f"Time taken: {elapsed:.2f} seconds")
        except Exception as e:
            logger.error(f"Error with {provider}: {e}")
    
    # Display comparison summary
    logger.info("\n=== Comparison Results ===")
    for provider, result in results.items():
        logger.info(f"\n{provider.capitalize()}:")
        logger.info(f"Response: {result['response'][:200]}...")
        logger.info(f"Time: {result['time']:.2f} seconds")


def run_demos():
    """Run all demo functions."""
    logger.info("DEMONSTRATING MULTI-PROVIDER LLM INTEGRATION")
    logger.info("This example shows how to use the generate_text_with_llm_multi function with various LLM providers:")
    logger.info("1. OpenAI (original implementation)")
    logger.info("2. Google Gemini")
    logger.info("3. Anthropic Claude")
    logger.info("4. Hugging Face")
    logger.info("5. AWS Bedrock")
    logger.info("\nNote: You need to set API keys as environment variables to run the examples.")
    
    # Run individual demos
    demo_openai()
    demo_google_gemini()
    demo_anthropic_claude()
    demo_huggingface_inference()
    demo_aws_bedrock()
    
    # Run multi-provider comparison
    comparison_prompt = "Explain the concept of prompt engineering and why it's important for working with large language models."
    run_multi_provider_comparison(comparison_prompt)


if __name__ == "__main__":
    run_demos()
"""
LLM Provider Factory for creating configurable LLM instances
"""
import os
import boto3
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock

from agents_playground.models import ProviderConfig


class LLMProviderFactory:
    """Factory class to create LLM provider instances."""
    
    @staticmethod
    def create_provider(provider_name: str, config: Optional[ProviderConfig]):
        """
        Create an LLM provider instance based on the provider name and config.
        
        Args:
            provider_name: Name of the provider ('openai' or 'bedrock')
            config: Provider configuration Pydantic model
            
        Returns:
            LLM provider instance
        """
        if not config:
            return None
            
        if provider_name == "openai":
            return LLMProviderFactory._create_openai_provider(config)
        elif provider_name == "bedrock":
            return LLMProviderFactory._create_bedrock_provider(config)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    @staticmethod
    def _create_openai_provider(config: ProviderConfig):
        """Create OpenAI provider instance."""
        api_key = os.getenv(config.api_key_env or "OPENAI_API_KEY")
        
        if not api_key:
            print("Warning: OpenAI API key not found in environment variables")
            return None
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=api_key
        )
    
    @staticmethod
    def _create_bedrock_provider(config: ProviderConfig):
        """Create AWS Bedrock provider instance."""
        aws_region = os.getenv(config.aws_region_env or "AWS_REGION", config.region or "us-east-1")
        aws_access_key_id = os.getenv(config.aws_access_key_env or "AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv(config.aws_secret_key_env or "AWS_SECRET_ACCESS_KEY")
        
        try:
            if not aws_access_key_id or not aws_secret_access_key:
                # Try to use default AWS credentials
                return ChatBedrock(
                    model_id=config.model_name,
                    region_name=aws_region,
                    model_kwargs={
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                    }
                )
            else:
                # Use explicit credentials
                bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                )
                
                return ChatBedrock(
                    client=bedrock_client,
                    model_id=config.model_name,
                    model_kwargs={
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                    }
                )
        except Exception as e:
            print(f"Warning: Could not initialize AWS Bedrock client: {e}")
            return None
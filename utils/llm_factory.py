"""Helper functions for creating LLM instances based on configuration."""
from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from .config import Config


def create_chat_llm(config: Config, temperature: Optional[float] = None, **kwargs):
    """
    Create a ChatLLM instance based on the configured provider.
    
    Args:
        config: Configuration object
        temperature: Optional temperature override
        **kwargs: Additional arguments to pass to the LLM
        
    Returns:
        ChatOpenAI or AzureChatOpenAI instance
    """
    temp = temperature if temperature is not None else config.temperature
    
    if config.is_azure():
        return AzureChatOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.azure_openai_deployment,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            temperature=temp,
            max_tokens=config.max_tokens,
            **kwargs
        )
    else:
        return ChatOpenAI(
            model=config.openai_model,
            temperature=temp,
            max_tokens=config.max_tokens,
            openai_api_key=config.openai_api_key,
            **kwargs
        )


def create_embeddings(config: Config):
    """
    Create an Embeddings instance based on the configured provider.
    
    Args:
        config: Configuration object
        
    Returns:
        OpenAIEmbeddings or AzureOpenAIEmbeddings instance
    """
    if config.is_azure():
        return AzureOpenAIEmbeddings(
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.azure_openai_deployment,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
        )
    else:
        return OpenAIEmbeddings(
            openai_api_key=config.openai_api_key
        )

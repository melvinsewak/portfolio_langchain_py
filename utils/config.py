"""Common configuration loader for langchain examples."""
import os
from dotenv import load_dotenv
from typing import Optional, Literal


class Config:
    """Configuration class to load and manage environment variables."""
    
    def __init__(self):
        """Initialize configuration by loading environment variables."""
        load_dotenv()
        
        # LLM Provider configuration (openai or azure)
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Azure OpenAI configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        # Common LLM settings
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        self.verbose = os.getenv("VERBOSE", "false").lower() == "true"
        
        # Vector store configuration
        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        # Document processing configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable or raise an error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(
                f"Missing required environment variable: {key}. "
                f"Please set it in your .env file or environment."
            )
        return value
    
    def validate(self) -> bool:
        """Validate that all required configurations are present."""
        try:
            if self.llm_provider == "azure":
                # Validate Azure OpenAI configuration
                if not self.azure_openai_api_key or self.azure_openai_api_key == "your_azure_openai_api_key_here":
                    print("⚠️  Warning: AZURE_OPENAI_API_KEY not configured properly.")
                    print("   Please update your .env file with a valid Azure OpenAI API key.")
                    return False
                if not self.azure_openai_endpoint:
                    print("⚠️  Warning: AZURE_OPENAI_ENDPOINT not configured.")
                    print("   Please set your Azure OpenAI endpoint in .env file.")
                    return False
                if not self.azure_openai_deployment:
                    print("⚠️  Warning: AZURE_OPENAI_DEPLOYMENT not configured.")
                    print("   Please set your Azure OpenAI deployment name in .env file.")
                    return False
            else:
                # Validate OpenAI configuration
                if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
                    print("⚠️  Warning: OPENAI_API_KEY not configured properly.")
                    print("   Please update your .env file with a valid API key.")
                    return False
            
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_llm_provider(self) -> Literal["openai", "azure"]:
        """Get the configured LLM provider."""
        return self.llm_provider
    
    def is_azure(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return self.llm_provider == "azure"


def get_config() -> Config:
    """Get the application configuration."""
    return Config()

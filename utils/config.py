"""Common configuration loader for langchain examples."""
import os
from dotenv import load_dotenv
from typing import Optional


class Config:
    """Configuration class to load and manage environment variables."""
    
    def __init__(self):
        """Initialize configuration by loading environment variables."""
        load_dotenv()
        
        self.openai_api_key = self._get_required_env("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
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
            if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
                print("⚠️  Warning: OPENAI_API_KEY not configured properly.")
                print("   Please update your .env file with a valid API key.")
                return False
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False


def get_config() -> Config:
    """Get the application configuration."""
    return Config()

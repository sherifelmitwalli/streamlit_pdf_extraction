import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with validation"""
    DEEPINFRA_API_KEY: str = Field(..., description="DeepInfra API key")
    MODEL_NAME: str = Field(
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",  # Updated to Llama 3.2
        description="Model name for text extraction"
    )
    API_BASE_URL: str = Field(
        default="https://api.deepinfra.com/v1/openai",
        description="Base URL for API"
    )
    TEMPERATURE: float = Field(default=0.7, ge=0, le=1)
    MAX_TOKENS: int = Field(default=4096, gt=0)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def validate_api_key(self) -> None:
        if not self.DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY is required in .env file")

settings = Settings()
settings.validate_api_key()

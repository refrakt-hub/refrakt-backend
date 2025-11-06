"""Application settings and configuration"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="./dev.env")


class Settings:
    """Application settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Cloudflare R2 Configuration
    R2_ACCOUNT_ID: Optional[str] = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID: Optional[str] = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY: Optional[str] = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "refrakt-artifacts")
    R2_PUBLIC_URL: str = os.getenv(
        "R2_PUBLIC_URL",
        "https://25b05206b0b66e008e4d3ae915f1d81c.r2.cloudflarestorage.com/refrakt-artifacts"
    )
    
    # Application Configuration
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    JOBS_DIR: Path = Path("./jobs")
    
    # Prompt Template Path
    PROMPT_TEMPLATE_PATH: Path = Path("config/prompt.md")
    
    def __init__(self):
        """Initialize settings and validate required configurations"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create jobs directory if it doesn't exist
        self.JOBS_DIR.mkdir(exist_ok=True)
    
    @property
    def is_r2_configured(self) -> bool:
        """Check if R2 is fully configured"""
        return all([
            self.R2_ACCOUNT_ID,
            self.R2_ACCESS_KEY_ID,
            self.R2_SECRET_ACCESS_KEY
        ])
    
    @property
    def r2_endpoint_url(self) -> Optional[str]:
        """Get R2 endpoint URL"""
        if self.R2_ACCOUNT_ID:
            return f"https://{self.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        return None


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


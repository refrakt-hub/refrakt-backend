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
    OPENAI_CONVERSATION_MODEL: str = os.getenv("OPENAI_CONVERSATION_MODEL", os.getenv("OPENAI_LIGHT_MODEL", "o4-mini"))
    OPENAI_CONFIG_MODEL: str = os.getenv("OPENAI_CONFIG_MODEL", OPENAI_MODEL)
    ASSISTANT_ENABLED: bool = os.getenv("ASSISTANT_ENABLED", "true").lower() in {"1", "true", "yes"}
    ASSISTANT_RETRIEVAL_ENABLED: bool = os.getenv("ASSISTANT_RETRIEVAL_ENABLED", "true").lower() in {"1", "true", "yes"}
    ASSISTANT_INDEX_PATH: Path = Path(os.getenv("ASSISTANT_INDEX_PATH", "./backend/cache/assistant_index")).resolve()
    ASSISTANT_EMBEDDING_MODEL: str = os.getenv("ASSISTANT_EMBEDDING_MODEL", "text-embedding-3-small")
    ASSISTANT_MAX_CONTEXT_CHUNKS: int = int(os.getenv("ASSISTANT_MAX_CONTEXT_CHUNKS", "4"))
    ASSISTANT_MAX_JOB_CHUNKS: int = int(os.getenv("ASSISTANT_MAX_JOB_CHUNKS", "3"))
    ASSISTANT_REINDEX_ON_START: bool = os.getenv("ASSISTANT_REINDEX_ON_START", "false").lower() in {"1", "true", "yes"}
    
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
    
    # Queue / background job configuration
    QUEUE_URL: Optional[str] = os.getenv("QUEUE_URL")
    QUEUE_NAME: str = os.getenv("QUEUE_NAME", "default")
    QUEUE_DEFAULT_TIMEOUT: int = int(os.getenv("QUEUE_DEFAULT_TIMEOUT", "7200"))
    QUEUE_RETRY_LIMIT: int = int(os.getenv("QUEUE_RETRY_LIMIT", "3"))
    QUEUE_RETRY_BACKOFF: float = float(os.getenv("QUEUE_RETRY_BACKOFF", "2.0"))
    QUEUE_VISIBILITY_TIMEOUT: int = int(os.getenv("QUEUE_VISIBILITY_TIMEOUT", "30"))
    QUEUE_WORKER_CONCURRENCY: int = int(os.getenv("QUEUE_WORKER_CONCURRENCY", "1"))
    QUEUE_RETRY_DELAY: float = float(os.getenv("QUEUE_RETRY_DELAY", "30"))

    # Prompt Template Path
    PROMPT_TEMPLATE_PATH: Path = Path("./backend/config/PROMPT.md")
    
    # Project root directory (parent of backend directory)
    # When running from project root: python refrakt-backend/main.py dev
    # We can detect project root by going up from refrakt-backend/config/settings.py
    # __file__ is refrakt-backend/config/settings.py
    # .parent is refrakt-backend/config/
    # .parent.parent is refrakt-backend/
    # .parent.parent.parent is refrakt/ (project root)
    # 
    # Alternative: Use current working directory if we're running from project root
    # This is more reliable when the script is executed from the project root
    _settings_file_path = Path(__file__)
    _backend_dir = _settings_file_path.parent.parent  # refrakt-backend/
    _calculated_root = _settings_file_path.parent.parent.parent  # refrakt/
    
    # Verify: if backend dir name is "refrakt-backend", then parent is project root
    if _backend_dir.name == "refrakt-backend" and _calculated_root.exists():
        PROJECT_ROOT: Path = _calculated_root
    else:
        # Fallback: use current working directory (assumes running from project root)
        PROJECT_ROOT: Path = Path.cwd()
    
    # JOBS_DIR is at project root: refrakt/jobs/
    # This will be set in __init__ after PROJECT_ROOT is available
    JOBS_DIR: Path = None  # type: ignore
    
    def __init__(self):
        """Initialize settings and validate required configurations"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Resolve JOBS_DIR relative to project root
        # jobs/ directory is at project root level
        self.JOBS_DIR = (self.PROJECT_ROOT / "jobs").resolve()
        
        # Debug output
        print(f"DEBUG: Settings initialized")
        print(f"DEBUG: PROJECT_ROOT: {self.PROJECT_ROOT}")
        print(f"DEBUG: PROJECT_ROOT exists: {self.PROJECT_ROOT.exists()}")
        print(f"DEBUG: JOBS_DIR: {self.JOBS_DIR}")
        print(f"DEBUG: Current working directory: {Path.cwd()}")
        
        if not self.QUEUE_URL:
            raise ValueError("QUEUE_URL environment variable is required for job processing")

        # Create jobs directory if it doesn't exist
        self.JOBS_DIR.mkdir(parents=True, exist_ok=True)

        if self.ASSISTANT_RETRIEVAL_ENABLED:
            self.ASSISTANT_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
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


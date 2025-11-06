"""Utility functions for loading prompt templates"""

from pathlib import Path
from config import get_settings


def load_prompt_template() -> str:
    """Load the prompt template from PROMPT.md"""
    settings = get_settings()
    prompt_path = settings.PROMPT_TEMPLATE_PATH
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


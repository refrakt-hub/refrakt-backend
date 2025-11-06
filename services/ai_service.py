"""AI service for generating YAML configs using OpenAI"""

import yaml
from typing import Dict, Optional
from openai import OpenAI
from config import get_settings


class AIService:
    """Service for interacting with OpenAI API"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        settings = get_settings()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    def generate_yaml_config(self, prompt: str, system_prompt: str) -> Dict:
        """
        Generate YAML configuration from user prompt using OpenAI
        
        Args:
            prompt: User's prompt describing the ML task
            system_prompt: System prompt template
            
        Returns:
            Parsed YAML configuration as dictionary
            
        Raises:
            ValueError: If YAML parsing fails
            Exception: If OpenAI API call fails
        """
        full_prompt = f"USER_REQUEST: {prompt}\n---\nYAML:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            yaml_text = response.choices[0].message.content.strip()
            
            # Clean the YAML text (remove markdown code blocks if present)
            yaml_text = self._clean_yaml_text(yaml_text)
            
            # Parse and validate YAML
            config = yaml.safe_load(yaml_text)
            
            if config is None:
                raise ValueError("Generated YAML is empty")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML generated: {str(e)}")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _clean_yaml_text(self, yaml_text: str) -> str:
        """Clean YAML text by removing markdown code blocks"""
        yaml_text = yaml_text.strip("` \n")
        
        # Remove common prefixes
        prefixes = ["yaml\n", "```yaml\n", "```\n"]
        for prefix in prefixes:
            if yaml_text.startswith(prefix):
                yaml_text = yaml_text[len(prefix):]
                break
        
        # Remove trailing backticks
        if yaml_text.endswith("```"):
            yaml_text = yaml_text[:-3]
        
        return yaml_text.strip()
    
    def test_connection(self) -> Dict[str, any]:
        """Test OpenAI API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=5
            )
            return {
                "status": "success",
                "response": response.choices[0].message.content,
                "api_key_configured": True
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "api_key_configured": False
            }


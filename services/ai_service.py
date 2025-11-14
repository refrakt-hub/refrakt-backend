"""AI service for generating YAML configs using OpenAI"""

import logging
import yaml
from typing import Any, Dict, Optional
from openai import OpenAI
from config import get_settings
from utils.config_validator import (
    ConfigValidator,
    ValidationError,
    ValidationWarning,
    UnsupportedModelError,
    FutureModelError,
)

logger = logging.getLogger(__name__)


class AIService:
    """Service for interacting with OpenAI API"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        settings = get_settings()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.config_model = settings.OPENAI_CONFIG_MODEL
        self.conversation_model = settings.OPENAI_CONVERSATION_MODEL
    
    def generate_yaml_config(
        self,
        prompt: str,
        system_prompt: str,
        dataset_hint: Optional[str] = None,
    ) -> Dict:
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
        prompt_parts = [f"USER_REQUEST: {prompt}"]
        if dataset_hint:
            prompt_parts.append(dataset_hint)
        prompt_parts.append("---")
        prompt_parts.append("YAML:")
        full_prompt = "\n".join(prompt_parts)
        
        validator = ConfigValidator()

        unsupported_reference = validator.find_unsupported_model_reference(prompt)
        if unsupported_reference:
            alias, canonical = unsupported_reference
            if validator.is_registered_model(canonical):
                supported = ", ".join(validator.supported_model_names())
                raise ValueError(
                    f'Refrakt does not currently support "{alias}" (requested model: {canonical}). '
                    f"Supported models: {supported}"
                )
            raise ValueError(
                f'Refrakt will support "{alias}" soon!'
            )

        try:
            response = self.client.chat.completions.create(
                model=self.config_model,
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
            raw_config = yaml.safe_load(yaml_text)
            
            if raw_config is None:
                raise ValueError("Generated YAML is empty")

            try:
                result = validator.validate(raw_config)
            except ValidationError as validation_error:
                model_name = self._extract_model_name(raw_config)
                if isinstance(validation_error, UnsupportedModelError):
                    supported = ", ".join(validator.supported_model_names())
                    raise ValueError(
                        f'Refrakt does not currently support "{model_name or "unknown"}". '
                        f"Supported models: {supported}"
                    ) from validation_error
                if isinstance(validation_error, FutureModelError):
                    raise ValueError(str(validation_error)) from validation_error

                if model_name:
                    logger.debug(
                        f"Validation failed for generated config; attempting fallback template "
                        f"(model={model_name}). Reason: {validation_error}"
                    )
                    try:
                        result = validator.fallback_to_template(model_name, overrides=raw_config)
                    except ValidationError as fallback_error:
                        raise ValueError(str(fallback_error)) from fallback_error
                    result.warnings.append(
                        ValidationWarning(
                            f"Applied fallback template due to validation error: {validation_error}"
                        )
                    )
                else:
                    raise ValueError(str(validation_error)) from validation_error

            for warning in result.warnings:
                logger.warning(f"Config validation warning: {warning.message}")

            return result.config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML generated: {str(e)}")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def generate_conversation_turn(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_completion_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate conversational response using the lightweight model."""
        params: Dict[str, Any] = {
            "model": self.conversation_model,
            "messages": messages,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_completion_tokens:
            params["max_completion_tokens"] = max_completion_tokens
        if response_format:
            params["response_format"] = response_format

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

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
    
    def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.conversation_model,
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_completion_tokens=10
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

    @staticmethod
    def _extract_model_name(config: Dict[str, Any]) -> Optional[str]:
        if not isinstance(config, dict):
            return None
        model_section = config.get("model")
        if isinstance(model_section, dict):
            name = model_section.get("name")
            if isinstance(name, str):
                return name
        return None


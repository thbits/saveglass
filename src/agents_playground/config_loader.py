"""
Configuration loader for AI Agents Playground
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import ValidationError

from agents_playground.models import (
    AgentConfig, 
    PromptsConfig, 
    UIConfig, 
    ProviderConfig,
    AgentSettings,
    ProviderType
)


class ConfigLoader:
    """Utility class to load configuration files from resources directory."""
    
    def __init__(self):
        """Initialize the config loader with resource paths."""
        self.project_root = Path(__file__).parent.parent.parent
        self.resources_dir = self.project_root / "resources"
        self.config_dir = self.resources_dir / "config"
        self.prompts_dir = self.resources_dir / "prompts"
        self.ui_dir = self.resources_dir / "ui"
    
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load and return YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"Warning: Configuration file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}")
            return {}
    
    def get_agent_config(self) -> AgentConfig:
        """Load agent configuration."""
        config_path = self.config_dir / "agents.yaml"
        config_data = self.load_yaml(config_path)
        
        try:
            # Convert providers to ProviderConfig instances
            if 'providers' in config_data:
                for provider_name, provider_config in config_data['providers'].items():
                    config_data['providers'][provider_name] = ProviderConfig(**provider_config)
            
            # Convert agent_settings to AgentSettings instance
            if 'agent_settings' in config_data:
                config_data['agent_settings'] = AgentSettings(**config_data['agent_settings'])
            
            return AgentConfig(**config_data)
        except ValidationError as e:
            print(f"Warning: Invalid agent configuration: {e}")
            # Return default configuration on validation error
            return AgentConfig()
    
    def get_prompts_config(self) -> PromptsConfig:
        """Load prompts configuration."""
        prompts_path = self.prompts_dir / "system_prompts.yaml"
        config_data = self.load_yaml(prompts_path)
        
        try:
            return PromptsConfig(**config_data)
        except ValidationError as e:
            print(f"Warning: Invalid prompts configuration: {e}")
            # Return default configuration on validation error
            return PromptsConfig()
    
    def get_ui_config(self) -> UIConfig:
        """Load UI configuration."""
        ui_path = self.ui_dir / "config.yaml"
        config_data = self.load_yaml(ui_path)
        
        try:
            return UIConfig(**config_data)
        except ValidationError as e:
            print(f"Warning: Invalid UI configuration: {e}")
            # Return default configuration on validation error
            return UIConfig()
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        agent_config = self.get_agent_config()
        return agent_config.providers.get(provider_name)
    
    def get_default_provider(self) -> str:
        """Get the name of the default provider."""
        agent_config = self.get_agent_config()
        return agent_config.default_provider.value
    
    def get_available_models(self, provider_name: str) -> list:
        """Get available models for a specific provider."""
        agent_config = self.get_agent_config()
        return agent_config.available_models.get(provider_name, [])
    
    def get_system_prompt(self, prompt_type: str = "default") -> str:
        """Get system prompt for the specified type."""
        prompts_config = self.get_prompts_config()
        return prompts_config.system_prompts.get(prompt_type, prompts_config.system_prompts.get("default", ""))
    
    def get_mock_response(self, response_type: str) -> str:
        """Get mock response template."""
        prompts_config = self.get_prompts_config()
        return prompts_config.mock_responses.get(response_type, "")
    
    def get_css_styles(self) -> str:
        """Load CSS styles from the styles.css file."""
        css_path = self.ui_dir / "styles.css"
        try:
            with open(css_path, 'r', encoding='utf-8') as file:
                css_content = file.read()
                # Format for Streamlit markdown with style tags
                return f"<style>\n{css_content}\n</style>"
        except FileNotFoundError:
            print(f"Warning: CSS file not found: {css_path}")
            return "<style></style>"
        except Exception as e:
            print(f"Error loading CSS file {css_path}: {e}")
            return "<style></style>"


# Global config loader instance
config_loader = ConfigLoader()
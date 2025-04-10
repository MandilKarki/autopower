# Configuration settings for report generation
from typing import Dict, List, Any, Optional
import os
import json


class ReportConfig:
    """Configuration for report generation"""
    
    DEFAULT_CONFIG = {
        "report": {
            "title": "Intelligent Report",
            "subtitle": "Generated Report",
            "author": "",
            "company": "",
            "logo": ""
        },
        "sections": {
            "executive_summary": {
                "title": "Executive Summary",
                "type": "user",
                "order": 0,
                "template": "This report provides a comprehensive analysis of {topic}."
            },
            "introduction": {
                "title": "Introduction",
                "type": "ai",
                "order": 1,
                "prompt": "Introduction to {topic} covering key background information."
            },
            "analysis": {
                "title": "Analysis",
                "type": "ai",
                "order": 2,
                "prompt": "Detailed analysis of {topic} including trends, challenges, and opportunities.",
                "detailed": True
            },
            "recommendations": {
                "title": "Recommendations",
                "type": "ai",
                "order": 3,
                "prompt": "Strategic recommendations based on the analysis of {topic}."
            },
            "conclusion": {
                "title": "Conclusion",
                "type": "ai",
                "order": 4,
                "prompt": "Conclusion summarizing key points about {topic}."
            }
        },
        "visualizations": {
            "auto_generate": True,
            "auto_count": 3,
            "custom": []
        },
        "output": {
            "markdown": "output/report.md",
            "powerpoint": "output/presentation.pptx",
            "visualization_dir": "temp"
        },
        "ai_model": {
            "name": "",  # Leave empty to use best available
            "temperature": 0.7
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        # Start with default config
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value"""
        if section not in self.config:
            return default
            
        if key is None:
            return self.config[section]
            
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
    
    def load_config(self, config_path: str) -> bool:
        """Load configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # Update config with loaded values
            for section, values in loaded_config.items():
                if section not in self.config:
                    self.config[section] = {}
                    
                if isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
                    
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """Save configuration to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def create_template(self, template_path: str, topic: str = "sample topic") -> bool:
        """Create a template configuration file"""
        template = self.DEFAULT_CONFIG.copy()
        
        # Update template with topic
        template["report"]["title"] = f"{topic.title()} Report"
        
        # Replace placeholders in section prompts
        for section_key, section in template["sections"].items():
            if "prompt" in section:
                section["prompt"] = section["prompt"].format(topic=topic)
            if "template" in section:
                section["template"] = section["template"].format(topic=topic)
        
        # Save template
        try:
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error creating template: {e}")
            return False

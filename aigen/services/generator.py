"""Agent code generator service."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from jinja2 import Environment, FileSystemLoader, exceptions

from aigen.services.models import AgentConfiguration

logger = logging.getLogger(__name__)

class AgentGeneratorService:
    """Service for generating agent code from configurations."""
    
    def __init__(self, template_dir: Optional[str] = None) -> None:
        """
        Initialize the generator service.
        
            template_dir: Path to template directory. If None, uses default templates.
        """
        if template_dir is None:
            template_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'templates'
            )
        
        self.template_dir = template_dir
        
        os.makedirs(template_dir, exist_ok=True)
        
        self._create_default_templates()
        
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
    
    def _create_default_templates(self) -> None:
        """
        Ensure default templates exist.
        
        Creates default templates if they don't exist in the template directory.
        """
        templates = {
            "agent_class.py.jinja": self._get_agent_class_template(),
            "agent_config.yaml.jinja": self._get_agent_config_template()
        }
        
        for filename, content in templates.items():
            file_path = os.path.join(self.template_dir, filename)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, "w") as f:
                    f.write(content)
    
    def generate_agent_code(self, config: AgentConfiguration) -> Tuple[bool, str]:
        """
        Generate Python code for an agent class.
        
            config: Agent configuration.
            
            Tuple containing:
                - Success flag (True if generation succeeded)
                - Generated code or error message
        """
        try:
            template = self.env.get_template("agent_class.py.jinja")
            
            code = template.render(
                config=config,
                framework_role=AgentConfiguration.to_framework_role(config.role.value)
            )
            
            return True, code
            
        except (exceptions.TemplateError, exceptions.TemplateNotFound) as e:
            error_msg = f"Template error: {str(e)}"
            logger.error(error_msg)
            return False, f"# Error generating code: {error_msg}"
        
        except Exception as e:
            error_msg = f"Error generating agent code: {str(e)}"
            logger.error(error_msg)
            return False, f"# Error generating code: {error_msg}"
    
    def generate_yaml_config(self, config: AgentConfiguration) -> Tuple[bool, str]:
        """
        Generate YAML configuration for an agent.
        
            config: Agent configuration.
            
            Tuple containing:
                - Success flag (True if generation succeeded)
                - Generated YAML or error message
        """
        try:
            template = self.env.get_template("agent_config.yaml.jinja")
            
            yaml_config = template.render(
                config=config,
                framework_role=AgentConfiguration.to_framework_role(config.role.value)
            )
            
            return True, yaml_config
            
        except (exceptions.TemplateError, exceptions.TemplateNotFound) as e:
            error_msg = f"Template error: {str(e)}"
            logger.error(error_msg)
            return False, f"# Error generating YAML: {error_msg}"
        
        except Exception as e:
            error_msg = f"Error generating YAML config: {str(e)}"
            logger.error(error_msg)
            return False, f"# Error generating YAML: {error_msg}"
    
    def _get_agent_class_template(self) -> str:
        """
        Get the default agent class template.
        
            Default agent class template string.
        """
        return """from typing import Dict, Any, Optional, List, Tuple
import logging
from pydantic import BaseModel

from agents import Agent, Runner
from agents.model_settings import ModelSettings
{% if config.output_type %}
from typing import List, Optional, Union, Dict, Any
{% endif %}

logger = logging.getLogger("{{ config.agent_type }}_agent")

{% if config.output_type %}
{{ config.output_type }}
{% endif %}

class {{ config.name | replace(" ", "") }}Agent:
    """
    {{ config.name }} - {{ config.role }} agent.
    
    
    {{ config.instructions | truncate(200) }}
    """
    
    def __init__(
        self, 
        agent_id: str = "{{ config.agent_type }}",
        **kwargs: Any
    ) -> None:
        """
        Initialize the agent.
        
            agent_id: Unique identifier for this agent instance.
            **kwargs: Additional arguments to pass to the agent.
        """
        self.agent_id = agent_id
        self.role = "{{ framework_role }}"
        
        self.system_prompt = kwargs.get(
            "instructions", 
            """{{ config.instructions }}"""
        )
        
        self._initialized = False
        self._openai_agent = None
        
        self.tools = []
        self.parameters = {
            {% for key, value in config.parameters.items() %}
            "{{ key }}": {{ value }},
            {% endfor %}
            **kwargs
        }
        
        self._init_tools()
    
    def _init_tools(self) -> None:
        """Initialize tools for this agent."""
        tool_names = {{ config.tools }}
        
        for tool_name in tool_names:
            try:
                from ..tools.factory import create_tool
                tool = create_tool(tool_name)
                self.tools.append(tool)
            except Exception as e:
                logger.warning(f"Failed to initialize tool {tool_name}: {str(e)}")
    
    def is_initialized(self) -> bool:
        """
        Check if the agent is initialized.
        
            True if the agent is initialized, False otherwise.
        """
        return self._initialized and self._openai_agent is not None
    
    async def initialize(self) -> None:
        """
        Initialize the OpenAI agent.
        
        """
        if self.is_initialized():
            return
            
        try:
            model_settings = ModelSettings(
                temperature={{ config.parameters.get("temperature", 0.7) }},
                {% if config.parameters.get("max_tokens") %}
                max_tokens={{ config.parameters.get("max_tokens") }},
                {% endif %}
            )
            
            self._openai_agent = Agent(
                name=self.agent_id,
                instructions=self.system_prompt,
                tools=self.tools,
                model="{{ config.model }}",
                {% if config.output_type %}
                output_type={{ config.output_type.split("class ")[1].split("(")[0] if config.output_type else "None" }},
                {% endif %}
                model_settings=model_settings
            )
            
            self._initialized = True
            logger.info(f"{{ config.name }} agent {self.agent_id} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {str(e)}")
            raise RuntimeError(f"Agent initialization failed: {str(e)}")
    
    async def execute(
        self, 
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the agent with the given input.
        
            input_text: Input text to send to the agent.
            context: Optional context information.
            
            Response dictionary with content and metadata.
            
            RuntimeError: If the agent fails to execute.
        """
        try:
            if not self.is_initialized():
                await self.initialize()
            
            result = await Runner.run(
                self._openai_agent,
                input=input_text,
                max_turns=self.parameters.get("max_turns", 10)
            )
            
            {% if config.output_type %}
            output = result.final_output
            {% else %}
            if hasattr(result, 'final_output') and result.final_output:
                output = result.final_output
            else:
                output = str(result)
            {% endif %}
            
            response = {
                "content": output,
                "agent_id": self.agent_id,
                "success": True,
                "metadata": {
                    "agent_type": "{{ config.agent_type }}",
                    "role": "{{ framework_role }}"
                }
            }
            
            return response
            
        except Exception as e:
            error_msg = f"Error executing agent {self.agent_id}: {str(e)}"
            logger.error(error_msg)
            
            return {
                "content": f"Error: {str(e)}",
                "agent_id": self.agent_id,
                "success": False,
                "metadata": {
                    "agent_type": "{{ config.agent_type }}",
                    "error": str(e)
                }
            }
"""
    
    def _get_agent_config_template(self) -> str:
        """
        Get the default agent config template.
        
            Default YAML configuration template string.
        """
        return """# Agent Configuration for {{ config.name }}
agent_type: {{ config.agent_type }}
name: {{ config.name }}
role: {{ framework_role }}
instructions: |
  {{ config.instructions | indent(2) }}
model: {{ config.model }}
parameters:
  temperature: {{ config.parameters.get("temperature", 0.7) }}
  {% if config.parameters.get("max_tokens") %}
  max_tokens: {{ config.parameters.get("max_tokens") }}
  {% endif %}
tools:
{% for tool in config.tools %}
  - {{ tool }}
{% endfor %}
handoffs:
{% for handoff in config.handoffs %}
  - {{ handoff }}
{% endfor %}
{% if config.output_type %}
output_type: |
  {{ config.output_type | indent(2) }}
{% endif %}
"""
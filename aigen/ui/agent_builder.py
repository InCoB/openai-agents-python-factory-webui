"""Agent builder UI components."""

import asyncio
import logging
import re
import json
import inspect
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Callable, Generator, AsyncGenerator

import gradio as gr

from aigen.services.models import AgentConfiguration, AgentRole
from aigen.services.generator import AgentGeneratorService
from aigen.services.registration import AgentRegistrationService
from aigen.services.testing import AgentTestingService
from aigen.services.persistence import AgentPersistenceService
from aigen.ui.utils import dict_to_string_adapter
from aigen.core.context import Context
from aigen.agents import factory

logger = logging.getLogger(__name__)

class AgentBuilderUI:
    """UI components for agent builder."""
    
    def __init__(self) -> None:
        """Initialize the agent builder UI."""
        self.generator = AgentGeneratorService()
        self.registrar = AgentRegistrationService()
        self.tester = AgentTestingService()
        self.persistence = AgentPersistenceService()
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools.
        
        """
        try:
            from aigen.tools.factory import tool_registry
            return sorted(tool_registry.list())
        except ImportError:
            logger.warning("Failed to import tool registry")
            return []
    
    def get_available_agents(self) -> List[str]:
        """
        Get list of available agents.
        
        """
        try:
            from aigen.agents.factory import agent_registry
            return sorted(agent_registry.list())
        except ImportError:
            logger.warning("Failed to import agent registry")
            return []
    
    def validate_agent_type(self, agent_type: str) -> Tuple[bool, str]:
        """
        Validate agent type.
        
            agent_type: Agent type to validate.
            
            Tuple containing:
                - Valid flag (True if valid)
                - Error message if invalid, empty if valid
        """
        if not agent_type:
            return False, "Agent type is required"
            
        pattern = r'^[a-z][a-z0-9_]*$'
        if not re.match(pattern, agent_type):
            return False, (
                "Agent type must start with a lowercase letter and "
                "contain only lowercase letters, numbers, and underscores"
            )
        
        try:
            from aigen.agents.factory import agent_registry
            if hasattr(agent_registry, "exists") and agent_registry.exists(agent_type):
                return False, f"Agent type '{agent_type}' already exists"
            elif hasattr(agent_registry, "_factories") and agent_type in agent_registry._factories:
                return False, f"Agent type '{agent_type}' already exists"
        except ImportError:
            pass
        
        return True, ""
    
    def validate_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate agent name.
        
            name: Agent name to validate.
            
            Tuple containing:
                - Valid flag (True if valid)
                - Error message if invalid, empty if valid
        """
        if not name:
            return False, "Name is required"
        return True, ""
    
    def validate_instructions(self, instructions: str) -> Tuple[bool, str]:
        """
        Validate agent instructions.
        
            instructions: Agent instructions to validate.
            
            Tuple containing:
                - Valid flag (True if valid)
                - Error message if invalid, empty if valid
        """
        if not instructions:
            return False, "Instructions are required"
        
        if len(instructions) < 20:
            return False, "Instructions should be at least 20 characters"
            
        return True, ""
    
    def generate_code(
        self, 
        agent_type: str, 
        name: str, 
        role: str, 
        model: str, 
        instructions: str, 
        temperature: float, 
        max_tokens: int, 
        tools: List[str], 
        handoffs: List[str], 
        use_output_type: bool, 
        output_type_code: str
    ) -> Tuple[str, str]:
        """
        Generate agent code from form inputs.
        
            agent_type: Agent type ID.
            name: Display name.
            role: Agent role.
            model: Model name.
            instructions: Agent instructions.
            temperature: Temperature parameter.
            max_tokens: Max tokens parameter.
            tools: Selected tools.
            handoffs: Selected handoffs.
            use_output_type: Whether to use structured output.
            output_type_code: Output type code.
            
            Tuple containing:
                - Generated Python code
                - Generated YAML config
        """
        try:
            valid, error = self.validate_agent_type(agent_type)
            if not valid:
                return f"# Error: {error}", f"# Error: {error}"
                
            valid, error = self.validate_name(name)
            if not valid:
                return f"# Error: {error}", f"# Error: {error}"
                
            valid, error = self.validate_instructions(instructions)
            if not valid:
                return f"# Error: {error}", f"# Error: {error}"
            
            config = AgentConfiguration(
                agent_type=agent_type,
                name=name,
                role=AgentRole(role),
                instructions=instructions,
                model=model,
                parameters={
                    "temperature": temperature, 
                    "max_tokens": max_tokens
                },
                tools=tools or [],
                handoffs=handoffs or [],
                output_type=output_type_code if use_output_type else None
            )
            
            success, code = self.generator.generate_agent_code(config)
            success_yaml, yaml_config = self.generator.generate_yaml_config(config)
            
            if not success:
                return code, "# Error generating YAML configuration"
                
            if not success_yaml:
                return code, yaml_config
                
            return code, yaml_config
            
        except Exception as e:
            error_msg = f"Error generating code: {str(e)}"
            logger.error(error_msg)
            return f"# Error: {error_msg}", f"# Error: {error_msg}"
    
    async def register_agent(
        self, 
        agent_type: str, 
        name: str, 
        role: str, 
        model: str, 
        instructions: str, 
        temperature: float, 
        max_tokens: int, 
        tools: List[str], 
        handoffs: List[str], 
        use_output_type: bool, 
        output_type_code: str, 
        code: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Register agent with the framework.
        
            agent_type: Agent type ID.
            name: Display name.
            role: Agent role.
            model: Model name.
            instructions: Agent instructions.
            temperature: Temperature parameter.
            max_tokens: Max tokens parameter.
            tools: Selected tools.
            handoffs: Selected handoffs.
            use_output_type: Whether to use structured output.
            output_type_code: Output type code.
            code: Generated Python code.
            
            Status messages during registration.
        """
        try:
            yield {"status": "progress", "message": "⏳ Validating inputs..."}
            
            valid, error = self.validate_agent_type(agent_type)
            if not valid:
                yield {"status": "error", "message": f"❌ {error}"}
                return
                
            valid, error = self.validate_name(name)
            if not valid:
                yield {"status": "error", "message": f"❌ {error}"}
                return
                
            valid, error = self.validate_instructions(instructions)
            if not valid:
                yield {"status": "error", "message": f"❌ {error}"}
                return
                
            if not code or code.startswith("# Error"):
                yield {"status": "error", "message": "❌ Invalid code. Please generate valid code first."}
                return
            
            yield {"status": "progress", "message": "⏳ Creating agent configuration..."}
            
            config = AgentConfiguration(
                agent_type=agent_type,
                name=name,
                role=AgentRole(role),
                instructions=instructions,
                model=model,
                parameters={
                    "temperature": temperature, 
                    "max_tokens": max_tokens
                },
                tools=tools or [],
                handoffs=handoffs or [],
                output_type=output_type_code if use_output_type else None
            )
            
            yield {"status": "progress", "message": "⏳ Registering agent..."}
            
            success, message = self.registrar.register_agent_type(config, code)
            
            if not success:
                yield {"status": "error", "message": f"❌ Registration failed: {message}"}
                return
            
            yield {"status": "progress", "message": "⏳ Saving configuration..."}
            
            success, message = self.persistence.save_agent_config(config)
            
            if not success:
                yield {"status": "warning", "message": f"⚠️ Agent registered but configuration could not be saved: {message}"}
                return
            
            yield {"status": "success", "message": f"✅ Agent '{name}' ({agent_type}) registered successfully!"}
            
        except Exception as e:
            error_msg = f"Error registering agent: {str(e)}"
            logger.error(error_msg)
            yield {"status": "error", "message": f"❌ {error_msg}"}
    
    async def test_agent(
        self,
        agent_type: str,
        agent_name: str,
        agent_instructions: str,
        test_input: str,
    ) -> Dict[str, Any]:
        """Test the agent by creating it with the given parameters and running it with a test prompt."""
        logger.debug(f"Testing agent: type={agent_type}, name={agent_name}")
        
        try:
            # Create context
            context = Context()
            context.store_output("system", test_input or "Please respond to this test message with a brief introduction of yourself.")
            
            # Create agent
            logger.debug(f"Creating agent with factory.create_agent({agent_type}, {agent_name})")
            agent = factory.create_agent(
                agent_type=agent_type,
                agent_id=agent_name,
                instructions=agent_instructions
            )
            
            # Initialize and execute
            logger.debug(f"Initializing agent {agent_name}")
            await agent.initialize()
            
            logger.debug(f"Executing agent {agent_name}")
            response = await agent.execute(context)
            
            # Return results
            logger.debug(f"Agent execution complete: {response}")
            return {
                "status": "success",
                "message": f"Agent test completed successfully!",
                "content": response.content,
                "agent_id": response.agent_id,
                "metadata": response.metadata
            }
        except Exception as e:
            logger.error(f"Error testing agent: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error testing agent: {str(e)}"
            }
    
    def build_ui(self) -> gr.Tab:
        """
        Build the agent builder UI tab.
        
            Gradio Tab component for the agent builder.
        """
        with gr.Tab("Agent Builder") as tab:
            gr.Markdown("## Create Custom Agent")
            gr.Markdown("""
            Build your own agent with specialized capabilities.
            
            tools, and other properties. Once created, these agents can be used in workflows.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    agent_type = gr.Textbox(
                        label="Agent Type ID",
                        placeholder="e.g., financial_analyst",
                        info="Unique identifier (lowercase, underscores)"
                    )
                    agent_type_validation = gr.HTML(visible=False)
                
                with gr.Column(scale=2):
                    agent_name = gr.Textbox(
                        label="Display Name",
                        placeholder="e.g., Financial Analysis Agent",
                        info="Human-readable name"
                    )
                    name_validation = gr.HTML(visible=False)
            
            with gr.Row():
                with gr.Column(scale=1):
                    agent_role = gr.Dropdown(
                        choices=[role.value for role in AgentRole],
                        value=AgentRole.CUSTOM.value,
                        label="Agent Role",
                        info="Functional role of the agent"
                    )
                
                with gr.Column(scale=1):
                    agent_model = gr.Dropdown(
                        choices=[
                            "gpt-4o",
                            "gpt-3.5-turbo",
                            "gpt-4-1106-preview",
                            "claude-3-opus-20240229",
                            "claude-3-sonnet-20240229"
                        ],
                        value="gpt-4o",
                        label="Model",
                        info="LLM to use for this agent"
                    )
            
            instructions = gr.Textbox(
                lines=8,
                label="Instructions/System Prompt",
                placeholder="You are a specialized agent that...",
                info="Detailed instructions for the agent's behavior"
            )
            instructions_validation = gr.HTML(visible=False)
            
            with gr.Accordion("Parameters", open=False):
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature",
                        info="Controls randomness (0.0 = deterministic, 2.0 = random)"
                    )
                    max_tokens = gr.Slider(
                        minimum=100, maximum=8000, value=4000, step=100,
                        label="Max Tokens",
                        info="Maximum response length"
                    )
            
            with gr.Accordion("Tools", open=True):
                tools = gr.CheckboxGroup(
                    choices=self.get_available_tools(),
                    label="Available Tools",
                    info="Select tools the agent can use"
                )
            
            with gr.Accordion("Handoffs", open=False):
                handoffs = gr.CheckboxGroup(
                    choices=self.get_available_agents(),
                    label="Handoff Targets",
                    info="Select agents this agent can hand off to"
                )
            
            with gr.Accordion("Output Type", open=False):
                use_output_type = gr.Checkbox(
                    label="Use Structured Output",
                    value=False,
                    info="Define a structured output type for this agent"
                )
                
                output_type_code = gr.Code(
                    language="python",
                    label="Output Type Definition",
                    value="class CustomOutput(BaseModel):\n    summary: str\n    recommendations: List[str]",
                    visible=False,
                    lines=8
                )
            
            with gr.Row():
                generate_btn = gr.Button("Generate Code", variant="primary")
                register_btn = gr.Button("Register Agent", variant="secondary")
                test_btn = gr.Button("Test Agent", variant="secondary")
            
            with gr.Accordion("Generated Code", open=True):
                preview_code = gr.Code(
                    language="python",
                    label="Agent Implementation",
                    lines=20
                )
                
                preview_yaml = gr.Code(
                    language="yaml",
                    label="YAML Configuration",
                    lines=10
                )
            
            with gr.Accordion("Test Your Agent", open=False):
                test_input = gr.Textbox(
                    label="Test Input",
                    placeholder="Enter test prompt here...",
                    lines=3
                )
                
                test_output = gr.JSON(
                    label="Test Output",
                )
            
            registration_result = gr.JSON(label="Registration Result")
            
            def on_agent_type_change(agent_type: str) -> Dict[str, Any]:
                valid, message = self.validate_agent_type(agent_type)
                if not valid:
                    return {
                        "visible": True,
                        "value": f"❌ {message}"
                    }
                return {
                    "visible": False,
                    "value": ""
                }
            
            def on_name_change(name: str) -> Dict[str, Any]:
                valid, message = self.validate_name(name)
                if not valid:
                    return {
                        "visible": True,
                        "value": f"❌ {message}"
                    }
                return {
                    "visible": False,
                    "value": ""
                }
            
            def on_instructions_change(instructions: str) -> Dict[str, Any]:
                valid, message = self.validate_instructions(instructions)
                if not valid:
                    return {
                        "visible": True,
                        "value": f"❌ {message}"
                    }
                return {
                    "visible": False,
                    "value": ""
                }
            
            def toggle_output_type(use_output: bool) -> Dict[str, Any]:
                return {"visible": use_output}
            
            agent_type.change(
                on_agent_type_change,
                inputs=[agent_type],
                outputs=[agent_type_validation]
            )
            
            agent_name.change(
                on_name_change,
                inputs=[agent_name],
                outputs=[name_validation]
            )
            
            instructions.change(
                on_instructions_change,
                inputs=[instructions],
                outputs=[instructions_validation]
            )
            
            use_output_type.change(
                toggle_output_type,
                inputs=[use_output_type],
                outputs=[output_type_code]
            )
            
            generate_btn.click(
                self.generate_code,
                inputs=[
                    agent_type, agent_name, agent_role, agent_model, instructions,
                    temperature, max_tokens, tools, handoffs, use_output_type, output_type_code
                ],
                outputs=[preview_code, preview_yaml]
            )
            
            register_btn.click(
                dict_to_string_adapter(self.register_agent),
                inputs=[
                    agent_type, agent_name, agent_role, agent_model, instructions,
                    temperature, max_tokens, tools, handoffs, use_output_type, 
                    output_type_code, preview_code
                ],
                outputs=[registration_result]
            )
            
            test_btn.click(
                dict_to_string_adapter(self.test_agent),
                inputs=[agent_type, agent_name, instructions, test_input],
                outputs=[test_output]
            )
            
            return tab
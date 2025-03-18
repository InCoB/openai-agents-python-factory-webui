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

        pattern = r"^[a-z][a-z0-9_]*$"
        if not re.match(pattern, agent_type):
            return False, (
                "Agent type must start with a lowercase letter and "
                "contain only lowercase letters, numbers, and underscores"
            )

        try:
            from aigen.agents.factory import agent_registry

            if hasattr(agent_registry, "exists") and agent_registry.exists(agent_type):
                return False, f"Agent type '{agent_type}' already exists"
            elif (
                hasattr(agent_registry, "get_registry")
                and agent_type in agent_registry.get_registry()
            ):
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

        Args:
            instructions: Instructions to validate

        Returns:
            Tuple containing:
                - Valid flag (True if valid)
                - Error message if invalid, empty if valid
        """
        return self.validate_instructions_content(instructions)

    def validate_instructions_content(self, instructions: str) -> Tuple[bool, str]:
        """
        Validate agent instructions content.

        Args:
            instructions: Agent instructions to validate.

        Returns:
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
        output_type_code: str,
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
                parameters={"temperature": temperature, "max_tokens": max_tokens},
                tools=tools or [],
                handoffs=handoffs or [],
                output_type=output_type_code if use_output_type else None,
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
        code: str,
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
                yield {
                    "status": "error",
                    "message": "❌ Invalid code. Please generate valid code first.",
                }
                return

            yield {
                "status": "progress",
                "message": "⏳ Creating agent configuration...",
            }

            config = AgentConfiguration(
                agent_type=agent_type,
                name=name,
                role=AgentRole(role),
                instructions=instructions,
                model=model,
                parameters={"temperature": temperature, "max_tokens": max_tokens},
                tools=tools or [],
                handoffs=handoffs or [],
                output_type=output_type_code if use_output_type else None,
            )

            yield {"status": "progress", "message": "⏳ Registering agent..."}

            success, message = self.registrar.register_agent_type(config, code)

            if not success:
                yield {
                    "status": "error",
                    "message": f"❌ Registration failed: {message}",
                }
                return

            yield {"status": "progress", "message": "⏳ Saving configuration..."}

            success, message = self.persistence.save_agent_config(config)

            if not success:
                yield {
                    "status": "warning",
                    "message": f"⚠️ Agent registered but configuration could not be saved: {message}",
                }
                return

            yield {
                "status": "success",
                "message": f"✅ Agent '{name}' ({agent_type}) registered successfully!",
            }

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
        logger.debug("Testing agent: type=%s, name=%s", agent_type, agent_name)

        try:
            # Create context
            context = Context()
            context.store_output(
                "system",
                test_input
                or "Please respond to this test message with a brief introduction of yourself.",
            )

            # Create agent
            logger.debug(
                "Creating agent with factory.create_agent(%s, %s)",
                agent_type,
                agent_name,
            )
            agent = factory.create_agent(
                agent_type=agent_type,
                agent_id=agent_name,
                instructions=agent_instructions,
            )

            # Initialize and execute
            logger.debug("Initializing agent %s", agent_name)
            await agent.initialize()

            logger.debug("Executing agent %s", agent_name)
            response = await agent.execute(context)

            # Return results
            logger.debug("Agent execution complete: %s", response)
            return {
                "status": "success",
                "message": f"Agent test completed successfully!",
                "content": response.content,
                "agent_id": response.agent_id,
                "metadata": response.metadata,
            }
        except Exception as e:
            logger.error("Error testing agent: %s", str(e), exc_info=True)
            return {"status": "error", "message": f"Error testing agent: {str(e)}"}

    def on_agent_type_change(self, agent_type: str) -> str:
        """
        Validate agent type as user types.

        Args:
            agent_type: Agent type to validate

        Returns:
            Validation message HTML
        """
        valid, error = self.validate_agent_type(agent_type)
        if not valid:
            return f"<span style='color: red'>❌ {error}</span>"
        else:
            return "<span style='color: green'>✓ Valid agent type</span>"

    def on_name_change(self, agent_type: str, name: str) -> str:
        """
        Validate agent name.

        Args:
            agent_type: Agent type
            name: Display name to validate

        Returns:
            Validation message HTML
        """
        return self.validate_display_name(agent_type, name)

    def validate_display_name(self, agent_type: str, display_name: str) -> str:
        """
        Validate that display name follows convention based on agent type.

        Args:
            agent_type: The agent type ID (snake_case)
            display_name: The display name to validate

        Returns:
            Validation message HTML
        """
        if not agent_type or not display_name:
            return ""

        # Check PascalCase format (each word should start with uppercase, no spaces)
        pascal_case_pattern = re.compile(r"^[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*$")
        if not pascal_case_pattern.match(display_name):
            return "<span style='color: red'>❌ Name must be in PascalCase (each word capitalized, no spaces)</span>"

        # Convert agent_type from snake_case to PascalCase for comparison
        parts = agent_type.split("_")
        expected_name = "".join(part.capitalize() for part in parts)

        # Check if display_name matches the PascalCase version of agent_type
        if display_name == expected_name:
            return "<span style='color: green'>✓ Display name matches convention</span>"
        else:
            return f"<span style='color: orange'>⚠️ Recommended: <b>{expected_name}</b> to match agent type</span>"

    def suggest_display_name(self, agent_type: str) -> str:
        """
        Suggest a display name based on agent type.

        Args:
            agent_type: The agent type ID (snake_case)

        Returns:
            Suggested display name in PascalCase
        """
        if not agent_type:
            return ""

        # Convert agent_type from snake_case to PascalCase
        parts = agent_type.split("_")
        suggested_name = "".join(part.capitalize() for part in parts)

        return suggested_name

    def on_instructions_change(self, instructions: str) -> str:
        """
        Validate instructions and return formatted HTML.

        Args:
            instructions: Instructions to validate

        Returns:
            Validation message HTML
        """
        valid, error = self.validate_instructions(instructions)
        if not valid:
            return f"<span style='color: red'>❌ {error}</span>"
        else:
            return "<span style='color: green'>✓ Valid instructions</span>"

    def build_ui(self) -> gr.Tab:
        """
        Build the agent builder UI tab.

            Gradio Tab component for the agent builder.
        """
        with gr.Tab("Agent Builder") as tab:
            gr.Markdown("## Create Custom Agent")
            gr.Markdown(
                """
            Build your own agent with specialized capabilities.
            
            tools, and other properties. Once created, these agents can be used in workflows.
            """
            )

            # Add helpful naming convention guidance
            gr.Markdown(
                """
            ### Naming Convention Guidelines
            - **Agent Type ID**: Use snake_case (lowercase with underscores) - e.g., `product_agent`
            - **Display Name**: Use matching PascalCase without spaces - e.g., `ProductAgent`
            
            The system will add "Agent" suffix to your Display Name when creating the class.
            """,
                elem_id="naming_guidelines",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    agent_type = gr.Textbox(
                        label="Agent Type ID",
                        placeholder="Unique identifier (lowercase, underscores)",
                        info="Must be snake_case: lowercase letters, numbers, underscores",
                    )
                    agent_type_status = gr.Markdown("")

                with gr.Column(scale=2):
                    agent_name = gr.Textbox(
                        label="Display Name",
                        placeholder="Human-readable name",
                        info="Should be PascalCase matching your agent type",
                    )
                    agent_name_status = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    agent_role = gr.Dropdown(
                        choices=[role.value for role in AgentRole],
                        value=AgentRole.CUSTOM.value,
                        label="Agent Role",
                        info="Functional role of the agent",
                    )

                with gr.Column(scale=1):
                    agent_model = gr.Dropdown(
                        choices=[
                            "gpt-4o",
                            "gpt-3.5-turbo",
                            "gpt-4-1106-preview",
                            "claude-3-opus-20240229",
                            "claude-3-sonnet-20240229",
                        ],
                        value="gpt-4o",
                        label="Model",
                        info="LLM to use for this agent",
                    )

            instructions = gr.Textbox(
                lines=8,
                label="Instructions/System Prompt",
                placeholder="You are a specialized agent that...",
                info="Detailed instructions for the agent's behavior",
            )
            instructions_validation = gr.HTML(visible=False)

            with gr.Accordion("Parameters", open=False):
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness (0.0 = deterministic, 2.0 = random)",
                    )
                    max_tokens = gr.Slider(
                        minimum=100,
                        maximum=8000,
                        value=4000,
                        step=100,
                        label="Max Tokens",
                        info="Maximum response length",
                    )

            with gr.Accordion("Tools", open=True):
                tools = gr.CheckboxGroup(
                    choices=self.get_available_tools(),
                    label="Available Tools",
                    info="Select tools the agent can use",
                )

            with gr.Accordion("Handoffs", open=False):
                handoffs = gr.CheckboxGroup(
                    choices=self.get_available_agents(),
                    label="Handoff Targets",
                    info="Select agents this agent can hand off to",
                )

            with gr.Accordion("Output Type", open=False):
                use_output_type = gr.Checkbox(
                    label="Use Structured Output",
                    value=False,
                    info="Define a structured output type for this agent",
                )

                output_type_code = gr.Code(
                    language="python",
                    label="Output Type Definition",
                    value="class CustomOutput(BaseModel):\n    summary: str\n    recommendations: List[str]",
                    visible=False,
                    lines=8,
                )

            with gr.Row():
                generate_btn = gr.Button("Generate Code", variant="primary")
                register_btn = gr.Button("Register Agent", variant="secondary")
                test_btn = gr.Button("Test Agent", variant="secondary")

            with gr.Accordion("Generated Code", open=True):
                preview_code = gr.Code(
                    language="python", label="Agent Implementation", lines=20
                )

                preview_yaml = gr.Code(
                    language="yaml", label="YAML Configuration", lines=10
                )

            with gr.Accordion("Test Your Agent", open=False):
                test_input = gr.Textbox(
                    label="Test Input", placeholder="Enter test prompt here...", lines=3
                )

                test_output = gr.JSON(
                    label="Test Output",
                )

            registration_result = gr.JSON(label="Registration Result")

            agent_type.change(
                self.on_agent_type_change,
                inputs=[agent_type],
                outputs=[agent_type_status],
            )

            agent_name.change(
                self.on_name_change,
                inputs=[agent_type, agent_name],
                outputs=[agent_name_status],
            )

            agent_type.change(
                self.suggest_display_name, inputs=[agent_type], outputs=[agent_name]
            )

            instructions.change(
                self.on_instructions_change,
                inputs=[instructions],
                outputs=[instructions_validation],
            )

            use_output_type.change(
                lambda use_output: {"visible": use_output},
                inputs=[use_output_type],
                outputs=[output_type_code],
            )

            generate_btn.click(
                self.generate_code,
                inputs=[
                    agent_type,
                    agent_name,
                    agent_role,
                    agent_model,
                    instructions,
                    temperature,
                    max_tokens,
                    tools,
                    handoffs,
                    use_output_type,
                    output_type_code,
                ],
                outputs=[preview_code, preview_yaml],
            )

            register_btn.click(
                dict_to_string_adapter(self.register_agent),
                inputs=[
                    agent_type,
                    agent_name,
                    agent_role,
                    agent_model,
                    instructions,
                    temperature,
                    max_tokens,
                    tools,
                    handoffs,
                    use_output_type,
                    output_type_code,
                    preview_code,
                ],
                outputs=[registration_result],
            )

            test_btn.click(
                dict_to_string_adapter(self.test_agent),
                inputs=[agent_type, agent_name, instructions, test_input],
                outputs=[test_output],
            )

            return tab

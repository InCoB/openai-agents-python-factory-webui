import os
import gradio as gr
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
import uuid
import logging

from ..core.logging import get_logger
from ..core.context import Context
from ..core.config import ConfigManager
from ..workflows.factory import create_workflow, workflow_registry
from ..agents.factory import agent_registry
from agents import trace, gen_trace_id
from .agent_builder import AgentBuilderUI
from .agent_manager import AgentManagerUI

logger = get_logger("gradio_interface")


class GradioInterface:
    """
    Gradio web interface for the agent workflow framework.
    Provides an interactive UI for creating and executing workflows.
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the Gradio interface.

            config: Optional configuration manager
        """
        self.config = config or ConfigManager()
        self.title = "AI Agent Workflow Framework"
        self.description = """
        
        Build and execute workflows with specialized AI agents. Create custom agent combinations 
        for research, content generation, analysis, and more.
        """

        self.available_workflows = self._get_available_workflows()
        self.available_agents = self._get_available_agents()

    def _get_available_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get available workflow templates."""
        workflows = {}

        for name in workflow_registry.list():
            metadata = workflow_registry.get_metadata(name) or {}
            workflows[name] = {
                "description": metadata.get("description", ""),
                "type": metadata.get("type", "workflow"),
            }

        return workflows

    def _get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get available agent templates."""
        agents = {}

        for name in agent_registry.list():
            metadata = agent_registry.get_metadata(name) or {}
            agents[name] = {
                "description": metadata.get("description", ""),
                "type": metadata.get("type", "agent"),
            }

        return agents

    def _get_workflow_choices(self) -> List[str]:
        """Get workflow choices for dropdown."""
        # Filter out base workflow types (handoff, deterministic) which are implementation details
        # Only show concrete workflow instances to users
        base_workflow_types = ["handoff", "deterministic"]
        filtered_workflows = [
            name
            for name in self.available_workflows.keys()
            if name not in base_workflow_types
        ]

        # Don't include "custom" in the dropdown - we'll use a separate tab for that
        return filtered_workflows

    def _get_agent_choices(self) -> List[str]:
        """Get agent choices for dropdown."""
        return list(self.available_agents.keys())

    def update_agent_selection(self, workflow_name: str) -> List[str]:
        """
        Get the agents for a selected workflow.

            workflow_name: Selected workflow name

            List[str]: Agents for the selected workflow
        """
        return self._get_workflow_agents(workflow_name)

    def _get_workflow_agents(self, workflow_name: str) -> List[str]:
        """Get the agents belonging to a workflow."""
        workflow_agents = {
            "research_only": ["research"],
            "content_generation": ["research", "writer", "editor"],
            "content_analysis": ["research", "editor"],
        }

        # Try to get actual agents from workflow registry
        if workflow_name:
            try:
                # This won't affect registration since we're not calling register_workflow_factory
                factory = workflow_registry.get(workflow_name)
                if callable(factory):
                    try:
                        temp_workflow = factory()
                        if hasattr(temp_workflow, "agents"):
                            return [agent.agent_id for agent in temp_workflow.agents]
                    except (TypeError, AttributeError, ValueError) as e:
                        logger.debug(f"Error accessing workflow agents from factory: {e}")
            except KeyError as e:
                logger.debug(f"Error retrieving workflow from registry: {e}")

        return workflow_agents.get(workflow_name, [])

    async def run_workflow(
        self,
        workflow_name: str,
        custom_agents: List[str],
        input_text: str,
        max_turns: int = 80,
        progress=gr.Progress(),
    ) -> str:
        """Run a workflow with the provided parameters."""
        progress(0, "Initializing workflow...")

        try:
            # Set up the workflow spec based on the selection
            if workflow_name == "custom":
                workflow_spec = custom_agents
                logger.info(
                    f"Creating custom workflow with agents: {', '.join(custom_agents)}"
                )
            else:
                workflow_spec = workflow_name
                logger.info(f"Creating standard workflow: {workflow_name}")

            # Create a context
            context = Context()

            # Create the workflow
            workflow = create_workflow(workflow_spec)

            # Create workflow engine
            from ..workflows.engine import WorkflowEngine

            engine = WorkflowEngine()

            progress(0.1, "Executing workflow...")

            # Set up progress callback
            def update_progress(current_step, total_steps, message):
                # Calculate progress percentage (from 0.1 to 0.9)
                percent = (
                    0.1 + (0.8 * (current_step / total_steps))
                    if total_steps > 0
                    else 0.5
                )
                progress(percent, message)

            # Add progress callback to context
            context.set_metadata("progress_callback", update_progress)

            # Execute the workflow - the engine will handle tracing
            result = await engine.execute(workflow, context, input_text)

            progress(0.9, "Completing workflow...")

            # Get the trace ID from the result
            workflow_id = result.get("workflow_id", "")
            trace_url = (
                f"https://platform.openai.com/traces/{workflow_id}"
                if workflow_id
                else ""
            )

            # Return the result with trace information - ensure it's always a string
            if result["status"] == "completed":
                # Always include trace info for better user experience
                trace_info = f"\n\n---\nTrace ID: {workflow_id}\nTrace URL: {trace_url}"
                return str(result["result"]) + trace_info
            else:
                error_msg = str(result.get("error", "Unknown error"))
                partial_result = str(result.get("result", ""))
                return f"Workflow failed: {error_msg}\n\n{partial_result}\n\nTrace ID: {workflow_id}"

        except Exception as e:
            logger.error(f"Error executing workflow {workflow_name}: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            progress(1.0, "Workflow complete")

    def build_ui(self) -> gr.Blocks:
        """
        Build the Gradio UI.

            gr.Blocks: Gradio Blocks app
        """
        # Use a predefined theme for better compatibility
        with gr.Blocks(
            title=self.title,
            theme=gr.themes.Default(),
            css="""
                * {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                }
                h1, h2, h3, h4 {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
                }
            """,
        ) as app:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)

            with gr.Tabs() as tabs:
                # Predefined workflow tab
                with gr.TabItem("Use Predefined Workflow"):
                    # Input section
                    gr.Markdown("## Input")
                    predefined_input_text = gr.Textbox(
                        placeholder="Enter your prompt or request here...",
                        lines=5,
                        label="Input Text",
                    )

                    # Configuration section
                    gr.Markdown("## Workflow Configuration")
                    workflow_name = gr.Dropdown(
                        choices=self._get_workflow_choices(),
                        value=(
                            "content_generation"
                            if "content_generation" in self._get_workflow_choices()
                            else None
                        ),
                        label="Workflow Type",
                        info="Select a predefined workflow",
                    )

                    # Display the agents in the selected workflow
                    workflow_agents_info = gr.Textbox(
                        value="Workflow contains: research, strategy, writer, editor",
                        label="Selected Agents",
                        interactive=False,
                    )

                    max_turns_predefined = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=80,
                        step=10,
                        label="Max Turns",
                        info="Maximum number of execution steps",
                    )

                    # Execute button for predefined workflows
                    execute_predefined_btn = gr.Button(
                        "Execute Workflow", variant="primary"
                    )

                    # Output section
                    gr.Markdown("## Output")
                    predefined_output_text = gr.Textbox(label="Result", lines=20)

                # Custom workflow tab
                with gr.TabItem("Create Custom Workflow"):
                    gr.Markdown("## Custom Workflow Builder")
                    gr.Markdown(
                        """
                    Create your own workflow by selecting agents to execute in sequence.
                    The input will be passed through each agent in order, and the final output will be displayed.
                    """
                    )

                    # Input for custom workflow
                    custom_input_text = gr.Textbox(
                        placeholder="Enter your prompt or request here...",
                        lines=5,
                        label="Input Text",
                    )

                    # Agent selection for custom workflow
                    gr.Markdown("## Select Agents")
                    gr.Markdown(
                        "*Select agents in the order they should execute. The order matters!*"
                    )

                    custom_agents = gr.Dropdown(
                        choices=self._get_agent_choices(),
                        multiselect=True,
                        allow_custom_value=True,
                        value=["research", "writer"],
                        label="Custom Workflow Agents",
                        info="Select multiple agents to execute in sequence",
                    )

                    max_turns_custom = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=80,
                        step=10,
                        label="Max Turns",
                        info="Maximum number of execution steps",
                    )

                    # Execute button for custom workflow
                    execute_custom_btn = gr.Button(
                        "Execute Custom Workflow", variant="primary"
                    )

                    # Output for custom workflow
                    gr.Markdown("## Output")
                    custom_output_text = gr.Textbox(label="Result", lines=20)

                # Workflow Management tab
                with gr.TabItem("Workflow Management"):
                    gr.Markdown("## Available Workflows")
                    workflows_table = gr.DataFrame(
                        value=[
                            [name, info["description"]]
                            for name, info in self.available_workflows.items()
                        ],
                        headers=["Name", "Description"],
                        label="Workflows",
                    )

                    gr.Markdown("## Available Agents")
                    agents_table = gr.DataFrame(
                        value=[
                            [name, info["description"]]
                            for name, info in self.available_agents.items()
                        ],
                        headers=["Name", "Description"],
                        label="Agents",
                    )

            # Event handlers
            # Update workflow agent info when workflow changes
            workflow_name.change(
                fn=lambda wf: f"Workflow contains: {', '.join(self._get_workflow_agents(wf))}",
                inputs=[workflow_name],
                outputs=[workflow_agents_info],
            )

            # Execute predefined workflow
            execute_predefined_btn.click(
                fn=self.run_workflow,
                inputs=[
                    workflow_name,
                    gr.State([]),
                    predefined_input_text,
                    max_turns_predefined,
                ],
                outputs=[predefined_output_text],
            )

            # Execute custom workflow
            execute_custom_btn.click(
                fn=self.run_workflow,
                inputs=[
                    gr.State("custom"),
                    custom_agents,
                    custom_input_text,
                    max_turns_custom,
                ],
                outputs=[custom_output_text],
            )

        return app

    def launch(self, **kwargs) -> None:
        """
        Launch the Gradio interface.

            **kwargs: Parameters to pass to gr.launch()
        """
        port = kwargs.get("server_port", self.config.get("ui.gradio_port", 7860))
        host = kwargs.get("server_name", self.config.get("ui.gradio_host", "127.0.0.1"))

        app = self.build_ui()
        app.launch(
            server_name=host,
            server_port=port,
            show_error=True,
            favicon_path=None,
            show_api=False,
            inbrowser=True,
            share=kwargs.get("share", False),
        )


class GradioApp:
    """Main Gradio application."""
    
    def __init__(
        self,
        title: str = "AI Gen Framework",
        description: str = "Generate and run AI agents"
    ) -> None:
        """
        Initialize the Gradio application.
        
            title: Application title.
            description: Application description.
        """
        self.title = title
        self.description = description
        
        self.agent_builder = AgentBuilderUI()
        self.agent_manager = AgentManagerUI()
    
    def build_ui(self) -> gr.Blocks:
        """
        Build the Gradio UI.
        
            Gradio Blocks application.
        """
        with gr.Blocks(title=self.title, theme=gr.themes.Default()) as app:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)
            
            with gr.Tabs() as tabs:
                with gr.TabItem("Use Predefined Workflow"):
                    gr.Markdown("Content for predefined workflows")
                
                with gr.TabItem("Create Custom Workflow"):
                    gr.Markdown("Content for custom workflows")
                
                self.agent_builder.build_ui()
                
                self.agent_manager.build_ui()
                
                with gr.TabItem("Workflow Management"):
                    gr.Markdown("Content for workflow management")
            
            return app
    
    def launch(
        self,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        share: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Launch the Gradio application.
        
            server_name: Server hostname.
            server_port: Server port.
            share: Whether to create a public link.
            **kwargs: Additional arguments to pass to gr.launch().
        """
        app = self.build_ui()
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            **kwargs
        )


def launch_ui(share: bool = False, **kwargs) -> None:
    """
    Launch the Gradio UI.

        share: Whether to create a publicly shareable link
        **kwargs: Additional parameters for gr.launch()
    """
    # Ensure factories are callable before launching UI
    from .. import _ensure_callable_factories

    _ensure_callable_factories()

    # Use GradioInterface by default for backward compatibility
    # To use the new GradioApp, set "use_new_ui" in kwargs
    if kwargs.pop("use_new_ui", False):
        app = GradioApp()
    else:
        app = GradioInterface()
        
    kwargs["share"] = share
    app.launch(**kwargs)

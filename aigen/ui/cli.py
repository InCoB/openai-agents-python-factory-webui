import asyncio
import argparse
import sys
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import socket

from ..core.logging import get_logger, configure_logging
from ..core.config import ConfigManager
from ..core.context import Context
from ..workflows.factory import create_workflow, workflow_registry
from ..agents.factory import agent_registry

logger = get_logger("cli")


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """
    Get input from the user with an optional default value.

        prompt: Prompt to display
        default: Optional default value

        str: User input or default
    """
    if default:
        user_input = input(f"{prompt} [{default}]: ")
        return user_input or default
    else:
        return input(f"{prompt}: ")


def get_selection(
    prompt: str, options: List[str], default: Optional[str] = None
) -> str:
    """
    Get a selection from a list of options.

        prompt: Prompt to display
        options: List of options
        default: Optional default option

        str: Selected option
    """
    print(f"\n{prompt}:")
    for i, option in enumerate(options, 1):
        default_marker = " (default)" if option == default else ""
        print(f"{i}. {option}{default_marker}")

    while True:
        selection = input("\nEnter selection (number): ")
        if not selection and default in options:
            return default

        try:
            index = int(selection) - 1
            if 0 <= index < len(options):
                return options[index]
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def save_output(output: str, file_path: Optional[str] = None) -> Optional[str]:
    """
    Save output to a file.

        output: Output content
        file_path: Optional file path

        Optional[str]: Path to saved file or None if failed
    """
    if not output:
        logger.warning("No output to save")
        return None

    if not file_path:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = f"output-{timestamp}.md"

    try:
        directory = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output)

        logger.info(f"Output saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        return None


async def interactive_mode() -> None:
    """Interactive CLI mode for workflow execution."""
    print("\n" + "=" * 50)
    print("AI Agent Workflow Framework".center(50))
    print("=" * 50)

    available_workflows = list(workflow_registry.list())
    available_agents = list(agent_registry.list())

    workflow_choices = ["custom"] + available_workflows
    workflow_selection = get_selection(
        "Select a workflow type",
        workflow_choices,
        default=(
            "content_generation"
            if "content_generation" in available_workflows
            else None
        ),
    )

    if workflow_selection == "custom":
        custom_agents = []
        print("\nSelect agents for your custom workflow (in order of execution).")
        print("Enter empty line when done.")

        while True:
            available = [a for a in available_agents if a not in custom_agents]
            if not available:
                break

            agent_options = available + ["Done"]
            selection = get_selection(
                f"Select agent #{len(custom_agents) + 1}", agent_options
            )

            if selection == "Done":
                break

            custom_agents.append(selection)

        if not custom_agents:
            print("No agents selected. Using default content generation workflow.")
            workflow_spec = "content_generation"
        else:
            print(f"\nSelected agents: {', '.join(custom_agents)}")
            workflow_spec = custom_agents
    else:
        workflow_spec = workflow_selection

    print("\nEnter your input text:")
    input_lines = []
    print("(Enter a blank line to finish)")

    while True:
        line = input()
        if not line:
            break
        input_lines.append(line)

    input_text = "\n".join(input_lines)

    if not input_text:
        print("No input provided. Exiting.")
        return

    print("\n" + "=" * 50)
    print("Executing Workflow".center(50))
    print("=" * 50)

    try:
        workflow = create_workflow(workflow_spec)

        from ..workflows.engine import WorkflowEngine

        engine = WorkflowEngine()

        context = Context()
        result = await engine.execute(workflow, context, input_text)

        if result["status"] == "completed":
            print("\n" + "=" * 50)
            print("Workflow Completed Successfully".center(50))
            print("=" * 50)
            print("\n" + result["result"])

            save = get_input("Save output to file? (y/n)", "y").lower() == "y"
            if save:
                file_path = get_input("Enter file path", "output.md")
                save_output(result["result"], file_path)
        else:
            print("\n" + "=" * 50)
            print("Workflow Failed".center(50))
            print("=" * 50)
            print(f"\nError: {result.get('error', 'Unknown error')}")

            if "result" in result:
                print("\nPartial result:")
                print(result["result"])

    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        print(f"\nError: {e}")


async def execute_workflow_command(args) -> int:
    """
    Execute a workflow from command-line arguments.

        args: Command-line arguments

        int: Exit code
    """
    input_text = None
    if args.input:
        input_text = args.input
    elif args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                input_text = f.read()
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            return 1
    else:
        logger.error("No input provided. Use --input or --input-file.")
        return 1

    if args.workflow:
        workflow_spec = args.workflow
    elif args.agents:
        workflow_spec = args.agents.split(",")
    else:
        workflow_spec = "content_generation"  # Default

    try:
        workflow = create_workflow(workflow_spec)

        context = Context()

        from ..workflows.engine import WorkflowEngine

        engine = WorkflowEngine(
            max_retries=args.max_retries, retry_delay=args.retry_delay
        )

        logger.info(f"Executing workflow: {workflow_spec}")
        result = await engine.execute(workflow, context, input_text)

        if result["status"] == "completed":
            output = result["result"]

            if args.output:
                save_output(output, args.output)
            else:
                print(output)

            return 0
        else:
            logger.error(f"Workflow failed: {result.get('error', 'Unknown error')}")

            if "result" in result:
                print(result["result"])

            return 1

    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        return 1


async def list_components_command(args) -> int:
    """
    List available components.

        args: Command-line arguments

        int: Exit code
    """
    if args.workflows or not (args.agents or args.tools):
        workflows = workflow_registry.list_with_metadata()

        print("\n=== Available Workflows ===\n")
        for name, metadata in workflows.items():
            print(f"- {name}")
            print(f"  Description: {metadata.get('description', 'No description')}")
            print()

    if args.agents or not (args.workflows or args.tools):
        agents = agent_registry.list_with_metadata()

        print("\n=== Available Agents ===\n")
        for name, metadata in agents.items():
            print(f"- {name}")
            print(f"  Description: {metadata.get('description', 'No description')}")
            print()

    if args.tools or not (args.workflows or args.agents):
        print("\n=== Available Tools ===\n")
        print("- tavily_search_tool")
        print("  Description: Search the web for information using Tavily")
        print()

    return 0


async def main_cli() -> int:
    """
    Main CLI entry point.

        int: Exit code
    """
    parser = argparse.ArgumentParser(description="AI Agent Workflow Framework")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    execute_parser = subparsers.add_parser("execute", help="Execute a workflow")
    execute_parser.add_argument("--workflow", help="Workflow name to execute")
    execute_parser.add_argument("--agents", help="Comma-separated list of agents")
    execute_parser.add_argument("--input", help="Input text")
    execute_parser.add_argument("--input-file", help="Input file path")
    execute_parser.add_argument("--output", help="Output file path")
    execute_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed agent executions",
    )
    execute_parser.add_argument(
        "--retry-delay", type=int, default=5, help="Delay between retries in seconds"
    )

    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")

    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument("--workflows", action="store_true", help="List workflows")
    list_parser.add_argument("--agents", action="store_true", help="List agents")
    list_parser.add_argument("--tools", action="store_true", help="List tools")

    web_parser = subparsers.add_parser("web", help="Launch web interface")
    web_parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    web_parser.add_argument(
        "--share", action="store_true", help="Create a publicly shareable link"
    )
    web_parser.add_argument(
        "--title", default="AI Generator Framework", help="Application title"
    )
    web_parser.add_argument(
        "--description",
        default="Create and manage custom AI agents",
        help="Application description",
    )

    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Log file path")

    args = parser.parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    configure_logging(log_level, args.log_file)

    if args.config:
        config = ConfigManager(args.config)
    else:
        config = ConfigManager()

    try:
        if args.command == "execute":
            return await execute_workflow_command(args)
        elif args.command == "interactive":
            await interactive_mode()
            return 0
        elif args.command == "list":
            return await list_components_command(args)
        elif args.command == "web":
            from .gradio_app import launch_ui

            # Resolve port conflicts
            port = args.port
            if port:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((args.host, port)) == 0:
                        # Port is in use, find an available one
                        logger.warning(
                            f"Port {port} is already in use, finding an available port"
                        )
                        for p in range(port, port + 100):
                            with socket.socket(
                                socket.AF_INET, socket.SOCK_STREAM
                            ) as s2:
                                if s2.connect_ex((args.host, p)) != 0:
                                    port = p
                                    logger.info(f"Using port {port} instead")
                                    break

            launch_ui(
                share=args.share,
                server_name=args.host,
                server_port=port,
                title=args.title,
                description=args.description,
            )
            return 0
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cli_entry_point() -> None:
    """CLI entry point for setuptools."""
    try:
        exit_code = asyncio.run(main_cli())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)

from typing import Dict, Any, Callable, Optional, List, Type
import importlib
import os
import inspect
import sys

from .base import AgentBase, AgentRole
from ..core.registry import Registry
from ..core.logging import get_logger

logger = get_logger("agent_factory")

agent_registry = Registry(Callable)


def register_agent_factory(
    name: str,
    factory: Callable[..., AgentBase],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register an agent factory function.

        name: Name to register the factory under
        factory: Function that creates agent instances
        metadata: Optional metadata for the factory
    """
    agent_registry.register_factory(name, factory, metadata)
    logger.info(f"Registered agent factory: {name}")


def register_agents_in_directory(directory: str) -> None:
    """
    Auto-register all agent classes found in Python files in the specified directory.

        directory: Path to directory containing agent files
    """
    # Get absolute path
    directory = os.path.abspath(directory)

    # Skip if directory doesn't exist
    if not os.path.exists(directory) or not os.path.isdir(directory):
        logger.warning(f"Directory {directory} does not exist or is not a directory")
        return

    # Find all Python files (excluding __init__.py)
    python_files = [
        f for f in os.listdir(directory) if f.endswith(".py") and not f.startswith("__")
    ]

    for filename in python_files:
        try:
            # Extract potential agent type from filename
            agent_type = filename.replace(".py", "").lower()

            # Skip already registered agents
            if agent_type in agent_registry.list():
                continue

            # Construct module path
            rel_path = os.path.relpath(
                directory, os.path.dirname(os.path.dirname(__file__))
            )
            module_path = f"aigen.{rel_path.replace(os.path.sep, '.')}.{agent_type}"

            # Import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                logger.debug(f"Could not import {module_path}: {str(e)}")
                continue

            # Find any class ending with "Agent"
            agent_classes = {}
            for name, obj in module.__dict__.items():
                if name.endswith("Agent") and inspect.isclass(obj):
                    agent_classes[name] = obj

            if not agent_classes:
                logger.debug(f"No agent classes found in {filename}")
                continue

            # Register each agent class
            for class_name, agent_class in agent_classes.items():
                # Create a factory function
                def make_factory(agent_cls, agent_name):
                    def factory(agent_id=None, **kwargs):
                        return agent_cls(agent_id=agent_id or agent_name, **kwargs)

                    return factory

                # Register with the registry
                register_agent_factory(
                    agent_type,
                    make_factory(agent_class, agent_type),
                    {"description": f"Agent defined in {filename}"},
                )

                logger.info(f"Auto-registered agent: {agent_type} ({class_name})")

        except Exception as e:
            logger.warning(f"Error auto-registering agent {filename}: {str(e)}")
            continue


def create_agent(
    agent_type: str, agent_id: Optional[str] = None, **kwargs
) -> AgentBase:
    """
    Create an agent instance by type.

        agent_type: Type of agent to create
        agent_id: Optional ID for the agent (defaults to agent_type if not provided)
        **kwargs: Additional parameters for the agent

        AgentBase: The created agent instance

        KeyError: If agent type is not registered
    """
    agent_id = agent_id or agent_type

    try:
        # First try direct instantiation from module as a fallback
        # This ensures we can create agents even if factory registration fails
        try:
            if agent_type == "research":
                from .research import ResearchAgent

                logger.info(f"Creating research agent {agent_id} directly")
                return ResearchAgent(agent_id=agent_id, **kwargs)
            elif agent_type == "writer":
                from .writer import WriterAgent

                logger.info(f"Creating writer agent {agent_id} directly")
                return WriterAgent(agent_id=agent_id, **kwargs)
            elif agent_type == "editor":
                from .editor import EditorAgent

                logger.info(f"Creating editor agent {agent_id} directly")
                return EditorAgent(agent_id=agent_id, **kwargs)
            elif agent_type == "strategy":
                from .strategy import StrategyAgent

                logger.info(f"Creating strategy agent {agent_id} directly")
                return StrategyAgent(agent_id=agent_id, **kwargs)
            elif agent_type == "test":
                from .test import testAgent

                logger.info(f"Creating test agent {agent_id} directly")
                return testAgent(agent_id=agent_id, **kwargs)
        except ImportError:
            logger.debug(
                f"Could not create agent {agent_type} directly, trying registry"
            )

        # Try using the registry
        if agent_type in agent_registry.list():
            factory = agent_registry.get(agent_type)
            logger.debug(f"Creating agent {agent_id} with factory {agent_type}")

            # Safely handle the factory - it might be a class instead of a callable function
            if callable(factory):
                try:
                    # Try to call it as a factory function
                    return factory(agent_id=agent_id, **kwargs)
                except TypeError as e:
                    logger.warning(f"Factory error for {agent_type}: {str(e)}")
                    # Fall back to dynamic import below
            else:
                logger.warning(
                    f"Factory for {agent_type} is not callable, trying dynamic import"
                )

        # Try dynamic import as a last resort
        logger.debug(f"No usable factory for {agent_type}, trying dynamic import")

        # First try direct module import
        module_path = f"aigen.agents.{agent_type.lower()}"
        try:
            module = importlib.import_module(module_path)
            logger.debug(f"Found module at {module_path}")
        except ImportError:
            # If not found, try from custom directory
            module_path = f"aigen.agents.custom.{agent_type.lower()}"
            try:
                module = importlib.import_module(module_path)
                logger.debug(f"Found module in custom directory: {module_path}")
            except ImportError:
                raise KeyError(f"No agent implementation found for type: {agent_type}")

        class_name = (
            "".join(word.capitalize() for word in agent_type.split("_")) + "Agent"
        )

        if not hasattr(module, class_name):
            # Try lowercase class name (e.g., testAgent instead of TestAgent)
            alt_class_name = agent_type.lower() + "Agent"
            if hasattr(module, alt_class_name):
                class_name = alt_class_name
            else:
                raise KeyError(
                    f"Agent module {module_path} does not contain class {class_name}"
                )

        agent_class = getattr(module, class_name)

        logger.debug(f"Creating agent {agent_id} with class {class_name}")
        return agent_class(agent_id=agent_id, **kwargs)

    except Exception as e:
        logger.error(f"Error creating agent {agent_id} of type {agent_type}: {str(e)}")
        raise


def register_standard_agents():
    """Register standard agent types with the registry."""

    # Define factories using lambda functions to ensure they're callable
    factories = [
        (
            "research",
            lambda agent_id=None, **kwargs: (
                __import__(
                    "aigen.agents.research", fromlist=["ResearchAgent"]
                ).ResearchAgent(agent_id=agent_id or "research", **kwargs)
            ),
            "Research agent that gathers information",
        ),
        (
            "writer",
            lambda agent_id=None, **kwargs: (
                __import__("aigen.agents.writer", fromlist=["WriterAgent"]).WriterAgent(
                    agent_id=agent_id or "writer", **kwargs
                )
            ),
            "Writer agent that creates content",
        ),
        (
            "editor",
            lambda agent_id=None, **kwargs: (
                __import__("aigen.agents.editor", fromlist=["EditorAgent"]).EditorAgent(
                    agent_id=agent_id or "editor", **kwargs
                )
            ),
            "Editor agent that improves and refines content",
        ),
        (
            "strategy",
            lambda agent_id=None, **kwargs: (
                __import__(
                    "aigen.agents.strategy", fromlist=["StrategyAgent"]
                ).StrategyAgent(agent_id=agent_id or "strategy", **kwargs)
            ),
            "Strategy agent that develops content plans and outlines",
        ),
        (
            "test",
            lambda agent_id=None, **kwargs: (
                __import__("aigen.agents.test", fromlist=["testAgent"]).testAgent(
                    agent_id=agent_id or "test", **kwargs
                )
            ),
            "Test agent for system testing",
        ),
    ]

    # Register each factory
    for agent_type, factory_func, description in factories:
        try:
            register_agent_factory(
                agent_type, factory_func, {"description": description}
            )
            logger.debug(f"Registered standard agent: {agent_type}")
        except Exception as e:
            logger.warning(f"Error registering agent {agent_type}: {str(e)}")
            continue


def auto_register_all_agents():
    """Register all agents found in standard directories."""
    # Register agents in the main agents directory
    agents_dir = os.path.dirname(os.path.abspath(__file__))
    register_agents_in_directory(agents_dir)

    # Register agents in the custom subdirectory
    custom_dir = os.path.join(agents_dir, "custom")
    register_agents_in_directory(custom_dir)


# Register standard agents first, then auto-register any others
register_standard_agents()
auto_register_all_agents()

"""Workflow factory system for creating workflow instances."""

from typing import Dict, Any, Callable, Optional, List, Union
import importlib
import yaml
from pathlib import Path

from ..core.registry import Registry
from ..core.logging import get_logger
from ..agents.factory import create_agent
from ..agents.base import AgentBase

logger = get_logger("workflow_factory")

# Registry for workflow factories
workflow_registry = Registry(Callable)

def register_workflow_factory(name: str, factory: Callable, 
                            metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a workflow factory function.
    
    Args:
        name: Name to register the factory under
        factory: Function that creates workflow instances
        metadata: Optional metadata for the factory
    """
    workflow_registry.register_factory(name, factory, metadata)
    logger.info(f"Registered workflow factory: {name}")

def create_workflow(workflow_spec: Union[str, List[str], Dict[str, Any], Path], **kwargs):
    """
    Create a workflow from a specification.
    
    Args:
        workflow_spec: Workflow specification. Can be:
            - String: Workflow type name, registered workflow name, or file path
            - List: List of agent names/objects
            - Dict: Configuration dictionary
            - Path: File path
        **kwargs: Additional parameters
        
    Returns:
        Workflow instance
    """
    try:
        if isinstance(workflow_spec, str):
            # Direct fallbacks for common workflows for reliability
            if workflow_spec == "research_only":
                logger.info("Creating research_only workflow directly")
                from .deterministic import DeterministicWorkflow
                from ..agents.factory import create_agent
                return DeterministicWorkflow(
                    agents=[create_agent("research")],
                    name="research_only"
                )
            elif workflow_spec == "content_generation":
                logger.info("Creating content_generation workflow directly")
                from .deterministic import DeterministicWorkflow
                from ..agents.factory import create_agent
                return DeterministicWorkflow(
                    agents=[
                        create_agent("research"),
                        create_agent("strategy"),
                        create_agent("writer"),
                        create_agent("editor")
                    ],
                    name="content_generation"
                )
            
            # Check if it's a registered workflow
            if workflow_spec in workflow_registry.list():
                logger.debug(f"Creating workflow from registered factory: {workflow_spec}")
                factory = workflow_registry.get(workflow_spec)
                
                # Safely handle the factory - it might be a class instead of a callable function
                if callable(factory):
                    try:
                        # Try to call it as a factory function
                        return factory(**kwargs)
                    except TypeError as e:
                        logger.warning(f"Workflow factory error for {workflow_spec}: {str(e)}")
                        # Fall back to other approaches below
                else:
                    logger.warning(f"Factory for {workflow_spec} is not callable, trying other approaches")
            
            # Check if it's a comma-separated list of agent names
            if ',' in workflow_spec:
                agent_names = [name.strip() for name in workflow_spec.split(',')]
                return create_workflow_from_agent_names(agent_names, **kwargs)
            
            # Check if it's a file path
            path = Path(workflow_spec)
            if path.exists():
                return create_workflow_from_file(path, **kwargs)
                
            # Assume it's a workflow type name
            return create_workflow_by_type(workflow_spec, **kwargs)
            
        elif isinstance(workflow_spec, list):
            # List of agent names/objects
            return create_workflow_from_agent_names(workflow_spec, **kwargs)
            
        elif isinstance(workflow_spec, dict):
            # Configuration dictionary
            return create_workflow_from_config(workflow_spec, **kwargs)
            
        elif isinstance(workflow_spec, Path):
            # File path
            return create_workflow_from_file(workflow_spec, **kwargs)
            
        else:
            raise ValueError(f"Unsupported workflow specification type: {type(workflow_spec)}")
            
    except Exception as e:
        logger.error(f"Error creating workflow: {str(e)}")
        raise

def create_workflow_from_file(file_path: Path, **kwargs):
    """
    Create a workflow from a configuration file.
    
    Args:
        file_path: Path to configuration file
        **kwargs: Additional parameters
        
    Returns:
        Workflow instance
    """
    # Load configuration from file
    with open(file_path, 'r') as f:
        if file_path.suffix.lower() in ('.yaml', '.yml'):
            config = yaml.safe_load(f)
        else:
            import json
            config = json.load(f)
    
    return create_workflow_from_config(config, **kwargs)

def create_workflow_from_config(config: Dict[str, Any], **kwargs):
    """
    Create a workflow from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional parameters
        
    Returns:
        Workflow instance
    """
    # Get workflow type
    workflow_type = config.get('type', 'deterministic')
    
    # Create agents from configuration
    agents = []
    for agent_config in config.get('agents', []):
        if isinstance(agent_config, str):
            # Simple agent name
            agents.append(create_agent(agent_config))
        elif isinstance(agent_config, dict):
            # Agent configuration dictionary
            agent_type = agent_config.get('type')
            if not agent_type:
                raise ValueError(f"Agent config missing 'type': {agent_config}")
            
            agent_params = agent_config.get('params', {})
            agents.append(create_agent(agent_type, **agent_params))
        else:
            raise ValueError(f"Invalid agent configuration: {agent_config}")
    
    # Configure handoffs if requested
    if config.get('setup_handoffs', True) and len(agents) > 1:
        for i in range(len(agents) - 1):
            agents[i].add_handoff(agents[i+1].agent_id)
    
    # Get workflow name
    name = config.get('name', 'custom_workflow')
    
    # Create the workflow
    return create_workflow_by_type(workflow_type, agents=agents, name=name, **kwargs)

def create_workflow_from_agent_names(agent_names: List[str], **kwargs):
    """
    Create a workflow from a list of agent names.
    
    Args:
        agent_names: List of agent names
        **kwargs: Additional parameters
        
    Returns:
        Workflow instance
    """
    # Convert agent names to agent instances
    agents = []
    for name in agent_names:
        # Handle agent objects directly
        if isinstance(name, AgentBase):
            agents.append(name)
        else:
            agents.append(create_agent(name))
    
    # Special handling for single-agent workflows (particularly research_only)
    if len(agents) == 1 and agents[0].agent_id == "research":
        logger.info("Creating research_only workflow from agent list")
        from .deterministic import DeterministicWorkflow
        return DeterministicWorkflow(
            agents=agents,
            name="research_only",
            **kwargs
        )
    
    # Configure handoffs if not disabled
    if kwargs.get('setup_handoffs', True) and len(agents) > 1:
        for i in range(len(agents) - 1):
            agents[i].add_handoff(agents[i+1].agent_id)
    
    # Get workflow type, defaulting to deterministic
    workflow_type = kwargs.pop('type', 'deterministic')
    
    # Create the workflow
    return create_workflow_by_type(workflow_type, agents=agents, **kwargs)

def create_workflow_by_type(workflow_type: str, **kwargs):
    """
    Create a workflow by type name.
    
    Args:
        workflow_type: Type of workflow to create
        **kwargs: Additional parameters
        
    Returns:
        Workflow instance
    """
    # Always check registry first
    if workflow_type in workflow_registry.list():
        factory = workflow_registry.get(workflow_type)
        if callable(factory):
            try:
                return factory(**kwargs)
            except Exception as e:
                logger.warning(f"Factory call failed: {str(e)}")
                # Fall back to other methods
    
    # Standard types fallback
    if workflow_type == 'deterministic':
        from .deterministic import DeterministicWorkflow
        return DeterministicWorkflow(**kwargs)
    elif workflow_type == 'handoff':
        from .handoff import HandoffWorkflow
        return HandoffWorkflow(**kwargs)
    else:
        # Try to dynamically import and instantiate
        try:
            module_path = f"aigen.workflows.{workflow_type.lower()}"
            class_name = "".join(word.capitalize() for word in workflow_type.split("_")) + "Workflow"
            
            module = importlib.import_module(module_path)
            workflow_class = getattr(module, class_name)
            
            return workflow_class(**kwargs)
        except (ImportError, AttributeError):
            raise ValueError(f"Unsupported workflow type: {workflow_type}")

# Register standard workflows
def register_standard_workflows():
    """Register standard workflow types."""
    
    # Define factories using lambda functions to ensure they're callable
    factories = [
        # Deterministic workflow
        (
            "deterministic",
            lambda agents=None, name="deterministic", **kwargs: (
                __import__("aigen.workflows.deterministic", fromlist=["DeterministicWorkflow"]).DeterministicWorkflow(
                    agents=agents or [], name=name, **kwargs
                )
            ),
            "Executes agents in a fixed sequence with explicit data passing"
        ),
        
        # Handoff workflow
        (
            "handoff",
            lambda agents=None, name="handoff", **kwargs: (
                __import__("aigen.workflows.handoff", fromlist=["HandoffWorkflow"]).HandoffWorkflow(
                    agents=agents or [], name=name, **kwargs
                )
            ),
            "Executes agents using the handoff mechanism for dynamic control flow"
        ),
        
        # Content generation workflow
        (
            "content_generation",
            lambda **kwargs: (
                __import__("aigen.workflows.deterministic", fromlist=["DeterministicWorkflow"]).DeterministicWorkflow(
                    agents=[
                        create_agent("research"),
                        create_agent("strategy"),
                        create_agent("writer"),
                        create_agent("editor")
                    ],
                    name="content_generation",
                    **kwargs
                )
            ),
            "Standard content generation workflow with research, strategy, writing, and editing phases"
        ),
        
        # Research-only workflow
        (
            "research_only",
            lambda **kwargs: (
                __import__("aigen.workflows.deterministic", fromlist=["DeterministicWorkflow"]).DeterministicWorkflow(
                    agents=[create_agent("research")],
                    name="research_only",
                    **kwargs
                )
            ),
            "Research-focused workflow that only gathers information"
        ),
        
        
    ]
    
    # Register each workflow factory
    for workflow_type, factory_func, description in factories:
        try:
            register_workflow_factory(
                workflow_type,
                factory_func,
                {"description": description}
            )
            logger.debug(f"Registered standard workflow: {workflow_type}")
        except Exception as e:
            logger.warning(f"Error registering workflow {workflow_type}: {str(e)}")
    
    logger.info("Registered standard workflows")

# Auto-register workflows on import
register_standard_workflows()
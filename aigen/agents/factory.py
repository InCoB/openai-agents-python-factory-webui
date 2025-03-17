from typing import Dict, Any, Callable, Optional, List, Type
import importlib

from .base import AgentBase, AgentRole
from ..core.registry import Registry
from ..core.logging import get_logger

logger = get_logger("agent_factory")

agent_registry = Registry(Callable)

def register_agent_factory(name: str, factory: Callable[..., AgentBase], 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register an agent factory function.
    
        name: Name to register the factory under
        factory: Function that creates agent instances
        metadata: Optional metadata for the factory
    """
    agent_registry.register_factory(name, factory, metadata)
    logger.info(f"Registered agent factory: {name}")

def create_agent(agent_type: str, agent_id: Optional[str] = None, **kwargs) -> AgentBase:
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
        except ImportError:
            logger.debug(f"Could not create agent {agent_type} directly, trying registry")
                
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
                logger.warning(f"Factory for {agent_type} is not callable, trying dynamic import")
        
        # Try dynamic import as a last resort
        logger.debug(f"No usable factory for {agent_type}, trying dynamic import")
        
        module_path = f"aigen.agents.{agent_type.lower()}"
        class_name = "".join(word.capitalize() for word in agent_type.split("_")) + "Agent"
        
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            raise KeyError(f"No agent implementation found for type: {agent_type}")
        
        if not hasattr(module, class_name):
            raise KeyError(f"Agent module {module_path} does not contain class {class_name}")
        
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
        ("research", 
         lambda agent_id=None, **kwargs: (
             __import__("aigen.agents.research", fromlist=["ResearchAgent"]).ResearchAgent(
                 agent_id=agent_id or "research", **kwargs
             )
         ),
         "Research agent that gathers information"
        ),
        ("writer", 
         lambda agent_id=None, **kwargs: (
             __import__("aigen.agents.writer", fromlist=["WriterAgent"]).WriterAgent(
                 agent_id=agent_id or "writer", **kwargs
             )
         ),
         "Writer agent that creates content"
        ),
        ("editor", 
         lambda agent_id=None, **kwargs: (
             __import__("aigen.agents.editor", fromlist=["EditorAgent"]).EditorAgent(
                 agent_id=agent_id or "editor", **kwargs
             )
         ),
         "Editor agent that improves and refines content"
        ),
        ("strategy", 
         lambda agent_id=None, **kwargs: (
             __import__("aigen.agents.strategy", fromlist=["StrategyAgent"]).StrategyAgent(
                 agent_id=agent_id or "strategy", **kwargs
             )
         ),
         "Strategy agent that develops content plans and outlines"
        )
    ]
    
    # Register each factory
    for agent_type, factory_func, description in factories:
        try:
            register_agent_factory(
                agent_type,
                factory_func,
                {"description": description}
            )
            logger.debug(f"Registered standard agent: {agent_type}")
        except Exception as e:
            logger.warning(f"Error registering agent {agent_type}: {str(e)}")
            continue

register_standard_agents()
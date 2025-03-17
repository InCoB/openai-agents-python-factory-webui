__version__ = "1.0.0"

from .core import (
    Context, Logger, ConfigManager, Registry,
    SystemError, ErrorSeverity, ErrorCategory
)

from .agents import (
    AgentBase, AgentResponse, AgentRole,
    create_agent, register_agent_factory
)

from .tools import (
    ToolBase, ToolResponse, ToolType,
    create_tool, register_tool_factory
)

from .workflows import (
    WorkflowEngine, DeterministicWorkflow, HandoffWorkflow,
    create_workflow, register_workflow_factory
)

from .agents.factory import register_standard_agents
from .workflows.factory import register_standard_workflows

from .core.logging import get_logger
from .workflows.factory import workflow_registry
from .agents.factory import agent_registry

__all__ = [
    'Context', 'Logger', 'ConfigManager', 'Registry',
    'SystemError', 'ErrorSeverity', 'ErrorCategory',
    
    'AgentBase', 'AgentResponse', 'AgentRole',
    'create_agent', 'register_agent_factory',
    
    'ToolBase', 'ToolResponse', 'ToolType',
    'create_tool', 'register_tool_factory',
    
    'WorkflowEngine', 'DeterministicWorkflow', 'HandoffWorkflow',
    'create_workflow', 'register_workflow_factory'
]

def init():
    """Initialize the framework."""
    from .core.logging import get_logger
    logger = get_logger("init")
    logger.info(f"Initializing AI Agent Workflow Framework v{__version__}")
    
    register_standard_agents()
    register_standard_workflows()
    
    # Fix any non-callable factories in the registry
    _ensure_callable_factories()
    
    logger.info("Framework initialized successfully")

def _ensure_callable_factories():
    """Ensure all factories in the registry are callable functions."""
    from .core.logging import get_logger
    
    logger = get_logger("registry_fix")
    
    # Fix agent factories
    for name in list(agent_registry.list()):
        try:
            factory = agent_registry._factories.get(name)
            if factory and not callable(factory):
                logger.warning(f"Fixing non-callable agent factory: {name}")
                
                # Create a proper factory function based on the agent type
                if name == "research":
                    def research_factory(agent_id=None, **kwargs):
                        from .agents.research import ResearchAgent
                        return ResearchAgent(agent_id=agent_id or "research", **kwargs)
                    agent_registry._factories[name] = research_factory
                
                elif name == "writer":
                    def writer_factory(agent_id=None, **kwargs):
                        from .agents.writer import WriterAgent
                        return WriterAgent(agent_id=agent_id or "writer", **kwargs)
                    agent_registry._factories[name] = writer_factory
                
                elif name == "editor":
                    def editor_factory(agent_id=None, **kwargs):
                        from .agents.editor import EditorAgent
                        return EditorAgent(agent_id=agent_id or "editor", **kwargs)
                    agent_registry._factories[name] = editor_factory
                
                else:
                    # Generic fallback
                    agent_type = name
                    def generic_agent_factory(agent_id=None, **kwargs):
                        module_path = f"aigen.agents.{agent_type.lower()}"
                        class_name = "".join(word.capitalize() for word in agent_type.split("_")) + "Agent"
                        module = __import__(module_path, fromlist=[class_name])
                        agent_class = getattr(module, class_name)
                        return agent_class(agent_id=agent_id or agent_type, **kwargs)
                    
                    agent_registry._factories[name] = generic_agent_factory
        
        except Exception as e:
            logger.error(f"Error fixing agent factory {name}: {str(e)}")
    
    # Fix workflow factories
    for name in list(workflow_registry.list()):
        try:
            factory = workflow_registry._factories.get(name)
            if factory and not callable(factory):
                logger.warning(f"Fixing non-callable workflow factory: {name}")
                
                # Create proper factory function based on workflow type
                if name == "research_only":
                    from .workflows.deterministic import DeterministicWorkflow
                    from .agents.factory import create_agent
                    
                    def research_only_factory(**kwargs):
                        agents = [create_agent("research", instructions="Perform thorough research and provide comprehensive, well-formatted results.")]
                        return DeterministicWorkflow(agents=agents, name="research_only", **kwargs)
                    
                    workflow_registry._factories[name] = research_only_factory
                    logger.info(f"Fixed factory for {name}")
                
                elif name == "deterministic":
                    def deterministic_factory(agents=None, name="deterministic", **kwargs):
                        return DeterministicWorkflow(agents=agents or [], name=name, **kwargs)
                    workflow_registry._factories[name] = deterministic_factory
                
                elif name == "handoff":
                    def handoff_factory(agents=None, name="handoff", **kwargs):
                        from .workflows.handoff import HandoffWorkflow
                        return HandoffWorkflow(agents=agents or [], name=name, **kwargs)
                    workflow_registry._factories[name] = handoff_factory
                
                elif name == "content_generation":
                    def content_gen_factory(**kwargs):
                        agents = [
                            create_agent("research"),
                            create_agent("writer"),
                            create_agent("editor")
                        ]
                        return DeterministicWorkflow(agents=agents, name="content_generation", **kwargs)
                    workflow_registry._factories[name] = content_gen_factory
                
                elif name == "content_analysis":
                    def content_analysis_factory(**kwargs):
                        agents = [
                            create_agent("research"),
                            create_agent("editor")
                        ]
                        return DeterministicWorkflow(agents=agents, name="content_analysis", **kwargs)
                    workflow_registry._factories[name] = content_analysis_factory
                
                else:
                    # Generic fallback for unknown workflow types
                    workflow_type = name
                    def generic_workflow_factory(**kwargs):
                        if workflow_type.endswith("_workflow"):
                            class_name = "".join(word.capitalize() for word in workflow_type.split("_")[:-1]) + "Workflow"
                        else:
                            class_name = "".join(word.capitalize() for word in workflow_type.split("_")) + "Workflow"
                        
                        module_path = f"aigen.workflows.{workflow_type.lower()}"
                        module = __import__(module_path, fromlist=[class_name])
                        workflow_class = getattr(module, class_name)
                        return workflow_class(**kwargs)
                    
                    workflow_registry._factories[name] = generic_workflow_factory
        
        except Exception as e:
            logger.error(f"Error fixing workflow factory {name}: {str(e)}")

# Run the factory fix on initialization
_ensure_callable_factories()
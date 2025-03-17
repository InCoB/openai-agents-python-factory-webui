from typing import Dict, Any, Callable, Optional, List, Type
import importlib

from .base import ToolBase, ToolType
from ..core.registry import Registry
from ..core.logging import get_logger

logger = get_logger("tool_factory")

tool_registry = Registry(Callable)

def register_tool_factory(name: str, factory: Callable[..., ToolBase], 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a tool factory function.
    
        name: Name to register the factory under
        factory: Function that creates tool instances
        metadata: Optional metadata for the factory
    """
    tool_registry.register_factory(name, factory, metadata)
    logger.info(f"Registered tool factory: {name}")

def create_tool(tool_type: str, tool_id: Optional[str] = None, **kwargs) -> ToolBase:
    """
    Create a tool instance by type.
    
        tool_type: Type of tool to create
        tool_id: Optional ID for the tool (defaults to tool_type if not provided)
        **kwargs: Additional parameters for the tool
        
        ToolBase: The created tool instance
        
        KeyError: If tool type is not registered
    """
    tool_id = tool_id or tool_type
    
    try:
        if tool_type in tool_registry.list():
            factory = tool_registry.get(tool_type)
            logger.debug(f"Creating tool {tool_id} with factory {tool_type}")
            return factory(tool_id=tool_id, **kwargs)
        
        logger.debug(f"No factory for {tool_type}, trying dynamic import")
        
        module_path = f"aigen.tools.{tool_type.lower()}"
        class_name = "".join(word.capitalize() for word in tool_type.split("_")) + "Tool"
        
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            raise KeyError(f"No tool implementation found for type: {tool_type}")
        
        if not hasattr(module, class_name):
            raise KeyError(f"Tool module {module_path} does not contain class {class_name}")
        
        tool_class = getattr(module, class_name)
        
        logger.debug(f"Creating tool {tool_id} with class {class_name}")
        return tool_class(tool_id=tool_id, **kwargs)
        
    except Exception as e:
        logger.error(f"Error creating tool {tool_id} of type {tool_type}: {str(e)}")
        raise

def register_standard_tools():
    """Register standard tool types with the registry."""
    standard_tools = [
        ("tavily_search", "TavilySearchTool"),
        ("content_analysis", "ContentAnalysisTool"),
        ("readability", "ReadabilityTool")
    ]
    
    for tool_type, class_name in standard_tools:
        try:
            if tool_type.startswith("tavily"):
                module_name = "research"
            elif tool_type in ("content_analysis", "readability"):
                module_name = "analysis"
            else:
                module_name = tool_type.split("_")[0]
            
            module = importlib.import_module(f"aigen.tools.{module_name}")
            
            if hasattr(module, class_name):
                tool_class = getattr(module, class_name)
                register_tool_factory(
                    tool_type,
                    lambda tool_id=None, cls=tool_class, **kwargs: cls(
                        tool_id=tool_id or tool_type, **kwargs
                    ),
                    {"description": tool_class.__doc__}
                )
                logger.debug(f"Registered standard tool: {tool_type}")
            
        except ImportError:
            logger.debug(f"Standard tool module not found: {module_name}")
            continue
        except Exception as e:
            logger.warning(f"Error registering tool {tool_type}: {str(e)}")
            continue

register_standard_tools()
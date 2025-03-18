"""Agent registration service."""

import os
import sys
import shutil
import importlib.util
import logging
from typing import Dict, Any, Optional, Union, Tuple

from aigen.services.models import AgentConfiguration

logger = logging.getLogger(__name__)

class AgentRegistrationService:
    """Service for registering agents with the framework."""
    
    def __init__(self, agents_dir: Optional[str] = None) -> None:
        """
        Initialize the registration service.
        
            agents_dir: Directory to store agent code. If None, uses default directory.
        """
        if agents_dir is None:
            agents_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'agents',
                'custom'
            )
        
        self.agents_dir = agents_dir
        
        os.makedirs(agents_dir, exist_ok=True)
        
        init_file = os.path.join(agents_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Dynamic agent modules\n")
    
    def register_agent_type(
        self, 
        config: AgentConfiguration, 
        code: str
    ) -> Tuple[bool, str]:
        """
        Register a new agent type with the framework.
        
        Uses a multi-strategy approach:
        
            config: Agent configuration.
            code: Generated agent code.
            
            Tuple containing:
                - Success flag (True if registration succeeded)
                - Result message
        """
        try:
            success, message = self._register_from_code(config, code)
            
            if success:
                return True, message
            
            logger.warning(
                f"Code-based registration failed for {config.agent_type}, "
                f"falling back to config-based: {message}"
            )
            success, message = self._register_from_config(config)
            
            return success, message
            
        except Exception as e:
            error_msg = f"Error registering agent {config.agent_type}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _register_from_code(
        self, 
        config: AgentConfiguration, 
        code: str
    ) -> Tuple[bool, str]:
        """
        Register agent by generating and loading Python code.
        
            config: Agent configuration.
            code: Generated Python code.
            
            Tuple containing:
                - Success flag (True if registration succeeded)
                - Result message
        """
        module_name = f"aigen.agents.custom.{config.agent_type}"
        file_name = f"{config.agent_type}.py"
        class_name = f"{config.name.replace(' ', '')}Agent"
        module_path = os.path.join(self.agents_dir, file_name)
        
        try:
            try:
                compile(code, module_path, 'exec')
            except SyntaxError as se:
                line_no = se.lineno if hasattr(se, 'lineno') else '?'
                return False, f"Syntax error in generated code at line {line_no}: {str(se)}"
            
            if os.path.exists(module_path):
                backup_path = f"{module_path}.bak"
                try:
                    shutil.copy2(module_path, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup: {str(e)}")
            
            with open(module_path, "w") as f:
                f.write(code)
            
            init_path = os.path.join(self.agents_dir, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w") as f:
                    f.write("# Auto-generated custom agents\n")
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                return False, "Failed to create module specification"
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            agent_class = getattr(module, class_name, None)
            if not agent_class:
                return False, f"Could not find class {class_name} in the generated module"
            
            try:
                from aigen.agents.factory import register_agent_factory
                
                register_agent_factory(
                    config.agent_type,
                    lambda agent_id=None, **kwargs: agent_class(
                        agent_id=agent_id or config.agent_type, 
                        **kwargs
                    ),
                    {"description": config.name}
                )
                
                return True, f"Successfully registered {config.name} ({config.agent_type})"
                
            except Exception as e:
                return False, f"Failed to register with agent factory: {str(e)}"
                
        except Exception as e:
            error_msg = f"Error registering agent from code: {str(e)}"
            logger.error(error_msg)
            
            if os.path.exists(module_path) and not os.path.exists(f"{module_path}.bak"):
                try:
                    os.unlink(module_path)
                except:
                    pass
                
            return False, error_msg
    
    def _register_from_config(
        self, 
        config: AgentConfiguration
    ) -> Tuple[bool, str]:
        """
        Register agent using a configuration-based factory.
        
            config: Agent configuration.
            
            Tuple containing:
                - Success flag (True if registration succeeded)
                - Result message
        """
        try:
            try:
                from aigen.agents.factory import register_agent_factory
                from aigen.agents.base import AgentBase, AgentRole
                from aigen.tools.factory import create_tool
            except ImportError as e:
                return False, f"Failed to import required modules: {str(e)}"
            
            def factory_function(agent_id: Optional[str] = None, **kwargs: Any) -> AgentBase:
                params = {**config.parameters, **kwargs}
                
                agent = AgentBase(
                    agent_id=agent_id or config.agent_type,
                    role=getattr(AgentRole, config.role.name, AgentRole.CUSTOM),
                    instructions=config.instructions,
                    **params
                )
                
                for tool_name in config.tools:
                    try:
                        tool = create_tool(tool_name)
                        agent.add_tool(tool)
                    except Exception as e:
                        logger.warning(f"Failed to add tool {tool_name}: {str(e)}")
                
                for handoff_name in config.handoffs:
                    agent.add_handoff(handoff_name)
                
                return agent
            
            register_agent_factory(
                config.agent_type,
                factory_function,
                {"description": config.name}
            )
            
            return True, f"Successfully registered {config.name} ({config.agent_type}) from config"
            
        except Exception as e:
            error_msg = f"Error registering agent from config: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def unregister_agent(self, agent_type: str) -> Tuple[bool, str]:
        """
        Unregister an agent.
        
            agent_type: Agent type to unregister.
            
            Tuple containing:
                - Success flag (True if unregistration succeeded)
                - Result message
        """
        try:
            try:
                from aigen.agents.factory import agent_registry
                
                if hasattr(agent_registry, "unregister"):
                    agent_registry.unregister(agent_type)
                elif hasattr(agent_registry, "_factories") and agent_type in agent_registry._factories:
                    del agent_registry._factories[agent_type]
                    if hasattr(agent_registry, "_metadata") and agent_type in agent_registry._metadata:
                        del agent_registry._metadata[agent_type]
                else:
                    return False, f"Agent type {agent_type} not found in registry"
            except Exception as e:
                return False, f"Failed to remove from registry: {str(e)}"
            
            module_path = os.path.join(self.agents_dir, f"{agent_type}.py")
            if os.path.exists(module_path):
                try:
                    os.unlink(module_path)
                except Exception as e:
                    return False, f"Failed to delete module file: {str(e)}"
            
            try:
                pyc_path = os.path.join(self.agents_dir, f"{agent_type}.pyc")
                if os.path.exists(pyc_path):
                    os.unlink(pyc_path)
                
                pycache_dir = os.path.join(self.agents_dir, "__pycache__")
                if os.path.exists(pycache_dir):
                    for filename in os.listdir(pycache_dir):
                        if filename.startswith(f"{agent_type}."):
                            os.unlink(os.path.join(pycache_dir, filename))
            except Exception as e:
                logger.warning(f"Error cleaning up cached files: {str(e)}")
            
            module_name = f"aigen.agents.custom.{agent_type}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return True, f"Successfully unregistered {agent_type}"
            
        except Exception as e:
            error_msg = f"Error unregistering agent {agent_type}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
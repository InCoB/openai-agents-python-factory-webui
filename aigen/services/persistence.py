"""Agent configuration persistence service."""

import os
import yaml
import shutil
import logging
from typing import List, Optional, Dict, Any, Tuple

from aigen.services.models import AgentConfiguration

logger = logging.getLogger(__name__)

class AgentPersistenceService:
    """Service for persisting and loading agent configurations."""
    
    def __init__(self, config_dir: Optional[str] = None) -> None:
        """
        Initialize the persistence service.
        
            config_dir: Directory to store configurations. If None, uses default directory.
        """
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'agents',
                'config'
            )
        
        self.config_dir = config_dir
        
        os.makedirs(config_dir, exist_ok=True)
        
        init_file = os.path.join(config_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Agent configuration storage\n")
    
    def save_agent_config(self, config: AgentConfiguration) -> Tuple[bool, str]:
        """
        Save agent configuration to disk.
        
            config: Agent configuration to save.
            
            Tuple containing:
                - Success flag (True if save succeeded)
                - Result message
        """
        try:
            config_dict = config.dict()
            
            file_path = os.path.join(self.config_dir, f"{config.agent_type}.yaml")
            
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                try:
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup: {str(e)}")
            
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            return True, f"Successfully saved configuration for {config.agent_type}"
            
        except Exception as e:
            error_msg = f"Error saving agent configuration: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def load_agent_config(self, agent_type: str) -> Tuple[bool, Optional[AgentConfiguration], str]:
        """
        Load agent configuration from disk.
        
            agent_type: Type of agent to load.
            
            Tuple containing:
                - Success flag (True if load succeeded)
                - Agent configuration or None if not found/error
                - Result message
        """
        try:
            file_path = os.path.join(self.config_dir, f"{agent_type}.yaml")
            
            if not os.path.exists(file_path):
                return False, None, f"Configuration for {agent_type} not found"
            
            with open(file_path, "r") as f:
                config_dict = yaml.safe_load(f)
            
            config = AgentConfiguration(**config_dict)
            
            return True, config, f"Successfully loaded configuration for {agent_type}"
            
        except Exception as e:
            error_msg = f"Error loading agent configuration: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def delete_agent_config(self, agent_type: str) -> Tuple[bool, str]:
        """
        Delete agent configuration.
        
            agent_type: Type of agent to delete.
            
            Tuple containing:
                - Success flag (True if deletion succeeded)
                - Result message
        """
        try:
            file_path = os.path.join(self.config_dir, f"{agent_type}.yaml")
            
            if not os.path.exists(file_path):
                return False, f"Configuration for {agent_type} not found"
            
            backup_path = f"{file_path}.bak"
            try:
                shutil.copy2(file_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
            
            os.unlink(file_path)
            
            return True, f"Successfully deleted configuration for {agent_type}"
            
        except Exception as e:
            error_msg = f"Error deleting agent configuration: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def list_saved_agents(self) -> List[Dict[str, str]]:
        """
        List all saved agent configurations with metadata.
        
        """
        try:
            result = []
            
            for filename in os.listdir(self.config_dir):
                if filename.endswith(".yaml") and not filename.endswith(".bak"):
                    try:
                        agent_type = os.path.splitext(filename)[0]
                        
                        success, config, _ = self.load_agent_config(agent_type)
                        
                        if success and config:
                            result.append({
                                "agent_type": agent_type,
                                "name": config.name,
                                "role": config.role.value,
                                "model": config.model
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing agent configurations: {str(e)}")
            return []
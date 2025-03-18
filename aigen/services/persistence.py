"""Agent configuration persistence service."""

import os
import yaml
import shutil
import logging
from typing import List, Optional, Dict, Tuple

from aigen.services.models import AgentConfiguration, AgentRole

logger = logging.getLogger(__name__)


# Configure YAML to handle AgentRole enum properly
class SafeAgentYamlLoader(yaml.SafeLoader):
    """Custom YAML loader that handles AgentRole enums."""

    pass


def agent_role_constructor(loader, node):
    """Convert YAML scalar to AgentRole enum."""
    value = loader.construct_scalar(node)
    try:
        return AgentRole(value)
    except ValueError:
        logger.warning("Invalid AgentRole value: %s, defaulting to CUSTOM", value)
        return AgentRole.CUSTOM


# Register constructor for role values
SafeAgentYamlLoader.add_constructor(
    "tag:yaml.org,2002:str", lambda loader, node: loader.construct_scalar(node)
)
yaml.add_representer(
    AgentRole,
    lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:str", data.value),
)

# Also handle old-style Python object serialization
YAML_TAG = "tag:yaml.org,2002:python/object/apply:aigen.services.models.AgentRole"
SafeAgentYamlLoader.add_constructor(YAML_TAG, lambda loader, node: AgentRole.CUSTOM)


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
                "agents",
                "config",
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
                    logger.info("Created backup at %s", backup_path)
                except Exception as e:
                    logger.warning("Failed to create backup: %s", str(e))

            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            return True, f"Successfully saved configuration for {config.agent_type}"

        except Exception as e:
            error_msg = f"Error saving agent configuration: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def load_agent_config(
        self, agent_type: str
    ) -> Tuple[bool, Optional[AgentConfiguration], str]:
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

            try:
                with open(file_path, "r") as f:
                    # Use our custom loader that handles AgentRole
                    config_dict = yaml.load(f, Loader=SafeAgentYamlLoader)

                config = AgentConfiguration(**config_dict)

                return (
                    True,
                    config,
                    f"Successfully loaded configuration for {agent_type}",
                )
            except Exception as e:
                logger.error("Error loading agent configuration: %s", str(e))
                # Try a backup approach - manually fix the role if it's the issue
                try:
                    with open(file_path, "r") as f:
                        config_dict = yaml.safe_load(f)

                    # Handle role specially
                    if "role" in config_dict:
                        if isinstance(config_dict["role"], dict):
                            # Likely a Python object serialization object
                            config_dict["role"] = AgentRole.CUSTOM
                        elif isinstance(config_dict["role"], str):
                            try:
                                config_dict["role"] = AgentRole(config_dict["role"])
                            except ValueError:
                                config_dict["role"] = AgentRole.CUSTOM

                    config = AgentConfiguration(**config_dict)
                    return (
                        True,
                        config,
                        f"Successfully loaded configuration for {agent_type} (fixed role)",
                    )
                except Exception as backup_e:
                    logger.error(
                        "Backup loading approach also failed: %s", str(backup_e)
                    )
                    return (
                        False,
                        None,
                        f"Error loading agent configuration: {str(e)}, backup error: {str(backup_e)}",
                    )

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
                logger.info("Created backup at %s", backup_path)
            except Exception as e:
                logger.warning("Failed to create backup: %s", str(e))

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
                            # Ensure all values are strings
                            result.append(
                                {
                                    "agent_type": str(agent_type),
                                    "name": str(config.name),
                                    "role": str(
                                        config.role.value
                                        if hasattr(config.role, "value")
                                        else config.role
                                    ),
                                    "model": str(config.model),
                                }
                            )
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", filename, str(e))

            return result

        except Exception as e:
            logger.error("Error listing agent configurations: %s", str(e))
            return []

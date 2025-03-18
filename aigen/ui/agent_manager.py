"""Agent manager UI components."""

import asyncio
import logging
import inspect
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Union
import os

import gradio as gr

from aigen.services.persistence import AgentPersistenceService
from aigen.services.registration import AgentRegistrationService
from aigen.ui.utils import dict_to_string_adapter

logger = logging.getLogger(__name__)

class AgentManagerUI:
    """UI components for agent management."""
    
    def __init__(self) -> None:
        """Initialize the agent manager UI."""
        self.persistence = AgentPersistenceService()
        self.registrar = AgentRegistrationService()
    
    def load_agent_list(self) -> List[List[str]]:
        """
        Load list of agents.
        
        """
        try:
            agents = self.persistence.list_saved_agents()
            return [
                [
                    a.get("agent_type", ""),
                    a.get("name", ""),
                    a.get("role", ""),
                    a.get("model", "")
                ]
                for a in agents
            ]
        except Exception as e:
            logger.error(f"Error loading agents: {str(e)}")
            return []
    
    def get_agent_type_choices(self) -> List[str]:
        """Get list of agent types for dropdown selection."""
        try:
            agents = self.persistence.list_saved_agents()
            return [a.get("agent_type", "") for a in agents if a.get("agent_type")]
        except Exception as e:
            logger.error(f"Error loading agent types: {str(e)}")
            return []
    
    async def delete_agent(self, agent_type: str) -> Dict[str, Any]:
        """
        Delete an agent.
        
            agent_type: Type of agent to delete.
            
            Result dictionary with status and message.
        """
        try:
            if not agent_type:
                return {"status": "error", "message": "❌ No agent selected"}
            
            # First delete the configuration file
            config_success, config_message = self.persistence.delete_agent_config(agent_type)
            
            if not config_success:
                return {"status": "error", "message": f"❌ Failed to delete agent configuration: {config_message}"}
            
            # Delete the implementation file if it exists
            try:
                implementation_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'agents',
                    'custom',
                    f"{agent_type}.py"
                )
                if os.path.exists(implementation_path):
                    os.unlink(implementation_path)
                    logger.info(f"Deleted implementation file for {agent_type}")
                    
                # Also check for compiled Python files
                compiled_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'agents',
                    'custom',
                    f"{agent_type}.pyc"
                )
                if os.path.exists(compiled_path):
                    os.unlink(compiled_path)
                    
                # And check for files in __pycache__
                pycache_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'agents',
                    'custom',
                    "__pycache__"
                )
                if os.path.exists(pycache_dir):
                    import glob
                    for cached_file in glob.glob(os.path.join(pycache_dir, f"{agent_type}.*")):
                        os.unlink(cached_file)
            except Exception as e:
                logger.warning(f"Error removing implementation file: {str(e)}")
            
            # Then try to unregister the agent (this may fail if it's not in the registry)
            try:
                success, message = self.registrar.unregister_agent(agent_type)
                
                if not success and "not found in registry" in message:
                    # This is normal for agents that exist only in config files
                    return {"status": "success", "message": f"✅ Agent '{agent_type}' deleted successfully"}
                
                if not success:
                    return {
                        "status": "warning", 
                        "message": f"⚠️ Agent configuration deleted but failed to unregister: {message}"
                    }
            except Exception as e:
                # Log but don't fail if unregistration has issues
                logger.warning(f"Error unregistering agent (config still deleted): {str(e)}")
                return {"status": "success", "message": f"✅ Agent '{agent_type}' deleted successfully"}
            
            return {"status": "success", "message": f"✅ Agent '{agent_type}' deleted successfully"}
            
        except Exception as e:
            error_msg = f"Error deleting agent: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": f"❌ {error_msg}"}
    
    def build_ui(self) -> gr.Tab:
        """
        Build the agent manager UI tab.
        
            Gradio Tab component for the agent manager.
        """
        with gr.Tab("Agent Management") as tab:
            gr.Markdown("## Manage Custom Agents")
            gr.Markdown("""
            View, update, and delete custom agents.
            
            select one to delete, or refresh the list to see newly created agents.
            """)
            
            refresh_btn = gr.Button("Refresh Agent List")
            
            agent_table = gr.DataFrame(
                headers=["Agent Type", "Name", "Role", "Model"],
                label="Custom Agents",
                interactive=False
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Select Agent to Delete")
                    
                    agent_dropdown = gr.Dropdown(
                        label="Select Agent Type",
                        choices=self.get_agent_type_choices(),
                        interactive=True
                    )
                    
                    delete_btn = gr.Button("Delete Selected Agent", variant="stop")
            
            delete_result = gr.JSON(label="Deletion Result")
            
            # Add an update function for the refresh button to update both the table and dropdown
            def on_refresh():
                agents_list = self.load_agent_list()
                agent_choices = [row[0] for row in agents_list if row[0]]
                return agents_list, gr.Dropdown(choices=agent_choices)
            
            # Add a function to delete and refresh
            def on_delete_and_refresh(agent_type: str):
                result = asyncio.run(self.delete_agent(agent_type))
                agents_list = self.load_agent_list()
                agent_choices = [row[0] for row in agents_list if row[0]]
                return result, agents_list, gr.Dropdown(choices=agent_choices, value=None)
            
            refresh_btn.click(
                on_refresh,
                outputs=[agent_table, agent_dropdown]
            )
            
            delete_btn.click(
                on_delete_and_refresh,
                inputs=[agent_dropdown],
                outputs=[delete_result, agent_table, agent_dropdown]
            )
            
            # Load initial data when UI is built
            agent_table.value = self.load_agent_list()
            
            return tab
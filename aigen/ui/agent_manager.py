"""Agent management UI components."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr

from aigen.services.persistence import AgentPersistenceService
from aigen.services.registration import AgentRegistrationService

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
    
    async def delete_agent(self, agent_type: str) -> str:
        """
        Delete an agent.
        
            agent_type: Type of agent to delete.
            
            Result message.
        """
        try:
            if not agent_type:
                return "❌ No agent selected"
            
            success, message = self.registrar.unregister_agent(agent_type)
            
            if not success:
                return f"❌ Failed to unregister agent: {message}"
            
            success, message = self.persistence.delete_agent_config(agent_type)
            
            if not success:
                return f"⚠️ Agent unregistered but failed to delete configuration: {message}"
            
            return f"✅ Agent '{agent_type}' deleted successfully"
            
        except Exception as e:
            error_msg = f"Error deleting agent: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
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
                gr.Markdown("### Selected Agent")
                
                selected_agent = gr.Textbox(
                    label="Selected Agent Type",
                    interactive=False
                )
            
            with gr.Row():
                delete_btn = gr.Button("Delete Selected Agent", variant="stop")
            
            delete_result = gr.Markdown()
            
            def on_agent_select(df: List[List[str]], evt: gr.SelectData) -> str:
                """Get agent type from selected row."""
                if evt.index is None or len(df) <= evt.index[0]:
                    return ""
                    
                return df[evt.index[0]][0]
            
            refresh_btn.click(
                self.load_agent_list,
                outputs=[agent_table]
            )
            
            agent_table.select(
                on_agent_select,
                inputs=[agent_table],
                outputs=[selected_agent]
            )
            
            delete_btn.click(
                self.delete_agent,
                inputs=[selected_agent],
                outputs=[delete_result]
            )
            
            tab.load(self.load_agent_list, outputs=[agent_table])
            
            return tab
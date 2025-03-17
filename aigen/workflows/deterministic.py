from typing import Dict, Any, List, Optional, Union
import asyncio
import time
import uuid

from agents import trace, gen_trace_id

from ..core.context import Context
from ..core.errors import WorkflowError
from ..core.logging import get_logger
from ..agents.base import AgentBase, AgentResponse
from .engine import WorkflowEngine

logger = get_logger("deterministic_workflow")

class DeterministicWorkflow:
    """
    Workflow that executes agents in a predetermined sequence.
    Agents are executed one after another in a fixed order.
    """
    
    def __init__(self, agents: List[AgentBase], name: str = "deterministic"):
        """
        Initialize a deterministic workflow.
        
            agents: List of agents to execute in sequence
            name: Workflow name
        """
        self.agents = agents
        self.name = name
        self.engine = WorkflowEngine()
    
    async def execute(self, context: Context, input_text: Optional[str] = None) -> str:
        """
        Execute the workflow using the deterministic approach.
        
            context: Execution context
            input_text: Optional direct input text
            
            str: Final output
        """
        return await self.execute_deterministic(context, input_text)
    
    async def execute_deterministic(self, context: Context, input_text: Optional[str] = None, trace_id: Optional[str] = None) -> str:
        """
        Execute agents in a fixed sequence with explicit data passing.
        
            context: Execution context
            input_text: Optional direct input text
            trace_id: Optional trace ID to use (if not provided, one will be generated)
            
            str: Final output
        """
        start_time = time.time()
        
        if not self.agents:
            raise WorkflowError("No agents in workflow", workflow_id=self.name)
        
        logger.info(f"Starting deterministic workflow with {len(self.agents)} agents")
        
        # Generate a unique trace ID for this workflow execution
        if trace_id is None:
            trace_id = gen_trace_id()
        
        # Wrap the entire agent sequence in a single trace for proper context flow
        with trace(f"Workflow: {self.name}", trace_id=trace_id):
            # Initialize current_input with the provided input_text
            current_input = input_text
            
            # Process agents in sequence
            for i, agent in enumerate(self.agents):
                agent_id = agent.agent_id
                logger.info(f"Running agent {i+1}/{len(self.agents)}: {agent_id}")
                
                # Execute the agent with retry logic
                response = await self.engine.execute_agent(agent, context, current_input)
                
                if not response.success:
                    # Handle agent failure
                    if i > 0:
                        logger.warning(f"Agent {agent_id} failed but we have partial results")
                        return self._get_best_content(context)
                    else:
                        error = f"Initial agent {agent_id} failed: {response.content}"
                        logger.error(error)
                        raise WorkflowError(error, workflow_id=self.name)
                
                # If this is a single-agent workflow, ensure the output is properly formatted
                if len(self.agents) == 1:
                    logger.info(f"Single-agent workflow completed successfully with {agent_id}")
                    # Ensure the result is stored in context
                    if not context.get_output(agent_id):
                        context.store_output(agent_id, response.content)
                    
                # Update current_input for the next agent
                if i < len(self.agents) - 1:
                    next_agent_id = self.agents[i+1].agent_id
                    current_input = f"""
                    Previous agent '{agent_id}' output:
                    {response.content}
                    
                    Original request: {input_text}
                    
                    Now you ({next_agent_id}) should continue processing based on this.
                    """
            
            # Get the final output
            final_agent_id = self.agents[-1].agent_id
            final_output = context.get_output(final_agent_id)
            
            if not final_output:
                logger.warning("No output from final agent, attempting to get best available content")
                return self._get_best_content(context)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Deterministic workflow completed in {elapsed_time:.2f}s")
            
            return final_output
    
    def _get_best_content(self, context: Context) -> str:
        """
        Get the best available content from the context.
        Falls back through different outputs based on availability.
        """
        for agent in reversed(self.agents):
            output = context.get_output(agent.agent_id)
            if output:
                return output
        
        return "No content was generated."
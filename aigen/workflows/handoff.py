from typing import Dict, Any, List, Optional
import asyncio
import time
import uuid

from agents import Runner, trace, gen_trace_id

from ..core.context import Context
from ..core.errors import WorkflowError
from ..core.logging import get_logger
from ..agents.base import AgentBase

logger = get_logger("handoff_workflow")


class HandoffWorkflow:
    """
    Workflow that uses the agent handoff approach.
    Agents pass control to each other dynamically via the handoff mechanism.
    """

    def __init__(self, agents: List[AgentBase], name: str = "handoff"):
        """
        Initialize a handoff workflow.

            agents: List of agents with proper handoffs configured
            name: Workflow name
        """
        self.agents = agents
        self.name = name

        self._verify_handoffs()

    def _verify_handoffs(self) -> None:
        """Verify that handoffs are properly configured."""
        if not self.agents:
            return

        for i, agent in enumerate(self.agents[:-1]):
            if not agent.handoffs:
                logger.warning(f"Agent {agent.agent_id} has no handoffs configured")

    async def execute(self, context: Context, input_text: Optional[str] = None) -> str:
        """
        Execute the workflow using the handoff approach.

            context: Execution context
            input_text: Optional direct input text

            str: Final output
        """
        return await self.execute_handoff(context, input_text)

    async def execute_handoff(
        self,
        context: Context,
        input_text: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Execute the workflow using agent handoffs.

            context: Execution context
            input_text: Optional direct input text
            trace_id: Optional trace ID to use (if not provided, one will be generated)

            str: Final output
        """
        # Record starting timestamp for performance tracking
        start_time = time.time()

        if not self.agents:
            raise WorkflowError("No agents in workflow", workflow_id=self.name)

        logger.info(f"Starting handoff workflow with {len(self.agents)} agents")

        # Use provided trace_id or generate a new one
        if not trace_id:
            trace_id = gen_trace_id()
            logger.debug(f"Generated new trace ID: {trace_id}")
        else:
            logger.debug(f"Using provided trace ID: {trace_id}")

        # Initialize the context with workflow execution metadata
        context.get_metadata().add_workflow(self.name)
        
        # Execute the first agent in the chain
        current_agent_idx = 0
        final_response = None
        
        # [Execute each agent and handle handoffs]
        
        # Calculate and log total execution time
        execution_time = time.time() - start_time
        logger.info(f"Handoff workflow completed in {execution_time:.2f} seconds")

        # Wrap the entire execution in a trace
        with trace(f"Handoff: {self.name}", trace_id=trace_id):
            # Initialize hooks
            from ..core.hooks import ContentWorkflowHooks

            hooks = ContentWorkflowHooks()

            # Start with the first agent
            entry_agent = self.agents[0]

            try:
                # Execute the workflow through OpenAI Runner
                result = await Runner.run(
                    entry_agent.openai_agent,
                    input=input_text,
                    context=context,
                    max_turns=80,  # Configurable
                    hooks=hooks,
                )

                final_agent_id = self.agents[-1].agent_id
                final_output = context.get_output(final_agent_id)

                if final_output:
                    logger.info(
                        f"Handoff workflow completed with output from final agent {final_agent_id}"
                    )

                    return final_output

                for agent in reversed(self.agents):
                    output = context.get_output(agent.agent_id)
                    if output:
                        logger.info(
                            f"Handoff workflow completed with partial output from agent {agent.agent_id}"
                        )
                        return output

                if hasattr(result, "final_output") and result.final_output:
                    logger.info("Using final_output from result object")
                    return str(result.final_output)

                logger.warning("No output found from any agent")
                return "No content was generated."

            except Exception as e:
                logger.error(f"Error in handoff workflow: {str(e)}")

                for agent in reversed(self.agents):
                    output = context.get_output(agent.agent_id)
                    if output:
                        logger.info(
                            f"Returning partial output from agent {agent.agent_id} after error"
                        )
                        return f"[PARTIAL RESULT] {output}"

                raise WorkflowError(
                    f"Handoff workflow failed: {str(e)}", workflow_id=self.name
                )

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import asyncio
import uuid
from datetime import datetime

from agents import gen_trace_id
from ..core.context import Context
from ..core.errors import WorkflowError
from ..core.logging import get_logger
from ..agents.base import AgentBase, AgentResponse

logger = get_logger("workflow_engine")


class WorkflowEngine:
    """
    Engine for executing agent workflows.
    Manages workflow execution, state, and error handling.
    """

    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """
        Initialize the workflow engine.

            max_retries: Maximum number of retries for failed agent executions
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.active_workflows = {}

    async def execute(
        self, workflow, context: Context, input_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow with the given context and input.

            workflow: Workflow to execute
            context: Execution context
            input_text: Optional direct input text

            Dict[str, Any]: Execution result
        """
        # Generate a valid trace ID using the Agents SDK function
        workflow_id = gen_trace_id()

        self.active_workflows[workflow_id] = {
            "start_time": datetime.now(),
            "status": "running",
            "agents_executed": [],
        }

        # Store the workflow ID in the context for tracking
        setattr(context, "_workflow_id", workflow_id)

        # Store the total number of agents in the workflow for progress tracking
        total_agents = len(getattr(workflow, "agents", []))
        setattr(context, "_total_agents", total_agents)

        try:
            logger.info(f"Starting workflow execution: {workflow_id}")

            # Execute the workflow based on its type
            if hasattr(workflow, "execute_with_trace"):
                # If workflow supports direct trace ID passing
                result = await workflow.execute_with_trace(
                    context, input_text, workflow_id
                )
            elif hasattr(workflow, "execute_deterministic"):
                result = await workflow.execute_deterministic(
                    context, input_text, trace_id=workflow_id
                )
            elif hasattr(workflow, "execute_handoff"):
                result = await workflow.execute_handoff(
                    context, input_text, trace_id=workflow_id
                )
            elif hasattr(workflow, "execute"):
                result = await workflow.execute(context, input_text)
            else:
                raise WorkflowError(
                    f"Unsupported workflow type: {type(workflow)}",
                    workflow_id=workflow_id,
                )

            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now()
            self.active_workflows[workflow_id]["result"] = result

            context.mark_complete()

            # Log the trace URL for debugging
            trace_url = f"https://platform.openai.com/traces/{workflow_id}"
            logger.info(f"Workflow trace available at: {trace_url}")

            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "result": result,
                "context": context,
            }

        except Exception as e:
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now()
            self.active_workflows[workflow_id]["error"] = str(e)

            context.record_error(str(e))

            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "context": context,
            }

    async def execute_agent(
        self, agent: AgentBase, context: Context, input_text: Optional[str] = None
    ) -> AgentResponse:
        """
        Execute a single agent with retry logic.

            agent: Agent to execute
            context: Execution context
            input_text: Optional direct input text

            AgentResponse: Agent execution response
        """
        retry_count = 0
        last_error = None

        # Get progress callback if available
        progress_callback = None
        if hasattr(context, "get_metadata"):
            progress_callback = context.get_user_metadata("progress_callback")

        # Track which agent is currently being executed for progress reporting
        active_agents = self.active_workflows.get(
            getattr(context, "_workflow_id", "unknown"), {}
        ).get("agents_executed", [])
        current_agent_idx = len(active_agents)
        total_agents = getattr(context, "_total_agents", 1)

        # Report progress if callback exists
        if progress_callback and callable(progress_callback):
            progress_callback(
                current_agent_idx, total_agents, f"Executing agent: {agent.agent_id}"
            )

        while retry_count <= self.max_retries:
            try:
                logger.info(
                    f"Executing agent {agent.agent_id} (attempt {retry_count + 1})"
                )

                response = await agent.execute(context, input_text)

                if response.success:
                    logger.info(f"Agent {agent.agent_id} executed successfully")

                    # Report progress on success if callback exists
                    if progress_callback and callable(progress_callback):
                        progress_callback(
                            current_agent_idx + 1,
                            total_agents,
                            f"Completed agent: {agent.agent_id}",
                        )

                    return response
                else:
                    error_msg = response.metadata.get("error", "Unknown error")
                    logger.warning(
                        f"Agent {agent.agent_id} reported failure: {error_msg}"
                    )
                    last_error = Exception(error_msg)
                    retry_count += 1

                    # Report retry progress if callback exists
                    if progress_callback and callable(progress_callback):
                        progress_callback(
                            current_agent_idx,
                            total_agents,
                            f"Retrying agent: {agent.agent_id} (attempt {retry_count + 1})",
                        )

                    if retry_count <= self.max_retries:
                        logger.info(
                            f"Retrying agent {agent.agent_id} in {self.retry_delay}s"
                        )
                        await asyncio.sleep(self.retry_delay)
                    else:
                        break

            except Exception as e:
                last_error = e
                retry_count += 1
                logger.error(f"Error executing agent {agent.agent_id}: {str(e)}")

                # Report error progress if callback exists
                if progress_callback and callable(progress_callback):
                    progress_callback(
                        current_agent_idx,
                        total_agents,
                        f"Error in agent: {agent.agent_id} - {str(e)[:50]}...",
                    )

                if retry_count <= self.max_retries:
                    logger.info(
                        f"Retrying agent {agent.agent_id} in {self.retry_delay}s"
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    break

        logger.error(f"Agent {agent.agent_id} failed after {self.max_retries} retries")
        error_message = f"Agent execution failed after {self.max_retries} retries: {str(last_error)}"
        return AgentResponse.error_response(agent.agent_id, error_message)

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow.

            workflow_id: ID of the workflow

            Dict[str, Any]: Workflow status
        """
        if workflow_id not in self.active_workflows:
            return {"status": "not_found"}

        return self.active_workflows[workflow_id]

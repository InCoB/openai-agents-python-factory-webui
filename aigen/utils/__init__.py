"""
Utility functions for the AI agent framework.
"""

import uuid
from typing import Dict, Any, Optional

from agents import trace

from ..core.context import Context
from ..core.logging import get_logger
from ..workflows.factory import create_workflow
from ..workflows.engine import WorkflowEngine

logger = get_logger("utils")


async def run_workflow(
    workflow_name: str, input_text: str, context: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Run a workflow with proper tracing and error handling.

    Args:
        workflow_name: Name of the workflow to run
        input_text: Input text for the workflow
        context: Optional context (will create one if not provided)

    Returns:
        Dict with workflow result and metadata
    """
    try:
        # Create a workflow instance
        workflow = create_workflow(workflow_name)

        # Create a context if one isn't provided
        if context is None:
            context = Context()

        # Create a workflow engine
        engine = WorkflowEngine()

        # Generate a trace ID for tracking
        trace_id = f"{workflow_name}_{uuid.uuid4().hex[:8]}"

        # Execute the workflow with a trace
        with trace(f"Workflow: {workflow_name}", trace_id=trace_id):
            result = await engine.execute(workflow, context, input_text)

            # Log the trace URL for debugging
            trace_url = f"https://platform.openai.com/traces/{trace_id}"
            logger.info(f"Workflow trace available at: {trace_url}")

            return result

    except Exception as e:
        logger.error(f"Error executing workflow {workflow_name}: {str(e)}")
        return {"status": "failed", "error": str(e), "result": None}

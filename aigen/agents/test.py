"""Simple test agent."""

from typing import Dict, Any, Optional
import asyncio

from agents import Agent, Runner
from agents.model_settings import ModelSettings

from .base import AgentBase, AgentRole, AgentResponse
from ..core.context import Context
from ..core.errors import AgentError
from ..core.logging import get_logger

logger = get_logger("test_agent")


class testAgent(AgentBase):
    """
    Simple test agent for testing workflows.
    """

    def __init__(self, agent_id: str = "test", **kwargs):
        """
        Initialize a test agent.

            agent_id: Unique identifier for the agent
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, role=AgentRole.CUSTOM, **kwargs)

        # Default instructions if not provided
        if not self.instructions:
            self.instructions = """
            You are a simple test agent.
            When called, reply with "Test completed successfully!" and a brief summary of the input.
            """

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self.is_initialized():
            return

        self.openai_agent = Agent(
            name=f"Test Agent ({self.agent_id})",
            instructions=self.instructions,
            tools=self.tools,
            model=self.parameters.get("model", "gpt-4o"),
            model_settings=ModelSettings(
                temperature=self.parameters.get("temperature", 0.7)
            ),
        )

        logger.info(f"Initialized test agent {self.agent_id}")

    async def execute(
        self, context: Context, input_text: Optional[str] = None
    ) -> AgentResponse:
        """
        Execute the test agent.

            context: Execution context
            input_text: Optional direct input text

            AgentResponse: The agent's response
        """
        try:
            # Initialize if not already done
            if not self.is_initialized():
                await self.initialize()

            # Get input - either direct or from context
            if input_text is None:
                input_text = context.get_latest_output() or "No input provided"

            logger.info(f"Executing test agent {self.agent_id}")

            # Execute the OpenAI agent
            response = await Runner.run(
                self.openai_agent,
                input=input_text,
                context=None,  # Don't pass the Context object directly to Runner
                max_turns=self.parameters.get("max_turns", 5)
            )

            # Get output from final_output property of RunResult
            result = response.final_output if hasattr(response, 'final_output') else str(response)

            # Store result in context
            context.store_output(self.agent_id, result)

            return AgentResponse(
                content=result,
                agent_id=self.agent_id,
                success=True,
                metadata={"role": "custom"}
            )

        except Exception as e:
            error_msg = f"Error executing test agent {self.agent_id}: {str(e)}"
            logger.error(error_msg)
            return AgentResponse.error_response(self.agent_id, str(e))
"""Strategy agent implementation."""

from typing import Dict, Any, Optional
import asyncio

from agents import Agent, RunContextWrapper
from agents.model_settings import ModelSettings

from .base import AgentBase, AgentRole, AgentResponse
from ..core.context import Context
from ..core.errors import AgentError
from ..core.logging import get_logger

logger = get_logger("strategy_agent")


class StrategyAgent(AgentBase):
    """
    Agent that generates content strategies and outlines.
    Specializes in planning and structuring content.
    """

    def __init__(self, agent_id: str = "strategy", **kwargs):
        """
        Initialize a strategy agent.

        Args:
            agent_id: Unique identifier for the agent
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, role=AgentRole.STRATEGY, **kwargs)

        # Default instructions if not provided
        if not self.instructions:
            self.instructions = """
            You are a content strategist who plans and outlines content.
            
            Your process:
            1. Analyze the request or research findings to identify key topics
            2. Create a logical structure for the content
            3. Include relevant headings and subheadings
            4. Add brief notes on what should be included in each section
            5. Consider the target audience and content purpose
            
            Focus on clarity, organization, and comprehensive coverage of the topic.
            """

    async def initialize(self) -> None:
        """Initialize the OpenAI agent."""
        if self.is_initialized():
            return

        from agents import Agent

        # Define instructions function
        async def instruction_fn(ctx, agent):
            return self.instructions

        # Create the OpenAI agent
        self.openai_agent = Agent(
            name=f"Strategy Agent ({self.agent_id})",
            instructions=instruction_fn,
            tools=self.tools,
            model=self.parameters.get("model", "gpt-4o"),
            model_settings=ModelSettings(
                temperature=self.parameters.get("temperature", 0.7)
            ),
        )

        logger.info(f"Initialized strategy agent {self.agent_id}")

    async def execute(
        self, context: Context, input_text: Optional[str] = None
    ) -> AgentResponse:
        """
        Execute the strategy agent.

        Args:
            context: Execution context
            input_text: Optional direct input text

        Returns:
            AgentResponse: The agent's response
        """
        await self.initialize()

        # Get input from context if not provided directly
        if input_text is None:
            input_text = context.get_latest_output() or ""

        try:
            logger.info(f"Executing strategy agent {self.agent_id}")

            # Prepare input by combining previous outputs if available
            enhanced_input = input_text

            # Look for research output
            research_output = context.get_output("research")
            if research_output:
                enhanced_input += (
                    f"\n\nAvailable research information:\n{research_output}"
                )

            # Execute the OpenAI agent
            from agents import Runner

            result = await Runner.run(
                self.openai_agent,
                input=enhanced_input,
                max_turns=self.parameters.get("max_turns", 10),
            )

            # Extract the result
            if hasattr(result, "final_output") and result.final_output:
                output = str(result.final_output)
            elif hasattr(result, "messages") and result.messages:
                output = str(result.messages[-1].content)
            else:
                output = str(result)

            # Save the result to context
            context.store_output(self.agent_id, output)

            # Determine next agent from handoffs if any
            next_agent = self.handoffs[0] if self.handoffs else None

            logger.info(f"Strategy agent {self.agent_id} completed successfully")

            # Return the response
            return AgentResponse(
                content=output,
                agent_id=self.agent_id,
                success=True,
                metadata={"word_count": len(output.split()), "char_count": len(output)},
                next_agent=next_agent,
            )

        except Exception as e:
            logger.error(f"Error executing strategy agent {self.agent_id}: {str(e)}")
            error = AgentError(
                f"Strategy agent execution failed: {str(e)}", agent_id=self.agent_id
            )
            context.record_error(str(error))
            return AgentResponse.error_response(self.agent_id, error)

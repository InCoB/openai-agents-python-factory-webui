from typing import Dict, Any, Optional
import asyncio

from agents import Agent, RunContextWrapper
from agents.model_settings import ModelSettings

from .base import AgentBase, AgentRole, AgentResponse
from ..core.context import Context
from ..core.errors import AgentError
from ..core.logging import get_logger

logger = get_logger("writer_agent")


class WriterAgent(AgentBase):
    """
    Agent that generates written content.
    Specializes in creating high-quality, engaging text.
    """

    def __init__(self, agent_id: str = "writer", **kwargs):
        """
        Initialize a writer agent.

            agent_id: Unique identifier for the agent
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, role=AgentRole.WRITER, **kwargs)

        if not self.instructions:
            self.instructions = """
            You are a professional content writer who creates high-quality, engaging content.
            
            Your process:
            
            Focus on providing value to the reader while meeting the content objectives.
            Be engaging but accurate, and maintain a consistent tone throughout.
            """

    async def initialize(self) -> None:
        """Initialize the OpenAI agent."""
        if self.is_initialized():
            return

        from agents import Agent

        async def instruction_fn(ctx, agent):
            return self.instructions

        self.openai_agent = Agent(
            name=f"Writer Agent ({self.agent_id})",
            instructions=instruction_fn,
            tools=self.tools,
            model=self.parameters.get("model", "gpt-4o"),
            model_settings=ModelSettings(
                temperature=self.parameters.get("temperature", 0.7)
            ),
        )

        logger.info(f"Initialized writer agent {self.agent_id}")

    async def execute(
        self, context: Context, input_text: Optional[str] = None
    ) -> AgentResponse:
        """
        Execute the writer agent.

            context: Execution context
            input_text: Optional direct input text

            AgentResponse: The agent's response
        """
        await self.initialize()

        if input_text is None:
            input_text = context.get_latest_output() or ""

        try:
            logger.info(f"Executing writer agent {self.agent_id}")

            enhanced_input = input_text

            research_output = context.get_output("research")
            if research_output:
                enhanced_input += (
                    f"\n\nAvailable research information:\n{research_output}"
                )

            strategy_output = context.get_output("strategy")
            if strategy_output:
                enhanced_input += f"\n\nContent outline:\n{strategy_output}"

            from agents import Runner

            result = await Runner.run(
                self.openai_agent,
                input=enhanced_input,
                max_turns=self.parameters.get("max_turns", 10),
            )

            if hasattr(result, "final_output") and result.final_output:
                output = str(result.final_output)
            elif hasattr(result, "messages") and result.messages:
                output = str(result.messages[-1].content)
            else:
                output = str(result)

            context.store_output(self.agent_id, output)

            next_agent = self.handoffs[0] if self.handoffs else None

            words = output.split()
            word_count = len(words)

            logger.info(
                f"Writer agent {self.agent_id} completed successfully: {word_count} words"
            )

            return AgentResponse(
                content=output,
                agent_id=self.agent_id,
                success=True,
                metadata={"word_count": word_count, "char_count": len(output)},
                next_agent=next_agent,
            )

        except Exception as e:
            logger.error(f"Error executing writer agent {self.agent_id}: {str(e)}")
            error = AgentError(
                f"Writer agent execution failed: {str(e)}", agent_id=self.agent_id
            )
            context.record_error(str(error))
            return AgentResponse.error_response(self.agent_id, error)

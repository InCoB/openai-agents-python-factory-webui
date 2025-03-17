from typing import Dict, Any, Optional
import asyncio

from agents import Agent, RunContextWrapper, Runner
from agents.model_settings import ModelSettings

from .base import AgentBase, AgentRole, AgentResponse
from ..core.context import Context
from ..core.errors import AgentError
from ..core.logging import get_logger
from ..tools.research import tavily_search_tool

logger = get_logger("research_agent")


class ResearchAgent(AgentBase):
    """
    Agent that performs research and information gathering.
    Specializes in finding relevant information from various sources.
    """

    def __init__(self, agent_id: str = "research", **kwargs):
        """
        Initialize a research agent.

            agent_id: Unique identifier for the agent
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, role=AgentRole.RESEARCH, **kwargs)

        if not self.instructions:
            self.instructions = """
            You are a specialized research agent who finds and organizes information.
            
            Your process:
            
            Focus on depth and accuracy rather than breadth. Quality research is the foundation of excellent results.
            """

        self.add_tool(tavily_search_tool)

    async def initialize(self) -> None:
        """Initialize the OpenAI agent."""
        if self.is_initialized():
            return

        from agents import Agent

        async def instruction_fn(ctx, agent):
            return self.instructions

        self.openai_agent = Agent(
            name=f"Research Agent ({self.agent_id})",
            instructions=instruction_fn,
            tools=self.tools,
            model=self.parameters.get("model", "gpt-4o"),
            model_settings=ModelSettings(
                temperature=self.parameters.get("temperature", 0.7), tool_choice="auto"
            ),
        )

        logger.info(f"Initialized research agent {self.agent_id}")

    async def execute(
        self, context: Context, input_text: Optional[str] = None
    ) -> AgentResponse:
        """
        Execute the agent to gather research information.

            context: Execution context
            input_text: Optional direct input text

            AgentResponse: The agent's response
        """
        try:
            # Initialize if not already done
            if not hasattr(self, "openai_agent") or self.openai_agent is None:
                await self.initialize()

            # Get input - either direct or from context
            if input_text is None:
                # Try to get input from context
                input_text = context.get_latest_output()

            if not input_text:
                return AgentResponse.error_response(
                    self.agent_id, "No input provided for research"
                )

            # Execute research using Runner instead of calling run() directly
            response = await Runner.run(self.openai_agent, input_text)

            # Process result - use final_output instead of content
            result = response.final_output

            # Format the result for better display in UI
            # This is especially important for research_only workflow where this is the final output
            if not result.startswith("# Research Results"):
                # Add proper formatting if it's not already formatted
                formatted_result = f"""# Research Results on: {input_text}

{result}

---
*Research conducted by {self.agent_id} agent*
"""
                result = formatted_result

            # Store in context
            context.store_output(self.agent_id, result)
            context.set_metadata("research", {"word_count": len(result.split())})

            # Return response
            return AgentResponse(
                content=result,
                agent_id=self.agent_id,
                success=True,
                metadata={"research_completed": True},
            )

        except Exception as e:
            logger.error(f"Error executing research agent: {str(e)}")
            error = AgentError(
                f"Research agent execution failed: {str(e)}", agent_id=self.agent_id
            )
            context.record_error(str(error))
            return AgentResponse.error_response(self.agent_id, error)

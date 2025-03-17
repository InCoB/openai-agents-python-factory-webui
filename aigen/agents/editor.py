from typing import Dict, Any, Optional
import asyncio

from agents import Agent, RunContextWrapper
from agents.model_settings import ModelSettings

from .base import AgentBase, AgentRole, AgentResponse
from ..core.context import Context
from ..core.errors import AgentError
from ..core.logging import get_logger

logger = get_logger("editor_agent")

class EditorAgent(AgentBase):
    """
    Agent that edits and refines content.
    Specializes in improving clarity, grammar, and overall quality.
    """
    
    def __init__(self, agent_id: str = "editor", **kwargs):
        """
        Initialize an editor agent.
        """
        super().__init__(agent_id=agent_id, role=AgentRole.EDITOR, **kwargs)
        
        # Default instructions if not provided
        if not self.instructions:
            self.instructions = """
            You are a professional content editor who improves and refines content.
            
            Your process:
            1. Review the provided content for clarity, coherence, and correctness
            2. Fix any grammatical errors or awkward phrasing
            3. Improve the flow and structure if needed
            4. Enhance the overall quality while preserving the original meaning
            5. Check for factual accuracy and logical consistency
            
            Focus on making the content both engaging and professional.
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
            name=f"Editor Agent ({self.agent_id})",
            instructions=instruction_fn,
            tools=self.tools,
            model=self.parameters.get("model", "gpt-4o"),
            model_settings=ModelSettings(
                temperature=self.parameters.get("temperature", 0.7)
            )
        )
        
        logger.info(f"Initialized editor agent {self.agent_id}")
    
    async def execute(self, context: Context, input_text: Optional[str] = None) -> AgentResponse:
        """
        Execute the editor agent.
        """
        await self.initialize()
        
        # Get input from context if not provided directly
        if input_text is None:
            input_text = context.get_latest_output() or ""
        
        try:
            logger.info(f"Executing editor agent {self.agent_id}")
            
            # Prepare input by combining previous outputs if available
            enhanced_input = input_text
            
            # Look for writer output if not directly provided
            if not input_text or len(input_text.strip()) < 100:  # Assume we need to find content to edit
                writer_output = context.get_output("writer")
                if writer_output:
                    enhanced_input = f"Please edit and improve the following content:\n\n{writer_output}"
            
            # Execute the OpenAI agent
            from agents import Runner
            
            result = await Runner.run(
                self.openai_agent,
                input=enhanced_input,
                max_turns=self.parameters.get("max_turns", 10)
            )
            
            # Extract the result
            if hasattr(result, 'final_output') and result.final_output:
                output = str(result.final_output)
            elif hasattr(result, 'messages') and result.messages:
                output = str(result.messages[-1].content)
            else:
                output = str(result)
            
            # Save the result to context
            context.store_output(self.agent_id, output)
            
            # Determine next agent from handoffs if any
            next_agent = self.handoffs[0] if self.handoffs else None
            
            logger.info(f"Editor agent {self.agent_id} completed successfully")
            
            # Return the response
            return AgentResponse(
                content=output,
                agent_id=self.agent_id,
                success=True,
                metadata={
                    "word_count": len(output.split()),
                    "char_count": len(output)
                },
                next_agent=next_agent
            )
            
        except Exception as e:
            logger.error(f"Error executing editor agent {self.agent_id}: {str(e)}")
            error = AgentError(f"Editor agent execution failed: {str(e)}", agent_id=self.agent_id)
            context.record_error(str(error))
            return AgentResponse.error_response(self.agent_id, error)
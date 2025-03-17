from typing import Dict, Any, List, Optional, Set, Protocol, Union, Callable
from enum import Enum
import asyncio
from datetime import datetime

from agents import Agent

from ..core.context import Context
from ..core.errors import AgentError

class AgentRole(Enum):
    """Enumeration of standard agent roles."""
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    STRATEGY = "strategy"
    WRITER = "writer"
    EDITOR = "editor"
    ANALYST = "analyst"
    CUSTOM = "custom"

class AgentResponse:
    """
    Standardized response object from agent execution.
    Provides consistent structure for agent outputs.
    """
    
    def __init__(self, 
                content: str, 
                agent_id: str,
                success: bool = True,
                metadata: Optional[Dict[str, Any]] = None,
                next_agent: Optional[str] = None):
        """
        Initialize an agent response.
        
            content: The main content output by the agent
            agent_id: ID of the agent that produced this response
            success: Whether the agent executed successfully
            metadata: Additional metadata about the execution
            next_agent: Optional ID of the next agent to execute
        """
        self.content = content
        self.agent_id = agent_id
        self.success = success
        self.metadata = metadata or {}
        self.next_agent = next_agent
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "content": self.content,
            "agent_id": self.agent_id,
            "success": self.success,
            "metadata": self.metadata,
            "next_agent": self.next_agent,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def error_response(cls, agent_id: str, error: Union[str, Exception]) -> 'AgentResponse':
        """
        Create an error response.
        
            agent_id: ID of the agent
            error: Error message or exception
            
            AgentResponse: Error response
        """
        return cls(
            content=str(error),
            agent_id=agent_id,
            success=False,
            metadata={"error": str(error)}
        )

class AgentBase:
    """
    Base class for all agents in the framework.
    Provides common functionality and interface.
    """
    
    def __init__(self, agent_id: str, role: AgentRole = AgentRole.CUSTOM, **kwargs):
        """
        Initialize an agent.
        
            agent_id: Unique identifier for the agent
            role: The agent's role in the system
            **kwargs: Additional agent parameters
        """
        self.agent_id = agent_id
        self.role = role
        self.openai_agent: Optional[Agent] = None
        self.tools = []
        self.handoffs = []
        self.parameters = kwargs
        self.instructions: Optional[str] = kwargs.get("instructions")
    
    async def initialize(self) -> None:
        """Initialize the agent, called before first execution."""
        pass
    
    async def execute(self, context: Context, input_text: Optional[str] = None) -> AgentResponse:
        """
        Execute the agent on the given context.
        
            context: Execution context
            input_text: Optional direct input text
            
            AgentResponse: The agent's response
            
            AgentError: If execution fails
        """
        raise NotImplementedError("Subclasses must implement execute method")
    
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self.openai_agent is not None
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if the agent has a specific tool."""
        return any(t.name == tool_name for t in self.tools)
    
    def add_tool(self, tool) -> None:
        """
        Add a tool to the agent.
        
            tool: The tool to add
        """
        if not self.has_tool(tool.name):
            self.tools.append(tool)
    
    def add_handoff(self, agent_id: str) -> None:
        """
        Add a handoff target.
        
            agent_id: ID of the agent to hand off to
        """
        if agent_id not in self.handoffs:
            self.handoffs.append(agent_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "tools": [t.name for t in self.tools] if self.tools else [],
            "handoffs": self.handoffs.copy() if self.handoffs else [],
            "parameters": {k: v for k, v in self.parameters.items() if k != "instructions"}
        }
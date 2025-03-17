"""Data models for agent configurations."""

import re
from enum import Enum
from typing import Dict, Any, List, Optional, ClassVar

from pydantic import BaseModel, Field, validator

class AgentRole(str, Enum):
    """Enum for agent roles."""
    RESEARCH = "RESEARCH"
    WRITER = "WRITER"
    EDITOR = "EDITOR"
    STRATEGY = "STRATEGY"
    ORCHESTRATOR = "ORCHESTRATOR"
    ANALYST = "ANALYST"
    CUSTOM = "CUSTOM"
    
    @classmethod
    def to_framework_role(cls, role_value: str) -> str:
        """
        Convert UI role enum to framework role value.
        
        Args:
            role_value: The role value from the UI enum.
            
        Returns:
            The corresponding framework role value.
        """
        role_map = {
            cls.RESEARCH.value: "research",
            cls.WRITER.value: "writer", 
            cls.EDITOR.value: "editor",
            cls.STRATEGY.value: "strategy",
            cls.ORCHESTRATOR.value: "orchestrator",
            cls.ANALYST.value: "analyst",
            cls.CUSTOM.value: "custom"
        }
        return role_map.get(role_value, "custom")


class AgentConfiguration(BaseModel):
    """Data model for agent configuration."""
    
    agent_type: str = Field(
        ..., 
        description="Unique identifier for this agent type"
    )
    name: str = Field(
        ..., 
        description="Display name for the agent"
    )
    role: AgentRole = Field(
        AgentRole.CUSTOM, 
        description="Agent's functional role"
    )
    instructions: str = Field(
        ..., 
        description="System prompt/instructions for the agent"
    )
    model: str = Field(
        "gpt-4o", 
        description="Model to use for the agent"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Agent parameters"
    )
    tools: List[str] = Field(
        default_factory=list, 
        description="Tools the agent can use"
    )
    handoffs: List[str] = Field(
        default_factory=list, 
        description="Agents this agent can hand off to"
    )
    output_type: Optional[str] = Field(
        None, 
        description="Structured output type specification"
    )
    
    # Class variables for validation
    AGENT_TYPE_PATTERN: ClassVar[re.Pattern] = re.compile(r'^[a-z][a-z0-9_]*$')
    
    @classmethod
    def to_framework_role(cls, role_value: str) -> str:
        """
        Convert UI role enum to framework role value.
        
        Args:
            role_value: The role value from the UI enum.
            
        Returns:
            The corresponding framework role value.
        """
        return AgentRole.to_framework_role(role_value)
    
    @staticmethod
    def validate_agent_type(v: str) -> str:
        """
        Validate agent_type format.
        
        Args:
            v: The agent type string to validate.
            
        Returns:
            The validated agent type string.
            
        Raises:
            ValueError: If the agent_type is invalid.
        """
        if not v:
            raise ValueError("agent_type cannot be empty")
        
        if not AgentConfiguration.AGENT_TYPE_PATTERN.match(v):
            raise ValueError(
                "agent_type must start with a lowercase letter and contain "
                "only lowercase letters, numbers, and underscores"
            )
        
        return v
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """
        Validate agent name.
        
            v: The agent name to validate.
            
            
            ValueError: If the name is invalid.
        """
        if not v:
            raise ValueError("name cannot be empty")
        return v
    
    @validator('instructions')
    def validate_instructions(cls, v: str) -> str:
        """
        Validate agent instructions.
        
            v: The instructions to validate.
            
            
            ValueError: If the instructions are invalid.
        """
        if not v:
            raise ValueError("instructions cannot be empty")
        return v
    
    class Config:
        """Configuration for the model."""
        schema_extra = {
            "example": {
                "agent_type": "financial_analyst",
                "name": "Financial Analysis Agent",
                "role": "ANALYST",
                "instructions": "You are a financial analysis agent...",
                "model": "gpt-4o",
                "parameters": {"temperature": 0.3, "max_tokens": 4000},
                "tools": ["calculator", "research_tool"],
                "handoffs": ["report_writer"],
                "output_type": "class FinancialReport(BaseModel):\n    summary: str\n    recommendations: List[str]"
            }
        }
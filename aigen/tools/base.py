from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import asyncio
from datetime import datetime

from ..core.errors import ToolError


class ToolType(Enum):
    """Enumeration of standard tool types."""

    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    COMPUTATION = "computation"
    UTILITIES = "utilities"
    CUSTOM = "custom"


class ToolResponse:
    """
    Standardized response object from tool execution.
    Provides consistent structure for tool outputs.
    """

    def __init__(
        self,
        data: Any,
        tool_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a tool response.

            data: The main data output by the tool
            tool_id: ID of the tool that produced this response
            success: Whether the tool executed successfully
            metadata: Additional metadata about the execution
        """
        self.data = data
        self.tool_id = tool_id
        self.success = success
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "data": self.data,
            "tool_id": self.tool_id,
            "success": self.success,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def error_response(
        cls, tool_id: str, error: Union[str, Exception]
    ) -> "ToolResponse":
        """
        Create an error response.

            tool_id: ID of the tool
            error: Error message or exception

            ToolResponse: Error response
        """
        return cls(
            data=None, tool_id=tool_id, success=False, metadata={"error": str(error)}
        )


class ToolBase:
    """
    Base class for all tools in the framework.
    Provides common functionality and interface.
    """

    def __init__(self, tool_id: str, tool_type: ToolType = ToolType.CUSTOM, **kwargs):
        """
        Initialize a tool.

            tool_id: Unique identifier for the tool
            tool_type: The tool's type
            **kwargs: Additional tool parameters
        """
        self.tool_id = tool_id
        self.tool_type = tool_type
        self.parameters = kwargs
        self.description = kwargs.get("description", "")
        self.name = kwargs.get("name", tool_id)

    async def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the tool with the given parameters.

            **kwargs: Tool execution parameters

            ToolResponse: The tool's response

            ToolError: If execution fails
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "tool_id": self.tool_id,
            "tool_type": self.tool_type.value,
            "name": self.name,
            "description": self.description,
            "parameters": {k: v for k, v in self.parameters.items()},
        }

"""
Errors Module - Defines exception types for the AI Agent Framework.

This module implements a comprehensive error handling system with hierarchical exception
classes. It provides structured error information including severity levels, error
categories, and detailed context for debugging and reporting.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
import traceback


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"  # Minor issue, non-disruptive
    MEDIUM = "medium"  # Significant issue, partial functionality affected
    HIGH = "high"  # Serious issue, major functionality affected
    FATAL = "fatal"  # Critical issue, system cannot continue


class ErrorCategory(Enum):
    """Categorization of error types."""

    CONFIGURATION = "configuration"  # Configuration-related errors
    VALIDATION = "validation"  # Input validation errors
    API = "api"  # External API errors
    NETWORK = "network"  # Network-related errors
    AGENT = "agent"  # Agent execution errors
    TOOL = "tool"  # Tool execution errors
    WORKFLOW = "workflow"  # Workflow execution errors
    UNKNOWN = "unknown"  # Uncategorized errors


class AISystemError(Exception):
    """
    Base exception class for all system errors.
    Provides structured error information and context.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize a system error.

            message: Error message
            category: Error category
            severity: Error severity
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.traceback = traceback.extract_stack() if cause else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        result = {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
        }

        if self.cause:
            result["cause"] = str(self.cause)

        return result

    def __str__(self) -> str:
        """String representation of the error."""
        base = f"{self.category.value.upper()} ERROR [{self.severity.value}]: {self.message}"

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({details_str})"

        if self.cause:
            base += f"\nCaused by: {self.cause}"

        return base


class ConfigurationError(AISystemError):
    """Error related to system configuration."""

    def __init__(self, message: str, **kwargs):
        """Initialize a configuration error."""
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class ValidationError(AISystemError):
    """Error related to input/output validation."""

    def __init__(self, message: str, **kwargs):
        """Initialize a validation error."""
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class AgentError(AISystemError):
    """Error related to agent execution."""

    def __init__(self, message: str, agent_id: str, **kwargs):
        """
        Initialize an agent error.

            message: Error message
            agent_id: ID of the agent that encountered the error
            **kwargs: Additional error parameters
        """
        details = kwargs.pop("details", {})
        details["agent_id"] = agent_id
        super().__init__(
            message, category=ErrorCategory.AGENT, details=details, **kwargs
        )


class ToolError(AISystemError):
    """Error related to tool execution."""

    def __init__(self, message: str, tool_id: str, **kwargs):
        """
        Initialize a tool error.

            message: Error message
            tool_id: ID of the tool that encountered the error
            **kwargs: Additional error parameters
        """
        details = kwargs.pop("details", {})
        details["tool_id"] = tool_id
        super().__init__(
            message, category=ErrorCategory.TOOL, details=details, **kwargs
        )


class WorkflowError(AISystemError):
    """Error related to workflow execution."""

    def __init__(self, message: str, workflow_id: str, **kwargs):
        """
        Initialize a workflow error.

            message: Error message
            workflow_id: ID of the workflow that encountered the error
            **kwargs: Additional error parameters
        """
        details = kwargs.pop("details", {})
        details["workflow_id"] = workflow_id
        super().__init__(
            message, category=ErrorCategory.WORKFLOW, details=details, **kwargs
        )

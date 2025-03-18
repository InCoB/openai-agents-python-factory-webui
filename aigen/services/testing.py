"""Agent testing service."""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentTestingService:
    """Service for testing agents before deployment."""

    def __init__(self, timeout: float = 30.0) -> None:
        """
        Initialize the testing service.

            timeout: Maximum execution time in seconds.
        """
        self.timeout = timeout

    async def test_agent(
        self, agent_type: str, test_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Test an agent with sample input.

            agent_type: Type of agent to test.
            test_input: Test input to send to the agent.
            context: Optional context to pass to the agent.

            Test result dictionary.
        """
        try:
            try:
                from aigen.agents.factory import agent_registry

                if not hasattr(agent_registry, "exists"):
                    if (
                        not hasattr(agent_registry, "_factories")
                        or agent_type not in agent_registry._factories
                    ):
                        return {
                            "success": False,
                            "error": f"Agent type '{agent_type}' not found in registry",
                            "content": f"Agent type '{agent_type}' not found",
                        }
                elif not agent_registry.exists(agent_type):
                    return {
                        "success": False,
                        "error": f"Agent type '{agent_type}' not found in registry",
                        "content": f"Agent type '{agent_type}' not found",
                    }
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"Failed to import agent registry: {str(e)}",
                    "content": "Internal error: Failed to import agent registry",
                }

            try:
                from aigen.agents.factory import create_agent
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"Failed to import agent factory: {str(e)}",
                    "content": "Internal error: Failed to import agent factory",
                }

            try:
                agent = create_agent(agent_type)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create agent: {str(e)}",
                    "content": f"Failed to create agent: {str(e)}",
                }

            try:
                coro = agent.execute(test_input, context or {})
                response = await asyncio.wait_for(coro, timeout=self.timeout)

                return {
                    "success": response.get("success", False),
                    "content": response.get("content", "No content returned"),
                    "metadata": response.get("metadata", {}),
                }
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Agent execution timed out after {self.timeout} seconds",
                    "content": "The agent took too long to respond",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error executing agent: {str(e)}",
                    "content": f"Error: {str(e)}",
                }

        except Exception as e:
            error_msg = f"Error testing agent {agent_type}: {str(e)}"
            logger.error(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "content": f"Failed to test agent: {error_msg}",
            }

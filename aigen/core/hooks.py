"""
Hooks Module - Defines the event hook system for the AI Agent Framework.

This module provides a flexible hook system that allows for intercepting and modifying
behavior at various points in the agent and workflow execution lifecycle. It enables
custom logic to be injected at key points such as agent start/end, tool execution,
and workflow transitions.
"""

from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from agents import RunHooks, RunContextWrapper

from .logging import get_logger
from .context import Context

logger = get_logger("workflow_hooks")


class ContentWorkflowHooks(RunHooks):
    """
    Hooks for monitoring and logging workflow execution.
    Tracks agent execution, handoffs, and tool usage.
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initialize hooks.

            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        logger.info("Initializing ContentWorkflowHooks")
        self.agent_outputs = {}
        self.agent_times = {}
        self.handoff_count = 0
        self.current_agent = None
        self.agent_history = []
        self.total_tokens = 0
        self.start_time = time.time()
        self.successful_completions = set()

    async def on_agent_start(self, context_wrapper: RunContextWrapper, agent) -> None:
        """Called when an agent starts running."""
        try:
            self.current_agent = agent.name
            self.agent_history.append(agent.name)

            logger.info(f"Agent '{agent.name}' starting")

            if self.debug_mode:
                logger.debug(f"Context type: {type(context_wrapper.context)}")
                if hasattr(context_wrapper.context, "get_state_summary"):
                    context_state = context_wrapper.context.get_state_summary()
                    logger.debug(
                        f"Context state before '{agent.name}' starts:", context_state
                    )

                if agent.name == "Content Strategist" and hasattr(
                    context_wrapper.context, "has_research"
                ):
                    if not context_wrapper.context.has_research():
                        logger.warning(
                            f"No research data found before '{agent.name}' starts!"
                        )

                if agent.name == "Content Writer" and hasattr(
                    context_wrapper.context, "has_outline"
                ):
                    if not context_wrapper.context.has_outline():
                        logger.warning(
                            f"No outline data found before '{agent.name}' starts!"
                        )

                if agent.name == "Content Editor" and hasattr(
                    context_wrapper.context, "has_draft"
                ):
                    if not context_wrapper.context.has_draft():
                        logger.warning(
                            f"No draft content found before '{agent.name}' starts!"
                        )

        except Exception as e:
            logger.error(f"Error in on_agent_start: {str(e)}")

    async def on_agent_end(
        self, context_wrapper: RunContextWrapper, agent, output
    ) -> None:
        """Called when an agent ends running."""
        try:
            if not agent or not hasattr(agent, "name"):
                return

            logger.debug(f"Agent '{agent.name}' ended, output type: {type(output)}")

            output_text = None
            if isinstance(output, str):
                output_text = output
                logger.debug(
                    f"Using direct string output from agent: {len(output_text)} chars"
                )
            elif (
                hasattr(agent, "messages")
                and agent.messages
                and len(agent.messages) > 0
            ):
                last_message = agent.messages[-1]
                if hasattr(last_message, "content"):
                    output_text = last_message.content
                    logger.debug(
                        f"Using content from agent's last message: {len(output_text)} chars"
                    )
                else:
                    output_text = str(last_message)
                    logger.debug(
                        f"Using string representation of agent's last message: {len(output_text)} chars"
                    )
            else:
                output_text = str(output)
                logger.debug(
                    f"Using stringified output parameter: {len(output_text)} chars"
                )

            if output_text:
                output_length = len(output_text)
                self.agent_outputs[agent.name] = {
                    "length": output_length,
                    "timestamp": datetime.now().isoformat(),
                }

                if hasattr(context_wrapper, "context"):
                    ctx = context_wrapper.context
                    storage_success = self._store_agent_output(
                        ctx, agent.name, output_text
                    )

                    if storage_success:
                        self.successful_completions.add(agent.name)
                        logger.success(
                            f"Agent '{agent.name}' finished successfully",
                            outputs_length=output_length,
                            output_sample=(
                                output_text[:100] + "..."
                                if len(output_text) > 100
                                else output_text
                            ),
                        )
                    else:
                        logger.warning(
                            f"Agent '{agent.name}' finished but output storage failed"
                        )

                    if self.debug_mode and hasattr(ctx, "get_state_summary"):
                        context_state = ctx.get_state_summary()
                        logger.debug(
                            f"Context state after '{agent.name}' completes:",
                            context_state,
                        )
            else:
                logger.warning(f"No output text could be extracted from {agent.name}")

        except Exception as e:
            logger.error(f"Error in on_agent_end: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

    def _store_agent_output(self, ctx, agent_name: str, output: str) -> bool:
        """Helper method to store agent output in context."""
        try:
            if hasattr(ctx, "store_output"):
                ctx.store_output(agent_name, output)
                return True

            if agent_name == "Research Specialist" and hasattr(
                ctx, "set_research_data"
            ):
                return ctx.set_research_data(output)
            elif agent_name == "Content Strategist" and hasattr(
                ctx, "set_outline_data"
            ):
                return ctx.set_outline_data(output)
            elif agent_name == "Content Writer" and hasattr(ctx, "set_draft_content"):
                return ctx.set_draft_content(output)
            elif agent_name == "Content Editor" and hasattr(ctx, "set_final_content"):
                return ctx.set_final_content(output)
            else:
                setattr(ctx, f"{agent_name.lower()}_output", output)
                return True

        except Exception as e:
            logger.error(f"Error in _store_agent_output: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    async def on_handoff(
        self, context: RunContextWrapper, from_agent, to_agent
    ) -> None:
        """Called when an agent hands off to another agent."""
        try:
            self.handoff_count += 1
            logger.info(f"🔄 HANDOFF: From '{from_agent.name}' to '{to_agent.name}'")

            if hasattr(context, "context"):
                ctx = context.context
                if hasattr(from_agent, "messages") and from_agent.messages:
                    last_message = from_agent.messages[-1]
                    output_text = (
                        last_message.content
                        if hasattr(last_message, "content")
                        else str(last_message)
                    )
                    logger.debug(
                        f"Handoff message from {from_agent.name}: {output_text[:100]}..."
                    )

                    if from_agent.name not in self.successful_completions:
                        storage_success = self._store_agent_output(
                            ctx, from_agent.name, output_text
                        )
                        if storage_success:
                            self.successful_completions.add(from_agent.name)

                    if self.debug_mode and hasattr(ctx, "get_state_summary"):
                        context_state = ctx.get_state_summary()
                        logger.debug(f"Context state during handoff:", context_state)

            self.current_agent = to_agent.name
        except Exception as e:
            logger.error(f"Error in on_handoff: {str(e)}")

    async def on_tool_start(
        self, context_wrapper: RunContextWrapper, agent, tool
    ) -> None:
        """Called when a tool starts execution."""
        try:
            if tool and agent and hasattr(agent, "name"):
                tool_name = tool.name if hasattr(tool, "name") else str(tool)
                logger.info(f"Tool '{tool_name}' starting in agent '{agent.name}'")
        except Exception as e:
            logger.error(f"Error in on_tool_start: {str(e)}")

    async def on_tool_end(
        self, context_wrapper: RunContextWrapper, agent, tool, result: str
    ) -> None:
        """Called when a tool ends execution."""
        try:
            if tool and agent and hasattr(agent, "name"):
                tool_name = tool.name if hasattr(tool, "name") else str(tool)
                logger.success(f"Tool '{tool_name}' completed in agent '{agent.name}'")

                if self.debug_mode:
                    result_snippet = (
                        str(result)[:100] + "..."
                        if len(str(result)) > 100
                        else str(result)
                    )
                    logger.debug(f"Tool result: {result_snippet}")
        except Exception as e:
            logger.error(f"Error in on_tool_end: {str(e)}")

    async def on_run_end(self, context_wrapper: RunContextWrapper) -> None:
        """Called when the entire run completes."""
        try:
            if hasattr(context_wrapper, "context"):
                ctx = context_wrapper.context
                if self.debug_mode and hasattr(ctx, "get_state_summary"):
                    context_state = ctx.get_state_summary()
                    logger.debug(
                        f"Final context state after running agents:", context_state
                    )

                elapsed = time.time() - self.start_time
                self._report_final_state(ctx, elapsed)
            else:
                logger.warning("No context available in on_run_end")
        except Exception as e:
            logger.error(f"Error in on_run_end: {str(e)}")

    def _report_final_state(self, ctx, elapsed: float) -> None:
        """Helper to report the final state of content generation."""
        try:
            has_final = hasattr(ctx, "has_final") and ctx.has_final()
            has_draft = hasattr(ctx, "has_draft") and ctx.has_draft()
            has_outline = hasattr(ctx, "has_outline") and ctx.has_outline()
            has_research = hasattr(ctx, "has_research") and ctx.has_research()

            if has_final:
                final_content = ctx.get_final() if hasattr(ctx, "get_final") else None
                logger.success(
                    f"Workflow complete with final content!",
                    characters=len(final_content) if final_content else 0,
                    time_taken=f"{elapsed:.2f}s",
                )
            elif has_draft:
                draft_content = ctx.get_draft() if hasattr(ctx, "get_draft") else None
                logger.success(
                    f"Workflow complete with draft content!",
                    characters=len(draft_content) if draft_content else 0,
                    time_taken=f"{elapsed:.2f}s",
                )
            elif has_outline:
                outline = ctx.get_outline() if hasattr(ctx, "get_outline") else None
                logger.warning(
                    f"Workflow stopped at outline stage!",
                    outline_length=len(outline) if outline else 0,
                    time_taken=f"{elapsed:.2f}s",
                )
            elif has_research:
                research = ctx.get_research() if hasattr(ctx, "get_research") else None
                logger.warning(
                    f"Workflow stopped at research stage!",
                    research_length=len(research) if research else 0,
                    time_taken=f"{elapsed:.2f}s",
                )
            else:
                outputs = []
                for agent in self.agent_history:
                    output = None
                    if hasattr(ctx, "get_output"):
                        output = ctx.get_output(agent)
                    elif hasattr(ctx, f"{agent.lower()}_output"):
                        output = getattr(ctx, f"{agent.lower()}_output")

                    if output:
                        outputs.append((agent, len(output)))

                if outputs:
                    last_agent, last_len = outputs[-1]
                    logger.info(
                        f"Workflow completed with output from {last_agent}",
                        characters=last_len,
                        time_taken=f"{elapsed:.2f}s",
                    )
                else:
                    logger.error(
                        f"Workflow completed but no content was generated!",
                        time_taken=f"{elapsed:.2f}s",
                        handoffs=self.handoff_count,
                        agents_run=len(self.agent_history),
                    )
        except Exception as e:
            logger.error(f"Error in _report_final_state: {str(e)}")

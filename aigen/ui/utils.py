"""UI utility functions."""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, List, Union, AsyncGenerator


def dict_to_string_adapter(fn: Callable) -> Callable:
    """
    Adapter to handle various return types and ensure compatibility with UI components.

    This adapter:
    1. Handles both async and non-async functions
    2. Preserves dictionaries for JSON components
    3. Collects async generator outputs for streaming
    4. Ensures proper error handling

    Args:
        fn: The function to adapt

    Returns:
        Wrapped function with compatible return types
    """

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            # Handle async generators (for streaming results)
            if inspect.isasyncgenfunction(fn):
                messages = []
                async for item in fn(*args, **kwargs):
                    if item is not None:
                        if isinstance(item, dict):
                            messages.append(item)
                        else:
                            messages.append(str(item))
                return messages

            # Handle regular async functions
            elif inspect.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
                # For JSON components, return the dict directly
                return result

            # Handle synchronous functions
            else:
                result = fn(*args, **kwargs)
                return result

        except Exception as e:
            # Provide a standardized error format
            return {
                "status": "error",
                "message": f"❌ Error: {str(e)}",
                "error": str(e),
            }

    return wrapper

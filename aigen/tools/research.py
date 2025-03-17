from typing import Dict, Any, Optional, List
import os
import time
import asyncio
import requests
from datetime import datetime

from agents import function_tool

from .base import ToolBase, ToolResponse, ToolType
from ..core.errors import ToolError
from ..core.logging import get_logger

logger = get_logger("research_tools")

class TavilySearchTool(ToolBase):
    """
    Tool for searching the web using the Tavily API.
    Provides relevant information from web searches.
    """
    
    def __init__(self, tool_id: str = "tavily_search", **kwargs):
        """
        Initialize a Tavily search tool.
        
            tool_id: Unique identifier for the tool
            **kwargs: Additional tool parameters
        """
        super().__init__(tool_id=tool_id, tool_type=ToolType.RESEARCH, **kwargs)
        
        self.name = kwargs.get("name", "tavily_search")
        self.description = kwargs.get("description", 
                                    "Search the web for relevant information on a topic using Tavily.")
    
    async def execute(self, query: str, results_count: int = 3, max_retries: int = 2) -> ToolResponse:
        """
        Execute the tool with the given parameters.
        
            query: Search query
            results_count: Number of results to return
            max_retries: Maximum number of retries on failure
            
            ToolResponse: The tool's response
        """
        start_time = time.time()
        
        if not query or not query.strip():
            error = "Empty search query provided"
            logger.error(error)
            return ToolResponse.error_response(self.tool_id, error)
        
        query = query.strip()
        logger.info(f"Searching web for: '{query}'", results_count=results_count)
        
        results_count = max(1, min(10, results_count))
        
        try:
            from tavily import TavilyClient
            has_tavily_package = True
        except ImportError:
            logger.warning("Tavily package not found. Using requests library instead.")
            has_tavily_package = False
        
        def create_mock_results(reason: str) -> List[Dict[str, str]]:
            return [
                {
                    "title": "AI Content Creation Techniques",
                    "snippet": f"Note: Using example data ({reason}). AI-powered content creation involves research, planning, drafting, and editing phases.",
                    "url": "https://example.com/ai-content-creation"
                },
                {
                    "title": "Content Strategy Best Practices",
                    "snippet": f"Note: Using example data ({reason}). Effective content strategy requires audience analysis, keyword research, and consistent publishing.",
                    "url": "https://example.com/content-strategy"
                }
            ]
        
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("No Tavily API key found, using mock data")
            return ToolResponse(
                data={
                    "query": query,
                    "results": create_mock_results("no API key"),
                    "timestamp": datetime.now().isoformat(),
                    "note": "Mock data - no Tavily API key provided"
                },
                tool_id=self.tool_id,
                metadata={"mock": True, "reason": "no_api_key"}
            )
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                logger.debug(f"Making API request to Tavily (attempt {retry_count + 1})")
                
                if has_tavily_package:
                    tavily_client = TavilyClient(api_key=api_key)
                    response_data = tavily_client.search(query, max_results=results_count)
                else:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    payload = {
                        "query": query,
                        "max_results": results_count
                    }
                    
                    response = requests.post(
                        "https://api.tavily.com/search",
                        json=payload,
                        headers=headers,
                        timeout=15
                    )
                    
                    if response.status_code != 200:
                        error_message = f"Tavily API error: HTTP {response.status_code}"
                        try:
                            error_detail = response.json().get("detail", "Unknown error")
                            error_message += f" - {error_detail}"
                        except:
                            pass
                            
                        logger.error(error_message)
                        last_error = Exception(error_message)
                        retry_count += 1
                        if retry_count <= max_retries:
                            await asyncio.sleep(2)  # Wait before retrying
                            continue
                        else:
                            raise last_error
                        
                    response_data = response.json()
                
                results = []
                if "results" in response_data:
                    for result in response_data["results"][:results_count]:
                        results.append({
                            "title": result.get("title", "No title"),
                            "snippet": result.get("content", "No content available"),
                            "url": result.get("url", "https://example.com/no-url")
                        })
                    
                    elapsed_time = time.time() - start_time
                    logger.success(f"Found {len(results)} results from Tavily", 
                                  time_taken=f"{elapsed_time:.2f}s")
                else:
                    logger.warning("No results found in Tavily response")
                
                if not results:
                    logger.warning("Using fallback mock data due to empty results")
                    results = create_mock_results("empty API response")
                        
                return ToolResponse(
                    data={
                        "query": query,
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                        "time_taken_seconds": round(time.time() - start_time, 2)
                    },
                    tool_id=self.tool_id,
                    metadata={"result_count": len(results)}
                )
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Error in search attempt {retry_count}: {str(e)}. Retrying...")
                    await asyncio.sleep(2)  # Wait before retrying
                else:
                    logger.error(f"All {max_retries+1} search attempts failed: {str(e)}")
                    return ToolResponse(
                        data={
                            "query": query,
                            "results": create_mock_results(f"API error after {max_retries+1} attempts"),
                            "timestamp": datetime.now().isoformat(),
                            "error": f"Error searching web: {str(e)}",
                            "note": "Using mock data due to search error"
                        },
                        tool_id=self.tool_id,
                        success=False,
                        metadata={"error": str(e), "mock": True}
                    )

tavily_tool_instance = TavilySearchTool()

@function_tool
async def tavily_search_tool(query: str, results_count: int) -> Dict[str, Any]:
    """
    Search the web for relevant information on a topic using Tavily.
    
        query: The search query string
        results_count: Number of results to return (1-10)
    """
    # Handle default inside the function implementation instead of schema
    if results_count is None:
        results_count = 3

    response = await tavily_tool_instance.execute(
        query=query, 
        results_count=results_count
    )
    return response.data
# OpenAI Agents Framework: Full Production Implementation

## 1. Overview

The `aigen` framework provides a complete production implementation of an AI agent workflow system built on top of the OpenAI Agents SDK. This framework enables developers to create, connect, and orchestrate specialized AI agents in flexible workflows with robust error handling and multiple interface options.

```
aigen/
├── core/                        # Core system components
├── agents/                      # Agent definitions and factory
├── tools/                       # Tool implementations
├── workflows/                   # Workflow engine and patterns
├── ui/                          # Interface implementations
├── __init__.py                  # Package initialization
├── main.py                      # Application entry point
└── requirements.txt             # Dependencies
```

## 2. Core Components

### 2.1 Configuration Management

The configuration system supports hierarchical configuration from multiple sources:

```python
from aigen.core.config import ConfigManager

# Load from various sources
config = ConfigManager("config.yaml")  

# Access configuration
api_key = config.get("api_keys.openai")
model = config.get("model_settings.default_model", "gpt-4o")

# Set runtime values
config.set("model_settings.temperature", 0.8)
```

### 2.2 Context System

The Context object manages state across agent executions:

```python
from aigen.core.context import Context

# Create context with initial data
context = Context(initial_data={"query": "What is the weather?"})

# Store and retrieve data
context.store_output("research_agent", research_results)
research_data = context.get_output("research_agent")

# Track execution
context.add_artifact("source_links", links)
context.mark_complete()
```

### 2.3 Logging System

Enhanced logging with structured output:

```python
from aigen.core.logging import get_logger, configure_logging

# Configure logging
configure_logging(level="INFO", log_file="aigen.log")

# Get a logger instance
logger = get_logger("my_component")

# Log with structured data
logger.info("Processing request", {"request_id": "12345"})
logger.success("Operation completed successfully")
```

### 2.4 Error Handling

Structured error system with categorization:

```python
from aigen.core.errors import ValidationError, AgentError, WorkflowError

# Raise specific error types
try:
    # Operation 
except Exception as e:
    raise AgentError("Failed to process text", agent_id="writer", cause=e)

# Handle errors with detailed info
try:
    # Operation
except WorkflowError as e:
    logger.error(f"Workflow error: {e}")
    print(f"Error severity: {e.severity.value}")
    print(f"Error details: {e.details}")
```

## 3. Agent System

### 3.1 Agent Base and Roles

```python
from aigen.agents.base import AgentBase, AgentRole, AgentResponse

# Create a custom agent
class MyAgent(AgentBase):
    def __init__(self, agent_id="my_agent", **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.CUSTOM, **kwargs)
        
    async def execute(self, context, input_text=None):
        # Agent logic
        return AgentResponse(
            content="Generated content",
            agent_id=self.agent_id,
            success=True,
            metadata={"word_count": 150}
        )
```

### 3.2 Factory System

```python
from aigen.agents.factory import create_agent, register_agent_factory

# Register a custom agent type
register_agent_factory(
    "my_agent_type",
    lambda agent_id=None, **kwargs: MyAgent(agent_id=agent_id, **kwargs),
    {"description": "My custom agent type"}
)

# Create an agent instance
agent = create_agent("my_agent_type", instructions="You are a helpful assistant.")
```

### 3.3 Built-in Agents

```python
from aigen.agents import create_agent

# Create standard agents
research_agent = create_agent("research", instructions="Research this topic thoroughly.")
writer_agent = create_agent("writer", instructions="Write an article based on research.")
editor_agent = create_agent("editor", instructions="Polish and improve the text.")
```

### 3.4 Creating and Registering New Agents

To extend the framework with your own custom agent, follow these steps:

#### Step 1: Create a New Agent Class

Create a new file in the `aigen/agents/` directory. For example, `summarizer.py`:

```python
from typing import Optional

from agents import Agent  # Import from OpenAI Agents SDK
from agents.model_settings import ModelSettings  # Import ModelSettings
from ..core.context import Context
from ..core.logging import get_logger
from ..core.errors import AgentError
from .base import AgentBase, AgentRole, AgentResponse

logger = get_logger("summarizer_agent")

class SummarizerAgent(AgentBase):
    """
    Agent that summarizes content into concise form.
    """
    
    def __init__(self, agent_id: str = "summarizer", **kwargs):
        """
        Initialize a summarizer agent.
        
            agent_id: Unique identifier for the agent
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id=agent_id, role=AgentRole.CUSTOM, **kwargs)
        self.system_prompt = kwargs.get("instructions", 
                                         "You are a skilled summarizer. "
                                         "Your task is to condense content into clear, "
                                         "concise summaries while preserving key information.")
    
    async def initialize(self) -> None:
        """Initialize the OpenAI agent."""
        try:
            # Create the agent with the OpenAI Agents SDK
            self.openai_agent = Agent(
                name=self.agent_id,
                description="Agent that creates concise summaries of content",
                instructions=self.system_prompt,
                tools=self.tools,
                # IMPORTANT: Pass the 'model' parameter here, NOT in ModelSettings
                model=self.parameters.get("model", "gpt-4o"),
                # ModelSettings should NOT include the 'model' parameter
                model_settings=ModelSettings(
                    temperature=self.parameters.get("temperature", 0.7)
                )
            )
            logger.info(f"Summarizer agent {self.agent_id} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize summarizer agent {self.agent_id}: {str(e)}")
            raise AgentError(f"Agent initialization failed: {str(e)}", agent_id=self.agent_id)
    
    async def execute(self, context: Context, input_text: Optional[str] = None) -> AgentResponse:
        """
        Execute the agent to summarize content.
        
            context: Execution context
            input_text: Optional direct input text
            
            AgentResponse: The agent's response
        """
        try:
            # Initialize if not already done
            if not self.is_initialized():
                await self.initialize()
            
            # Get input - either direct or from context
            if input_text is None:
                input_text = context.get_latest_output()
                
            if not input_text:
                return AgentResponse.error_response(
                    self.agent_id, 
                    "No input provided for summarization"
                )
            
            # Call the OpenAI agent
            response = await self.openai_agent.run(input_text)
            
            # Store result in context
            result = response.content
            context.store_output(self.agent_id, result)
            
            # Return successful response
            return AgentResponse(
                content=result,
                agent_id=self.agent_id,
                success=True,
                metadata={"source": "summarizer"}
            )
            
        except Exception as e:
            logger.error(f"Error executing summarizer agent {self.agent_id}: {str(e)}")
            error = AgentError(f"Summarizer agent execution failed: {str(e)}", agent_id=self.agent_id)
            context.record_error(str(error))
            return AgentResponse.error_response(self.agent_id, error)

#### Common Agent Implementation Pitfalls to Avoid

When implementing a new agent, be careful to avoid these common issues:

1. **ModelSettings Configuration**: The `model` parameter must be passed directly to the `Agent` constructor, not to the `ModelSettings` constructor:

   ```python
   # CORRECT:
   self.openai_agent = Agent(
       name="My Agent",
       instructions=instruction_fn,
       tools=self.tools,
       model=self.parameters.get("model", "gpt-4o"),  # ✅ model here
       model_settings=ModelSettings(
           temperature=0.7,                           # ✅ only temperature/other settings
           tool_choice="auto"                         # ✅ no model parameter here
       )
   )
   
   # INCORRECT - will cause runtime error:
   self.openai_agent = Agent(
       name="My Agent",
       instructions=instruction_fn,
       tools=self.tools,
       model_settings=ModelSettings(
           model="gpt-4o",                           # ❌ Wrong! model should not be here
           temperature=0.7
       )
   )
   ```

2. **Trace ID Format**: When using the `trace()` function, ensure trace IDs always start with `trace_`:

   ```python
   # CORRECT - use the SDK-provided function for guaranteed valid IDs:
   from agents import gen_trace_id
   trace_id = gen_trace_id()
   
   # CORRECT - trace ID with proper prefix:
   trace_id = f"trace_my_custom_id_{uuid.uuid4().hex[:8]}"
   
   # INCORRECT - will cause API error:
   trace_id = f"my_custom_id_{uuid.uuid4().hex[:8]}"
   trace_id = f"workflow_{workflow_name}_{uuid.uuid4().hex[:8]}"
   trace_id = str(uuid.uuid4())
   ```
   
   Using `gen_trace_id()` from the Agents SDK is strongly recommended as it automatically generates properly formatted trace IDs that are accepted by the OpenAI API. This is much safer than manually formatting trace IDs, which can lead to errors if the format changes in the future.

3. **Agent Initialization**: Always check if the agent is already initialized before initializing:

   ```python
   async def initialize(self) -> None:
       if self.is_initialized():
           return
       
       # Initialization logic
   ```

4. **Context Aware Results**: Make your agent context-aware by retrieving previous outputs:

   ```python
   # Get input from relevant previous agents
   research_output = context.get_output("research")
   if research_output:
       enhanced_input += f"\n\nResearch findings:\n{research_output}"
   ```

#### Step 2: Register the Agent in the Factory System

Option 1: Update the `register_standard_agents()` function in `aigen/agents/factory.py`:

```python
def register_standard_agents():
    """Register standard agent types with the registry."""
    
    # Define factories using lambda functions to ensure they're callable
    factories = [
        # ... existing agents ...
        
        # Add your new agent
        ("summarizer", 
         lambda agent_id=None, **kwargs: (
             __import__("aigen.agents.summarizer", fromlist=["SummarizerAgent"]).SummarizerAgent(
                 agent_id=agent_id or "summarizer", **kwargs
             )
         ),
         "Summarizer agent that creates concise summaries of content"
        )
    ]
    
    # Register each factory
    for agent_type, factory_func, description in factories:
        try:
            register_agent_factory(
                agent_type,
                factory_func,
                {"description": description}
            )
            logger.debug(f"Registered standard agent: {agent_type}")
        except Exception as e:
            logger.warning(f"Error registering agent {agent_type}: {str(e)}")
            continue
```

Option 2: Register your agent during application initialization:

```python
# In your application initialization code
from aigen.agents.factory import register_agent_factory
from aigen.agents.summarizer import SummarizerAgent

# Register the agent
register_agent_factory(
    "summarizer",
    lambda agent_id=None, **kwargs: SummarizerAgent(agent_id=agent_id or "summarizer", **kwargs),
    {"description": "Summarizer agent that creates concise summaries of content"}
)
```

#### Step 3: Update Direct Instantiation Support (Optional)

For improved reliability, also update the direct instantiation logic in the `create_agent` function:

```python
# In aigen/agents/factory.py, in the create_agent function:
try:
    # First try direct instantiation from module as a fallback
    try:
        # ... existing agent types ...
        elif agent_type == "summarizer":
            from .summarizer import SummarizerAgent
            logger.info(f"Creating summarizer agent {agent_id} directly")
            return SummarizerAgent(agent_id=agent_id, **kwargs)
    except ImportError:
        logger.debug(f"Could not create agent {agent_type} directly, trying registry")
```

#### Step 4: Use Your New Agent

Once registered, you can use your agent like any built-in agent:

```python
from aigen.agents import create_agent
from aigen.core.context import Context

# Create your agent
summarizer = create_agent("summarizer", instructions="Summarize in bullet points.")

# Use in workflows
workflow = create_workflow(["research", "writer", "summarizer"])

# Or use directly
context = Context()
response = await summarizer.execute(context, "Long text to summarize...")
summary = response.content
```

#### Step 5: Add to Workflows (Optional)

To include your agent in predefined workflows, update the workflow configurations:

```python
# In aigen/ui/gradio_app.py or similar
def _get_workflow_agents(self, workflow_name: str) -> List[str]:
    """Get the agent names for a workflow."""
    workflow_agents = {
        # ... existing workflows ...
        "summarization": ["research", "summarizer"],
        # Add to existing workflows
        "content_generation": ["research", "strategy", "writer", "editor", "summarizer"]
    }
    
    return workflow_agents.get(workflow_name, [])
```

## 4. Tools Implementation

### 4.1 Tool Base

```python
from aigen.tools.base import ToolBase, ToolType, ToolResponse

# Create a custom tool
class WeatherTool(ToolBase):
    def __init__(self, tool_id="weather_tool", **kwargs):
        super().__init__(tool_id=tool_id, tool_type=ToolType.UTILITIES, **kwargs)
        self.name = kwargs.get("name", "weather")
        self.description = "Get weather information for a location"
    
    async def execute(self, location):
        # Tool logic
        return ToolResponse(
            data={"temperature": 72, "conditions": "sunny"},
            tool_id=self.tool_id,
            success=True
        )
```

### 4.2 Tool Factory

```python
from aigen.tools.factory import create_tool, register_tool_factory

# Register custom tool
register_tool_factory(
    "weather", 
    lambda tool_id=None, **kwargs: WeatherTool(tool_id=tool_id, **kwargs),
    {"description": "Weather information tool"}
)

# Create a tool
weather_tool = create_tool("weather", api_key="abc123")
```

### 4.3 Function Tools

```python
from agents import function_tool

# Define a function tool using the decorator pattern
@function_tool
async def tavily_search_tool(query: str, results_count: int = 3) -> dict:
    """
    Search the web for relevant information on a topic using Tavily.
    
        query: The search query string
        results_count: Number of results to return (1-10)
    """
    # Implementation that uses a tool instance properly
    from .base import ToolResponse
    
    # Create or get a tool instance (singleton pattern)
    tool_instance = get_tool_instance()
    
    # Execute the tool properly with kwargs
    response = await tool_instance.execute(
        query=query, 
        results_count=results_count
    )
    
    # Return only the data
    return response.data
```

### 4.4 Research Tools

```python
from aigen.tools.research import tavily_search_tool, tavily_tool_instance
from ..core.logging import get_logger

# Get a logger instance
logger = get_logger("my_module")

# Using the Tavily tool instance directly (recommended approach)
try:
    # Note the proper keyword arguments pattern for logging extra information
    logger.info(f"Searching for: '{query}'", query_length=len(query))
    
    result = await tavily_tool_instance.execute(
        query="Latest AI research", 
        results_count=5
    )
    
    # Process successful results
    if result.success:
        # Note the correct logging pattern with kwargs, not a dictionary
        logger.success("Search completed successfully", result_count=len(result.data.get("results", [])))
        print(result.data)
    else:
        logger.warning(f"Search failed: {result.metadata.get('error')}")
        
except Exception as e:
    # Always log errors appropriately
    logger.error(f"Error during search: {str(e)}")

# Alternative: Using the function tool pattern via Agents SDK
from agents import function_tool_call
result = await function_tool_call(
    tavily_search_tool,
    {"query": "Latest AI research", "results_count": 5}
)

#### API Key Configuration

The Tavily tool requires an API key, which can be set as an environment variable:

```bash
export TAVILY_API_KEY="your-tavily-key"
```

If no API key is provided, the tool will return mock data for development purposes.

## 5. Workflow Engine

### 5.1 Engine Core

```python
from aigen.workflows.engine import WorkflowEngine
from aigen.core.context import Context

# Create workflow engine
engine = WorkflowEngine(max_retries=3, retry_delay=5)

# Execute a workflow with a context
context = Context()
result = await engine.execute(workflow, context, "Input prompt")

# Get workflow status
status = engine.get_workflow_status(result["workflow_id"])
```

### 5.2 Deterministic Workflow

Sequential execution of agents:

```python
from aigen.workflows import DeterministicWorkflow
from aigen.agents import create_agent

# Create agents
research_agent = create_agent("research")
writer_agent = create_agent("writer")
editor_agent = create_agent("editor")

# Create workflow with fixed sequence
workflow = DeterministicWorkflow(
    agents=[research_agent, writer_agent, editor_agent],
    name="content_generation"
)

# Execute
result = await engine.execute(workflow, context, "Generate content about AI")
final_content = result["result"]
```

### 5.3 Handoff Workflow

Dynamic handoffs between agents:

```python
from aigen.workflows import HandoffWorkflow
from aigen.agents import create_agent

# Create agents
triage_agent = create_agent("orchestrator")
research_agent = create_agent("research")
writer_agent = create_agent("writer")

# Configure handoffs
triage_agent.add_handoff(research_agent.agent_id)
research_agent.add_handoff(writer_agent.agent_id)

# Create handoff workflow
workflow = HandoffWorkflow(
    agents=[triage_agent, research_agent, writer_agent],
    name="dynamic_content"
)

# Execute - agents will determine the flow dynamically
result = await engine.execute(workflow, context, "Research and write about quantum computing")
```

### 5.4 Workflow Factory

```python
from aigen.workflows.factory import create_workflow, register_workflow_factory

# Register custom workflow
register_workflow_factory(
    "my_workflow",
    lambda **kwargs: DeterministicWorkflow(agents=[...], **kwargs),
    {"description": "Custom workflow implementation"}
)

# Create a workflow by name
workflow = create_workflow("content_generation")

# Create from agent list
workflow = create_workflow(["research", "writer", "editor"])

# Create from config file
workflow = create_workflow("workflows/my_workflow.yaml")
```

## 6. User Interfaces

### 6.1 Command Line Interface

```bash
# Run a workflow
aigen execute --workflow content_generation --input "Generate an article about AI" --output article.md

# Interactive mode
aigen interactive

# List available components
aigen list --workflows --agents --tools

# Launch web interface
aigen web --port 8080 --host localhost
```

### 6.2 Web Interface (Gradio)

```python
from aigen.ui.gradio_app import launch_ui

# Launch the UI
launch_ui(share=True, port=8080)
```

#### User Interface Organization

The Gradio web interface is organized into multiple tabs for a better user experience:

1. **Use Predefined Workflow**: Run standard workflows like content generation or research
   ```python
   # In the Gradio UI code
   with gr.TabItem("Use Predefined Workflow"):
       # Input section
       predefined_input_text = gr.Textbox(...)
       
       # Workflow selection dropdown - only shows concrete workflows
       workflow_name = gr.Dropdown(
           choices=self._get_workflow_choices(),
           value="content_generation",
           label="Workflow Type"
       )
       
       # Shows which agents are in the selected workflow
       workflow_agents_info = gr.Markdown(...)
   ```

2. **Create Custom Workflow**: Build workflows by selecting agent combinations
   ```python
   # In the Gradio UI code
   with gr.TabItem("Create Custom Workflow"):
       # Custom workflow builder
       custom_input_text = gr.Textbox(...)
       
       # Multi-select dropdown for agents
       custom_agents = gr.Dropdown(
           choices=self._get_agent_choices(),
           multiselect=True,
           value=["research", "writer"]
       )
   ```

3. **Workflow Management**: View available workflows and agents
   ```python
   # In the Gradio UI code
   with gr.TabItem("Workflow Management"):
       # Lists all available workflows with descriptions
       workflows_table = gr.DataFrame(...)
       
       # Lists all available agents with descriptions
       agents_table = gr.DataFrame(...)
   ```

The interface automatically filters out implementation details like base workflow types ("handoff" and "deterministic"), showing only concrete workflow instances to users.

#### Dynamic Workflow Information

The interface dynamically shows which agents are part of each workflow:

```python
# Update workflow agent info when workflow changes
workflow_name.change(
    fn=lambda wf: f"Workflow contains: {', '.join(self._get_workflow_agents(wf))}",
    inputs=[workflow_name],
    outputs=[workflow_agents_info]
)
```

This helps users understand what each workflow does before executing it.

#### Custom Workflow Builder

The custom workflow tab allows users to build their own workflows by selecting agents to execute in sequence:

```python
# Execute custom workflow
execute_custom_btn.click(
    fn=self.run_workflow,
    inputs=[gr.State("custom"), custom_agents, custom_input_text, max_turns_custom],
    outputs=[custom_output_text]
)
```

The order of selected agents matters, as they will be executed in sequence with the output of each agent being passed to the next one in the chain.

## 7. Example: Complete Content Generation System

```python
import asyncio
from aigen.core.context import Context
from aigen.workflows.factory import create_workflow
from aigen.workflows.engine import WorkflowEngine

async def generate_content(topic: str, output_file: str) -> None:
    # Create context
    context = Context()
    
    # Create workflow for content generation
    workflow = create_workflow("content_generation")
    
    # Create engine
    engine = WorkflowEngine()
    
    # Execute workflow
    result = await engine.execute(workflow, context, topic)
    
    # Save output
    if result["status"] == "completed":
        with open(output_file, "w") as f:
            f.write(result["result"])
        print(f"Content generated and saved to {output_file}")
    else:
        print(f"Workflow failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(generate_content(
        "The future of renewable energy",
        "energy_article.md"
    ))
```

## 8. Advanced Features

### 8.1 Hooks for Execution Monitoring

```python
from aigen.core.hooks import ContentWorkflowHooks

# Create custom hooks
hooks = ContentWorkflowHooks(debug_mode=True)

# Use hooks with workflow execution
result = await engine.execute(workflow, context, input_text, hooks=hooks)
```

### 8.2 Registry System

```python
from aigen.core.registry import Registry

# Create a registry for custom components
my_registry = Registry(MyComponentType)

# Register a component
my_registry.register("component_name", component_instance, metadata)

# Get a component
component = my_registry.get("component_name")
```

## 9. Installation and Setup

```bash
# Install from PyPI
pip install aigen

# Install from source
git clone https://github.com/yourusername/aigen.git
cd aigen
pip install -e .

# Set environment variables for API keys
export OPENAI_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-key"  # Required for Tavily web search tool
```

## 10. Configuration Example

```yaml
# config.yaml
api_keys:
  openai: "your-openai-key"
  tavily: "your-tavily-key"  # Required for research capabilities

model_settings:
  default_model: "gpt-4o"
  temperature: 0.7
  max_tokens: 4096

logging:
  level: "INFO"
  
research_tools:
  tavily:
    enabled: true
    max_results: 5
    max_retries: 3
```

This framework provides a comprehensive foundation for developing production-grade AI agent applications with clean separation of concerns, robust error handling, and flexible workflow management. The modular architecture allows for easy extension and customization to fit different business needs.

## 11. Troubleshooting

### 11.1 Factory Registration Issues

If you encounter errors related to factory registration, such as `'ResearchAgent' object is not callable` or `Factory for content_generation is not callable`, this indicates an issue with how agents or workflows are registered in the system's registry.

#### Common Factory Errors

```
ERROR: Error creating agent research of type research: 'ResearchAgent' object is not callable
ERROR: Registered factory for content_generation is not callable
```

#### Solution 1: Fix by Adding Registry Cleanup Function

Add a registry cleanup function to your initialization code:

```python
# In your __init__.py or main initialization file
def _ensure_callable_factories():
    """Ensure all factories in the registry are callable functions."""
    from .agents.factory import agent_registry
    from .workflows.factory import workflow_registry
    from .core.logging import get_logger
    
    logger = get_logger("registry_fix")
    
    # Fix agent factories
    for name in list(agent_registry.list()):
        try:
            factory = agent_registry._factories.get(name)
            if factory and not callable(factory):
                logger.warning(f"Fixing non-callable agent factory: {name}")
                
                # Create a proper factory function based on the agent type
                if name == "research":
                    def research_factory(agent_id=None, **kwargs):
                        from .agents.research import ResearchAgent
                        return ResearchAgent(agent_id=agent_id or "research", **kwargs)
                    agent_registry._factories[name] = research_factory
                
                # ... Fix other agent types similarly
        
        except Exception as e:
            logger.error(f"Error fixing agent factory {name}: {str(e)}")
    
    # Fix workflow factories in a similar way
    # ...
```

#### Solution 2: Use Lambda Functions for Registration

When registering factories, use lambda functions to ensure they're callable:

```python
# Instead of directly registering the class or instance:
register_agent_factory("research", ResearchAgent)  # WRONG

# Use a lambda function:
register_agent_factory(
    "research", 
    lambda agent_id=None, **kwargs: ResearchAgent(agent_id=agent_id or "research", **kwargs)
)
```

#### Solution 3: Implement Fallback in Factory Methods

Add robust error handling and fallbacks in your creation methods:

```python
def create_agent(agent_type, agent_id=None, **kwargs):
    agent_id = agent_id or agent_type
    
    try:
        # First try direct instantiation as a fallback
        try:
            if agent_type == "research":
                from .research import ResearchAgent
                return ResearchAgent(agent_id=agent_id, **kwargs)
                
            # ... Handle other agent types
                
        except ImportError:
            pass
            
        # Then try the registry
        if agent_type in agent_registry.list():
            factory = agent_registry.get(agent_type)
            
            if callable(factory):
                try:
                    return factory(agent_id=agent_id, **kwargs)
                except TypeError:
                    pass
                    
        # Last resort: dynamic import
        # ...
            
    except Exception as e:
        logger.error(f"Error creating agent {agent_type}: {str(e)}")
        raise
```

These solutions ensure that your factory system will work reliably even if there are registration issues.

## 12. Implementing Workflow Tracing

Proper workflow tracing enables you to monitor and debug the execution flow of agents within your workflows by integrating with OpenAI's tracing system.

### 12.1 Deterministic Workflow Tracing

In `aigen/workflows/deterministic.py`, wrap the `execute_deterministic` method in a trace:

```python
from agents import trace, gen_trace_id

async def execute_deterministic(self, context: Context, input_text: Optional[str] = None) -> str:
    """Execute agents in a fixed sequence with explicit data passing."""
    start_time = time.time()
    
    if not self.agents:
        raise WorkflowError("No agents in workflow", workflow_id=self.name)
    
    logger.info(f"Starting deterministic workflow with {len(self.agents)} agents")
    
    # Generate a unique trace ID for this workflow execution using the SDK
    trace_id = gen_trace_id()
    
    # Wrap the entire agent sequence in a single trace for proper context flow
    with trace(f"Workflow: {self.name}", trace_id=trace_id):
        # Initialize current_input with the provided input_text
        current_input = input_text
        
        # Process agents in sequence with step-by-step data passing
        for i, agent in enumerate(self.agents):
            agent_id = agent.agent_id
            logger.info(f"Running agent {i+1}/{len(self.agents)}: {agent_id}")
            
            # Execute the agent with retry logic
            response = await self.engine.execute_agent(agent, context, current_input)
            
            if not response.success:
                # Handle agent failure
                if i > 0:
                    logger.warning(f"Agent {agent_id} failed but we have partial results")
                    return self._get_best_content(context)
                else:
                    error = f"Initial agent {agent_id} failed: {response.content}"
                    logger.error(error)
                    raise WorkflowError(error, workflow_id=self.name)
            
            # Update current_input for the next agent
            if i < len(self.agents) - 1:
                next_agent_id = self.agents[i+1].agent_id
                current_input = f"""
                Previous agent '{agent_id}' output:
                {response.content}
                
                Original request: {input_text}
                
                Now you ({next_agent_id}) should continue processing based on this.
                """
        
        # Get the final output
        final_agent_id = self.agents[-1].agent_id
        final_output = context.get_output(final_agent_id)
        
        if not final_output:
            logger.warning("No output from final agent, attempting to get best available content")
            return self._get_best_content(context)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Deterministic workflow completed in {elapsed_time:.2f}s")
        
        return final_output
```

### 12.2 Handoff Workflow Tracing

Similarly update the `execute_handoff` method in `aigen/workflows/handoff.py`:

```python
from agents import trace, gen_trace_id

async def execute_handoff(self, context: Context, input_text: Optional[str] = None) -> str:
    """Execute the workflow using agent handoffs."""
    start_time = time.time()
    
    if not self.agents:
        raise WorkflowError("No agents in workflow", workflow_id=self.name)
    
    logger.info(f"Starting handoff workflow with {len(self.agents)} agents")
    
    # Generate a unique trace ID using the SDK
    trace_id = gen_trace_id()
    
    # Wrap the entire execution in a trace
    with trace(f"Handoff: {self.name}", trace_id=trace_id):
        # Initialize hooks
        from ..core.hooks import ContentWorkflowHooks
        hooks = ContentWorkflowHooks()
        
        # Start with the first agent
        entry_agent = self.agents[0]
        
        try:
            # Execute the workflow through OpenAI Runner
            result = await Runner.run(
                entry_agent.openai_agent,
                input=input_text,
                context=context,
                max_turns=80,
                hooks=hooks
            )
```

### 12.4 Helper Function for Workflow Execution

Create a helper function that properly handles tracing for all workflows:

```python
from agents import trace, gen_trace_id

async def run_workflow(workflow_name: str, input_text: str, context=None) -> Dict[str, Any]:
    """Run a workflow with proper tracing and error handling."""
    try:
        # Create a workflow instance
        workflow = create_workflow(workflow_name)
        
        # Create a context if one isn't provided
        if context is None:
            context = Context()
        
        # Create a workflow engine
        engine = WorkflowEngine()
        
        # Generate a trace ID for tracking using the SDK
        trace_id = gen_trace_id()
        
        # Execute the workflow with a trace
        with trace(f"Workflow: {workflow_name}", trace_id=trace_id):
            result = await engine.execute(workflow, context, input_text)
            
            # Log the trace URL for debugging
            trace_url = f"https://platform.openai.com/traces/{trace_id}"
            logger.info(f"Workflow trace available at: {trace_url}")
            
            return result
            
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_name}: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "result": None
        }
```

### 12.5 Integrating Tracing with Gradio

Update your Gradio app's `run_workflow` method:

```python
from agents import trace, gen_trace_id

async def run_workflow(self, 
                      workflow_name: str,
                      custom_agents: List[str],
                      input_text: str,
                      max_turns: int = 80,
                      progress=gr.Progress()) -> str:
    """Run a workflow with the provided parameters."""
    progress(0, "Initializing workflow...")
    
    try:
        # Generate a valid trace ID using the SDK
        trace_id = gen_trace_id()
        
        # Set up the workflow spec based on the selection
        if workflow_name == "custom":
            workflow_spec = custom_agents
        else:
            workflow_spec = workflow_name
        
        # Create a context
        context = Context()
        
        # Create the workflow
        workflow = create_workflow(workflow_spec)
        
        # Create workflow engine
        from ..workflows.engine import WorkflowEngine
        engine = WorkflowEngine()
        
        progress(0.1, "Executing workflow...")
        
        # Wrap execution in a trace context manager
        with trace(f"Workflow: {workflow_name}", trace_id=trace_id):
            # Execute the workflow
            result = await engine.execute(workflow, context, input_text)
            
            progress(0.9, "Completing workflow...")
            
            # Return the result
            if result["status"] == "completed":
                return result["result"]
            else:
                return f"Workflow failed: {result.get('error', 'Unknown error')}\n\n{result.get('result', '')}"
                
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        progress(1.0, "Workflow complete")
```

With these implementations, each workflow execution will be wrapped in a single `trace()` context, ensuring all agent executions are properly grouped and context flows correctly between agents. This creates a unified view in the OpenAI dashboard, making debugging and monitoring much easier.

## 13. Troubleshooting Common Implementation Issues

### 13.1 Agent Implementation with OpenAI Agents SDK

When implementing agents that use the OpenAI Agents SDK, follow these best practices to avoid common errors:

#### Use Runner.run() Instead of agent.run()

```python
# INCORRECT - Will cause error: 'Agent' object has no attribute 'run'
response = await self.openai_agent.run(input_text)

# CORRECT - Always use Runner.run() to execute an agent
from agents import Runner
response = await Runner.run(self.openai_agent, input_text)
```

#### Access RunResult Properties Correctly

When processing the results returned by `Runner.run()`, use the correct property names:

```python
# INCORRECT - Will cause error: 'RunResult' object has no attribute 'content'
result = response.content

# CORRECT - Use final_output for the main text response
result = response.final_output
```

#### Example of Correctly Implemented Agent.execute() Method

```python
async def execute(self, context: Context, input_text: Optional[str] = None) -> AgentResponse:
    """Execute the agent with proper SDK patterns."""
    try:
        # Initialize if not already done
        if not hasattr(self, 'openai_agent') or self.openai_agent is None:
            await self.initialize()
        
        # Get input - either direct or from context
        if input_text is None:
            input_text = context.get_latest_output()
            
        if not input_text:
            return AgentResponse.error_response(
                self.agent_id, 
                "No input provided"
            )
        
        # CORRECT: Use Runner.run() from the SDK
        from agents import Runner
        response = await Runner.run(
            self.openai_agent,
            input=input_text,
            max_turns=self.parameters.get("max_turns", 10)
        )
        
        # CORRECT: Extract result using final_output
        if hasattr(response, 'final_output') and response.final_output:
            output = str(response.final_output)
        else:
            output = "No output generated"
        
        # Store in context
        context.store_output(self.agent_id, output)
        
        # CORRECT: Use set_metadata not add_metadata
        context.set_metadata(self.agent_id, {"word_count": len(output.split())})
        
        return AgentResponse(
            content=output,
            agent_id=self.agent_id,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        return AgentResponse.error_response(self.agent_id, str(e))
```

### 13.2 Workflow Factory Registration Issues

The system can encounter factory registration issues where workflow factories are not callable. To prevent or fix these problems:

#### Direct Workflow Creation in create_workflow()

Add direct creation paths for commonly used workflows:

```python
def create_workflow(workflow_spec: str, **kwargs):
    """Create a workflow with direct fallbacks for reliability."""
    
    # Direct fallbacks for common workflows
    if workflow_spec == "research_only":
        logger.info("Creating research_only workflow directly")
        from .deterministic import DeterministicWorkflow
        from ..agents.factory import create_agent
        return DeterministicWorkflow(
            agents=[create_agent("research")],
            name="research_only"
        )
    elif workflow_spec == "content_generation":
        logger.info("Creating content_generation workflow directly")
        from .deterministic import DeterministicWorkflow
        from ..agents.factory import create_agent
        return DeterministicWorkflow(
            agents=[
                create_agent("research"),
                create_agent("strategy"),
                create_agent("writer"),
                create_agent("editor")
            ],
            name="content_generation"
        )
    
    # Continue with standard workflow creation logic...
```

#### Always Check Registry First in workflow_by_type

Ensure `create_workflow_by_type` always checks the registry before trying other methods:

```python
def create_workflow_by_type(workflow_type: str, **kwargs):
    """Create a workflow by type with proper registry checking."""
    
    # Always check registry first
    if workflow_type in workflow_registry.list():
        factory = workflow_registry.get(workflow_type)
        if callable(factory):
            try:
                return factory(**kwargs)
            except Exception as e:
                logger.warning(f"Factory call failed: {str(e)}")
                # Fall back to other methods
    
    # Standard types fallback
    if workflow_type == 'deterministic':
        from .deterministic import DeterministicWorkflow
        return DeterministicWorkflow(**kwargs)
    elif workflow_type == 'handoff':
        from .handoff import HandoffWorkflow
        return HandoffWorkflow(**kwargs)
    
    # Dynamic import as last resort
    try:
        module = importlib.import_module(f"aigen.workflows.{workflow_type.lower()}")
        class_name = "".join(word.capitalize() for word in workflow_type.split("_")) + "Workflow"
        workflow_class = getattr(module, class_name)
        return workflow_class(**kwargs)
    except (ImportError, AttributeError):
        raise ValueError(f"Unsupported workflow type: {workflow_type}")
```

#### Workflow Registration Precedence Issue

A critical issue can occur when workflows are registered twice - once properly with lambda functions and once incorrectly with workflow instances. The incorrect registration can overwrite the proper one, leading to "not callable" errors:

```python
# CORRECT - Will work properly
register_workflow_factory(
    "research_only",
    lambda **kwargs: DeterministicWorkflow(agents=[create_agent("research")], **kwargs),
    {"description": "Research workflow"}
)

# INCORRECT - Will break the workflow if registered later
# This overwrites the previous registration!
register_workflow_factory(
    "research_only",
    DeterministicWorkflow(agents=[create_agent("research")]),  # Not a factory function!
    {"description": "Research workflow"}
)
```

To fix this issue:

1. Always use lambda functions when registering workflows
2. Add direct fallbacks in `create_workflow()` for critical workflows
3. Check for duplicate registrations in startup logs
4. When modifying workflow configurations, make sure to preserve the lambda pattern

#### Ensuring UI Components Don't Override Factory Registration

If you're building UI components that need to reference workflows, make sure they don't re-register workflows with instances:

```python
# INCORRECT - UI component registering workflow instances
def init_ui_workflows(self):
    # This causes issues by registering instances instead of factories
    register_workflow_factory(
        "research_only", 
        DeterministicWorkflow(agents=[create_agent("research")])  # ❌ Wrong!
    )

# CORRECT - UI component referencing workflows without re-registering
def get_workflow_config(self, name):
    # This just references workflow configurations without changing registration
    workflow_configs = {
        "research_only": {
            "agents": ["research"],
            "description": "Research-only workflow"
        }
    }
    return workflow_configs.get(name, {})
```

By following these patterns, you'll avoid the "Factory for X is not callable" errors that can break workflow execution.

#### Dynamic Workflow Resolution in UI Components

UI components like the Gradio interface don't need to hardcode all available workflows. The system now includes a fallback mechanism to automatically retrieve workflow agent information:

```python
def _get_workflow_agents(self, workflow_name: str) -> List[str]:
    """Get the agent names for a workflow."""
    # Hardcoded workflow agent configurations for common workflows
    workflow_agents = {
        "content_generation": ["research", "strategy", "writer", "editor"],
        "research_only": ["research"],
    }
    
    # Fall back to checking the registry if not found in hardcoded list
    if workflow_name not in workflow_agents and workflow_name in workflow_registry.list():
        try:
            # Try to create a temporary workflow to get its agents
            factory = workflow_registry.get(workflow_name)
            if callable(factory):
                temp_workflow = factory()
                if hasattr(temp_workflow, 'agents'):
                    return [agent.agent_id for agent in temp_workflow.agents]
        except Exception:
            pass
            
    return workflow_agents.get(workflow_name, [])
```

This allows any registered workflow to automatically appear in the UI without manual configuration.

#### Workflow Registration Architecture

The framework includes a two-level workflow registration architecture:

##### Base Workflow Types vs Concrete Workflow Instances

There are two kinds of registered workflows:

1. **Base Workflow Types** (like "deterministic" and "handoff"):
   - These are registered as workflow types in the factory system
   - They provide the execution strategy (sequential, handoff-based, etc.)
   - They're registered so they can be referenced by name in configuration files
   - Example: `create_workflow("deterministic")`

2. **Concrete Workflow Instances** (like "content_generation" and "research_only"):
   - These are pre-configured workflows with specific agent combinations
   - They're created using one of the base workflow types
   - They're registered for user convenience
   - Example: `create_workflow("content_generation")`

```python
# Registering a base workflow type
register_workflow_factory(
    "deterministic",  # Base type
    lambda agents=None, **kwargs: DeterministicWorkflow(agents=agents or [], **kwargs),
    {"description": "Sequential execution workflow"}
)

# Registering a concrete workflow instance
register_workflow_factory(
    "content_generation",  # Concrete instance
    lambda **kwargs: DeterministicWorkflow(
        agents=[
            create_agent("research"),
            create_agent("writer"),
            create_agent("editor")
        ],
        **kwargs
    ),
    {"description": "Complete content creation pipeline"}
)
```

##### Why Utilities Like Factory and Engine Aren't Registered

Components like `factory.py` and `engine.py` aren't registered because they're:

1. **Singletons**: Only one instance exists in the system
2. **Infrastructure**: They provide functionality rather than being configurable components
3. **Not Referenced by Name**: They're imported directly, not accessed through a registry

The registry pattern is primarily used for components that:
1. Have multiple variants (like different workflow types)
2. Need to be user-selectable by name
3. Can be dynamically extended with plugins or custom implementations

This separation creates a clear distinction between the framework's infrastructure and its configurable components.

### 13.3 Single-Agent Workflow Issues

When creating single-agent workflows like "research_only", ensure the agent's output is properly formatted for direct display:

```python
# In research agent's execute method:
if not result.startswith("# Research Results"):
    # Add proper formatting for single-agent display
    formatted_result = f"""# Research Results on: {input_text}

{result}

---
*Research conducted by {self.agent_id} agent*
"""
    result = formatted_result
```

In DeterministicWorkflow, add special handling for single-agent workflows:

```python
# Single-agent workflow special handling
if len(self.agents) == 1:
    logger.info(f"Single-agent workflow completed with {agent_id}")
    if not context.get_output(agent_id):
        context.store_output(agent_id, response.content)
```

### 13.4 Understanding Response Objects at Different Levels

A common source of errors is confusing the different response objects returned at different levels of the system:

#### Response Object Hierarchy

1. **OpenAI Agents SDK Level**:
   - `Runner.run()` returns a `RunResult` object
   - Access text output via `.final_output`
   - Example: `result = response.final_output`

2. **Agent Level (Your Framework)**:
   - `agent.execute()` returns an `AgentResponse` object
   - Access text output via `.content`
   - Example: `output = response.content`

3. **Workflow Engine Level**:
   - `engine.execute_agent()` returns an `AgentResponse` 
   - `engine.execute()` returns a dictionary with result info
   - Example: `if result["status"] == "completed": return result["result"]`

#### Common Mistake: Mixing Response Types

```python
# INCORRECT - Mixing response types
# In a method that receives an AgentResponse but treats it like a RunResult:
def process_agent_output(response):
    text = response.final_output  # ❌ Wrong! AgentResponse has .content, not .final_output
    return text
```

```python
# CORRECT - Handle each response type appropriately
def process_agent_output(response):
    # Check the response type
    if hasattr(response, 'final_output'):  # It's a RunResult from OpenAI SDK
        return response.final_output
    elif hasattr(response, 'content'):     # It's an AgentResponse from our framework
        return response.content
    elif isinstance(response, dict) and "result" in response:  # It's from engine.execute()
        return response["result"]
    else:
        return str(response)  # Fallback
```

These improvements ensure that all workflow types (including single-agent workflows) work correctly in both programmatic and web UI contexts.

### 13.5 Correct Logging Patterns

The logging system uses keyword arguments (`**kwargs`) for structured logging. A common mistake is passing dictionaries as positional arguments instead of using keyword arguments:

#### Incorrect vs Correct Logging Patterns

```python
# INCORRECT - Passing a dictionary as a positional argument
# Will cause: "Logger.info() takes 2 positional arguments but 3 were given"
logger.info(f"Searching for: '{query}'", {"results_count": results_count})  # ❌ Wrong!

# INCORRECT - Same issue with other logging methods
logger.success(f"Found {len(results)} results", {"time_taken": f"{elapsed_time:.2f}s"})  # ❌ Wrong!
```

```python
# CORRECT - Use keyword arguments instead of a dictionary
logger.info(f"Searching for: '{query}'", results_count=results_count)  # ✅ Correct

# CORRECT - Same pattern for all logging methods
logger.success(f"Found {len(results)} results", time_taken=f"{elapsed_time:.2f}s")
logger.warning(f"Process slow", threshold=threshold, actual_time=elapsed_time)
logger.error(f"Operation failed", error_code=code, error_message=str(e))
```

#### Complete Example with Proper Logging

```python
async def execute_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Execute a search with proper logging."""
    start_time = time.time()
    
    try:
        # Log the search parameters correctly
        logger.info(f"Starting search for '{query}'", 
                   max_results=max_results, 
                   timestamp=datetime.now().isoformat())
        
        # Perform search operation...
        results = await search_api.execute(query, max_results)
        
        # Log success with keyword arguments
        elapsed_time = time.time() - start_time
        logger.success(f"Search completed successfully", 
                      results_count=len(results),
                      time_taken=f"{elapsed_time:.2f}s")
        
        return {
            "status": "success",
            "results": results,
            "meta": {"query": query, "time": elapsed_time}
        }
        
    except Exception as e:
        # Log errors with keyword arguments
        logger.error(f"Search failed", 
                    error_type=type(e).__name__,
                    error_message=str(e),
                    query=query)
        
        return {
            "status": "error",
            "error": str(e),
            "meta": {"query": query}
        }
```

These improvements ensure that all workflow types (including single-agent workflows) work correctly in both programmatic and web UI contexts.

### 13.6 UI Component Type Mismatches with Gradio

When working with the Gradio-based UI, you may encounter errors related to component type mismatches. These typically occur when a function returns a data type incompatible with the UI component it's connected to.

#### Common Error: Dictionaries Passed to Markdown Components

A frequent error occurs when a dictionary is passed to a Gradio Markdown component, which expects a string:

```
Error: Cannot convert dict to str implicitly
```

This often happens after refactoring functions to return more structured data (dictionaries) while the UI components remain configured for string outputs.

#### Solution 1: Use Appropriate Component Types

Match your component types to your function return types:

```python
# INCORRECT - Will cause error if function returns a dictionary
result_display = gr.Markdown()

# CORRECT - Use JSON component for dictionary return types
result_display = gr.JSON()
```

#### Solution 2: Use Adapter Functions

Add adapter functions that convert between types:

```python
def dict_to_string_adapter(func):
    """Adapter to convert dictionary outputs to strings for Markdown components."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        if isinstance(result, dict):
            # Format dictionary nicely for display
            if "status" in result and "message" in result:
                return result["message"]
            return str(result)
        return result
    return wrapper

# Then use it in your UI setup:
btn.click(
    dict_to_string_adapter(my_func),
    inputs=[...],
    outputs=[result_markdown]
)
```

#### Solution 3: Move Shared Adapter Functions to Utilities

For frequently used adapter functions, place them in a shared utilities module:

```python
# In aigen/ui/utils.py
import inspect
from functools import wraps
from typing import Any, Callable, Dict

def dict_to_string_adapter(func: Callable) -> Callable:
    """Adapter to convert dictionary outputs to strings for Markdown components."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if inspect.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
            
        if isinstance(result, dict):
            if "status" in result and "message" in result:
                return result["message"]
            return str(result)
        return result
    return wrapper

# Then import it where needed:
from aigen.ui.utils import dict_to_string_adapter
```

#### Best Practice: Consistency in Return Types

To prevent these issues:
- Design UI functions to have consistent return types
- Document expected return types for all UI-connected functions
- Use typing annotations to make return types explicit
- When refactoring, review all UI component connections
- Consider using more flexible components like HTML or JSON instead of Markdown
- Test UI changes thoroughly before deployment

## 14. Advanced Component Creation and Registration

### 14.1 Factory System Architecture

The framework uses a registry-based factory system to manage component creation. Understanding this architecture is key to extending the system effectively:

```
Registry → Factory Function → Component Instance
```

Each registry maps string identifiers to factory functions:

```python
# Simplified internal registry structure
{
    "research": lambda agent_id=None, **kwargs: ResearchAgent(agent_id, **kwargs),
    "writer": lambda agent_id=None, **kwargs: WriterAgent(agent_id, **kwargs),
    # ...other components
}
```

When calling `create_agent("research")`, the system:
1. Looks up "research" in the registry
2. Retrieves the associated factory function
3. Calls the factory with any provided arguments
4. Returns the resulting instance

### 14.2 Agent Creation Best Practices

When creating new agents, follow these patterns to ensure they integrate properly with the framework:

#### Provide Multiple Registration Methods

Register your agent in multiple ways to ensure reliability:

```python
# Method 1: Factory registration (primary approach)
register_agent_factory(
    "my_agent",
    lambda agent_id=None, **kwargs: MyAgent(agent_id=agent_id or "my_agent", **kwargs),
    {"description": "My custom agent"}
)

# Method 2: Direct import fallback in create_agent
# In aigen/agents/factory.py:
def create_agent(agent_type, agent_id=None, **kwargs):
    # ...existing code...
    
    # Add direct instantiation for your agent
    if agent_type == "my_agent":
        from .my_agent import MyAgent
        return MyAgent(agent_id=agent_id or "my_agent", **kwargs)
    
    # ...rest of the function...

# Method 3: Dynamic import fallback
# Ensure your agent follows naming conventions:
# - File: aigen/agents/my_agent.py
# - Class: MyAgent
```

#### Implement Multiple Initialization Patterns

Make your agent resilient to different initialization approaches:

```python
class MyAgent(AgentBase):
    def __init__(self, agent_id="my_agent", **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.CUSTOM, **kwargs)
        
        # Support initialization via both 'instructions' and 'system_prompt'
        self.system_prompt = kwargs.get("instructions") or kwargs.get("system_prompt") or DEFAULT_PROMPT
        
        # Accept model configuration in multiple formats
        self.model = kwargs.get("model") or kwargs.get("model_name") or "gpt-4o"
        self.temperature = kwargs.get("temperature", 0.7)
        
        # Initialize tools with fallbacks
        self.initialize_tools(**kwargs)
    
    def initialize_tools(self, **kwargs):
        """Initialize tools with fallback options."""
        tool_names = kwargs.get("tools", [])
        
        # Try multiple approaches to get tools
        if tool_names:
            # Approach 1: Direct tool names
            for tool_name in tool_names:
                try:
                    tool = create_tool(tool_name)
                    self.add_tool(tool)
                except Exception as e:
                    logger.warning(f"Failed to add tool {tool_name}: {str(e)}")
        
        # Approach 2: Add default tools based on role
        if self.role == AgentRole.RESEARCH:
            try:
                from ..tools.research import tavily_search_tool
                self.add_tool(tavily_search_tool)
            except Exception as e:
                logger.warning(f"Failed to add default research tools: {str(e)}")
```

### 14.3 Workflow Creation Patterns

Use these patterns to create reliable, maintainable workflows:

#### Multi-Registration Approach

Register workflows in multiple ways to ensure they can be created reliably:

```python
# 1. Register with a factory function
register_workflow_factory(
    "my_workflow",
    lambda **kwargs: DeterministicWorkflow(
        agents=[
            create_agent("research"),
            create_agent("my_agent"),
            create_agent("editor")
        ],
        name="my_workflow",
        **kwargs
    ),
    {"description": "My custom workflow"}
)

# 2. Add a direct creation path in create_workflow
# In aigen/workflows/factory.py:
def create_workflow(workflow_spec, **kwargs):
    # Add direct paths for common workflows
    if workflow_spec == "my_workflow":
        from .deterministic import DeterministicWorkflow
        from ..agents.factory import create_agent
        return DeterministicWorkflow(
            agents=[
                create_agent("research"),
                create_agent("my_agent"),
                create_agent("editor")
            ],
            name="my_workflow"
        )
    
    # ...existing code...
```

#### Composable Workflow Creation

Build workflows from reusable components:

```python
def create_research_agents(domain="general", **kwargs):
    """Create a list of research-focused agents."""
    research_agents = []
    
    # Core research agent
    research_agent = create_agent("research", instructions=f"Research {domain} topics")
    research_agents.append(research_agent)
    
    # Domain-specific research if needed
    if domain == "technical":
        tech_agent = create_agent("research", 
                               agent_id="technical_research",
                               instructions="Focus on technical details")
        research_agents.append(tech_agent)
    
    return research_agents

def create_content_agents(**kwargs):
    """Create a list of content creation agents."""
    return [
        create_agent("strategy"),
        create_agent("writer"),
        create_agent("editor")
    ]

# Create a workflow from composable agent groups
def create_custom_workflow(domain="general", **kwargs):
    """Create a custom workflow with composable agent groups."""
    agents = create_research_agents(domain) + create_content_agents()
    
    return DeterministicWorkflow(
        agents=agents,
        name=f"{domain}_content_workflow"
    )
```

### 14.4 Dynamic Registration of Components

For more flexible applications, dynamically register components based on configuration or discovery:

```python
def register_components_from_directory(directory_path: str) -> None:
    """Dynamically register all components from a directory."""
    import os
    import importlib
    import inspect
    
    # Get all Python files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            # Import the module
            module_name = filename[:-3]  # Remove .py extension
            module_path = f"your_package.{os.path.basename(directory_path)}.{module_name}"
            
            try:
                module = importlib.import_module(module_path)
                
                # Find all classes that inherit from AgentBase
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, AgentBase) and obj != AgentBase:
                        # Register the agent
                        agent_type = module_name.lower()
                        logger.info(f"Dynamically registering agent: {agent_type}")
                        
                        register_agent_factory(
                            agent_type,
                            lambda agent_id=None, cls=obj, **kwargs: cls(
                                agent_id=agent_id or agent_type, **kwargs
                            ),
                            {"description": obj.__doc__ or f"Dynamically registered {agent_type} agent"}
                        )
            except Exception as e:
                logger.error(f"Error registering components from {module_path}: {str(e)}")
```

### 14.5 Factory System Debugging

When troubleshooting factory and registry issues, use these diagnostic techniques:

```python
def diagnose_registry(registry_name="all"):
    """Print diagnostic information about registries."""
    from aigen.agents.factory import agent_registry
    from aigen.workflows.factory import workflow_registry
    from aigen.tools.factory import tool_registry
    
    registries = {
        "agent": agent_registry,
        "workflow": workflow_registry,
        "tool": tool_registry
    }
    
    if registry_name != "all" and registry_name not in registries:
        print(f"Unknown registry: {registry_name}")
        return
    
    targets = [registries[registry_name]] if registry_name != "all" else registries.values()
    
    for name, registry in list(filter(lambda x: x[0] != "all", registries.items())):
        if registry in targets:
            print(f"\n{name.upper()} REGISTRY:")
            print("=" * 50)
            
            for item_name in registry.list():
                factory = registry._factories.get(item_name)
                metadata = registry._metadata.get(item_name, {})
                
                print(f"- {item_name}:")
                print(f"  Callable: {callable(factory)}")
                print(f"  Type: {type(factory)}")
                print(f"  Metadata: {metadata}")
                
                # Try to safely call the factory
                try:
                    if callable(factory):
                        instance = factory()
                        print(f"  Creates: {type(instance).__name__}")
                except Exception as e:
                    print(f"  Creation Error: {str(e)}")
            
            print("-" * 50)
            print(f"Total items: {len(registry.list())}")
```

By following these advanced patterns, you can create robust, maintainable components that integrate seamlessly with the framework, even in complex applications.

### 14.6 Specialized Workflow Patterns

Beyond the standard deterministic and handoff workflows, you can create specialized workflow patterns for specific domains:

#### Domain-Specific Agent Configuration

Configure agents differently based on the domain they're operating in:

```python
def create_domain_workflow(domain: str, **kwargs) -> Any:
    """Create a workflow optimized for a specific domain."""
    
    # Domain-specific agent configurations
    domain_configs = {
        "technical": {
            "research": {
                "instructions": "Focus on technical information from reliable sources.",
                "temperature": 0.2,  # Lower temperature for more precise responses
                "tools": ["tavily_search_tool"]
            },
            "writer": {
                "instructions": "Write technical content with clear explanations.",
                "temperature": 0.5
            },
            "editor": {
                "instructions": "Check for technical accuracy and clarity.",
                "temperature": 0.3
            }
        },
        "creative": {
            "research": {
                "instructions": "Gather diverse and inspiring information.",
                "temperature": 0.7  # Higher temperature for more creative responses
            },
            "writer": {
                "instructions": "Create engaging, creative content with vivid language.",
                "temperature": 0.8
            },
            "editor": {
                "instructions": "Enhance the creative elements while ensuring readability.",
                "temperature": 0.6
            }
        }
    }
    
    # Use default domain if specified domain not found
    config = domain_configs.get(domain, domain_configs.get("technical"))
    
    # Create agents with domain-specific configurations
    agents = []
    for agent_type, agent_config in config.items():
        agent = create_agent(agent_type, **agent_config)
        agents.append(agent)
    
    # Create workflow with the configured agents
    return DeterministicWorkflow(
        agents=agents,
        name=f"{domain}_workflow"
    )
```

#### Branching Workflows

Create workflows that can take different paths based on context or input:

```python
async def execute_branching_workflow(context: Context, input_text: str) -> str:
    """Execute a workflow with conditional branching based on content type."""
    
    # First, analyze the input to determine the content type
    analysis_agent = create_agent("strategy", 
                                 instructions="Analyze the input and classify it as technical, creative, or general.")
    
    analysis_response = await analysis_agent.execute(context, input_text)
    content_type = _extract_content_type(analysis_response.content)
    
    # Create different workflow paths based on content type
    if content_type == "technical":
        workflow = create_workflow([
            "research", 
            "technical_writer",  # Specialized technical writer
            "technical_editor"   # Specialized technical editor
        ])
    elif content_type == "creative":
        workflow = create_workflow([
            "creative_research",  # More open-ended research
            "writer",
            "creative_editor"     # Focus on style and engagement
        ])
    else:  # General content
        workflow = create_workflow("content_generation")  # Standard workflow
    
    # Execute the selected workflow
    engine = WorkflowEngine()
    result = await engine.execute(workflow, context, input_text)
    
    return result["result"]

def _extract_content_type(analysis_text: str) -> str:
    """Extract content type from analysis text."""
    analysis_text = analysis_text.lower()
    
    if "technical" in analysis_text:
        return "technical"
    elif "creative" in analysis_text:
        return "creative"
    else:
        return "general"
```

#### Parallel Workflows with Aggregation

Run multiple workflows in parallel and combine their results:

```python
async def execute_parallel_workflows(context: Context, input_text: str) -> str:
    """Execute multiple workflows in parallel and aggregate results."""
    
    # Create different workflow configurations
    workflows = {
        "technical": create_domain_workflow("technical"),
        "creative": create_domain_workflow("creative"),
        "general": create_workflow("content_generation")
    }
    
    # Create isolated contexts for each workflow
    contexts = {name: Context() for name in workflows.keys()}
    
    # Execute workflows in parallel
    engine = WorkflowEngine()
    tasks = {
        name: engine.execute(workflow, contexts[name], input_text)
        for name, workflow in workflows.items()
    }
    
    # Wait for all workflows to complete
    import asyncio
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    workflow_results = dict(zip(tasks.keys(), results))
    
    # Create an aggregation agent to combine results
    aggregator = create_agent("editor", 
                            instructions="Combine the different content versions into a cohesive final piece.")
    
    # Prepare combined input for the aggregator
    combined_input = f"Original request: {input_text}\n\n"
    for name, result in workflow_results.items():
        if isinstance(result, Exception):
            combined_input += f"{name.upper()} VERSION: Error - {str(result)}\n\n"
        else:
            combined_input += f"{name.upper()} VERSION:\n{result.get('result', 'No result')}\n\n"
    
    # Get the aggregated result
    aggregated_response = await aggregator.execute(context, combined_input)
    
    return aggregated_response.content
```

#### Self-Improving Workflows

Create workflows that can evaluate and improve their own output:

```python
async def execute_self_improving_workflow(context: Context, input_text: str, max_iterations: int = 3) -> str:
    """Execute a workflow that can critique and improve its own output."""
    
    # Create the base content generation workflow
    workflow = create_workflow("content_generation")
    
    # Create a critique agent
    critique_agent = create_agent("editor", 
                                instructions="Critically evaluate the content and suggest specific improvements.")
    
    # Create a revision agent
    revision_agent = create_agent("writer", 
                                instructions="Revise the content based on the critique.")
    
    # Initial content generation
    engine = WorkflowEngine()
    result = await engine.execute(workflow, context, input_text)
    current_content = result["result"]
    
    # Iterative improvement loop
    for i in range(max_iterations):
        # Get critique
        critique_input = f"Original request: {input_text}\n\nCurrent content:\n{current_content}"
        critique_response = await critique_agent.execute(context, critique_input)
        critique = critique_response.content
        
        # Check if critique suggests major improvements
        if "excellent" in critique.lower() and "no improvements needed" in critique.lower():
            logger.info(f"Content reached excellent quality at iteration {i+1}")
            break
        
        # Revise based on critique
        revision_input = f"Original request: {input_text}\n\nCurrent content:\n{current_content}\n\nCritique:\n{critique}"
        revision_response = await revision_agent.execute(context, revision_input)
        current_content = revision_response.content
        
        logger.info(f"Completed revision iteration {i+1}/{max_iterations}")
    
    return current_content
```

These specialized workflow patterns demonstrate the flexibility of the framework for complex real-world applications.

### 14.7 Framework Extension Points

The `aigen` framework is designed with multiple extension points that allow you to customize its behavior without modifying core code:

#### Custom Agent State Management

Extend the `AgentBase` class to add custom state management:

```python
class StatefulAgent(AgentBase):
    """Agent with enhanced state management capabilities."""
    
    def __init__(self, agent_id="stateful_agent", **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self.state = {}
        self.state_schema = kwargs.get("state_schema", {})
        self.history = []
    
    async def initialize(self) -> None:
        """Initialize agent with state persistence."""
        await super().initialize()
        
        # Load persisted state if available
        if hasattr(self, 'persistence_provider'):
            try:
                persisted_state = await self.persistence_provider.load_state(self.agent_id)
                if persisted_state:
                    self.state.update(persisted_state)
                    logger.info(f"Loaded persisted state for {self.agent_id}")
            except Exception as e:
                logger.warning(f"Failed to load state for {self.agent_id}: {str(e)}")
    
    async def execute(self, context, input_text=None):
        """Execute with state tracking."""
        # Record state before execution
        pre_state = self.state.copy()
        self.history.append({"time": datetime.now().isoformat(), "state": pre_state})
        
        # Execute normally
        response = await super().execute(context, input_text)
        
        # Update state based on execution
        self._update_state(context, response)
        
        # Persist state if needed
        if hasattr(self, 'persistence_provider'):
            try:
                await self.persistence_provider.save_state(self.agent_id, self.state)
            except Exception as e:
                logger.warning(f"Failed to persist state for {self.agent_id}: {str(e)}")
        
        return response
    
    def _update_state(self, context, response):
        """Update agent state based on execution result."""
        # Implement custom state update logic
        if hasattr(context, '_execution_count'):
            self.state['execution_count'] = context._execution_count
            
        if response.success:
            self.state['last_successful_execution'] = datetime.now().isoformat()
            
        if hasattr(context, 'get_output'):
            for key in self.state_schema.get('tracked_outputs', []):
                output = context.get_output(key)
                if output:
                    self.state[f'last_{key}_output'] = output[:500]
```

#### Custom Workflow Execution Strategies

Create custom workflow execution strategies by extending the base workflow classes:

```python
class RetryingWorkflow(DeterministicWorkflow):
    """Workflow that implements advanced retry strategies."""
    
    def __init__(self, agents, name="retrying_workflow", max_retries=3, **kwargs):
        super().__init__(agents=agents, name=name, **kwargs)
        self.max_retries = max_retries
        self.retry_delays = kwargs.get("retry_delays", [1, 5, 15])
        self.retry_strategies = kwargs.get("retry_strategies", {})
    
    async def execute(self, context, input_text=None):
        """Execute with retry logic."""
        for retry in range(self.max_retries + 1):
            try:
                return await super().execute(context, input_text)
            
            except WorkflowError as e:
                if retry < self.max_retries:
                    # Get agent that failed
                    failed_agent_id = getattr(e, 'agent_id', None)
                    
                    # Apply retry strategy
                    if failed_agent_id and failed_agent_id in self.retry_strategies:
                        strategy = self.retry_strategies[failed_agent_id]
                        logger.info(f"Applying retry strategy for {failed_agent_id}: {strategy}")
                        
                        if strategy == "different_model":
                            # Try with a different model
                            for agent in self.agents:
                                if agent.agent_id == failed_agent_id:
                                    current_model = agent.parameters.get("model", "gpt-4o")
                                    fallback_models = ["gpt-4-1106-preview", "gpt-3.5-turbo"]
                                    
                                    for model in fallback_models:
                                        if model != current_model:
                                            logger.info(f"Retrying with model {model}")
                                            agent.parameters["model"] = model
                                            break
                    
                    # Wait before retrying
                    delay = self.retry_delays[min(retry, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying workflow in {delay} seconds (attempt {retry + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded
                    raise
```

#### Plugin System

Create a plugin system to extend the framework with new functionality:

```python
class AIgenPlugin:
    """Base class for framework plugins."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", self.__class__.__name__)
        self.version = kwargs.get("version", "0.1.0")
        self.enabled = kwargs.get("enabled", True)
        self.priority = kwargs.get("priority", 50)  # 0-100, higher is higher priority
    
    async def initialize(self, app=None):
        """Initialize the plugin."""
        pass
    
    async def on_agent_created(self, agent):
        """Called when an agent is created."""
        pass
    
    async def on_workflow_created(self, workflow):
        """Called when a workflow is created."""
        pass
    
    async def on_execution_start(self, context, input_text):
        """Called before workflow execution."""
        pass
    
    async def on_execution_complete(self, context, result):
        """Called after workflow execution."""
        pass

class PluginManager:
    """Manager for framework plugins."""
    
    def __init__(self):
        self.plugins = []
        self.initialized = False
    
    def register_plugin(self, plugin):
        """Register a plugin."""
        self.plugins.append(plugin)
        self.plugins.sort(key=lambda p: p.priority, reverse=True)
        
    async def initialize_plugins(self, app=None):
        """Initialize all registered plugins."""
        if self.initialized:
            return
            
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    await plugin.initialize(app)
                    logger.info(f"Initialized plugin: {plugin.name} v{plugin.version}")
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {plugin.name}: {str(e)}")
        
        self.initialized = True
    
    async def trigger_event(self, event_name, *args, **kwargs):
        """Trigger an event on all plugins."""
        for plugin in self.plugins:
            if not plugin.enabled:
                continue
                
            method = getattr(plugin, f"on_{event_name}", None)
            if method and callable(method):
                try:
                    await method(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin.name}.on_{event_name}: {str(e)}")
```

#### Example Analytics Plugin

Here's an example plugin that collects usage analytics:

```python
class AnalyticsPlugin(AIgenPlugin):
    """Plugin for collecting usage analytics."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Analytics", version="1.0.0", **kwargs)
        self.analytics = {
            "agent_created": 0,
            "workflow_created": 0,
            "execution_started": 0,
            "execution_completed": 0,
            "execution_errors": 0,
            "agent_usage": {},
            "workflow_usage": {},
            "average_execution_time": 0,
            "total_executions": 0
        }
    
    async def on_agent_created(self, agent):
        """Track agent creation."""
        self.analytics["agent_created"] += 1
        
        agent_type = getattr(agent, 'role', 'unknown')
        if agent_type not in self.analytics["agent_usage"]:
            self.analytics["agent_usage"][agent_type] = 0
        self.analytics["agent_usage"][agent_type] += 1
    
    async def on_workflow_created(self, workflow):
        """Track workflow creation."""
        self.analytics["workflow_created"] += 1
        
        workflow_type = workflow.name
        if workflow_type not in self.analytics["workflow_usage"]:
            self.analytics["workflow_usage"][workflow_type] = 0
        self.analytics["workflow_usage"][workflow_type] += 1
    
    async def on_execution_start(self, context, input_text):
        """Track execution start."""
        self.analytics["execution_started"] += 1
        
        # Store start time in context for later calculation
        context.set_metadata("analytics_start_time", time.time())
    
    async def on_execution_complete(self, context, result):
        """Track execution completion."""
        self.analytics["execution_completed"] += 1
        
        # Calculate execution time
        start_time = context.get_metadata("analytics_start_time")
        if start_time:
            execution_time = time.time() - start_time
            
            # Update average execution time
            current_avg = self.analytics["average_execution_time"]
            current_total = self.analytics["total_executions"]
            
            new_total = current_total + 1
            new_avg = ((current_avg * current_total) + execution_time) / new_total
            
            self.analytics["average_execution_time"] = new_avg
            self.analytics["total_executions"] = new_total
        
        # Check for errors
        if result.get("status") != "completed":
            self.analytics["execution_errors"] += 1
    
    def get_analytics_report(self):
        """Generate an analytics report."""
        return {
            "summary": self.analytics,
            "top_agent_types": sorted(
                self.analytics["agent_usage"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_workflows": sorted(
                self.analytics["workflow_usage"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "error_rate": (
                self.analytics["execution_errors"] / self.analytics["execution_completed"]
                if self.analytics["execution_completed"] > 0 else 0
            )
        }
```

By leveraging these extension points, you can adapt the framework to meet specific needs without modifying its core implementation.

## 10. Troubleshooting

### 10.1 Custom Agent Issues

The agent system supports both built-in agents and custom agents. Custom agents can be created through the Agent Builder UI or by manually creating Python files. Here are solutions to common issues:

#### 10.1.1 Agent Not Found Issues

If your custom agent isn't being found by the system, check the following:

1. **Location**: Agents should be placed in either:
   - Main agents directory: `aigen/agents/*.py`
   - Custom agents directory: `aigen/agents/custom/*.py`

2. **Module Structure**: Ensure both directories have an `__init__.py` file to make them proper Python packages.

3. **Class Naming**: The agent class should follow either:
   - PascalCase: `TestAgent` (automatically detected)
   - camelCase: `testAgent` (automatically detected as fallback)

#### 10.1.2 Import Errors

Custom agents must use absolute imports rather than relative ones:

```python
# CORRECT - Use absolute imports
from aigen.agents.base import AgentBase, AgentRole, AgentResponse
from aigen.core.context import Context
from aigen.core.errors import AgentError

# INCORRECT - Don't use relative imports 
from .base import AgentBase  # This will fail in custom subdirectories
```

#### 10.1.3 Interface Compatibility

All agents must:

1. **Inherit from AgentBase**:
   ```python
   class MyCustomAgent(AgentBase):
       # Implementation
   ```

2. **Implement the correct method signatures**:
   ```python
   async def execute(
       self, 
       context: Context,  # Must accept Context object first
       input_text: Optional[str] = None  # Input text is optional second param
   ) -> AgentResponse:  # Must return AgentResponse object
       # Implementation
   ```

3. **Handle context correctly**:
   ```python
   # Get input from context if not provided directly
   if input_text is None:
       input_text = context.get_latest_output() or "No input provided"
       
   # Store results in context
   context.store_output(self.agent_id, result)
   ```

4. **Return properly structured responses**:
   ```python
   return AgentResponse(
       content=result,
       agent_id=self.agent_id,
       success=True,
       metadata={"role": "custom"}
   )
   ```

#### 10.1.4 Agent Builder Generated Code

When using the Agent Builder to create custom agents:

1. The system generates code in the `aigen/agents/custom/` directory
2. The agent is automatically registered with the factory
3. The generated agent follows the correct interface pattern

If you encounter issues with a generated agent:
- Check the logs for specific error messages
- Verify that the agent was successfully registered by looking for "Registered agent factory: [name]" in logs
- If needed, manually edit the agent code to fix any issues

#### 10.1.5 Agent Factory System

The agent factory system uses multiple strategies to find and instantiate agents:

1. **Direct instantiation** for built-in agents
2. **Factory functions** for registered agents 
3. **Dynamic import** as a fallback, which:
   - First checks in main agents directory
   - Then checks in custom agents directory
   - Tries both PascalCase and camelCase class names

If you're having trouble with agent registration:
```python
from aigen.agents.factory import register_agent_factory

# Register with a factory function
register_agent_factory(
    "my_agent",  # Agent type ID
    lambda agent_id=None, **kwargs: MyAgent(agent_id=agent_id or "my_agent", **kwargs),
    {"description": "My custom agent"}  # Optional metadata
)
```

##### Common Warning: "Factory for [agent_name] is not callable"

If you see a warning like:
```
[12:08:33] ⚠️ WARNING: Factory for agent_22 is not callable, trying dynamic import
```

This indicates that:

1. The system found a registered factory for the agent, but it's not a callable function
2. This typically happens when an agent instance (object) was registered instead of a factory function
3. The system will fall back to dynamic import, which might still work but is less efficient

To fix this issue:

*No action is required** when you see this warning. The system will automatically attempt to:
1. Use dynamic import to find the appropriate agent module
2. Instantiate the agent class directly
3. Continue normal operation without interruption.
4. This is a known issue. However, you can move your custom agent to the agents folder and update it following examples of other agents.



#### 10.1.6 Runner Context Issues

When using the OpenAI Agents SDK Runner:
```python
# CORRECT: Don't pass the Context object directly to Runner
response = await Runner.run(
    self.openai_agent,
    input=input_text,
    context=None,  # Don't use our Context object here
    max_turns=self.parameters.get("max_turns", 10)
)
```

### 10.2 Other Common Issues

#### 10.2.1 Context Class Metadata

The `Context` class provides two methods for handling metadata:

```python
# Use get_metadata() to get the execution metadata object
metadata = context.get_metadata()  # Returns ExecutionMetadata object

# Use set_metadata() and get_user_metadata() for custom metadata
context.set_metadata("key", value)  # Set custom metadata
value = context.get_user_metadata("key")  # Get custom metadata
```

These methods must not be confused with each other.
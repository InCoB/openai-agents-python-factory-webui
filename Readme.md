# AIgen: OpenAI Agents Framework

AIgen is a production-ready implementation of an AI agent orchestration framework built on the OpenAI Agents SDK. The framework provides pre-built agents, tools, and workflows while enabling extensive customization and extension.

## Features

- **Ready-to-use AI Agents**: Includes research, writer, editor, and strategy agents
- **Flexible Workflows**: Deterministic (sequential) and Handoff (dynamic) workflow patterns // handoff must be tested more properly//
- **Extensible Architecture**: Registry-based component system for adding custom agents and tools
- **Multiple Interfaces**: CLI, Web UI (Gradio), and Programmatic API
- **Robust Error Handling**: Built-in retry mechanisms and error classification

## Getting Started

```bash
# Install from source
git clone https://github.com/yourusername/aigen.git
cd aigen
pip install -e .

# Set required API keys
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # For research capabilities
```

## CLI Usage

AIgen includes a comprehensive command-line interface for working with workflows:

```bash
# List all available components
aigen list

# Execute workflows
aigen execute --workflow content_generation --input "Write about AI frameworks" --output article.md

# Run workflows with specific agents
aigen execute --agents research,writer,editor --input "Explain quantum computing" --output quantum.md

# Interactive guided workflow creation
aigen interactive

# Launch web interface
aigen --web --port 8080 --share
```

### Available Workflows

- **content_generation**: Full pipeline with research, strategy, writing, and editing
- **research_only**: Single-agent workflow that performs research only
- **Custom workflows**: Create your own by specifying agent combinations

## Agent Builder

The framework includes a powerful Agent Builder tool that allows you to create custom agents through a visual interface without writing code:

### Features

- **No-Code Agent Creation**: Define agents using a simple form interface
- **Custom Instructions**: Provide specialized instructions to guide agent behavior
- **Tool Selection**: Choose which tools your agent can access
- **Structured Output**: Optionally define structured output formats using Pydantic models
- **Testing Interface**: Test your agents with sample inputs before deployment
- **Agent Management**: List, edit, and delete custom agents

![Agent Builder Interface](documentation/images/agent_builder_screenshot.png)

*The Agent Builder provides a user-friendly interface for creating custom agents without coding*

### Usage

```bash
# Launch the Agent Builder interface
aigen --builder

# Or access it through the main web interface
aigen --web --port 8080
```

### Integration

Custom agents created with the Agent Builder are automatically registered with the framework and can be:

- Used in workflows through the CLI and API
- Selected in the web interface
- Combined with built-in agents
- Used as handoff targets for other agents

For detailed technical documentation on the Agent Builder module, see [8. agent_builder_module.md](documentation/8.%20agent_builder_module.md).

### CLI Commands

| Command | Description |
|---------|-------------|
| `execute` | Run workflows with input/output options |
| `interactive` | Guided workflow creation and execution |
| `list` | Display available components |
| `web` | Launch the Gradio web interface |

### Web Interface

The web UI offers an intuitive way to interact with the framework:

- **Predefined Workflows**: Select and run standard workflow pipelines
- **Custom Workflow Builder**: Create custom agent sequences through a visual interface
- **Workflow Management**: View available components and their descriptions

## Programmatic API

```python
import asyncio
from aigen import create_workflow, Context
from aigen.workflows.engine import WorkflowEngine

async def generate_content():
    workflow = create_workflow("content_generation")
    context = Context()
    engine = WorkflowEngine()
    
    result = await engine.execute(workflow, context, "Write about AI frameworks")
    
    if result["status"] == "completed":
        print(result["result"])

# Run the async function
asyncio.run(generate_content())
```

## Extending the Framework

### Creating Custom Agents

```python
from aigen.agents.base import AgentBase, AgentRole, AgentResponse
from aigen.agents.factory import register_agent_factory

# Create custom agent
class MyAgent(AgentBase):
    def __init__(self, agent_id="my_agent", **kwargs):
        super().__init__(agent_id=agent_id, role=AgentRole.CUSTOM, **kwargs)
        
    async def execute(self, context, input_text=None):
        # Agent logic
        return AgentResponse(
            content="Generated content",
            agent_id=self.agent_id,
            success=True
        )

# Register with the factory system
register_agent_factory(
    "my_agent",
    lambda agent_id=None, **kwargs: MyAgent(agent_id=agent_id, **kwargs),
    {"description": "My custom agent"}
)
```

### Using Agents Created with Agent Builder

Agents created with the Agent Builder are automatically registered with the factory system and can be used programmatically:

```python
from aigen.agents import create_agent

# Use a custom agent created with the Agent Builder
custom_agent = create_agent("my_custom_agent")

# Execute the agent
response = await custom_agent.execute("What's the weather in Paris?")
print(response.content)

# Use in workflows
from aigen.workflows import DeterministicWorkflow

workflow = DeterministicWorkflow(
    agents=[
        create_agent("research"),
        create_agent("my_custom_agent"),  # Your Agent Builder created agent
        create_agent("editor")
    ],
    name="custom_workflow"
)

result = await workflow.execute("Research quantum computing trends")
```

### Creating Custom Workflows

```python
from aigen.workflows import DeterministicWorkflow
from aigen.agents import create_agent

# Create custom workflow
def create_custom_workflow(**kwargs):
    return DeterministicWorkflow(
        agents=[
            create_agent("research"),
            create_agent("my_agent"),  # Your custom agent
            create_agent("editor")
        ],
        name="my_workflow"
    )

# Always use lambda for proper factory registration
register_workflow_factory(
    "my_workflow",
    lambda **kwargs: create_custom_workflow(**kwargs),
    {"description": "My custom workflow pipeline"}
)
```

## Troubleshooting

### Factory Registration Issues

If you encounter errors like `Factory for research_only is not callable`, ensure all factories are registered using lambda functions:

```python
# CORRECT - Using lambda to ensure callable factory
register_workflow_factory(
    "research_only",
    lambda **kwargs: DeterministicWorkflow(
        agents=[create_agent("research")],
        name="research_only",
        **kwargs
    ),
    {"description": "Research-only workflow"}
)

# INCORRECT - Will cause "not callable" errors
register_workflow_factory(
    "research_only",
    DeterministicWorkflow(agents=[create_agent("research")]),
    {"description": "Research-only workflow"}
)
```

## Tools System

AIgen includes a powerful tools system that enables agents to interact with external services:

### Built-in Tools

```python
from aigen.tools import create_tool

# Create and use a search tool
search_tool = create_tool("tavily_search", api_key="your-tavily-key")
results = await search_tool.execute(query="Latest AI advancements", results_count=3)

# Access results
if results.success:
    for result in results.data.get("results", []):
        print(f"Title: {result.get('title')}")
        print(f"Content: {result.get('content')}")
```

### Creating Custom Tools

```python
from aigen.tools.base import ToolBase, ToolType, ToolResponse
from aigen.tools.factory import register_tool_factory

# Create a custom tool
class WeatherTool(ToolBase):
    def __init__(self, tool_id="weather_tool", **kwargs):
        super().__init__(tool_id=tool_id, tool_type=ToolType.UTILITIES, **kwargs)
        self.api_key = kwargs.get("api_key", "")
        
    async def execute(self, location="New York", **kwargs):
        # Tool implementation logic
        return ToolResponse(
            data={"temperature": 72, "conditions": "sunny"},
            tool_id=self.tool_id,
            success=True
        )

# Register with factory
register_tool_factory(
    "weather",
    lambda tool_id=None, **kwargs: WeatherTool(tool_id=tool_id, **kwargs),
    {"description": "Weather information tool"}
)
```

## Architecture

The framework is built with extensibility in mind:

- **Registry-Based Components**: All agents, tools, and workflows are managed through registries
- **Factory Pattern**: Components are created through factory functions
- **Plugin System**: Extend functionality with custom plugins
- **Extension Points**: Customize agent state management, workflow execution, and more

## License

[MIT License](LICENSE)
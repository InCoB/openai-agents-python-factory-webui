from typing import Dict, Any, Optional, List, TypeVar, Generic, Union
from datetime import datetime
import uuid
import json

T = TypeVar('T')

class ExecutionMetadata:
    """Tracks execution metadata for observability and debugging."""
    
    def __init__(self):
        """Initialize execution metadata."""
        self.execution_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.completion_time = None
        self.agent_sequence = []
        self.error_count = 0
        self.last_error = None
        self.total_tokens = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "elapsed_seconds": self.get_elapsed_seconds(),
            "agent_sequence": self.agent_sequence,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "total_tokens": self.total_tokens
        }
    
    def get_elapsed_seconds(self) -> float:
        """Get the elapsed time in seconds."""
        end_time = self.completion_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def mark_complete(self) -> None:
        """Mark the execution as complete."""
        self.completion_time = datetime.now()
    
    def add_agent(self, agent_id: str) -> None:
        """Add an agent to the execution sequence."""
        self.agent_sequence.append(agent_id)
    
    def add_tokens(self, token_count: int) -> None:
        """Add tokens to the total token count."""
        self.total_tokens += token_count
    
    def record_error(self, error: str) -> None:
        """Record an error that occurred during execution."""
        self.error_count += 1
        self.last_error = error

class Context(Generic[T]):
    """
    Context object for managing state across agent execution.
    Provides mechanisms for storing data, tracking execution, and maintaining state.
    """
    
    def __init__(self, initial_data: Optional[T] = None):
        """
        Initialize a new context.
        
            initial_data: Optional initial data for the context
        """
        self._data = initial_data
        self._metadata = ExecutionMetadata()
        self._intermediate_outputs: Dict[str, Any] = {}
        self._artifacts: Dict[str, Any] = {}
    
    def get_data(self) -> T:
        """Get the primary data object."""
        return self._data
    
    def set_data(self, data: T) -> None:
        """Set the primary data object."""
        self._data = data
    
    def get_metadata(self) -> ExecutionMetadata:
        """Get the execution metadata."""
        return self._metadata
    
    def store_output(self, agent_id: str, output: Any) -> None:
        """
        Store output from an agent.
        
            agent_id: The ID of the agent
            output: The output to store
        """
        self._intermediate_outputs[agent_id] = output
        self._metadata.add_agent(agent_id)
    
    def get_output(self, agent_id: str) -> Any:
        """
        Get output from an agent.
        
            agent_id: The ID of the agent
            
        """
        return self._intermediate_outputs.get(agent_id)
    
    def get_latest_output(self) -> Any:
        """
        Get the most recent agent output.
        
        """
        if not self._metadata.agent_sequence:
            return None
        
        latest_agent = self._metadata.agent_sequence[-1]
        return self._intermediate_outputs.get(latest_agent)
    
    def add_artifact(self, key: str, artifact: Any) -> None:
        """
        Store an execution artifact.
        
            key: The artifact key
            artifact: The artifact value
        """
        self._artifacts[key] = artifact
    
    def get_artifact(self, key: str) -> Any:
        """
        Get an execution artifact.
        
            key: The artifact key
            
        """
        return self._artifacts.get(key)
    
    def list_artifacts(self) -> List[str]:
        """
        Get a list of all artifact keys.
        
            List[str]: List of artifact keys
        """
        return list(self._artifacts.keys())
    
    def mark_complete(self) -> None:
        """Mark the execution as complete."""
        self._metadata.mark_complete()
    
    def record_error(self, error: Union[str, Exception]) -> None:
        """Record an execution error."""
        error_str = str(error)
        self._metadata.record_error(error_str)
    
    def add_tokens(self, count: int) -> None:
        """Add tokens to the total token count."""
        self._metadata.add_tokens(count)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata on the context.
        
            key: The metadata key
            value: The metadata value
        """
        if not hasattr(self, '_user_metadata'):
            self._user_metadata = {}
        self._user_metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from the context.
        
            key: The metadata key
            default: Default value if key doesn't exist
            
            Any: The metadata value or default
        """
        if not hasattr(self, '_user_metadata'):
            return default
        return self._user_metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to a dictionary representation.
        
            Dict[str, Any]: Dictionary representation of the context
        """
        try:
            if hasattr(self._data, 'to_dict'):
                data_dict = self._data.to_dict()
            elif hasattr(self._data, '__dict__'):
                data_dict = self._data.__dict__
            else:
                data_dict = json.loads(json.dumps(self._data, default=str))
        except:
            data_dict = str(self._data)
        
        # Include user metadata in the output
        user_metadata = getattr(self, '_user_metadata', {})
        user_metadata_str = {k: str(v) for k, v in user_metadata.items()}
        
        return {
            "metadata": self._metadata.to_dict(),
            "data": data_dict,
            "intermediate_outputs": {k: str(v) for k, v in self._intermediate_outputs.items()},
            "artifacts": {k: str(v) for k, v in self._artifacts.items()},
            "user_metadata": user_metadata_str
        }
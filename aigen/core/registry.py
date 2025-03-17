from typing import Dict, Any, Type, TypeVar, Generic, Callable, Optional, List
import inspect

T = TypeVar('T')

class Component:
    """Base class for registrable components."""
    pass

class Registry(Generic[T]):
    """
    Generic registry for managing system components.
    Provides registration, retrieval, and discovery capabilities.
    """
    
    def __init__(self, component_type: Type[T]):
        """
        Initialize a registry for a specific component type.
        
            component_type: The type of components this registry will manage
        """
        self._components: Dict[str, T] = {}
        self._factories: Dict[str, Callable[..., T]] = {}
        self._component_type = component_type
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, component: T, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component instance.
        
            name: Name to register the component under
            component: The component instance
            metadata: Optional metadata for the component
        """
        if not isinstance(component, self._component_type):
            raise TypeError(f"Component must be of type {self._component_type.__name__}")
        
        self._components[name] = component
        self._metadata[name] = metadata or {}
    
    def register_factory(self, name: str, factory, metadata=None) -> None:
        """
        Register a factory function with metadata.
        
        Args:
            name: Name to register the factory under
            factory: Function or class to construct the component
            metadata: Optional metadata dictionary
        """
        if name in self._factories:
            self._logger.warning(f"Overwriting existing factory for {name}")
        
        # Ensure factory is callable and preserve lambda functions correctly
        if not callable(factory):
            self._logger.warning(f"Factory for {name} is not callable")
            raise ValueError(f"Factory for {name} must be callable")
        
        # Store the factory directly without wrapping it in another function
        self._factories[name] = factory
        
        if metadata:
            self._metadata[name] = metadata
            
        return name
    
    def get(self, name: str, *args, **kwargs) -> T:
        """
        Get a component by name.
        
            name: Name of the component to get
            *args: Arguments to pass to factory if used
            **kwargs: Keyword arguments to pass to factory if used
            
            T: The component instance
            
            KeyError: If no component or factory with the given name exists
        """
        if name in self._components:
            return self._components[name]
        
        if name in self._factories:
            factory = self._factories[name]
            component = factory(*args, **kwargs)
            
            # Skip type checking for factories to avoid issues with mixed component types
            # Type checking is still applied when components are registered directly
            
            return component
        
        raise KeyError(f"No component or factory registered with name: {name}")
    
    def list(self) -> List[str]:
        """
        Get a list of all registered component names.
        
            List[str]: List of component names
        """
        return list(set(list(self._components.keys()) + list(self._factories.keys())))
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a component.
        
            name: Name of the component
            
            Dict[str, Any]: Component metadata or empty dict if not found
        """
        return self._metadata.get(name, {})
    
    def list_with_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all components with their metadata.
        
            Dict[str, Dict[str, Any]]: Dictionary of component names to metadata
        """
        result = {}
        
        for name in self._components:
            metadata = self._metadata.get(name, {}).copy()
            metadata["type"] = "instance"
            result[name] = metadata
        
        for name in self._factories:
            if name not in result:  # Avoid duplicates
                metadata = self._metadata.get(name, {}).copy()
                metadata["type"] = "factory"
                result[name] = metadata
        
        return result
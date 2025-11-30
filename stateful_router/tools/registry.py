"""
Tool Registry for the Enhanced Stateful Router.

This module manages all available tools and provides a unified interface
for tool discovery, registration, and execution.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

from loguru import logger

from .base import BaseTool, create_simple_tool
from .deterministic import DETERMINISTIC_TOOLS
from .probabilistic import ParseTool, CompareTool, SearchTool
from ..config.settings import settings


@dataclass
class ToolInfo:
    """Information about a registered tool."""
    name: str
    function: Callable
    deterministic: bool
    category: str
    description: str
    expected_confidence: float
    timeout: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "deterministic": self.deterministic,
            "category": self.category,
            "description": self.description,
            "expected_confidence": self.expected_confidence,
            "timeout": self.timeout
        }


class ToolRegistry:
    """
    Central registry for all tools in the system.
    
    Manages tool registration, discovery, and provides a unified interface
    for executing tools with proper error handling and timeout management.
    """
    
    def __init__(self, model_client=None):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolInfo] = {}
        self._register_default_tools(model_client) # <-- Pass client here
        logger.info(f"Initialized ToolRegistry with {len(self.tools)} tools")
    
    def _register_default_tools(self, model_client=None) -> None:
        """Register all default tools."""
        # Register deterministic tools
        for name, tool in DETERMINISTIC_TOOLS.items():
            self.register_tool(
                name=name,
                function=tool,
                deterministic=True,
                category="arithmetic",
                description=f"Deterministic {name.replace('tool_', '')} operation",
                expected_confidence=1.0
            )
        prob_tools = {
            "tool_parse": ParseTool(model_client=model_client),
            "tool_compare": CompareTool(model_client=model_client),
            "tool_search": SearchTool(), # Doesn't need client
        }

        # Register probabilistic tools
        for name, tool in prob_tools.items():
            self.register_tool(
                name=name,
                function=tool,
                deterministic=False,
                category="extraction" if "parse" in name else "comparison",
                description=f"Probabilistic {name.replace('tool_', '')} operation",
                expected_confidence=0.85 # hard set confidence
            )
    
    def register_tool(self,
                     name: str,
                     function: Callable,
                     deterministic: bool,
                     category: str,
                     description: str,
                     expected_confidence: float = 0.9,
                     timeout: Optional[float] = None) -> None:
        """
        Register a new tool.
        
        Args:
            name: Tool name
            function: Tool function or BaseTool instance
            deterministic: Whether tool is deterministic
            category: Tool category
            description: Tool description
            expected_confidence: Expected confidence score
            timeout: Execution timeout
        """
        if timeout is None:
            timeout = settings.TOOL_TIMEOUT.get(name, 5.0)
        
        self.tools[name] = ToolInfo(
            name=name,
            function=function,
            deterministic=deterministic,
            category=category,
            description=description,
            expected_confidence=expected_confidence,
            timeout=timeout
        )
        
        logger.debug(f"Registered tool: {name} (category: {category})")
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool info dictionary or None
        """
        if name in self.tools:
            tool_info = self.tools[name]
            return {
                "name": tool_info.name,
                "function": tool_info.function,
                "deterministic": tool_info.deterministic,
                "category": tool_info.category,
                "description": tool_info.description,
                "expected_confidence": tool_info.expected_confidence,
                "timeout": tool_info.timeout
            }
        return None
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            logger.error(f"Tool not found: {name}")
            return {
                "result": None,
                "confidence": 0.0,
                "error": f"Tool not found: {name}"
            }
        
        tool_info = self.tools[name]
        
        try:
            # Execute the tool
            if isinstance(tool_info.function, BaseTool):
                result = tool_info.function(**kwargs)
            else:
                result = tool_info.function(**kwargs)
            
            # Ensure result is in expected format
            if not isinstance(result, dict):
                result = {
                    "result": result,
                    "confidence": tool_info.expected_confidence
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {
                "result": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool information dictionaries
        """
        tools = []
        
        for tool_info in self.tools.values():
            if category is None or tool_info.category == category:
                tools.append(tool_info.to_dict())
        
        return tools
    
    def get_categories(self) -> List[str]:
        """
        Get all tool categories.
        
        Returns:
            List of category names
        """
        categories = set(tool.category for tool in self.tools.values())
        return sorted(list(categories))
    
    def register_custom_tool(self,
                           name: str,
                           function: Callable,
                           deterministic: bool = False,
                           category: str = "custom",
                           description: str = "") -> None:
        """
        Register a custom user-defined tool.
        
        Args:
            name: Tool name
            function: Python function to wrap
            deterministic: Whether tool is deterministic
            category: Tool category
            description: Tool description
        """
        # Wrap the function as a BaseTool
        tool = create_simple_tool(
            name=name,
            function=function,
            deterministic=deterministic,
            confidence=0.9 if not deterministic else 1.0
        )
        
        self.register_tool(
            name=name,
            function=tool,
            deterministic=deterministic,
            category=category,
            description=description or f"Custom tool: {name}"
        )
        
        logger.info(f"Registered custom tool: {name}")
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was unregistered
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry stats
        """
        stats = {
            "total_tools": len(self.tools),
            "deterministic_tools": sum(1 for t in self.tools.values() if t.deterministic),
            "probabilistic_tools": sum(1 for t in self.tools.values() if not t.deterministic),
            "categories": self.get_categories(),
            "tools_by_category": {}
        }
        
        for category in stats["categories"]:
            stats["tools_by_category"][category] = sum(
                1 for t in self.tools.values() if t.category == category
            )
        
        return stats


# Global registry instance
_global_registry = None


def get_global_registry() -> ToolRegistry:
    """
    Get or create the global tool registry.
    
    Returns:
        The global ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_global_tool(name: str,
                        function: Callable,
                        deterministic: bool = False,
                        category: str = "custom",
                        description: str = "") -> None:
    """
    Register a tool in the global registry.
    
    Args:
        name: Tool name
        function: Tool function
        deterministic: Whether tool is deterministic
        category: Tool category
        description: Tool description
    """
    registry = get_global_registry()
    registry.register_custom_tool(name, function, deterministic, category, description)

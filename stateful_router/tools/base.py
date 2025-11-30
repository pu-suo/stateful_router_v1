"""
Base interface for tools in the Enhanced Stateful Router.

This module defines the base classes and protocols that all tools must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import time

from loguru import logger


@dataclass
class ToolResult:
    """Standard result format for all tools."""
    result: Any
    confidence: float
    operation: str
    deterministic: bool
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result,
            "confidence": self.confidence,
            "operation": self.operation,
            "deterministic": self.deterministic,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "error": self.error
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    All tools must inherit from this class and implement the execute method.
    """
    
    def __init__(self, name: str, deterministic: bool = False):
        """
        Initialize the tool.
        
        Args:
            name: Name of the tool
            deterministic: Whether the tool always produces the same output
        """
        self.name = name
        self.deterministic = deterministic
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with the execution result
        """
        pass
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Make the tool callable, handling timing and error catching.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            Dictionary with result and metadata
        """
        start_time = time.time()
        
        try:
            # Execute the tool
            result = self.execute(**kwargs)
            
            # Update statistics
            self.execution_count += 1
            self.total_execution_time += result.execution_time
            
            # Log execution
            logger.debug(f"Tool {self.name} executed successfully in {result.execution_time:.3f}s")
            
            return result.to_dict()
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool {self.name} failed: {e}")
            
            # Return error result
            return ToolResult(
                result=None,
                confidence=0.0,
                operation=self.name,
                deterministic=self.deterministic,
                execution_time=execution_time,
                error=str(e)
            ).to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for the tool.
        
        Returns:
            Dictionary with execution stats
        """
        avg_time = (self.total_execution_time / self.execution_count 
                   if self.execution_count > 0 else 0)
        
        return {
            "name": self.name,
            "deterministic": self.deterministic,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_time
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_count = 0
        self.total_execution_time = 0.0


def create_simple_tool(name: str, 
                      function: Callable,
                      deterministic: bool = False,
                      confidence: float = 1.0) -> BaseTool:
    """
    Create a simple tool from a function.
    
    Args:
        name: Name of the tool
        function: The function to wrap
        deterministic: Whether the tool is deterministic
        confidence: Default confidence for the tool
        
    Returns:
        A BaseTool instance wrapping the function
    """
    
    class SimpleTool(BaseTool):
        def __init__(self):
            super().__init__(name, deterministic)
            self.function = function
            self.default_confidence = confidence
        
        def execute(self, **kwargs) -> ToolResult:
            start_time = time.time()
            result = self.function(**kwargs)
            
            return ToolResult(
                result=result,
                confidence=self.default_confidence,
                operation=self.name,
                deterministic=self.deterministic,
                execution_time=time.time() - start_time
            )
    
    return SimpleTool()

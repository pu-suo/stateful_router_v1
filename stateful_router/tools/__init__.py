"""
Tools package for the Enhanced Stateful Router.
"""

from .base import BaseTool, ToolResult, create_simple_tool
from .deterministic import (
    AddTool, SubtractTool, MultiplyTool, DivideTool,
    PowerTool, ModuloTool, AbsoluteTool, RoundTool,
    tool_add, tool_subtract, tool_multiply, tool_divide,
    DETERMINISTIC_TOOLS
)
from .probabilistic import (
    ParseTool, CompareTool, SearchTool,
    tool_parse, tool_compare, tool_search,
    PROBABILISTIC_TOOLS
)
from .registry import (
    ToolRegistry, ToolInfo,
    get_global_registry, register_global_tool
)

__all__ = [
    # Base
    'BaseTool', 'ToolResult', 'create_simple_tool',
    
    # Deterministic tools
    'AddTool', 'SubtractTool', 'MultiplyTool', 'DivideTool',
    'PowerTool', 'ModuloTool', 'AbsoluteTool', 'RoundTool',
    'tool_add', 'tool_subtract', 'tool_multiply', 'tool_divide',
    'DETERMINISTIC_TOOLS',
    
    # Probabilistic tools
    'ParseTool', 'CompareTool', 'SearchTool',
    'tool_parse', 'tool_compare', 'tool_search',
    'PROBABILISTIC_TOOLS',
    
    # Registry
    'ToolRegistry', 'ToolInfo',
    'get_global_registry', 'register_global_tool'
]

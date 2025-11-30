"""
Stateful Router - A verifiable reasoning architecture.
"""

# Import the main entry point from the core package
from .core import StatefulRouterOrchestrator, ExecutionResult

# Import the main tool registry functions
from .tools import get_global_registry, register_global_tool

__all__ = [
    'StatefulRouterOrchestrator',
    'ExecutionResult',
    'get_global_registry',
    'register_global_tool'
]
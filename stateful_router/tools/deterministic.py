"""
Deterministic tools for the Enhanced Stateful Router.

These tools provide 100% reliable operations with confidence = 1.0.
They are primarily arithmetic and logical operations that always
produce the same output for the same input.
"""

import time
import math
from typing import Union, Any, Dict

from .base import BaseTool, ToolResult
from loguru import logger


class AddTool(BaseTool):
    """Tool for addition of any number of values."""

    def __init__(self):
        super().__init__(name="tool_add", deterministic=True)

    def execute(self, values: list = None, a: Union[int, float] = None, b: Union[int, float] = None, **kwargs) -> ToolResult:
        """
        Add any number of values together.

        Args:
            values: List of numbers to add (can be 2 or more)
            a: First number (legacy)
            b: Second number (legacy)

        Returns:
            ToolResult with sum
        """
        start_time = time.time()

        # Handle variable-length values
        if values is not None:
            if not isinstance(values, (list, tuple)):
                values = [values]
            if len(values) < 2:
                return ToolResult(
                    result=None,
                    confidence=1.0,
                    operation="addition",
                    deterministic=True,
                    execution_time=time.time() - start_time,
                    error="Addition requires at least 2 values"
                )
        # Handle legacy a, b parameters
        elif a is not None and b is not None:
            values = [a, b]
        else:
            return ToolResult(
                result=None,
                confidence=1.0,
                operation="addition",
                deterministic=True,
                execution_time=time.time() - start_time,
                error="Addition requires either 'values' or 'a' and 'b' parameters"
            )

        # Perform addition
        result = sum(values)

        return ToolResult(
            result=result,
            confidence=1.0,
            operation="addition",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"values": list(values)}, "count": len(values)}
        )


class SubtractTool(BaseTool):
    """Tool for subtraction."""
    
    def __init__(self):
        super().__init__(name="tool_subtract", deterministic=True)
    
    def execute(self, a: Union[int, float], b: Union[int, float]) -> ToolResult:
        """
        Subtract b from a.
        
        Args:
            a: Number to subtract from
            b: Number to subtract
            
        Returns:
            ToolResult with difference
        """
        start_time = time.time()
        
        # Perform subtraction
        result = a - b
        
        return ToolResult(
            result=result,
            confidence=1.0,
            operation="subtraction",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"a": a, "b": b}}
        )


class MultiplyTool(BaseTool):
    """Tool for multiplication of any number of values."""

    def __init__(self):
        super().__init__(name="tool_multiply", deterministic=True)

    def execute(self, values: list = None, a: Union[int, float] = None, b: Union[int, float] = None, **kwargs) -> ToolResult:
        """
        Multiply any number of values together.

        Args:
            values: List of numbers to multiply (can be 2 or more)
            a: First number (legacy)
            b: Second number (legacy)

        Returns:
            ToolResult with product
        """
        start_time = time.time()

        # Handle variable-length values
        if values is not None:
            if not isinstance(values, (list, tuple)):
                values = [values]
            if len(values) < 2:
                return ToolResult(
                    result=None,
                    confidence=1.0,
                    operation="multiplication",
                    deterministic=True,
                    execution_time=time.time() - start_time,
                    error="Multiplication requires at least 2 values"
                )
        # Handle legacy a, b parameters
        elif a is not None and b is not None:
            values = [a, b]
        else:
            return ToolResult(
                result=None,
                confidence=1.0,
                operation="multiplication",
                deterministic=True,
                execution_time=time.time() - start_time,
                error="Multiplication requires either 'values' or 'a' and 'b' parameters"
            )

        # Perform multiplication
        result = 1
        for val in values:
            result *= val

        return ToolResult(
            result=result,
            confidence=1.0,
            operation="multiplication",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"values": list(values)}, "count": len(values)}
        )


class DivideTool(BaseTool):
    """Tool for division."""
    
    def __init__(self):
        super().__init__(name="tool_divide", deterministic=True)
    
    def execute(self, a: Union[int, float], b: Union[int, float]) -> ToolResult:
        """
        Divide a by b.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            ToolResult with quotient
        """
        start_time = time.time()
        
        # Check for division by zero
        if b == 0:
            return ToolResult(
                result=None,
                confidence=1.0,
                operation="division",
                deterministic=True,
                execution_time=time.time() - start_time,
                error="Division by zero"
            )
        
        # Perform division
        result = a / b
        
        return ToolResult(
            result=result,
            confidence=1.0,
            operation="division",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"a": a, "b": b}}
        )


class PowerTool(BaseTool):
    """Tool for exponentiation."""
    
    def __init__(self):
        super().__init__(name="tool_power", deterministic=True)
    
    def execute(self, base: Union[int, float], exponent: Union[int, float]) -> ToolResult:
        """
        Raise base to the power of exponent.
        
        Args:
            base: Base number
            exponent: Exponent
            
        Returns:
            ToolResult with result
        """
        start_time = time.time()
        
        try:
            result = base ** exponent
            
            # Check for overflow/invalid results
            if math.isnan(result) or math.isinf(result):
                return ToolResult(
                    result=None,
                    confidence=1.0,
                    operation="exponentiation",
                    deterministic=True,
                    execution_time=time.time() - start_time,
                    error=f"Invalid result: {result}"
                )
            
            return ToolResult(
                result=result,
                confidence=1.0,
                operation="exponentiation",
                deterministic=True,
                execution_time=time.time() - start_time,
                metadata={"inputs": {"base": base, "exponent": exponent}}
            )
            
        except Exception as e:
            return ToolResult(
                result=None,
                confidence=1.0,
                operation="exponentiation",
                deterministic=True,
                execution_time=time.time() - start_time,
                error=str(e)
            )


class ModuloTool(BaseTool):
    """Tool for modulo operation."""
    
    def __init__(self):
        super().__init__(name="tool_modulo", deterministic=True)
    
    def execute(self, a: Union[int, float], b: Union[int, float]) -> ToolResult:
        """
        Calculate a modulo b.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            ToolResult with remainder
        """
        start_time = time.time()
        
        # Check for modulo by zero
        if b == 0:
            return ToolResult(
                result=None,
                confidence=1.0,
                operation="modulo",
                deterministic=True,
                execution_time=time.time() - start_time,
                error="Modulo by zero"
            )
        
        # Perform modulo
        result = a % b
        
        return ToolResult(
            result=result,
            confidence=1.0,
            operation="modulo",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"a": a, "b": b}}
        )


class AbsoluteTool(BaseTool):
    """Tool for absolute value."""
    
    def __init__(self):
        super().__init__(name="tool_abs", deterministic=True)
    
    def execute(self, value: Union[int, float]) -> ToolResult:
        """
        Calculate absolute value.
        
        Args:
            value: Input value
            
        Returns:
            ToolResult with absolute value
        """
        start_time = time.time()
        
        result = abs(value)
        
        return ToolResult(
            result=result,
            confidence=1.0,
            operation="absolute",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"value": value}}
        )


class RoundTool(BaseTool):
    """Tool for rounding numbers."""
    
    def __init__(self):
        super().__init__(name="tool_round", deterministic=True)
    
    def execute(self, value: Union[int, float], decimals: int = 0) -> ToolResult:
        """
        Round a number to specified decimal places.
        
        Args:
            value: Number to round
            decimals: Number of decimal places
            
        Returns:
            ToolResult with rounded value
        """
        start_time = time.time()
        
        result = round(value, decimals)
        
        return ToolResult(
            result=result,
            confidence=1.0,
            operation="round",
            deterministic=True,
            execution_time=time.time() - start_time,
            metadata={"inputs": {"value": value, "decimals": decimals}}
        )

class StoreTool(BaseTool):
    """Tool for storing a value (identity function)."""

    def __init__(self):
        super().__init__(name="tool_store", deterministic=True)

    def execute(self, value: Union[int, float, str, Any]) -> ToolResult:
        """Store a value directly into a variable."""
        return ToolResult(
            result=value,
            confidence=1.0,
            operation="store",
            deterministic=True,
            execution_time=0.0,
            metadata={"inputs": {"value": value}}
        )

# Factory functions for backwards compatibility
def tool_add(a: Union[int, float], b: Union[int, float]) -> Dict[str, Any]:
    """Add two numbers."""
    tool = AddTool()
    return tool(a=a, b=b)


def tool_subtract(a: Union[int, float], b: Union[int, float]) -> Dict[str, Any]:
    """Subtract b from a."""
    tool = SubtractTool()
    return tool(a=a, b=b)


def tool_multiply(a: Union[int, float], b: Union[int, float]) -> Dict[str, Any]:
    """Multiply two numbers."""
    tool = MultiplyTool()
    return tool(a=a, b=b)


def tool_divide(a: Union[int, float], b: Union[int, float]) -> Dict[str, Any]:
    """Divide a by b."""
    tool = DivideTool()
    return tool(a=a, b=b)


# Collection of all deterministic tools
DETERMINISTIC_TOOLS = {
    "tool_add": AddTool(),
    "tool_subtract": SubtractTool(),
    "tool_multiply": MultiplyTool(),
    "tool_divide": DivideTool(),
    "tool_power": PowerTool(),
    "tool_modulo": ModuloTool(),
    "tool_abs": AbsoluteTool(),
    "tool_round": RoundTool(),
    "tool_store": StoreTool(),
}

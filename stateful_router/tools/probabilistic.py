"""
Probabilistic tools for the Enhanced Stateful Router.

These tools involve some uncertainty and return confidence scores < 1.0.
They include information extraction, comparison, and search operations.
"""

import re
import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .base import BaseTool, ToolResult
from ..config.settings import settings
from ..config.prompts import TOOL_PARSER_PROMPT


class ParseTool(BaseTool):
    """Tool for extracting information from text."""
    
    def __init__(self, model_client=None):
        super().__init__(name="tool_parse", deterministic=False)
        self.model_client = model_client
        self.model_name = settings.TOOL_PARSER_MODEL
        self.parse_attempts = settings.PARSE_ATTEMPTS
    
    def execute(self, context: str, query: str) -> ToolResult:
        """
        Extract information from context based on query.
        
        Args:
            context: Text to extract from
            query: What to extract
            
        Returns:
            ToolResult with extracted information
        """
        start_time = time.time()
        
        # First try rule-based extraction
        rule_result = self._rule_based_extraction(context, query)
        if rule_result["confidence"] > 0.9:
            return ToolResult(
                result=rule_result["result"],
                confidence=rule_result["confidence"],
                operation="parse",
                deterministic=False,
                execution_time=time.time() - start_time,
                metadata={"method": "rule-based", "context_length": len(context)}
            )
        
        # If model client available, use LLM extraction
        if self.model_client:
            llm_result = self._llm_extraction(context, query)
            return ToolResult(
                result=llm_result["result"],
                confidence=llm_result["confidence"],
                operation="parse",
                deterministic=False,
                execution_time=time.time() - start_time,
                metadata={"method": "llm", "attempts": llm_result.get("attempts", 1)}
            )
        
        # Fallback to rule-based result
        return ToolResult(
            result=rule_result["result"],
            confidence=rule_result["confidence"],
            operation="parse",
            deterministic=False,
            execution_time=time.time() - start_time,
            metadata={"method": "fallback"}
        )
    
    def _rule_based_extraction(self, context: str, query: str) -> Dict[str, Any]:
        """
        Try to extract information using rules and patterns.
        
        Args:
            context: Text to extract from
            query: What to extract
            
        Returns:
            Dictionary with result and confidence
        """
        context_lower = context.lower()
        query_lower = query.lower()
        
        # Look for numbers
        if any(word in query_lower for word in ["how many", "number", "count", "quantity"]):
            # Find numbers in context
            numbers = re.findall(r'\b(\d+\.?\d*)\b', context)
            if numbers:
                # Return the first number found
                try:
                    num = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                    return {"result": num, "confidence": 0.85}
                except ValueError:
                    pass
        
        # Look for specific patterns
        if "name" in query_lower:
            # Try to find capitalized words (likely names)
            names = re.findall(r'\b[A-Z][a-z]+\b', context)
            if names:
                return {"result": names[0], "confidence": 0.75}
        
        # Look for yes/no questions
        if any(word in query_lower for word in ["is", "are", "was", "were", "does", "do"]):
            if any(word in context_lower for word in ["yes", "true", "correct"]):
                return {"result": "true", "confidence": 0.7}
            elif any(word in context_lower for word in ["no", "false", "incorrect"]):
                return {"result": "false", "confidence": 0.7}
        
        # Try to find exact query match in context
        if query_lower in context_lower:
            # Extract surrounding context
            idx = context_lower.index(query_lower)
            start = max(0, idx - 20)
            end = min(len(context), idx + len(query) + 20)
            snippet = context[start:end].strip()
            return {"result": snippet, "confidence": 0.5}
        
        return {"result": "NOT_FOUND", "confidence": 0.3}
    
    def _llm_extraction(self, context: str, query: str) -> Dict[str, Any]:
        """
        Use LLM for extraction with multiple attempts for confidence.
        
        Args:
            context: Text to extract from
            query: What to extract
            
        Returns:
            Dictionary with result and confidence
        """
        extractions = []
        
        for i in range(self.parse_attempts):
            try:
                prompt = TOOL_PARSER_PROMPT.format(context=context, query=query)
                
                if "gpt" in self.model_name.lower():
                    response = self.model_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=settings.PARSER_TEMPERATURE * (1 + 0.1 * i),  # Slight variation
                        max_tokens=100
                    )
                    extraction = response.choices[0].message.content.strip()
                elif "claude" in self.model_name.lower():
                    response = self.model_client.messages.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=settings.PARSER_TEMPERATURE * (1 + 0.1 * i),
                        max_tokens=100
                    )
                    extraction = response.content[0].text.strip()
                else:
                    extraction = "NOT_FOUND"
                
                extractions.append(extraction)
                
            except Exception as e:
                logger.warning(f"LLM extraction attempt {i+1} failed: {e}")
                extractions.append("NOT_FOUND")
        
        # Calculate confidence based on agreement
        if all(e == extractions[0] for e in extractions):
            confidence = 0.99  # High confidence if all agree
        else:
            # Count most common extraction
            from collections import Counter
            counter = Counter(extractions)
            most_common = counter.most_common(1)[0]
            confidence = most_common[1] / len(extractions)
        
        # Get the most common extraction
        result = max(set(extractions), key=extractions.count)
        
        # Try to parse as number if applicable
        try:
            if '.' in result:
                result = float(result)
            else:
                result = int(result)
        except (ValueError, AttributeError):
            pass  # Keep as string
        
        return {
            "result": result,
            "confidence": confidence,
            "attempts": len(extractions)
        }


class CompareTool(BaseTool):
    """Tool for comparing items."""
    
    def __init__(self, model_client=None):
        super().__init__(name="tool_compare", deterministic=False)
        self.model_client = model_client
    
    def execute(self, item1: Any, item2: Any, attribute: str) -> ToolResult:
        """
        Compare two items on a given attribute.
        
        Args:
            item1: First item
            item2: Second item
            attribute: What to compare
            
        Returns:
            ToolResult with comparison result
        """
        start_time = time.time()
        
        # Try numeric comparison
        if isinstance(item1, (int, float)) and isinstance(item2, (int, float)):
            if item1 > item2:
                result = "greater"
            elif item1 < item2:
                result = "less"
            else:
                result = "equal"
            
            return ToolResult(
                result=result,
                confidence=1.0,  # Numeric comparison is deterministic
                operation="comparison",
                deterministic=True,
                execution_time=time.time() - start_time,
                metadata={"type": "numeric", "item1": item1, "item2": item2}
            )
        
        # Try string comparison
        str1 = str(item1).lower()
        str2 = str(item2).lower()
        
        attribute_lower = attribute.lower()
        
        # Length comparison
        if "length" in attribute_lower or "size" in attribute_lower:
            if len(str1) > len(str2):
                result = "greater"
            elif len(str1) < len(str2):
                result = "less"
            else:
                result = "equal"
            
            return ToolResult(
                result=result,
                confidence=0.95,
                operation="comparison",
                deterministic=False,
                execution_time=time.time() - start_time,
                metadata={"type": "length", "len1": len(str1), "len2": len(str2)}
            )
        
        # Alphabetical comparison
        if "alphabetical" in attribute_lower or "lexical" in attribute_lower:
            if str1 > str2:
                result = "greater"
            elif str1 < str2:
                result = "less"
            else:
                result = "equal"
            
            return ToolResult(
                result=result,
                confidence=0.95,
                operation="comparison",
                deterministic=False,
                execution_time=time.time() - start_time,
                metadata={"type": "alphabetical"}
            )
        
        # Default string equality
        result = "equal" if str1 == str2 else "different"
        
        return ToolResult(
            result=result,
            confidence=0.7,
            operation="comparison",
            deterministic=False,
            execution_time=time.time() - start_time,
            metadata={"type": "equality"}
        )


class SearchTool(BaseTool):
    """Tool for searching within text or lists."""
    
    def __init__(self):
        super().__init__(name="tool_search", deterministic=False)
    
    def execute(self, haystack: Union[str, List], needle: str) -> ToolResult:
        """
        Search for needle in haystack.
        
        Args:
            haystack: Text or list to search in
            needle: What to search for
            
        Returns:
            ToolResult with search result
        """
        start_time = time.time()
        
        if isinstance(haystack, str):
            # Text search
            needle_lower = needle.lower()
            haystack_lower = haystack.lower()
            
            if needle_lower in haystack_lower:
                # Find the position
                position = haystack_lower.index(needle_lower)
                
                # Extract surrounding context
                context_start = max(0, position - 50)
                context_end = min(len(haystack), position + len(needle) + 50)
                context = haystack[context_start:context_end]
                
                return ToolResult(
                    result={
                        "found": True,
                        "position": position,
                        "context": context
                    },
                    confidence=0.95,
                    operation="search",
                    deterministic=False,
                    execution_time=time.time() - start_time,
                    metadata={"type": "text", "haystack_length": len(haystack)}
                )
            else:
                return ToolResult(
                    result={"found": False},
                    confidence=0.95,
                    operation="search",
                    deterministic=False,
                    execution_time=time.time() - start_time,
                    metadata={"type": "text"}
                )
        
        elif isinstance(haystack, list):
            # List search
            needle_lower = needle.lower()
            
            for i, item in enumerate(haystack):
                if needle_lower in str(item).lower():
                    return ToolResult(
                        result={
                            "found": True,
                            "index": i,
                            "item": item
                        },
                        confidence=0.95,
                        operation="search",
                        deterministic=False,
                        execution_time=time.time() - start_time,
                        metadata={"type": "list", "list_length": len(haystack)}
                    )
            
            return ToolResult(
                result={"found": False},
                confidence=0.95,
                operation="search",
                deterministic=False,
                execution_time=time.time() - start_time,
                metadata={"type": "list"}
            )
        
        else:
            return ToolResult(
                result=None,
                confidence=0.0,
                operation="search",
                deterministic=False,
                execution_time=time.time() - start_time,
                error=f"Invalid haystack type: {type(haystack)}"
            )


# Factory functions for backwards compatibility
def tool_parse(context: str, query: str) -> Dict[str, Any]:
    """Parse information from context."""
    tool = ParseTool()
    return tool(context=context, query=query)


def tool_compare(item1: Any, item2: Any, attribute: str) -> Dict[str, Any]:
    """Compare two items."""
    tool = CompareTool()
    return tool(item1=item1, item2=item2, attribute=attribute)


def tool_search(haystack: Union[str, List], needle: str) -> Dict[str, Any]:
    """Search for needle in haystack."""
    tool = SearchTool()
    return tool(haystack=haystack, needle=needle)


# Collection of all probabilistic tools
PROBABILISTIC_TOOLS = {
    "tool_parse": ParseTool(),
    "tool_compare": CompareTool(),
    "tool_search": SearchTool(),
}

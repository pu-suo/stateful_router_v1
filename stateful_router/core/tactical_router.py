"""
Tactical Router Module for the Enhanced Stateful Router.

This module executes individual steps from the strategic plan, generating
thoughts and actions that are then executed by the appropriate tools.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import settings
from ..config.prompts import (
    TACTICAL_ROUTER_SYSTEM_PROMPT,
    TACTICAL_ROUTER_USER_PROMPT,
    ERROR_RECOVERY_PROMPT
)
from .strategic_planner import PlanStep
from .memory import HierarchicalMemory


@dataclass
class RouterOutput:
    """Output from the tactical router."""
    thought: str
    action: str
    tool_name: str
    tool_args: Dict[str, Any]
    confidence: float = 0.9
    output_var: Optional[str] = None


class TacticalRouter:
    """
    The Tactical Router executes individual steps from the strategic plan.
    
    It generates thoughts explaining reasoning and creates tool calls
    that can be executed and verified.
    """
    
    def __init__(self, model_client=None):
        """
        Initialize the Tactical Router.
        
        Args:
            model_client: Client for interacting with the LLM
        """
        self.model_client = model_client
        self.model_name = settings.TACTICAL_ROUTER_MODEL
        self.temperature = settings.ROUTER_TEMPERATURE
        
        # Patterns for parsing router output
        self.thought_pattern = re.compile(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', re.DOTALL | re.IGNORECASE)
        self.action_pattern = re.compile(r'ACTION:\s*(.+?)(?=$)', re.DOTALL | re.IGNORECASE)
        
        # Pattern for parsing tool calls
        self.tool_call_pattern = re.compile(r'(\w+)\((.*)\)')
        
        logger.info(f"Initialized TacticalRouter with model: {self.model_name}")
    
    def execute_step(self,
                    plan_step: PlanStep,
                    memory: HierarchicalMemory,
                    problem: str,
                    is_final_step: bool = False) -> RouterOutput:
        """
        Execute a single step from the strategic plan.

        Args:
            plan_step: The step to execute
            memory: The hierarchical memory system
            problem: The original problem
            is_final_step: Whether this is the final step in the plan

        Returns:
            RouterOutput containing thought and action
        """
        logger.info(f"Executing step {plan_step.step}: {plan_step.description}")

        # Get current state from memory
        observations = memory.get_observations()
        history = memory.get_history_text()

        # Generate thought and action
        output = self._generate_output(
            problem=problem,
            plan_step=plan_step,
            observations=observations,
            history=history,
            memory=memory,
            is_final_step=is_final_step
        )
        
        logger.debug(f"Generated thought: {output.thought[:100]}...")
        logger.debug(f"Generated action: {output.action}")
        
        return output
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_output(self,
                        problem: str,
                        plan_step: PlanStep,
                        observations: List[Any],
                        history: str,
                        memory: HierarchicalMemory,
                        is_final_step: bool = False) -> RouterOutput:
        """
        Generate thought and action using the LLM.

        Args:
            problem: The original problem
            plan_step: Current step to execute
            observations: Previous observations
            history: Execution history
            memory: Memory system for accessing variables
            is_final_step: Whether this is the final step

        Returns:
            RouterOutput with thought and action
        """
        # Format observations for the prompt
        obs_text = self._format_observations(observations)

        # Get available variables from memory
        variables = memory.get_all_variables()
        var_text = "\n".join([f"{k} = {v}" for k, v in variables.items()]) if variables else "None"

        # Determine suggested output variable
        if plan_step.output_var:
            # Use explicitly specified output var
            suggested_var = plan_step.output_var
        elif is_final_step:
            # Last step should use final_answer
            suggested_var = "final_answer"
        else:
            # Intermediate steps get unique names
            suggested_var = f"var_step_{plan_step.step}"

        # Create the prompt
        user_prompt = TACTICAL_ROUTER_USER_PROMPT.format(
            problem=problem,
            current_step=plan_step.step,
            step_description=plan_step.description,
            expected_tool=plan_step.expected_tool,
            observations=obs_text,
            history=history,
            current_step_output_var=suggested_var
        )

        # Add variables section to the prompt
        user_prompt += f"\n\nAvailable Intermediate Variables:\n{var_text}"
        
        # Call LLM
        response = self._call_llm(
            system_prompt=TACTICAL_ROUTER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature
        )
        
        # Parse the response
        return self._parse_response(response, plan_step.expected_tool)
    
    def _format_observations(self, observations: List[Any]) -> str:
        """
        Format observations for inclusion in the prompt.
        
        Args:
            observations: List of observations from previous steps
            
        Returns:
            Formatted string of observations
        """
        if not observations:
            return "No observations yet."
        
        lines = []
        for i, obs in enumerate(observations, 1):
            if isinstance(obs, dict):
                obs_str = json.dumps(obs, indent=2)
            else:
                obs_str = str(obs)
            lines.append(f"Observation {i}: {obs_str}")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str, expected_tool: str) -> RouterOutput:
        """
        Parse the LLM response to extract thought and action.
        
        Args:
            response: Raw LLM response
            expected_tool: The tool expected by the plan
            
        Returns:
            RouterOutput with parsed components
        """
        # Extract thought
        thought_match = self.thought_pattern.search(response)
        thought = thought_match.group(1).strip() if thought_match else "No explicit thought provided"
        
        # Extract action
        action_match = self.action_pattern.search(response)
        action = action_match.group(1).strip() if action_match else ""
        
        # If no action found, try to find any tool call pattern
        if not action:
            tool_match = self.tool_call_pattern.search(response)
            if tool_match:
                action = tool_match.group(0)
            else:
                # Fallback to expected tool with generic parameters
                logger.warning("No action found in response, using expected tool")
                action = f"{expected_tool}()"
        
        # Parse the tool call
        tool_name, tool_args, output_var = self._parse_tool_call(action)

        # Check if tool matches expectation
        confidence = 0.95
        if tool_name != expected_tool:
            logger.warning(f"Tool mismatch: expected {expected_tool}, got {tool_name}")
            confidence = 0.7

        return RouterOutput(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            confidence=confidence,
            output_var=output_var
        )
    
    def _parse_tool_call(self, action: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """
        Parse a tool call string into name, arguments, and optional output variable.

        Args:
            action: Tool call string like "tool_add(3, 5)" or "tool_add(3, 5) -> var_1"

        Returns:
            Tuple of (tool_name, tool_args, output_var_name)
        """
        # Remove any extra whitespace
        action = action.strip()

        # LLM sometimes generates multiple actions - take only the FIRST line
        if '\n' in action:
            first_line = action.split('\n')[0].strip()
            logger.warning(f"Multiple actions detected, using only first: {first_line}")
            action = first_line

        # Check for variable assignment (e.g., "tool_add(3, 5) -> var_1")
        output_var = None
        if '->' in action:
            parts = action.split('->')
            action = parts[0].strip()
            if len(parts) > 1:
                # Clean output var - take only the first word (before any newline, space, or punctuation)
                raw_var = parts[1].strip()
                # Extract just the variable name (alphanumeric + underscores)
                import re
                var_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', raw_var)
                if var_match:
                    output_var = var_match.group(1)
                else:
                    output_var = raw_var.split()[0] if raw_var else None

                if output_var != raw_var:
                    logger.debug(f"Cleaned output var: '{raw_var}' -> '{output_var}'")

        # Try to match tool call pattern
        match = self.tool_call_pattern.match(action)
        if not match:
            logger.error(f"Could not parse tool call: {action}")
            return "unknown", {}, output_var

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        tool_args = self._parse_arguments(args_str)

        return tool_name, tool_args, output_var
    
    def _parse_arguments(self, args_str: str) -> Dict[str, Any]:
        """
        Parse tool arguments from a string.

        Args:
            args_str: Arguments string like "3, 5" or "context='text', query='question'"

        Returns:
            Dictionary of parsed arguments
        """
        if not args_str.strip():
            return {}

        # Try to identify if arguments are positional or keyword
        if '=' in args_str:
            # Keyword arguments
            args_dict = {}
            # Match keyword=value pairs
            pattern = re.compile(r'(\w+)\s*=\s*(["\']?)(.+?)\2(?:,|$)')
            for match in pattern.finditer(args_str):
                key = match.group(1)
                value = match.group(3)
                args_dict[key] = self._parse_value(value)
            return args_dict
        else:
            # Positional arguments - need to map based on tool
            values = []
            # Split by comma, handling quoted strings
            parts = re.split(r',(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str)
            for part in parts:
                part = part.strip()
                if part:
                    values.append(self._parse_value(part))

            # Helper function to check if a value is numeric or a variable reference
            def is_numeric_or_var(v):
                """Check if value is numeric or a variable that will be resolved to numeric."""
                if isinstance(v, (int, float)):
                    return True
                if isinstance(v, str):
                    # Any identifier-like string could be a variable reference
                    # (e.g., final_answer, var_1, wallet_cost, etc.)
                    return v.replace('_', '').replace('-', '').isalnum()
                return False

            # Map positional arguments based on common tool patterns
            if len(values) == 1:
                return {"value": values[0]}
            elif len(values) == 2:
                # Check if these are numeric arguments (or variable references)
                if all(is_numeric_or_var(v) for v in values):
                    # Arithmetic tool pattern (legacy 2-arg mode)
                    return {"a": values[0], "b": values[1]}
                else:
                    # Parse tool pattern
                    return {"context": str(values[0]), "query": str(values[1])}
            elif len(values) == 3:
                # Check if all numeric/vars - if so, it's arithmetic with 3 operands
                if all(is_numeric_or_var(v) for v in values):
                    # Variable-length arithmetic: pass as list that will be unpacked
                    return {"values": values}
                else:
                    # Comparison tool
                    return {"item1": str(values[0]), "item2": str(values[1]), "attribute": str(values[2])}
            elif len(values) > 3:
                # 4+ values - assume arithmetic operation
                if all(is_numeric_or_var(v) for v in values):
                    return {"values": values}
                else:
                    # Generic numbered arguments
                    return {f"arg{i}": v for i, v in enumerate(values)}
    
    def _parse_value(self, value_str: str) -> Any:
        """
        Parse a string value into appropriate Python type.

        Args:
            value_str: String representation of value

        Returns:
            Parsed value
        """
        value_str = value_str.strip()

        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Try to parse as boolean
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False

        # Keep as string (might be a variable name)
        return value_str

    def resolve_variables(self, tool_args: Dict[str, Any], memory: HierarchicalMemory) -> Dict[str, Any]:
        """
        Resolve variable references in tool arguments.

        Args:
            tool_args: Dictionary of tool arguments that may contain variable names
            memory: Memory system to look up variable values

        Returns:
            Dictionary with variables resolved to their values
        """
        resolved_args = {}
        all_variables = memory.get_all_variables()

        for key, value in tool_args.items():
            if isinstance(value, str):
                # Check if this string is a variable name in memory
                if value in all_variables:
                    resolved_value = all_variables[value]
                    resolved_args[key] = resolved_value
                    logger.debug(f"Resolved variable {value} to {resolved_value}")
                else:
                    # Not a variable, keep original
                    resolved_args[key] = value
            elif isinstance(value, (list, tuple)):
                # Resolve each item in the list
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and item in all_variables:
                        resolved_list.append(all_variables[item])
                        logger.debug(f"Resolved variable {item} to {all_variables[item]}")
                    else:
                        resolved_list.append(item)
                resolved_args[key] = resolved_list
            else:
                resolved_args[key] = value

        return resolved_args
    
    def generate_alternative(self, 
                           plan_step: PlanStep,
                           failed_output: RouterOutput,
                           failure_reason: str,
                           memory: HierarchicalMemory,
                           problem: str) -> RouterOutput:
        """
        Generate an alternative approach when a step fails verification.
        
        Args:
            plan_step: The step that failed
            failed_output: The output that failed verification
            failure_reason: Why it failed
            memory: The hierarchical memory
            problem: The original problem
            
        Returns:
            Alternative RouterOutput
        """
        logger.info(f"Generating alternative for step {plan_step.step}")
        
        # Create error recovery prompt
        context = {
            "problem": problem,
            "step_description": plan_step.description,
            "previous_thought": failed_output.thought,
            "previous_action": failed_output.action,
            "observations": memory.get_observations()
        }
        
        user_prompt = ERROR_RECOVERY_PROMPT.format(
            failed_step=json.dumps({
                "step": plan_step.step,
                "description": plan_step.description,
                "thought": failed_output.thought,
                "action": failed_output.action
            }, indent=2),
            failure_reason=failure_reason,
            context=json.dumps(context, indent=2)
        )
        
        # Call LLM for alternative
        response = self._call_llm(
            system_prompt="Generate an alternative approach. Respond with ALTERNATIVE_THOUGHT and ALTERNATIVE_ACTION.",
            user_prompt=user_prompt,
            temperature=self.temperature + 0.1  # Slightly higher temperature for creativity
        )
        
        # Parse alternative
        thought_pattern = re.compile(r'ALTERNATIVE_THOUGHT:\s*(.+?)(?=ALTERNATIVE_ACTION:|$)', re.DOTALL | re.IGNORECASE)
        action_pattern = re.compile(r'ALTERNATIVE_ACTION:\s*(.+?)(?=$)', re.DOTALL | re.IGNORECASE)
        
        thought_match = thought_pattern.search(response)
        action_match = action_pattern.search(response)
        
        thought = thought_match.group(1).strip() if thought_match else "Retrying with alternative approach"
        action = action_match.group(1).strip() if action_match else failed_output.action

        tool_name, tool_args, output_var = self._parse_tool_call(action)

        return RouterOutput(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            confidence=0.7,  # Lower confidence for alternatives
            output_var=output_var
        )
    
    def _call_llm(self, system_prompt: str, user_prompt: str, 
                 temperature: float = None) -> str:
        """
        Call the LLM with the given prompts.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature override
            
        Returns:
            LLM response as string
        """
        if temperature is None:
            temperature = self.temperature
        
        if self.model_client is None:
            # Return mock response for testing
            logger.warning("No model client provided, returning mock output")
            return "THOUGHT: This is a mock thought for testing.\nACTION: tool_parse(context='test', query='test')"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if "gpt" in self.model_name.lower():
                # OpenAI format
                response = self.model_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content
            elif "claude" in self.model_name.lower():
                # Anthropic format
                response = self.model_client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=500
                )
                return response.content[0].text
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

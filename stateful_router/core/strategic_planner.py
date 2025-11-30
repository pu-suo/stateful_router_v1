"""
Strategic Planner Module for the Enhanced Stateful Router.

This module is responsible for high-level problem analysis and planning.
It classifies problems, selects appropriate strategies, and creates detailed
execution plans that can be verified and executed by the tactical router.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import settings
from ..config.prompts import (
    STRATEGIC_PLANNER_SYSTEM_PROMPT,
    STRATEGIC_PLANNER_USER_PROMPT,
    PROBLEM_CLASSIFICATION_PROMPT,
    PARALLEL_STEP_DETECTION_PROMPT
)


class Strategy(Enum):
    """Available problem-solving strategies."""
    ARITHMETIC_SEQUENCE = "ARITHMETIC_SEQUENCE"
    LOGICAL_DEDUCTION = "LOGICAL_DEDUCTION"
    INFORMATION_EXTRACTION = "INFORMATION_EXTRACTION"
    COMPARISON = "COMPARISON"
    ALGEBRAIC = "ALGEBRAIC"
    HYBRID = "HYBRID"


@dataclass
class PlanStep:
    """Represents a single step in the execution plan."""
    step: int
    description: str
    expected_tool: str
    depends_on: List[int] = None
    output_var: Optional[str] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "expected_tool": self.expected_tool,
            "depends_on": self.depends_on,
            "output_var": self.output_var
        }


@dataclass
class StrategicPlan:
    """Complete strategic plan for solving a problem."""
    strategy: Strategy
    plan: List[PlanStep]
    confidence: float
    reasoning: str
    fallback_strategy: Optional[Strategy] = None
    parallel_groups: Optional[List[List[int]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "plan": [step.to_dict() for step in self.plan],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "fallback_strategy": self.fallback_strategy.value if self.fallback_strategy else None,
            "parallel_groups": self.parallel_groups
        }


class StrategicPlanner:
    """
    The Strategic Planner analyzes problems and creates high-level execution plans.
    
    This component performs problem classification, strategy selection, and 
    generates detailed step-by-step plans that can be executed and verified.
    """
    
    def __init__(self, model_client=None):
        """
        Initialize the Strategic Planner.
        
        Args:
            model_client: Client for interacting with the LLM (OpenAI/Anthropic)
        """
        self.model_client = model_client
        self.model_name = settings.STRATEGIC_PLANNER_MODEL
        self.temperature = settings.PLANNER_TEMPERATURE
        
        # Strategy patterns for problem classification
        self.strategy_patterns = {
            Strategy.ARITHMETIC_SEQUENCE: [
                r'\b(add|subtract|multiply|divide|sum|total|difference|product)\b',
                r'\b(how many|how much|calculate|compute)\b',
                r'\b\d+\s*(plus|minus|times|divided by)\s*\d+\b'
            ],
            Strategy.INFORMATION_EXTRACTION: [
                r'\b(what|who|when|where|which|how)\b',
                r'\b(extract|find|identify|locate|determine)\b',
                r'\b(mentioned|stated|described|said)\b'
            ],
            Strategy.COMPARISON: [
                r'\b(compare|versus|vs|better|worse|more|less|greater|smaller)\b',
                r'\b(difference between|similar|different)\b',
                r'\b(rank|order|sort)\b'
            ],
            Strategy.LOGICAL_DEDUCTION: [
                r'\b(if|then|therefore|because|implies|concludes)\b',
                r'\b(all|some|none|every|any)\b',
                r'\b(must be|cannot be|proves)\b'
            ],
            Strategy.ALGEBRAIC: [
                r'\b(solve for|equation|variable|unknown)\b',
                r'\b[x|y|z]\s*=',
                r'\b(formula|expression)\b'
            ]
        }
        
        logger.info(f"Initialized StrategicPlanner with model: {self.model_name}")
    
    def plan(self, problem: str, previous_attempts: Optional[List[Dict]] = None) -> StrategicPlan:
        """
        Create a strategic plan for solving the given problem.
        
        Args:
            problem: The problem to solve
            previous_attempts: Previous failed attempts (for replanning)
            
        Returns:
            StrategicPlan object containing the strategy and execution steps
        """
        logger.info(f"Creating strategic plan for problem: {problem[:100]}...")
        
        # First classify the problem
        problem_type = self._classify_problem(problem)
        logger.debug(f"Classified problem as: {problem_type}")
        
        # Generate the plan using LLM
        plan_json = self._generate_plan(problem, problem_type, previous_attempts)
        
        # Parse and validate the plan
        strategic_plan = self._parse_plan(plan_json)
        
        # Identify parallel execution opportunities
        strategic_plan.parallel_groups = self._identify_parallel_steps(strategic_plan)
        
        logger.info(f"Created plan with strategy: {strategic_plan.strategy.value}, "
                   f"steps: {len(strategic_plan.plan)}, confidence: {strategic_plan.confidence:.2f}")
        
        return strategic_plan
    
    def _classify_problem(self, problem: str) -> Strategy:
        """
        Classify the problem type using pattern matching and LLM.
        
        Args:
            problem: The problem text
            
        Returns:
            The most appropriate strategy
        """
        # First try pattern matching for quick classification
        problem_lower = problem.lower()
        pattern_scores = {}
        
        for strategy, patterns in self.strategy_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, problem_lower))
            if score > 0:
                pattern_scores[strategy] = score
        
        # If clear winner from patterns, use it
        if pattern_scores:
            best_strategy = max(pattern_scores, key=pattern_scores.get)
            if pattern_scores[best_strategy] >= 2:
                logger.debug(f"Pattern matching selected strategy: {best_strategy.value}")
                return best_strategy
        
        # Otherwise, use LLM for classification
        if self.model_client:
            try:
                response = self._call_llm(
                    system_prompt="Classify the problem type. Respond with only: ARITHMETIC, EXTRACTION, LOGICAL, COMPARISON, ALGEBRAIC, or MIXED",
                    user_prompt=PROBLEM_CLASSIFICATION_PROMPT.format(problem=problem),
                    temperature=0.1
                )
                
                classification = response.strip().upper()
                strategy_map = {
                    "ARITHMETIC": Strategy.ARITHMETIC_SEQUENCE,
                    "EXTRACTION": Strategy.INFORMATION_EXTRACTION,
                    "LOGICAL": Strategy.LOGICAL_DEDUCTION,
                    "COMPARISON": Strategy.COMPARISON,
                    "ALGEBRAIC": Strategy.ALGEBRAIC,
                    "MIXED": Strategy.HYBRID
                }
                
                if classification in strategy_map:
                    return strategy_map[classification]
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
        
        # Default to HYBRID if uncertain
        return Strategy.HYBRID
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_plan(self, problem: str, problem_type: Strategy, 
                      previous_attempts: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate a detailed execution plan using the LLM.
        
        Args:
            problem: The problem to solve
            problem_type: The classified problem type
            previous_attempts: Previous failed attempts
            
        Returns:
            JSON dictionary containing the plan
        """
        # Prepare the prompt
        user_prompt = STRATEGIC_PLANNER_USER_PROMPT.format(
            problem=problem,
            previous_attempts=json.dumps(previous_attempts, indent=2) if previous_attempts else "None"
        )
        
        # Add hint about expected strategy
        user_prompt += f"\n\nSuggested strategy based on analysis: {problem_type.value}"
        
        # Call LLM
        response = self._call_llm(
            system_prompt=STRATEGIC_PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
            response_format="json"
        )
        
        # Parse JSON response
        try:
            plan_json = json.loads(response)
            return plan_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse plan from LLM response: {response}")
    
    def _parse_plan(self, plan_json: Dict[str, Any]) -> StrategicPlan:
        """
        Parse and validate the plan JSON into a StrategicPlan object.
        
        Args:
            plan_json: Raw JSON from LLM
            
        Returns:
            Validated StrategicPlan object
        """
        # Parse strategy
        strategy_str = plan_json.get("strategy", "HYBRID")
        try:
            strategy = Strategy[strategy_str]
        except KeyError:
            logger.warning(f"Unknown strategy: {strategy_str}, defaulting to HYBRID")
            strategy = Strategy.HYBRID
        
        # Parse plan steps
        plan_steps = []
        for step_dict in plan_json.get("plan", []):
            step = PlanStep(
                step=step_dict.get("step", len(plan_steps) + 1),
                description=step_dict.get("description", ""),
                expected_tool=step_dict.get("expected_tool", "tool_parse"),
                depends_on=step_dict.get("depends_on", [])
            )
            plan_steps.append(step)
        
        # Validate and fix dependencies
        plan_steps = self._validate_dependencies(plan_steps)
        
        # Parse other fields
        confidence = float(plan_json.get("confidence", 0.5))
        reasoning = plan_json.get("reasoning", "No reasoning provided")
        
        # Parse fallback strategy
        fallback_str = plan_json.get("fallback_strategy")
        fallback_strategy = None
        if fallback_str:
            try:
                fallback_strategy = Strategy[fallback_str]
            except KeyError:
                pass
        
        return StrategicPlan(
            strategy=strategy,
            plan=plan_steps,
            confidence=confidence,
            reasoning=reasoning,
            fallback_strategy=fallback_strategy
        )
    
    def _validate_dependencies(self, plan_steps: List[PlanStep]) -> List[PlanStep]:
        """
        Validate and fix step dependencies to ensure they're valid.
        
        Args:
            plan_steps: List of plan steps
            
        Returns:
            List of plan steps with validated dependencies
        """
        step_numbers = {step.step for step in plan_steps}
        
        for step in plan_steps:
            # Remove invalid dependencies
            step.depends_on = [dep for dep in step.depends_on if dep in step_numbers and dep < step.step]
            
            # If no explicit dependencies, assume depends on previous step (except first)
            if not step.depends_on and step.step > 1:
                # Check if this seems like an independent extraction
                if "extract" in step.description.lower() and step.step <= 3:
                    # Early extraction steps might be independent
                    step.depends_on = []
                else:
                    # Otherwise, depends on previous step
                    step.depends_on = [step.step - 1]
        
        return plan_steps
    
    def _identify_parallel_steps(self, plan: StrategicPlan) -> List[List[int]]:
        """
        Identify which steps can be executed in parallel.
        
        Args:
            plan: The strategic plan
            
        Returns:
            List of step groups that can run in parallel
        """
        if not settings.ENABLE_PARALLEL_EXECUTION:
            # Return sequential execution
            return [[step.step] for step in plan.plan]
        
        # Build dependency graph
        dependencies = {step.step: set(step.depends_on) for step in plan.plan}
        
        # Group steps by dependency level
        groups = []
        processed = set()
        
        while len(processed) < len(plan.plan):
            # Find steps that can be executed now
            current_group = []
            for step in plan.plan:
                if step.step not in processed:
                    # Check if all dependencies are processed
                    if dependencies[step.step].issubset(processed):
                        current_group.append(step.step)
            
            if not current_group:
                # Avoid infinite loop if there's a dependency cycle
                logger.warning("Dependency cycle detected, falling back to sequential execution")
                remaining = [s.step for s in plan.plan if s.step not in processed]
                groups.extend([[s] for s in remaining])
                break
            
            groups.append(current_group)
            processed.update(current_group)
        
        logger.debug(f"Identified parallel groups: {groups}")
        return groups
    
    def _call_llm(self, system_prompt: str, user_prompt: str, 
                 temperature: float = None, response_format: str = "text") -> str:
        """
        Call the LLM with the given prompts.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature override
            response_format: "text" or "json"
            
        Returns:
            LLM response as string
        """
        if temperature is None:
            temperature = self.temperature
        
        if self.model_client is None:
            # Return a mock response for testing
            logger.warning("No model client provided, returning mock plan")
            return json.dumps({
                "strategy": "ARITHMETIC_SEQUENCE",
                "plan": [
                    {"step": 1, "description": "Extract first number", "expected_tool": "tool_parse", "depends_on": []},
                    {"step": 2, "description": "Extract second number", "expected_tool": "tool_parse", "depends_on": []},
                    {"step": 3, "description": "Add numbers", "expected_tool": "tool_add", "depends_on": [1, 2]}
                ],
                "confidence": 0.9,
                "reasoning": "Mock plan for testing",
                "fallback_strategy": "INFORMATION_EXTRACTION"
            })
        
        # Use actual model client (implementation depends on whether using OpenAI or Anthropic)
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
                    response_format={"type": "json_object"} if response_format == "json" else None
                )
                return response.choices[0].message.content
            elif "claude" in self.model_name.lower():
                # Anthropic format
                response = self.model_client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.content[0].text
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def replan(self, problem: str, previous_plan: StrategicPlan, 
              failure_reason: str) -> StrategicPlan:
        """
        Create a new plan after a previous plan failed.
        
        Args:
            problem: The original problem
            previous_plan: The plan that failed
            failure_reason: Why the previous plan failed
            
        Returns:
            New strategic plan
        """
        logger.info(f"Replanning due to: {failure_reason}")
        
        previous_attempt = {
            "strategy": previous_plan.strategy.value,
            "failure_reason": failure_reason,
            "failed_steps": len(previous_plan.plan)
        }
        
        # Try fallback strategy if available
        if previous_plan.fallback_strategy:
            logger.info(f"Trying fallback strategy: {previous_plan.fallback_strategy.value}")
            # Force the fallback strategy in classification
            self.strategy_patterns = {
                previous_plan.fallback_strategy: [r'.*']  # Match everything
            }
        
        return self.plan(problem, previous_attempts=[previous_attempt])

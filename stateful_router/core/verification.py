"""
Verification Module for the Enhanced Stateful Router.

This module provides continuous verification of reasoning steps,
checking consistency between thoughts, actions, and observations,
and tracking confidence throughout the execution.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from loguru import logger

from ..config.settings import settings
from ..config.prompts import (
    VERIFICATION_THOUGHT_ACTION_PROMPT,
    VERIFICATION_OBSERVATION_PROMPT,
    CONFIDENCE_CALIBRATION_PROMPT
)
from .memory import VerificationStatus
from .strategic_planner import PlanStep
from .tactical_router import RouterOutput


@dataclass
class VerificationResult:
    """Result of a verification check."""
    score: float
    status: VerificationStatus
    checks: Dict[str, float]
    issues: List[str]
    suggestions: List[str]


class VerificationModule:
    """
    The Verification Module checks consistency and tracks confidence.
    
    It verifies that:
    1. Actions match thoughts (thought-action alignment)
    2. Observations are plausible given actions (observation validity)
    3. Execution follows the strategic plan (plan adherence)
    4. Confidence is properly calibrated
    """
    
    def __init__(self, model_client=None, consistency_threshold: float = None):
        """
        Initialize the Verification Module.
        
        Args:
            model_client: Client for LLM-based verification
            consistency_threshold: Threshold for passing verification
        """
        self.model_client = model_client
        self.consistency_threshold = consistency_threshold or settings.VERIFICATION_THRESHOLD
        
        # Initialize embedding model for similarity checks
        self.embedding_model = None
        if settings.EMBEDDING_MODEL:
            try:
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"Initialized embedding model: {settings.EMBEDDING_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
        
        # Verification history for pattern detection
        self.verification_history: List[VerificationResult] = []
        
        # Tool output patterns for validation
        self.tool_patterns = {
            "tool_add": r'^-?\d+\.?\d*$',
            "tool_subtract": r'^-?\d+\.?\d*$',
            "tool_multiply": r'^-?\d+\.?\d*$',
            "tool_divide": r'^-?\d+\.?\d*$',
            "tool_parse": r'.+',  # Any non-empty string
            "tool_compare": r'^(greater|less|equal|true|false)',
        }
        
        logger.info(f"Initialized VerificationModule with threshold: {self.consistency_threshold}")
    
    def verify_step(self,
                   router_output: RouterOutput,
                   observation: Any,
                   plan_step: PlanStep) -> VerificationResult:
        """
        Verify a single execution step.
        
        Args:
            router_output: Output from the tactical router
            observation: Observation from tool execution
            plan_step: The planned step
            
        Returns:
            VerificationResult with scores and status
        """
        logger.debug(f"Verifying step {plan_step.step}")
        
        checks = {}
        issues = []
        suggestions = []
        
        # Check 1: Thought-Action Consistency
        thought_action_score = self.check_thought_action_consistency(
            router_output.thought,
            router_output.action
        )
        checks["thought_action_consistency"] = thought_action_score
        
        if thought_action_score < 0.7:
            issues.append("Thought and action are inconsistent")
            suggestions.append("Ensure the action directly follows from the thought")
        
        # Check 2: Observation Plausibility
        observation_score = self.check_observation_plausibility(
            router_output.tool_name,
            router_output.tool_args,
            observation
        )
        checks["observation_plausibility"] = observation_score
        
        if observation_score < 0.7:
            issues.append("Observation seems implausible for the action")
            suggestions.append("Verify tool execution and parameters")
        
        # Check 3: Plan Adherence
        plan_score = self.check_plan_adherence(
            router_output.tool_name,
            plan_step.expected_tool
        )
        checks["plan_adherence"] = plan_score
        
        if plan_score < 0.8:
            issues.append(f"Deviated from plan (expected {plan_step.expected_tool})")
            suggestions.append("Consider if deviation is justified or revert to plan")
        
        # Check 4: Output Format Validity
        format_score = self.check_output_format(
            router_output.tool_name,
            observation
        )
        checks["output_format"] = format_score
        
        if format_score < 0.9:
            issues.append("Output format doesn't match expected pattern")
            suggestions.append("Check tool implementation and parameters")
        
        # Calculate overall score
        overall_score = np.mean(list(checks.values()))
        
        # Determine status
        if overall_score >= self.consistency_threshold:
            status = VerificationStatus.PASS
        elif overall_score >= 0.5:
            status = VerificationStatus.REVIEW
        else:
            status = VerificationStatus.FAIL
        
        result = VerificationResult(
            score=overall_score,
            status=status,
            checks=checks,
            issues=issues,
            suggestions=suggestions
        )
        
        # Add to history
        self.verification_history.append(result)
        
        logger.info(f"Verification result: {status.value} (score: {overall_score:.2f})")
        
        return result
    
    def check_thought_action_consistency(self, thought: str, action: str) -> float:
        """
        Check if the action is consistent with the thought.
        
        Args:
            thought: The reasoning thought
            action: The action taken
            
        Returns:
            Consistency score (0-1)
        """
        # First try rule-based checking
        score = self._rule_based_consistency(thought, action)
        
        # If embedding model available, use semantic similarity
        if self.embedding_model and score < 0.9:
            embedding_score = self._embedding_consistency(thought, action)
            # Weighted average
            score = 0.3 * score + 0.7 * embedding_score
        
        # If LLM available and score is borderline, use LLM for verification
        if self.model_client and 0.5 < score < 0.8:
            llm_score = self._llm_consistency(thought, action)
            # Weighted average
            score = 0.2 * score + 0.8 * llm_score
        
        return min(1.0, max(0.0, score))
    
    def _rule_based_consistency(self, thought: str, action: str) -> float:
        """
        Rule-based consistency checking.
        
        Args:
            thought: The reasoning thought
            action: The action taken
            
        Returns:
            Consistency score
        """
        thought_lower = thought.lower()
        action_lower = action.lower()
        
        # Check for keyword matches
        score = 0.5  # Base score
        
        # Arithmetic operations
        if "add" in thought_lower and "tool_add" in action_lower:
            score = 1.0
        elif "subtract" in thought_lower and "tool_subtract" in action_lower:
            score = 1.0
        elif "multiply" in thought_lower and "tool_multiply" in action_lower:
            score = 1.0
        elif "divide" in thought_lower and "tool_divide" in action_lower:
            score = 1.0
        
        # Extraction operations
        elif "extract" in thought_lower and "tool_parse" in action_lower:
            score = 0.95
        elif "find" in thought_lower and "tool_parse" in action_lower:
            score = 0.9
        elif "identify" in thought_lower and "tool_parse" in action_lower:
            score = 0.9
        
        # Comparison operations
        elif "compare" in thought_lower and "tool_compare" in action_lower:
            score = 0.95
        elif ("greater" in thought_lower or "less" in thought_lower) and "tool_compare" in action_lower:
            score = 0.9
        
        # Check for numbers mentioned in thought appearing in action
        thought_numbers = re.findall(r'\b\d+\.?\d*\b', thought)
        action_numbers = re.findall(r'\b\d+\.?\d*\b', action)
        
        if thought_numbers and action_numbers:
            matching_numbers = set(thought_numbers) & set(action_numbers)
            if matching_numbers:
                score = min(1.0, score + 0.1 * len(matching_numbers))
        
        return score
    
    def _embedding_consistency(self, thought: str, action: str) -> float:
        """
        Check consistency using embedding similarity.
        
        Args:
            thought: The reasoning thought
            action: The action taken
            
        Returns:
            Similarity score
        """
        if not self.embedding_model:
            return 0.5
        
        try:
            # Get embeddings
            thought_embedding = self.embedding_model.encode([thought])
            action_embedding = self.embedding_model.encode([action])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(thought_embedding, action_embedding)[0][0]
            
            # Scale to 0-1 range (similarity is already in [-1, 1], but typically positive)
            return max(0.0, similarity)
        except Exception as e:
            logger.warning(f"Embedding consistency check failed: {e}")
            return 0.5
    
    def _llm_consistency(self, thought: str, action: str) -> float:
        """
        Check consistency using LLM.
        
        Args:
            thought: The reasoning thought
            action: The action taken
            
        Returns:
            Consistency score from LLM
        """
        if not self.model_client:
            return 0.5
        
        try:
            prompt = VERIFICATION_THOUGHT_ACTION_PROMPT.format(
                thought=thought,
                action=action
            )
            
            response = self._call_llm(prompt, temperature=0.1)
            
            # Parse score from response
            score_match = re.search(r'(\d*\.?\d+)', response)
            if score_match:
                return float(score_match.group(1))
            else:
                logger.warning(f"Could not parse score from LLM response: {response}")
                return 0.5
        except Exception as e:
            logger.warning(f"LLM consistency check failed: {e}")
            return 0.5
    
    def check_observation_plausibility(self, 
                                      tool_name: str,
                                      tool_args: Dict[str, Any],
                                      observation: Any) -> float:
        """
        Check if the observation is plausible given the action.
        
        Args:
            tool_name: Name of the tool executed
            tool_args: Arguments passed to the tool
            observation: The observation returned
            
        Returns:
            Plausibility score (0-1)
        """
        # Check if observation is None or error
        if observation is None:
            return 0.0
        
        if isinstance(observation, dict) and "error" in observation:
            return 0.1  # Low score for errors
        
        # Tool-specific checks
        if tool_name in ["tool_add", "tool_subtract", "tool_multiply", "tool_divide"]:
            # Check if numeric
            try:
                float(observation)
                score = 0.9
                
                # Sanity checks for arithmetic
                if tool_name == "tool_divide" and "b" in tool_args:
                    if tool_args["b"] == 0:
                        # Division by zero should error
                        return 0.0 if not isinstance(observation, dict) else 0.9
                
                return score
            except (ValueError, TypeError):
                return 0.0
        
        elif tool_name == "tool_parse":
            # Check if non-empty string or valid extraction
            if observation == "NOT_FOUND":
                return 0.7  # Valid but indicates extraction failure
            elif observation and isinstance(observation, (str, int, float)):
                return 0.95
            else:
                return 0.3
        
        elif tool_name == "tool_compare":
            # Check if valid comparison result
            valid_results = ["greater", "less", "equal", "true", "false"]
            if str(observation).lower() in valid_results:
                return 0.95
            else:
                return 0.2
        
        # Default plausibility
        return 0.5
    
    def check_plan_adherence(self, actual_tool: str, expected_tool: str) -> float:
        """
        Check if execution follows the strategic plan.
        
        Args:
            actual_tool: Tool that was actually used
            expected_tool: Tool expected by the plan
            
        Returns:
            Adherence score (0-1)
        """
        if actual_tool == expected_tool:
            return 1.0
        
        # Check if tools are similar (e.g., both arithmetic)
        arithmetic_tools = ["tool_add", "tool_subtract", "tool_multiply", "tool_divide"]
        if actual_tool in arithmetic_tools and expected_tool in arithmetic_tools:
            return 0.7  # Similar category
        
        # Check if deviation might be justified
        if expected_tool == "tool_parse" and actual_tool in arithmetic_tools:
            # Might have skipped extraction if numbers were obvious
            return 0.6
        
        return 0.3  # Major deviation
    
    def check_output_format(self, tool_name: str, observation: Any) -> float:
        """
        Check if output matches expected format for the tool.
        
        Args:
            tool_name: Name of the tool
            observation: The observation
            
        Returns:
            Format validity score (0-1)
        """
        if tool_name not in self.tool_patterns:
            return 0.8  # Unknown tool, be lenient
        
        pattern = self.tool_patterns[tool_name]
        obs_str = str(observation)
        
        if re.match(pattern, obs_str):
            return 1.0
        else:
            return 0.2
    
    def calculate_cumulative_confidence(self, 
                                       strategic_confidence: float,
                                       tactical_confidences: List[float],
                                       tool_confidences: List[float]) -> float:
        """
        Calculate overall confidence for the solution.
        
        Args:
            strategic_confidence: Confidence from strategic planning
            tactical_confidences: Confidence scores from tactical execution
            tool_confidences: Confidence scores from tools
            
        Returns:
            Cumulative confidence score
        """
        # Get weights from settings
        weights = settings.CONFIDENCE_WEIGHTS
        
        # Calculate weighted components
        components = []
        
        # Strategic component
        components.append(weights["strategic_planning"] * strategic_confidence)
        
        # Tactical component (average of step confidences)
        if tactical_confidences:
            tactical_avg = np.mean(tactical_confidences)
            components.append(weights["tactical_execution"] * tactical_avg)
        else:
            components.append(0)
        
        # Tool component (multiplicative for critical tools)
        if tool_confidences:
            # For deterministic tools (confidence = 1.0), use average
            # For probabilistic tools, use geometric mean
            deterministic = [c for c in tool_confidences if c == 1.0]
            probabilistic = [c for c in tool_confidences if c < 1.0]
            
            if probabilistic:
                # Geometric mean for probabilistic tools
                prob_confidence = np.exp(np.mean(np.log(probabilistic + 1e-10)))
            else:
                prob_confidence = 1.0
            
            tool_confidence = prob_confidence
            components.append(weights["tool_reliability"] * tool_confidence)
        else:
            components.append(weights["tool_reliability"])
        
        # Verification component (average verification scores)
        if self.verification_history:
            verification_scores = [v.score for v in self.verification_history]
            verification_avg = np.mean(verification_scores)
            components.append(weights["verification_score"] * verification_avg)
        else:
            components.append(weights["verification_score"])
        
        # Sum weighted components
        cumulative_confidence = sum(components)
        
        # Apply penalties for failures
        failure_count = sum(1 for v in self.verification_history 
                          if v.status == VerificationStatus.FAIL)
        if failure_count > 0:
            penalty = 0.9 ** failure_count  # 10% reduction per failure
            cumulative_confidence *= penalty
        
        return min(1.0, max(0.0, cumulative_confidence))
    
    def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in verification history.
        
        Returns:
            Dictionary of detected patterns and issues
        """
        if len(self.verification_history) < 3:
            return {"patterns": [], "recurring_issues": []}
        
        patterns = []
        recurring_issues = {}
        
        # Count issue frequencies
        for result in self.verification_history:
            for issue in result.issues:
                recurring_issues[issue] = recurring_issues.get(issue, 0) + 1
        
        # Find recurring issues (appearing in >30% of verifications)
        threshold = len(self.verification_history) * 0.3
        recurring = [issue for issue, count in recurring_issues.items() 
                    if count >= threshold]
        
        # Detect declining confidence pattern
        recent_scores = [v.score for v in self.verification_history[-5:]]
        if len(recent_scores) >= 3:
            if all(recent_scores[i] <= recent_scores[i-1] for i in range(1, len(recent_scores))):
                patterns.append("declining_confidence")
        
        # Detect consistent failures in specific check types
        check_failures = {}
        for result in self.verification_history:
            for check_name, score in result.checks.items():
                if score < 0.5:
                    check_failures[check_name] = check_failures.get(check_name, 0) + 1
        
        for check_name, count in check_failures.items():
            if count >= threshold:
                patterns.append(f"consistent_{check_name}_failures")
        
        return {
            "patterns": patterns,
            "recurring_issues": recurring,
            "average_score": np.mean([v.score for v in self.verification_history]),
            "failure_rate": sum(1 for v in self.verification_history 
                               if v.status == VerificationStatus.FAIL) / len(self.verification_history)
        }
    
    def should_abort(self) -> bool:
        """
        Determine if execution should be aborted based on verification history.
        
        Returns:
            True if execution should stop
        """
        if len(self.verification_history) < 2:
            return False
        
        # Abort if too many consecutive failures
        recent_failures = sum(1 for v in self.verification_history[-3:]
                            if v.status == VerificationStatus.FAIL)
        if recent_failures >= 3:
            logger.warning("Too many consecutive verification failures")
            return True
        
        # Abort if average score too low
        if len(self.verification_history) >= 5:
            avg_score = np.mean([v.score for v in self.verification_history[-5:]])
            if avg_score < 0.3:
                logger.warning(f"Average verification score too low: {avg_score:.2f}")
                return True
        
        # Abort if confidence has collapsed
        if self.verification_history[-1].score < 0.1:
            logger.warning("Verification confidence collapsed")
            return True
        
        return False
    
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Call the LLM for verification tasks.
        
        Args:
            prompt: The prompt to send
            temperature: Temperature for generation
            
        Returns:
            LLM response
        """
        if not self.model_client:
            return "0.5"  # Default middle score
        
        try:
            model_name = settings.TOOL_PARSER_MODEL  # Use cheaper model for verification
            
            if "gpt" in model_name.lower():
                response = self.model_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=50
                )
                return response.choices[0].message.content
            elif "claude" in model_name.lower():
                response = self.model_client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=50
                )
                return response.content[0].text
            else:
                return "0.5"
        except Exception as e:
            logger.error(f"LLM verification call failed: {e}")
            return "0.5"
    
    def reset(self) -> None:
        """Reset the verification module for a new problem."""
        self.verification_history.clear()
        logger.debug("Reset verification history")

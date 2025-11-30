"""
Hierarchical Memory System for the Enhanced Stateful Router.

This module implements a three-level logging system that maintains:
1. Strategic-level decisions and plans
2. Tactical-level step executions
3. Low-level tool execution details

The memory system ensures complete auditability and verifiability of the reasoning process.
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid

from loguru import logger


class LogLevel(Enum):
    """Enumeration of log levels in the hierarchy."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    EXECUTION = "execution"


class VerificationStatus(Enum):
    """Status of verification for a log entry."""
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    PENDING = "pending"


@dataclass
class StrategicLogEntry:
    """Entry for strategic-level planning decisions."""
    timestamp: float
    problem: str
    strategy: str
    plan: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    fallback_strategy: Optional[str] = None
    attempt_number: int = 1
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class TacticalLogEntry:
    """Entry for tactical-level step execution."""
    timestamp: float
    step_number: int
    step_description: str
    thought: str
    action: str
    observation: Any
    confidence: float
    verification_status: VerificationStatus
    verification_score: float
    expected_tool: str
    actual_tool: str
    retry_count: int = 0
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_strategic_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['verification_status'] = self.verification_status.value
        return data


@dataclass
class ExecutionLogEntry:
    """Entry for low-level tool execution."""
    timestamp: float
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    execution_time: float
    confidence: float
    deterministic: bool
    error: Optional[str] = None
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_tactical_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class HierarchicalMemory:
    """
    Manages the three-level memory system for the Stateful Router.

    This class maintains synchronized logs at strategic, tactical, and execution levels,
    providing methods to add entries, query history, and generate audit trails.
    """

    def __init__(self, save_to_disk: bool = True, trace_dir: Optional[Path] = None):
        """
        Initialize the hierarchical memory system.

        Args:
            save_to_disk: Whether to persist logs to disk
            trace_dir: Directory for saving execution traces
        """
        self.strategic_log: List[StrategicLogEntry] = []
        self.tactical_log: List[TacticalLogEntry] = []
        self.execution_log: List[ExecutionLogEntry] = []

        self.current_strategic_id: Optional[str] = None
        self.current_tactical_id: Optional[str] = None

        # Intermediate variables for storing computation results
        self.intermediate_vars: Dict[str, Any] = {}
        self.var_counter = 0

        # Variable lifecycle tracking
        self.var_history: List[Dict[str, Any]] = []  # Track all variable operations

        self.save_to_disk = save_to_disk
        self.trace_dir = trace_dir or Path("./traces")
        if save_to_disk:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.session_id = str(uuid.uuid4())
            self.session_file = self.trace_dir / f"session_{self.session_id}.jsonl"

        logger.info(f"Initialized HierarchicalMemory (save_to_disk={save_to_disk})")
    
    def add_strategic_entry(self, 
                          problem: str,
                          strategy: str,
                          plan: List[Dict[str, Any]],
                          confidence: float,
                          reasoning: str,
                          fallback_strategy: Optional[str] = None,
                          attempt_number: int = 1) -> str:
        """
        Add a strategic-level planning entry.
        
        Returns:
            The entry ID for linking tactical entries
        """
        entry = StrategicLogEntry(
            timestamp=time.time(),
            problem=problem,
            strategy=strategy,
            plan=plan,
            confidence=confidence,
            reasoning=reasoning,
            fallback_strategy=fallback_strategy,
            attempt_number=attempt_number
        )
        
        self.strategic_log.append(entry)
        self.current_strategic_id = entry.entry_id
        
        if self.save_to_disk:
            self._save_entry(LogLevel.STRATEGIC, entry.to_dict())
        
        logger.debug(f"Added strategic entry: {entry.entry_id} - Strategy: {strategy}")
        return entry.entry_id
    
    def add_tactical_entry(self,
                         step_number: int,
                         step_description: str,
                         thought: str,
                         action: str,
                         observation: Any,
                         confidence: float,
                         verification_status: VerificationStatus,
                         verification_score: float,
                         expected_tool: str,
                         actual_tool: str,
                         retry_count: int = 0) -> str:
        """
        Add a tactical-level execution entry.
        
        Returns:
            The entry ID for linking execution entries
        """
        entry = TacticalLogEntry(
            timestamp=time.time(),
            step_number=step_number,
            step_description=step_description,
            thought=thought,
            action=action,
            observation=observation,
            confidence=confidence,
            verification_status=verification_status,
            verification_score=verification_score,
            expected_tool=expected_tool,
            actual_tool=actual_tool,
            retry_count=retry_count,
            parent_strategic_id=self.current_strategic_id
        )
        
        self.tactical_log.append(entry)
        self.current_tactical_id = entry.entry_id
        
        if self.save_to_disk:
            self._save_entry(LogLevel.TACTICAL, entry.to_dict())
        
        logger.debug(f"Added tactical entry: {entry.entry_id} - Step {step_number}: {action}")
        return entry.entry_id
    
    def add_execution_entry(self,
                          tool_name: str,
                          tool_input: Dict[str, Any],
                          tool_output: Any,
                          execution_time: float,
                          confidence: float,
                          deterministic: bool,
                          error: Optional[str] = None) -> str:
        """
        Add an execution-level tool entry.
        
        Returns:
            The entry ID
        """
        entry = ExecutionLogEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            execution_time=execution_time,
            confidence=confidence,
            deterministic=deterministic,
            error=error,
            parent_tactical_id=self.current_tactical_id
        )
        
        self.execution_log.append(entry)
        
        if self.save_to_disk:
            self._save_entry(LogLevel.EXECUTION, entry.to_dict())
        
        logger.debug(f"Added execution entry: {entry.entry_id} - Tool: {tool_name}")
        return entry.entry_id
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the system including all observations.
        
        Returns:
            Dictionary containing the current problem, plan, and observations
        """
        if not self.strategic_log:
            return {"problem": None, "plan": None, "observations": []}
        
        current_strategy = self.strategic_log[-1]
        observations = []
        
        for tactical_entry in self.tactical_log:
            if tactical_entry.parent_strategic_id == current_strategy.entry_id:
                observations.append({
                    "step": tactical_entry.step_number,
                    "thought": tactical_entry.thought,
                    "action": tactical_entry.action,
                    "observation": tactical_entry.observation,
                    "confidence": tactical_entry.confidence
                })
        
        return {
            "problem": current_strategy.problem,
            "strategy": current_strategy.strategy,
            "plan": current_strategy.plan,
            "observations": observations
        }
    
    def get_observations(self) -> List[Any]:
        """
        Get all observations from the current strategic context.

        Returns:
            List of observations from tactical executions
        """
        if not self.current_strategic_id:
            return []

        observations = []
        for entry in self.tactical_log:
            if entry.parent_strategic_id == self.current_strategic_id:
                observations.append(entry.observation)

        return observations

    def store_variable(self, value: Any, var_name: Optional[str] = None, allow_overwrite: bool = False) -> str:
        """
        Store an intermediate computation result as a named variable.

        Args:
            value: The value to store
            var_name: Optional custom variable name (auto-generated if None)
            allow_overwrite: Whether to allow overwriting existing variables

        Returns:
            The variable name used
        """
        original_requested_name = var_name
        if var_name is None:
            self.var_counter += 1
            var_name = f"var_{self.var_counter}"

        # Check for overwrites
        operation_type = "create"
        old_value = None
        if var_name in self.intermediate_vars:
            old_value = self.intermediate_vars[var_name]
            if not allow_overwrite:
                # Generate a unique name to avoid collision
                logger.warning(f"Variable '{var_name}' already exists with value {old_value}. "
                             f"Creating unique variable to avoid collision.")
                self.var_counter += 1
                original_name = var_name
                var_name = f"{var_name}_{self.var_counter}"
                logger.info(f"Renamed '{original_name}' to '{var_name}' to avoid collision")
                operation_type = "create_renamed"
            else:
                logger.warning(f"Overwriting variable '{var_name}': {old_value} -> {value}")
                operation_type = "overwrite"

        self.intermediate_vars[var_name] = value

        # Track variable lifecycle
        self.var_history.append({
            "timestamp": time.time(),
            "operation": operation_type,
            "var_name": var_name,
            "requested_name": original_requested_name,
            "value": value,
            "old_value": old_value,
            "step": len(self.tactical_log)
        })

        logger.debug(f"Stored intermediate variable: {var_name} = {value}")
        return var_name

    def get_variable(self, var_name: str) -> Any:
        """
        Retrieve an intermediate variable by name.

        Args:
            var_name: The variable name to retrieve

        Returns:
            The stored value, or None if not found
        """
        value = self.intermediate_vars.get(var_name)

        # Track variable access
        self.var_history.append({
            "timestamp": time.time(),
            "operation": "read" if value is not None else "read_missing",
            "var_name": var_name,
            "value": value,
            "step": len(self.tactical_log)
        })

        if value is None:
            logger.warning(f"Variable not found: {var_name}")
        return value

    def get_all_variables(self) -> Dict[str, Any]:
        """
        Get all intermediate variables.

        Returns:
            Dictionary of all stored variables
        """
        return self.intermediate_vars.copy()

    def clear_variables(self) -> None:
        """Clear all intermediate variables."""
        self.intermediate_vars.clear()
        self.var_counter = 0
        self.var_history.clear()
        logger.debug("Cleared all intermediate variables")

    def get_variable_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete variable lifecycle history.

        Returns:
            List of all variable operations in chronological order
        """
        return self.var_history.copy()

    def get_variable_summary(self) -> Dict[str, Any]:
        """
        Get a summary of variable usage.

        Returns:
            Dictionary with variable statistics and summary
        """
        summary = {
            "total_variables": len(self.intermediate_vars),
            "total_operations": len(self.var_history),
            "variables": {}
        }

        # Analyze each variable's history
        for var_name in self.intermediate_vars.keys():
            var_ops = [op for op in self.var_history if op.get("var_name") == var_name]
            summary["variables"][var_name] = {
                "current_value": self.intermediate_vars[var_name],
                "operations": len(var_ops),
                "created_at_step": var_ops[0]["step"] if var_ops else None,
                "operations_detail": [
                    {"operation": op["operation"], "step": op["step"]}
                    for op in var_ops
                ]
            }

        # Count operation types
        summary["operation_counts"] = {
            "create": sum(1 for op in self.var_history if op["operation"] == "create"),
            "create_renamed": sum(1 for op in self.var_history if op["operation"] == "create_renamed"),
            "overwrite": sum(1 for op in self.var_history if op["operation"] == "overwrite"),
            "read": sum(1 for op in self.var_history if op["operation"] == "read"),
            "read_missing": sum(1 for op in self.var_history if op["operation"] == "read_missing"),
        }

        return summary
    
    def get_audit_trail(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a complete audit trail of the reasoning process.
        
        Returns:
            Dictionary with strategic, tactical, and execution logs
        """
        return {
            "strategic": [entry.to_dict() for entry in self.strategic_log],
            "tactical": [entry.to_dict() for entry in self.tactical_log],
            "execution": [entry.to_dict() for entry in self.execution_log]
        }
    
    def get_history_text(self) -> str:
        """
        Get a human-readable text representation of the history.
        
        Returns:
            Formatted string of the execution history
        """
        lines = []
        
        if self.strategic_log:
            latest_strategic = self.strategic_log[-1]
            lines.append(f"Problem: {latest_strategic.problem}")
            lines.append(f"Strategy: {latest_strategic.strategy}")
            lines.append("")
        
        for entry in self.tactical_log:
            if entry.parent_strategic_id == self.current_strategic_id:
                lines.append(f"Step {entry.step_number}: {entry.step_description}")
                lines.append(f"  Thought: {entry.thought}")
                lines.append(f"  Action: {entry.action}")
                lines.append(f"  Observation: {entry.observation}")
                lines.append(f"  Confidence: {entry.confidence:.2f}")
                lines.append("")
        
        return "\n".join(lines)
    
    def calculate_cumulative_confidence(self) -> float:
        """
        Calculate the cumulative confidence across all steps.
        
        Returns:
            Cumulative confidence score
        """
        if not self.strategic_log:
            return 0.0
        
        strategic_confidence = self.strategic_log[-1].confidence
        
        tactical_confidences = [
            entry.confidence 
            for entry in self.tactical_log 
            if entry.parent_strategic_id == self.current_strategic_id
        ]
        
        if not tactical_confidences:
            return strategic_confidence
        
        # Multiply all confidences
        cumulative = strategic_confidence
        for conf in tactical_confidences:
            cumulative *= conf
        
        return cumulative
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get a summary of verification results.
        
        Returns:
            Dictionary with verification statistics
        """
        if not self.current_strategic_id:
            return {
                "total_steps": 0,
                "passed": 0,
                "review": 0,
                "failed": 0,
                "average_score": 0.0
            }
        
        relevant_entries = [
            entry for entry in self.tactical_log
            if entry.parent_strategic_id == self.current_strategic_id
        ]
        
        if not relevant_entries:
            return {
                "total_steps": 0,
                "passed": 0,
                "review": 0,
                "failed": 0,
                "average_score": 0.0
            }
        
        passed = sum(1 for e in relevant_entries if e.verification_status == VerificationStatus.PASS)
        review = sum(1 for e in relevant_entries if e.verification_status == VerificationStatus.REVIEW)
        failed = sum(1 for e in relevant_entries if e.verification_status == VerificationStatus.FAIL)
        avg_score = sum(e.verification_score for e in relevant_entries) / len(relevant_entries)
        
        return {
            "total_steps": len(relevant_entries),
            "passed": passed,
            "review": review,
            "failed": failed,
            "average_score": avg_score
        }
    
    def _save_entry(self, level: LogLevel, entry_data: Dict[str, Any]) -> None:
        """Save an entry to disk in JSONL format."""
        if not self.save_to_disk:
            return
        
        entry_with_metadata = {
            "level": level.value,
            "session_id": self.session_id,
            "timestamp": time.time(),
            "data": entry_data
        }
        
        with open(self.session_file, "a") as f:
            f.write(json.dumps(entry_with_metadata) + "\n")
    
    def save_final_trace(self, result: Dict[str, Any]) -> Path:
        """
        Save the complete execution trace with the final result.
        
        Args:
            result: The final result dictionary
            
        Returns:
            Path to the saved trace file
        """
        trace_data = {
            "session_id": self.session_id if self.save_to_disk else str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "audit_trail": self.get_audit_trail(),
            "verification_summary": self.get_verification_summary(),
            "cumulative_confidence": self.calculate_cumulative_confidence()
        }
        
        filename = self.trace_dir / f"trace_{trace_data['session_id']}_complete.json"
        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)
        
        logger.info(f"Saved complete trace to {filename}")
        return filename
    
    def reset(self) -> None:
        """Reset the memory system for a new problem."""
        self.strategic_log.clear()
        self.tactical_log.clear()
        self.execution_log.clear()
        self.current_strategic_id = None
        self.current_tactical_id = None
        self.clear_variables()

        if self.save_to_disk:
            self.session_id = str(uuid.uuid4())
            self.session_file = self.trace_dir / f"session_{self.session_id}.jsonl"

        logger.info("Reset memory system")

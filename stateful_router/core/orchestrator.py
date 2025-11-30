"""
Orchestrator Module for the Enhanced Stateful Router.

This module coordinates all components to solve problems through
hierarchical planning, execution, verification, and confidence tracking.
"""

import time
import asyncio
import inspect
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI
from anthropic import Anthropic

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config.settings import settings
from ..config.prompts import FINAL_ANSWER_PROMPT
from .memory import HierarchicalMemory, VerificationStatus
from .strategic_planner import StrategicPlanner, StrategicPlan
from .tactical_router import TacticalRouter
from .verification import VerificationModule
from ..tools.registry import ToolRegistry
from ..tools.base import BaseTool


@dataclass
class ExecutionResult:
    """Result of solving a problem."""
    answer: Any
    confidence: float
    strategy: str
    steps_executed: int
    verification_summary: Dict[str, Any]
    audit_trail: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None
    chain_of_thought: str = ""


class StatefulRouterOrchestrator:
    """
    Main orchestrator that coordinates all components of the Stateful Router.
    
    This class manages the complete problem-solving pipeline:
    1. Strategic planning
    2. Tactical execution
    3. Tool invocation
    4. Verification
    5. Error recovery
    """
    
    def __init__(self,
                planner: Optional[StrategicPlanner] = None,
                router: Optional[TacticalRouter] = None,
                verifier: Optional[VerificationModule] = None,
                memory: Optional[HierarchicalMemory] = None,
                tool_registry: Optional[ToolRegistry] = None,
                verbose: bool = False):
        """
        Initialize the orchestrator.
        
        Args:
            planner: Strategic planner instance
            router: Tactical router instance
            verifier: Verification module instance
            memory: Hierarchical memory instance
            tool_registry: Tool registry instance
            verbose: Whether to print detailed progress
        """
        # Create the API client based on the model name in settings
        self.client = None
        if "gpt" in settings.STRATEGIC_PLANNER_MODEL.lower():
            if not settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY not found in .env file.")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        elif "claude" in settings.STRATEGIC_PLANNER_MODEL.lower():
            if not settings.ANTHROPIC_API_KEY:
                logger.error("ANTHROPIC_API_KEY not found in .env file.")
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        if self.client is None:
             logger.warning("No API client created. Check API keys and model names. System will run in MOCK mode.")
        
        self.planner = planner or StrategicPlanner(model_client=self.client)
        self.router = router or TacticalRouter(model_client=self.client)
        self.verifier = verifier or VerificationModule(model_client=self.client)
        
        # The registry needs the client for the ParseTool
        self.tool_registry = tool_registry or ToolRegistry(model_client=self.client)
        self.memory = memory or HierarchicalMemory()
        self.tool_registry = tool_registry or ToolRegistry()
        
        self.verbose = verbose
        self.console = Console() if verbose else None
        
        # Execution limits
        self.max_steps = settings.MAX_STEPS_PER_PROBLEM
        self.max_plan_attempts = settings.MAX_PLAN_ATTEMPTS
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_PARALLEL_TOOLS)
        
        logger.info("Initialized StatefulRouterOrchestrator")
    
    def solve(self, problem: str) -> ExecutionResult:
        """
        Solve a problem using the complete pipeline.
        
        Args:
            problem: The problem to solve
            
        Returns:
            ExecutionResult with answer and metadata
        """
        start_time = time.time()
        
        if self.verbose:
            self.console.print(f"[bold blue]Solving:[/bold blue] {problem}")
        
        try:
            # Reset components for new problem
            self.memory.reset()
            self.verifier.reset()
            
            # Phase 1: Strategic Planning
            plan = self._strategic_planning_phase(problem)
            if not plan:
                return self._create_failure_result(
                    "Strategic planning failed",
                    time.time() - start_time
                )
            
            # Phase 2: Tactical Execution
            execution_success = self._tactical_execution_phase(problem, plan)
            
            # Phase 3: Answer Extraction
            answer = self._extract_answer(problem)
            
            # Phase 4: Confidence Calculation
            confidence = self._calculate_final_confidence(plan)
            
            # Create result
            result = ExecutionResult(
                answer=answer,
                confidence=confidence,
                strategy=plan.strategy.value,
                steps_executed=len(self.memory.tactical_log),
                verification_summary=self.memory.get_verification_summary(),
                audit_trail=self.memory.get_audit_trail(),
                execution_time=time.time() - start_time,
                success=execution_success and confidence >= settings.CONFIDENCE_THRESHOLD
            )
            
            # Save trace if configured
            if settings.SAVE_EXECUTION_TRACES:
                trace_path = self.memory.save_final_trace(result.__dict__)
                logger.info(f"Saved execution trace to {trace_path}")
            
            if self.verbose:
                self._print_result_summary(result)
            
            result.chain_of_thought = self.get_chain_of_thought()

            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return self._create_failure_result(
                str(e),
                time.time() - start_time
            )
    
    def _strategic_planning_phase(self, problem: str) -> Optional[StrategicPlan]:
        """
        Execute the strategic planning phase.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Strategic plan or None if planning fails
        """
        if self.verbose:
            self.console.print("\n[bold yellow]Phase 1: Strategic Planning[/bold yellow]")
        
        attempts = 0
        previous_attempts = []
        
        while attempts < self.max_plan_attempts:
            attempts += 1
            
            try:
                # Generate plan
                plan = self.planner.plan(problem, previous_attempts)
                
                # Log to memory
                self.memory.add_strategic_entry(
                    problem=problem,
                    strategy=plan.strategy.value,
                    plan=[step.to_dict() for step in plan.plan],
                    confidence=plan.confidence,
                    reasoning=plan.reasoning,
                    fallback_strategy=plan.fallback_strategy.value if plan.fallback_strategy else None,
                    attempt_number=attempts
                )
                
                if self.verbose:
                    self._print_plan(plan)
                
                # Validate plan
                if len(plan.plan) == 0:
                    logger.warning("Empty plan generated")
                    previous_attempts.append({
                        "attempt": attempts,
                        "failure": "Empty plan"
                    })
                    continue
                
                if plan.confidence < 0.3:
                    logger.warning(f"Low confidence plan: {plan.confidence}")
                    previous_attempts.append({
                        "attempt": attempts,
                        "failure": f"Low confidence: {plan.confidence}"
                    })
                    continue
                
                return plan
                
            except Exception as e:
                logger.error(f"Planning attempt {attempts} failed: {e}")
                previous_attempts.append({
                    "attempt": attempts,
                    "failure": str(e)
                })
        
        logger.error("All planning attempts failed")
        return None
    
    def _tactical_execution_phase(self, problem: str, plan: StrategicPlan) -> bool:
        """
        Execute the tactical execution phase.
        
        Args:
            problem: The problem to solve
            plan: The strategic plan to execute
            
        Returns:
            True if execution succeeded
        """
        if self.verbose:
            self.console.print("\n[bold yellow]Phase 2: Tactical Execution[/bold yellow]")
        
        steps_executed = 0
        
        # Execute plan step by step or in parallel groups
        for group in plan.parallel_groups or [[step.step for step in plan.plan]]:
            if len(group) == 1:
                # Sequential execution
                step_num = group[0]
                step = plan.plan[step_num - 1]
                
                success = self._execute_single_step(problem, step, plan)
                if not success:
                    logger.error(f"Step {step_num} failed")
                    return False
                    
            else:
                # Parallel execution
                success = self._execute_parallel_steps(problem, group, plan)
                if not success:
                    logger.error(f"Parallel group {group} failed")
                    return False
            
            steps_executed += len(group)
            
            # Check if we should abort
            if self.verifier.should_abort():
                logger.error("Verification module triggered abort")
                return False
            
            # Check step limit
            if steps_executed >= self.max_steps:
                logger.warning(f"Reached max steps limit: {self.max_steps}")
                break
        
        return True
    
    def _execute_single_step(self, problem: str, step: Any, plan: StrategicPlan) -> bool:
        """
        Execute a single step of the plan.
        
        Args:
            problem: The problem being solved
            step: The step to execute
            plan: The complete strategic plan
            
        Returns:
            True if step succeeded
        """
        retry_count = 0
        max_retries = settings.MAX_RETRIES
        
        while retry_count < max_retries:
            try:
                if self.verbose:
                    self.console.print(f"\n[cyan]Step {step.step}:[/cyan] {step.description}")

                # Check if this is the final step
                is_final_step = (step.step == len(plan.plan))

                # Generate thought and action
                router_output = self.router.execute_step(step, self.memory, problem, is_final_step)

                if self.verbose:
                    self.console.print(f"  [dim]Thought:[/dim] {router_output.thought}")
                    self.console.print(f"  [dim]Action:[/dim] {router_output.action}")

                # Resolve any variable references in tool arguments
                resolved_args = self.router.resolve_variables(router_output.tool_args, self.memory)

                # Execute tool with resolved arguments
                tool_result = self._execute_tool(
                    router_output.tool_name,
                    resolved_args
                )

                # Store result in variable if specified
                actual_var_name = None
                if router_output.output_var:
                    # Store the variable and get the actual name used (may be different if collision detected)
                    actual_var_name = self.memory.store_variable(
                        tool_result['observation'],
                        router_output.output_var,
                        allow_overwrite=False  # Never allow overwrite - create unique names instead
                    )
                    if self.verbose:
                        if actual_var_name != router_output.output_var:
                            self.console.print(f"  [dim]Stored in:[/dim] {actual_var_name} (renamed from {router_output.output_var} to avoid collision)")
                        else:
                            self.console.print(f"  [dim]Stored in:[/dim] {actual_var_name}")

                if self.verbose:
                    self.console.print(f"  [dim]Observation:[/dim] {tool_result['observation']}")
                
                # Verify step
                verification = self.verifier.verify_step(
                    router_output,
                    tool_result['observation'],
                    step
                )
                
                # Log to memory
                self.memory.add_tactical_entry(
                    step_number=step.step,
                    step_description=step.description,
                    thought=router_output.thought,
                    action=router_output.action,
                    observation=tool_result['observation'],
                    confidence=router_output.confidence * tool_result['confidence'],
                    verification_status=verification.status,
                    verification_score=verification.score,
                    expected_tool=step.expected_tool,
                    actual_tool=router_output.tool_name,
                    retry_count=retry_count
                )
                
                # Check verification result
                if verification.status == VerificationStatus.PASS:
                    return True
                elif verification.status == VerificationStatus.REVIEW:
                    logger.warning(f"Step {step.step} under review: {verification.issues}")
                    # Continue but with caution
                    return True
                else:
                    # Verification failed
                    logger.warning(f"Step {step.step} failed verification: {verification.issues}")
                    
                    # Try alternative approach
                    if retry_count < max_retries - 1:
                        router_output = self.router.generate_alternative(
                            step,
                            router_output,
                            str(verification.issues),
                            self.memory,
                            problem
                        )
                        retry_count += 1
                        continue
                    else:
                        return False
                        
            except Exception as e:
                logger.error(f"Step {step.step} execution error: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    return False
                
                time.sleep(settings.RETRY_DELAY * retry_count)
        
        return False
    
    def _execute_parallel_steps(self, problem: str, group: List[int], 
                               plan: StrategicPlan) -> bool:
        """
        Execute multiple steps in parallel.
        
        Args:
            problem: The problem being solved
            group: List of step numbers to execute in parallel
            plan: The complete strategic plan
            
        Returns:
            True if all steps succeeded
        """
        if self.verbose:
            self.console.print(f"\n[cyan]Parallel execution:[/cyan] Steps {group}")
        
        futures = []
        step_map = {}
        
        # Submit all steps to thread pool
        for step_num in group:
            step = plan.plan[step_num - 1]
            future = self.executor.submit(
                self._execute_single_step,
                problem,
                step,
                plan
            )
            futures.append(future)
            step_map[future] = step_num
        
        # Wait for all to complete
        success = True
        for future in as_completed(futures):
            step_num = step_map[future]
            try:
                result = future.result(timeout=30)
                if not result:
                    logger.error(f"Parallel step {step_num} failed")
                    success = False
            except Exception as e:
                logger.error(f"Parallel step {step_num} exception: {e}")
                success = False
        
        return success
    
    def _filter_tool_args(self, tool_function: Any, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter tool arguments to only include parameters accepted by the tool.

        Args:
            tool_function: The tool function or BaseTool instance
            tool_args: Dictionary of all provided arguments

        Returns:
            Filtered dictionary with only valid parameters
        """
        try:
            # Get the actual execute method for BaseTool instances
            if isinstance(tool_function, BaseTool):
                method = tool_function.execute
            else:
                method = tool_function

            # Get the signature of the execute method
            sig = inspect.signature(method)
            valid_params = set(sig.parameters.keys())

            # Filter arguments to only include valid parameters
            filtered_args = {k: v for k, v in tool_args.items() if k in valid_params}

            # Log any filtered parameters
            filtered_keys = set(tool_args.keys()) - set(filtered_args.keys())
            if filtered_keys:
                logger.debug(f"Filtered out invalid parameters: {filtered_keys}")

            return filtered_args

        except Exception as e:
            logger.warning(f"Could not filter tool args: {e}, using all args")
            return tool_args

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dictionary with observation and confidence
        """
        start_time = time.time()

        try:
            # Get tool from registry
            tool_info = self.tool_registry.get_tool(tool_name)
            if not tool_info:
                logger.error(f"Tool not found: {tool_name}")
                return {
                    "observation": {"error": f"Tool not found: {tool_name}"},
                    "confidence": 0.0
                }

            # Filter arguments to only include valid parameters
            filtered_args = self._filter_tool_args(tool_info["function"], tool_args)

            # Execute tool with filtered arguments
            result = tool_info["function"](**filtered_args)

            # Handle different result formats
            if isinstance(result, dict):
                observation = result.get("result", result)
                confidence = result.get("confidence", 0.9)
            else:
                observation = result
                confidence = 1.0 if tool_info["deterministic"] else 0.9

            # Log execution
            self.memory.add_execution_entry(
                tool_name=tool_name,
                tool_input=filtered_args,
                tool_output=observation,
                execution_time=time.time() - start_time,
                confidence=confidence,
                deterministic=tool_info["deterministic"]
            )

            return {
                "observation": observation,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")

            # Log error
            self.memory.add_execution_entry(
                tool_name=tool_name,
                tool_input=tool_args,
                tool_output=None,
                execution_time=time.time() - start_time,
                confidence=0.0,
                deterministic=False,
                error=str(e)
            )

            return {
                "observation": {"error": str(e)},
                "confidence": 0.0
            }
    
    def _extract_answer(self, problem: str) -> Any:
        """
        Extract the final answer from the execution.
        
        Args:
            problem: The original problem
            
        Returns:
            The extracted answer
        """
        # Get final observations
        observations = self.memory.get_observations()
        
        if not observations:
            return None
        
        # Simple heuristic: last observation is often the answer
        last_observation = observations[-1]
        
        # For arithmetic problems, the last number is usually the answer
        if isinstance(last_observation, (int, float)):
            return last_observation
        
        # For extraction problems, return the extracted value
        if isinstance(last_observation, str) and last_observation != "NOT_FOUND":
            return last_observation
        
        # For comparison problems, return the result
        if last_observation in ["greater", "less", "equal", "true", "false"]:
            return last_observation
        
        # If complex, try to use LLM to extract answer (if available)
        if self.router.model_client:
            try:
                execution_summary = self.memory.get_history_text()
                prompt = FINAL_ANSWER_PROMPT.format(
                    problem=problem,
                    execution_summary=execution_summary,
                    final_observations=str(observations[-3:]) if len(observations) >= 3 else str(observations)
                )
                
                # Use router's LLM to extract answer
                answer = self.router._call_llm(
                    system_prompt="Extract the final answer from the execution trace.",
                    user_prompt=prompt,
                    temperature=0.1
                )
                
                # Try to parse as number if applicable
                try:
                    return float(answer.strip())
                except ValueError:
                    return answer.strip()
                    
            except Exception as e:
                logger.warning(f"Answer extraction with LLM failed: {e}")
        
        # Fallback to last observation
        return last_observation
    
    def _calculate_final_confidence(self, plan: StrategicPlan) -> float:
        """
        Calculate the final confidence score.
        
        Args:
            plan: The strategic plan that was executed
            
        Returns:
            Final confidence score
        """
        # Get confidence components
        strategic_confidence = plan.confidence
        
        tactical_confidences = [
            entry.confidence
            for entry in self.memory.tactical_log
            if entry.parent_strategic_id == self.memory.current_strategic_id
        ]
        
        tool_confidences = [
            entry.confidence
            for entry in self.memory.execution_log
        ]
        
        # Use verification module to calculate cumulative confidence
        confidence = self.verifier.calculate_cumulative_confidence(
            strategic_confidence,
            tactical_confidences,
            tool_confidences
        )
        
        return confidence
    
    def _create_failure_result(self, error: str, execution_time: float) -> ExecutionResult:
        """
        Create a failure result.
        
        Args:
            error: Error message
            execution_time: Time taken
            
        Returns:
            ExecutionResult indicating failure
        """
        return ExecutionResult(
            answer=None,
            confidence=0.0,
            strategy="NONE",
            steps_executed=len(self.memory.tactical_log),
            verification_summary=self.memory.get_verification_summary(),
            audit_trail=self.memory.get_audit_trail(),
            execution_time=execution_time,
            success=False,
            error=error
        )
    
    def _print_plan(self, plan: StrategicPlan) -> None:
        """Print the strategic plan in a formatted way."""
        if not self.console:
            return
        
        table = Table(title=f"Strategy: {plan.strategy.value}")
        table.add_column("Step", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Tool", style="green")
        table.add_column("Dependencies", style="yellow")
        
        for step in plan.plan:
            deps = ", ".join(map(str, step.depends_on)) if step.depends_on else "-"
            table.add_row(
                str(step.step),
                step.description,
                step.expected_tool,
                deps
            )
        
        self.console.print(table)
        self.console.print(f"[dim]Confidence: {plan.confidence:.2f}[/dim]")
        self.console.print(f"[dim]Reasoning: {plan.reasoning}[/dim]")
    
    def _print_result_summary(self, result: ExecutionResult) -> None:
        """Print a summary of the execution result."""
        if not self.console:
            return
        
        self.console.print("\n[bold green]Execution Summary[/bold green]")
        self.console.print(f"Answer: [bold]{result.answer}[/bold]")
        self.console.print(f"Confidence: {result.confidence:.2%}")
        self.console.print(f"Strategy: {result.strategy}")
        self.console.print(f"Steps executed: {result.steps_executed}")
        self.console.print(f"Execution time: {result.execution_time:.2f}s")
        
        # Verification summary
        if result.verification_summary["total_steps"] > 0:
            self.console.print(f"\nVerification:")
            self.console.print(f"  Passed: {result.verification_summary['passed']}")
            self.console.print(f"  Review: {result.verification_summary['review']}")
            self.console.print(f"  Failed: {result.verification_summary['failed']}")
            self.console.print(f"  Average score: {result.verification_summary['average_score']:.2f}")
    
    def get_chain_of_thought(self) -> str:
        """Extract clean chain-of-thought."""
        if not hasattr(self, 'memory') or not self.memory.tactical_log:
            return "No execution found"
        
        lines = []
        for entry in self.memory.tactical_log:
            if entry.verification_status.value == 'fail':
                continue
            lines.append(f"Step {entry.step_number}: {entry.thought}")
            lines.append(f"→ Result: {entry.observation}\n")
        
        return "\n".join(lines)
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

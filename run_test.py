# --- File: run_test.py ---
from stateful_router.core.orchestrator import StatefulRouterOrchestrator
from rich.console import Console

# (Note: This assumes you have organized your files as shown
# in the directory structure and run 'pip install -e .')

console = Console()

# 1. Initialize the system
# This will automatically load models and settings from config/settings.py
console.print("[bold yellow]Initializing Stateful Router...[/bold yellow]")
router = StatefulRouterOrchestrator(verbose=True)

# 2. Define a problem to solve
problem = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

# 3. Solve the problem
console.print("[bold green]Solving problem...[/bold green]")
result = router.solve(problem)

# 4. Access the verifiable audit trail
console.print("\n--- [bold]Final Result[/bold] ---")
console.print(f"Answer: {result.answer}")
console.print(f"Confidence: {result.confidence:.2%}")

# Print the full, verifiable chain of thought
console.print("\n--- [bold]Verifiable Audit Trail[/bold] ---")
for entry in result.audit_trail['tactical']:
    console.print(f"[cyan]Step {entry['step_number']}[/cyan]: {entry['step_description']}")
    console.print(f"  [dim]Thought:[/dim] {entry['thought']}")
    console.print(f"  [dim]Action:[/dim] {entry['action']}")
    console.print(f"  [dim]Observation:[/dim] {entry['observation']}")
    console.print(f"  [green]Verification:[/green] {entry['verification_status']}")
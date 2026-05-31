"""
Centralized CLI Display System for OmicsNavigator Pipeline.

Provides a singleton CLIDisplay class that all agents import for
structured, rich-formatted terminal output with automatic phase
headers and visual emphasis on key results.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

logger = logging.getLogger(__name__)

# Phase configuration: maps phase number to name, color, and list of agents
PHASE_CONFIG = {
    1: {"name": "Planning Module",      "color": "cyan",    "agents": ["DataAnalyst", "LiteratureReviewer", "Planner"]},
    2: {"name": "Interpretation Module", "color": "magenta", "agents": ["VisualProfiler", "OmicsProfiler", "OmicsInterpreter"]},
    3: {"name": "Analysis Module",       "color": "yellow",  "agents": ["Retriever", "Validator"]},
}

# Reverse lookup: agent_name -> (phase_num, phase_config)
_AGENT_PHASE = {}
for _pnum, _pinfo in PHASE_CONFIG.items():
    for _agent in _pinfo["agents"]:
        _AGENT_PHASE[_agent] = (_pnum, _pinfo)


class CLIDisplay:
    """
    Centralized display manager for the multi-agent pipeline.

    Usage:
        display = CLIDisplay.get()
        display.agent_start("DataAnalyst", "Analyzing dataset...")
        display.agent_progress("DataAnalyst", "Report generated (2370 chars)")
        display.agent_done("DataAnalyst", "report saved -> analysis_report.md")
    """

    _instance: Optional["CLIDisplay"] = None

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._current_phase: int = 0

    @classmethod
    def get(cls, console: Optional[Console] = None) -> "CLIDisplay":
        """Get or create the singleton display instance."""
        if cls._instance is None:
            cls._instance = cls(console=console)
        return cls._instance

    def _check_phase(self, agent_name: str):
        """Auto-show phase header if agent belongs to a new phase."""
        info = _AGENT_PHASE.get(agent_name)
        if info and info[0] != self._current_phase:
            self._current_phase = info[0]
            self.console.print()
            self.console.print(Rule(
                f"Phase {info[0]}: {info[1]['name']}",
                style=f"bold {info[1]['color']}",
            ))
            self.console.print()

    # --- Agent-level methods ---

    def agent_start(self, name: str, desc: str = ""):
        """Show agent starting with colored name and description."""
        self._check_phase(name)
        color = _AGENT_PHASE.get(name, (0, {"color": "white"}))[1]["color"]
        self.console.print(f"[{color}]{name:20s}[/{color}] {desc}")

    def agent_progress(self, name: str, msg: str):
        """Show a dimmed progress update from the agent."""
        self.console.print(f"[dim]  {name:20s} {msg}[/dim]")

    def agent_done(self, name: str, summary: str = ""):
        """Show agent completion with a green check mark."""
        self.console.print(f"[green]\u2713[/green] [bold]{name:20s}[/bold] {summary}")

    def agent_error(self, name: str, error: str):
        """Show agent error with a red cross mark."""
        self.console.print(f"[red]\u2717[/red] [bold red]{name}:[/bold red] {error}")

    # --- Key result display methods ---

    def show_blueprint(self, blueprint: dict):
        """Display a formatted blueprint summary in a bordered panel."""
        tv = blueprint.get("target_variables", [])
        dag = blueprint.get("statistical_dag", {}).get("nodes", [])
        corr = blueprint.get("multiple_testing_correction", {})

        var_names = ", ".join(v.get("name", "?") for v in tv[:4])
        if len(tv) > 4:
            var_names += ", ..."

        node_ids = " -> ".join(n.get("id", "?") for n in dag[:5])
        if len(dag) > 5:
            node_ids += " -> ..."

        content = (
            f"{len(tv)} target variables | {len(dag)} DAG nodes | "
            f"{corr.get('method', '?')} (\u03b1={corr.get('alpha', '?')})\n"
            f"Variables: {var_names}\n"
            f"DAG: {node_ids}"
        )
        self.console.print(Panel(
            content,
            title="Verification Blueprint",
            border_style="cyan",
        ))

    def show_validation(self, interpretation: dict):
        """Display validation conclusion with colored border panel."""
        conclusion = interpretation.get("conclusion", "?")
        confidence = interpretation.get("confidence", 0)
        border = {"VERIFIED": "green", "FALSIFIED": "red"}.get(conclusion, "yellow")

        lines = [f"[bold]{conclusion}[/bold] (confidence: {confidence:.2f})"]
        for finding in interpretation.get("key_findings", []):
            mark = "[green]+[/green]" if finding.get("significant") else "[red]x[/red]"
            p_val = finding.get("corrected_p_value", finding.get("p_value", "?"))
            feature = finding.get("feature", "?")
            try:
                lines.append(f"  [{mark}] {feature}: p={p_val:.3f}")
            except (ValueError, TypeError):
                lines.append(f"  [{mark}] {feature}: p={p_val}")

        self.console.print(Panel(
            "\n".join(lines),
            title="Hypothesis Validation",
            border_style=border,
        ))

    # --- Pipeline-level methods ---

    def show_pipeline_header(self, hypothesis_id: str, sample_id: str):
        """Display the top-level pipeline banner."""
        self.console.print(Rule("OmicsNavigator Pipeline", style="bold white"))
        self.console.print(f"[dim]Hypothesis: {hypothesis_id} | Sample: {sample_id}[/dim]")

    def show_pipeline_complete(self, session_dir: str):
        """Display pipeline completion banner."""
        self.console.print()
        self.console.print(Rule("Pipeline Complete", style="bold green"))
        self.console.print(f"[dim]Output: {session_dir}[/dim]")

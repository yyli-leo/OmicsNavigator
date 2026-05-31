"""
Pipeline orchestrator for OmicsNavigator CLI.
Manages the execution of the 7-action pipeline with step-by-step interaction.
Supports both mock demo mode and real LangGraph agent execution.
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.markdown import Markdown

from .actions import PipelineActions
from .progress import PipelineProgress

if TYPE_CHECKING:
    from cli.agent import SpatialOmicsAgent


class PipelineOrchestrator:
    """
    Manages the execution of the 7-action pipeline demo.

    Handles:
    - State validation (READY state check)
    - Sequential mock and real pipeline execution
    - Progress tracking and formatting
    - Phase and action header display
    """

    # Pipeline structure definition
    PHASES = [
        {
            "num": 1,
            "name": "Plan Module",
            "description": "Establishing physical boundaries and theoretical boundaries",
            "actions": [
                (1, "DataAnalyst"),
                (2, "LiteratureReviewer"),
                (3, "Planner (PI)"),
            ]
        },
        {
            "num": 2,
            "name": "Interpretation Module",
            "description": "Data sampling and high-throughput interpretation",
            "actions": [
                (4, "Anchor-Cluster-Expand Sampling"),
                (5, "High-throughput Interpretation"),
            ]
        },
        {
            "num": 3,
            "name": "Analysis Module",
            "description": "Semantic retrieval and hypothesis validation",
            "actions": [
                (6, "SemanticRetriever"),
                (7, "HypothesisValidator"),
            ]
        },
    ]

    def __init__(
        self,
        console: Console,
        session: PromptSession,
        cli_state: str = "READY",
        config: Optional[dict] = None,
        execution_mode: str = "mock",
        active_phases: Optional[list] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            console: Rich console instance for output
            session: PromptSession instance for user interaction
            cli_state: Current CLI state (should be "READY")
            config: Optional configuration dictionary
            execution_mode: Execution mode ("mock" or "real")
            active_phases: List of phase numbers to execute [1, 2, 3]
            debug_mode: If True, show detailed error messages and warnings
        """
        self.console = console
        self.session = session
        self.cli_state = cli_state
        self.config = config or {}
        self.execution_mode = execution_mode
        self.active_phases = active_phases or [1, 2, 3]
        self.debug_mode = debug_mode

        # Get mockdata directory
        self.mockdata_dir = Path(__file__).parent.parent / "mockdata"
        self.mock_snapshot_name = "reference_pipeline_snapshot"
        self.mock_snapshot_dir = self.mockdata_dir / "reference_pipeline_snapshot"

        # Initialize actions
        self.actions = PipelineActions(console, self.mockdata_dir)

        # Determine if we should use real agents based on execution_mode
        self.use_real_agents = (execution_mode == "real")

        # Create session directory for CLI outputs
        base_output_dir = config.get("session", {}).get("output_dir", "./outputs") if config else "./outputs"
        session_id_prefix = "cli_complete_pipeline"
        session_id = f"{session_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = Path(base_output_dir) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "api_calls").mkdir(exist_ok=True)
        (self.session_dir / "interpretation").mkdir(exist_ok=True)

        self.console.print(f"[dim]Session directory: {self.session_dir}[/dim]")

        # Initialize LangGraph orchestrator if real agents are enabled
        self.langgraph_orchestrator = None
        if self.use_real_agents:
            try:
                from src.workflows.langgraph_orchestrator import LangGraphOrchestrator
                self.langgraph_orchestrator = LangGraphOrchestrator(
                    self.config,
                    session_id_prefix=session_id_prefix
                )
            except ImportError as e:
                self.console.print(f"[yellow]Warning:[/yellow] Could not import LangGraph: {e}")
                self.console.print("[yellow]Falling back to mock pipeline.[/yellow]")
                self.use_real_agents = False

    def _validate_ready_state(self) -> bool:
        """
        Ensure cli_state == 'READY' before starting.

        Returns:
            True if state is valid, False otherwise
        """
        if self.cli_state != "READY":
            self.console.print("[bold red]Error:[/bold red] Pipeline execution requires READY state")
            self.console.print("[yellow]Hint:[/yellow] Run [bold]/status[/bold] to verify system readiness")
            return False
        return True

    async def execute_pipeline(self) -> None:
        """
        Main pipeline execution entry point.

        Executes all 7 actions in order after the initial start confirmation.
        Uses either mock pipeline or real LangGraph agents based on configuration.
        """
        # Validate state before starting
        if not self._validate_ready_state():
            return

        # Print pipeline start banner
        await self._print_pipeline_banner()

        try:
            if self.use_real_agents and self.langgraph_orchestrator:
                # Use real LangGraph agents
                await self._execute_real_pipeline()
            else:
                # Use mock pipeline (existing behavior)
                await self._execute_mock_pipeline()

            # Print completion message
            self._print_completion_message()

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        except Exception as e:
            if self.debug_mode:
                import traceback
                self.console.print(f"\n[bold red]Pipeline error:[/bold red] {e}")
                self.console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
            else:
                self.console.print("\n[yellow]Pipeline completed with warnings.[/yellow]")
                self.console.print("[dim]Run /debug on for detailed error output.[/dim]")

    async def _execute_phase(self, phase: dict) -> None:
        """
        Execute a single phase with all its actions.

        Args:
            phase: Phase configuration dictionary
        """
        # Print phase header
        PipelineProgress.format_phase_header(
            self.console,
            phase["num"],
            phase["name"]
        )

        # Execute each action in the phase
        for action_num, action_name in phase["actions"]:
            await self._execute_action(action_num, action_name)

    async def _execute_action(self, action_num: int, action_name: str) -> None:
        """
        Execute a single action with header.

        Args:
            action_num: Action number (1-7)
            action_name: Action name for display
        """
        # Print action header
        PipelineProgress.format_action_header(
            self.console,
            action_num,
            action_name
        )

        # Execute the action
        await self.actions.execute_action(action_num)

    async def _execute_mock_pipeline(self) -> None:
        """Execute the mock pipeline with pre-defined actions."""
        # Filter phases based on active_phases
        active_phases = [p for p in self.PHASES if p["num"] in self.active_phases]

        if not active_phases:
            self.console.print("[yellow]No active phases to execute.[/yellow]")
            return

        # Display which phases will be executed
        phase_nums = ", ".join([str(p["num"]) for p in active_phases])
        self.console.print(f"[dim]Executing mock pipeline: Phase {phase_nums}[/dim]")

        # Create session metadata
        self._create_session_metadata("mock")
        self._copy_mock_snapshot_outputs()

        # Execute each active phase
        for phase in active_phases:
            await self._execute_phase(phase)

        # Create completion summary
        self._create_completion_summary("mock")

    async def _execute_real_pipeline(self) -> None:
        """Execute the real LangGraph agent workflow."""
        self.console.print("[bold cyan]Executing with Real AI Agents (LangGraph)...[/bold cyan]")
        self.console.print()

        # Get hypothesis from config
        hypotheses = self.config.get("hypotheses", [])
        if not hypotheses:
            self.console.print("[bold red]Error:[/bold red] No hypotheses defined in config")
            return

        hypothesis = hypotheses[0]
        session_config = self.config.get("session", {})
        data_root = session_config.get("data_path", "./data/s255/")
        sample_id = session_config.get("sample_id", "s255")

        # Note: Phase filtering is not yet implemented for real pipeline
        # The full pipeline will always execute in real mode
        if len(self.active_phases) < 3:
            self.console.print("[yellow]Note:[/yellow] Phase filtering is not yet supported in real mode.")
            self.console.print("[yellow]Running full pipeline (all phases) in real mode.[/yellow]")
            self.console.print()

        # Create session metadata
        self._create_session_metadata("real")

        # Display execution status without progress bar
        self.console.print("[bold yellow]Pipeline Status:[/bold yellow]")
        self.console.print("  [dim]Initializing agents...[/dim]")

        # Execute LangGraph workflow
        result = await self.langgraph_orchestrator.execute(
            hypothesis_id=hypothesis.get("id", "H1"),
            hypothesis_description=hypothesis.get("description", ""),
            data_root=data_root,
            sample_id=sample_id
        )

        self.console.print("  [green]✓[/green] Pipeline execution complete")

        # Display results
        self._display_agent_results(result)

        # Create completion summary
        self._create_completion_summary("real", result)

    def _display_agent_results(self, state) -> None:
        """Display results from agent execution."""
        from rich.table import Table

        # Handle both dict and object types
        def get_attr(obj, key, default=None):
            """Get attribute from object or dict."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Phase 1 Results
        dataset_profile = get_attr(state, "dataset_profile")
        if dataset_profile:
            self.console.print("\n[bold yellow]Phase 1: Dataset Analysis[/bold yellow]")
            summary = get_attr(dataset_profile, "summary", "")
            if summary:
                self.console.print(Markdown(summary))

        literature_summaries = get_attr(state, "literature_summaries")
        if literature_summaries:
            self.console.print("\n[bold yellow]Phase 1: Literature Review[/bold yellow]")
            for paper in literature_summaries[:3]:  # Show first 3
                if isinstance(paper, dict):
                    title = paper.get('title', 'Paper')
                else:
                    title = getattr(paper, 'title', 'Paper')
                self.console.print(f"  • {title}")

        analysis_plan = get_attr(state, "analysis_plan")
        if analysis_plan:
            self.console.print("\n[bold yellow]Phase 1: Analysis Plan[/bold yellow]")
            self.console.print("  [green]✓[/green] Analysis plan generated")

        # Phase 2 Results
        roi_count = get_attr(state, "roi_count")
        if roi_count:
            self.console.print(f"\n[bold yellow]Phase 2: ROI Sampling[/bold yellow]")
            self.console.print(f"  Generated [bold cyan]{roi_count}[/bold cyan] ROIs")

        interpretation_reports_path = get_attr(state, "interpretation_reports_path")
        if interpretation_reports_path:
            self.console.print(f"\n[bold yellow]Phase 2: Feature Extraction[/bold yellow]")
            self.console.print(f"  [green]✓[/green] Interpretation complete")

        # Phase 3 Results
        semantic_search_results = get_attr(state, "semantic_search_results")
        if semantic_search_results:
            self.console.print(f"\n[bold yellow]Phase 3: Semantic Search[/bold yellow]")
            matched = get_attr(semantic_search_results, "matched_rois", 0)
            keyword = get_attr(semantic_search_results, "keyword", "")
            self.console.print(f"  [green]✓[/green] Found {matched} ROIs matching '{keyword}'")

        validation_results = get_attr(state, "validation_results")
        if validation_results:
            self.console.print(f"\n[bold yellow]Phase 3: Hypothesis Validation[/bold yellow]")
            conclusion = get_attr(validation_results, "conclusion", "UNKNOWN")
            confidence = get_attr(validation_results, "confidence", 0)

            color = "green" if conclusion == "VERIFIED" else "red"
            self.console.print(f"  Conclusion: [bold {color}]{conclusion}[/bold {color}]")
            self.console.print(f"  Confidence: {confidence:.1%}")

        # Show errors if any
        errors = get_attr(state, "errors", [])
        if errors and callable(get_attr(state, "has_errors", None)):
            has_errors = get_attr(state, "has_errors")()
            if has_errors:
                self.console.print("\n[bold red]Errors encountered:[/bold red]")
                for error in errors[:5]:  # Show first 5
                    self.console.print(f"  • {error}")

    async def _print_pipeline_banner(self) -> None:
        """Print the pipeline start banner."""
        self.console.print()
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]  Starting Hypothesis Verification Pipeline[/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print()
        self.console.print("[yellow]Press Enter to begin the pipeline...[/yellow]")

        # Wait for user to start
        try:
            await self.session.prompt_async("", enable_suspend=True)
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt()

        self.console.print()

    def _print_completion_message(self) -> None:
        """Print the pipeline completion message."""
        self.console.print()
        self.console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
        self.console.print("[bold green]  Pipeline Execution Complete[/bold green]")
        self.console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
        self.console.print()
        hypothesis_id = self.config.get("hypotheses", [{}])[0].get("id", "H1") if self.config.get("hypotheses") else "H1"
        self.console.print("[dim]Pipeline actions have been executed successfully.[/dim]")
        self.console.print(f"[dim]Review completion_summary.json for the final status of hypothesis {hypothesis_id}.[/dim]")
        self.console.print()
        self.console.print(f"[cyan]Outputs saved to: {self.session_dir}[/cyan]")
        self.console.print()
        self.console.print("[cyan]Run /status to check system status or /help for more commands.[/cyan]")
        self.console.print()

    def _copy_mock_snapshot_outputs(self) -> None:
        """Copy lightweight real-run fixture artifacts into the current mock session."""
        if not self.mock_snapshot_dir.exists():
            self.console.print(f"[yellow]Warning:[/yellow] Mock snapshot not found: {self.mock_snapshot_dir}")
            return

        for source in self.mock_snapshot_dir.iterdir():
            target = self.session_dir / source.name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)

    def _load_mock_snapshot_results(self) -> dict:
        """Load validation summary values from the bundled mock snapshot."""
        summary_path = self.mock_snapshot_dir / "display_summary.json"
        if not summary_path.exists():
            return {}

        with open(summary_path, 'r', encoding='utf-8') as f:
            display_summary = json.load(f)

        validator = display_summary.get("validator", {})
        return {
            "hypothesis_id": self.config.get("hypotheses", [{}])[0].get("id", "H1") if self.config.get("hypotheses") else "H1",
            "conclusion": validator.get("conclusion", "unknown"),
            "confidence": validator.get("confidence", 0),
            "data_source": validator.get("data_source", "unknown"),
            "validation_rois": validator.get("validation_rois", 0),
            "significant_findings": validator.get("significant_findings", 0),
            "total_findings": validator.get("total_findings", 0),
            "source_snapshot": self.mock_snapshot_name
        }

    def _create_session_metadata(self, execution_type: str) -> None:
        """Create session metadata file."""
        import json

        session_config = self.config.get("session", {}) if self.config else {}
        metadata = {
            "session_id": self.session_dir.name,
            "execution_type": execution_type,  # "mock" or "real"
            "execution_mode": self.execution_mode,
            "active_phases": self.active_phases,
            "start_time": datetime.now().isoformat(),
            "output_path": str(self.session_dir),
            "source_snapshot": self.mock_snapshot_name if execution_type == "mock" else None,
            "config": {
                "model": self.config.get("system", {}).get("default_model") if self.config else "unknown",
                "sample_id": session_config.get("sample_id", "unknown"),
                "data_path": session_config.get("data_path", "unknown"),
                "hypothesis_id": self.config.get("hypotheses", [{}])[0].get("id", "H1") if self.config and self.config.get("hypotheses") else "H1"
            }
        }

        metadata_path = self.session_dir / "session_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _create_completion_summary(self, execution_type: str, result=None) -> None:
        """Create completion summary file."""
        import json

        summary = {
            "session_id": self.session_dir.name,
            "execution_type": execution_type,
            "completion_time": datetime.now().isoformat(),
            "status": "completed",
            "active_phases": self.active_phases,
            "sample_id": self.config.get("session", {}).get("sample_id", "unknown") if self.config else "unknown",
            "output_path": str(self.session_dir)
        }

        if execution_type == "mock":
            summary["results"] = self._load_mock_snapshot_results()

        # Add real pipeline results if available
        if execution_type == "real" and result:
            summary["results"] = {
                "hypothesis_id": result.get("hypothesis_id", "unknown"),
                "conclusion": result.get("validation_results", {}).get("conclusion", "unknown") if result.get("validation_results") else "unknown",
                "roi_count": result.get("roi_count", 0),
                "errors": result.get("errors", []) if hasattr(result, 'get') else []
            }

        summary_path = self.session_dir / "completion_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

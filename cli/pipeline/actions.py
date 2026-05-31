"""
Pipeline action implementations for OmicsNavigator CLI demo.
Each action corresponds to a step in the spatial omics analysis pipeline.
"""

import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .progress import PipelineProgress


class PipelineActions:
    """Individual action implementations with mock data."""

    MOCK_AGENT_PROGRESS_SECONDS = 5.0

    def __init__(self, console: Console, mockdata_dir: Path):
        self.console = console
        self.mockdata_dir = mockdata_dir
        self.snapshot_dir = mockdata_dir / "reference_pipeline_snapshot"
        self.display_summary = self._load_json("display_summary.json")

    def _load_mock_data(self, filename: str) -> str:
        """Load mock data from file."""
        file_path = self.mockdata_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def _load_json(self, filename: str) -> dict:
        """Load structured mock snapshot data."""
        file_path = self.snapshot_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    async def _show_agent_progress(self, agent_name: str, message: str) -> None:
        """Show a dynamic mock progress bar before rendering the agent's stored output."""
        await PipelineProgress.show_progress(
            self.console,
            f"{agent_name} is {message}...",
            self.MOCK_AGENT_PROGRESS_SECONDS
        )

    async def action_1_data_analyst(self) -> None:
        """
        Action 1: DataAnalyst
        Print the factual dataset profile from the real pipeline snapshot.
        """
        profile = self.display_summary.get("data_analyst", {})
        await self._show_agent_progress(
            "DataAnalyst",
            "inspecting the s255 registry snapshot"
        )

        overview = Table(title="Dataset Profile", border_style="cyan")
        overview.add_column("Metric", style="bold")
        overview.add_column("Value", style="green")
        overview.add_row("Sample", profile.get("sample", "unknown"))
        overview.add_row("Total cells", f"{profile.get('total_cell_count', 0):,}")
        overview.add_row("Biomarkers", str(profile.get("biomarker_count", "unknown")))
        overview.add_row("Cell types", str(profile.get("cell_type_count", "unknown")))
        self.console.print(overview)
        self.console.print()

        composition = Table(title="Top Cell Types", border_style="blue")
        composition.add_column("Cell Type", style="bold")
        composition.add_column("Count", justify="right")
        composition.add_column("Percentage", justify="right")
        for item in profile.get("top_cell_types", []):
            composition.add_row(
                item.get("name", "unknown"),
                f"{item.get('count', 0):,}",
                f"{item.get('percentage', 0):.2f}%"
            )
        self.console.print(composition)
        self.console.print()

        markers = Table(title="Selected Marker Dynamic Range", border_style="magenta")
        markers.add_column("Marker", style="bold")
        markers.add_column("Median", justify="right")
        markers.add_column("95th pct.", justify="right")
        markers.add_column("Max", justify="right")
        for marker in profile.get("marker_dynamic_range", []):
            markers.add_row(
                marker.get("marker", "unknown"),
                f"{marker.get('median', 0):.2f}",
                f"{marker.get('p95', 0):.2f}",
                f"{marker.get('max', 0):.2f}"
            )
        self.console.print(markers)
        self.console.print()

    async def action_2_literature_reviewer(self) -> None:
        """
        Action 2: LiteratureReviewer
        Print the real snapshot's focused query context and conservative takeaways.
        """
        literature = self.display_summary.get("literature_reviewer", {})
        query_fields = literature.get("query_fields", {})

        await self._show_agent_progress(
            "LiteratureReviewer",
            "reviewing the query context and literature synthesis"
        )

        query_table = Table(title="Deep Research Query Context", border_style="blue")
        query_table.add_column("Field", style="bold")
        query_table.add_column("Value", style="white")
        query_table.add_row("Domain", query_fields.get("domain_expertise", "unknown"))
        query_table.add_row("Disease", query_fields.get("disease_context", "unknown"))
        query_table.add_row("Question", query_fields.get("mechanistic_question", "unknown"))
        query_table.add_row("Microenvironment", query_fields.get("target_microenvironment", "unknown"))
        self.console.print(query_table)
        self.console.print()

        takeaways = "\n".join(f"- {item}" for item in literature.get("takeaways", []))
        self.console.print(Panel(takeaways, title="Conservative Literature Takeaways", border_style="cyan"))
        self.console.print()

    async def action_3_planner(self) -> None:
        """
        Action 3: Planner (PI)
        Print target variables and statistical routing from the real blueprint.
        """
        planner = self.display_summary.get("planner", {})
        await self._show_agent_progress(
            "Planner",
            "thinking through target variables and statistical guardrails"
        )

        variable_table = Table(title=f"Blueprint v{planner.get('blueprint_version', 'unknown')} Target Variables", border_style="yellow")
        variable_table.add_column("Variable", style="bold")
        variable_table.add_column("Expected Trend", style="cyan")
        variable_table.add_column("Measurement", style="white")
        for variable in planner.get("target_variables", []):
            variable_table.add_row(
                variable.get("name", "unknown"),
                variable.get("expected_trend", "unknown"),
                variable.get("method", "unknown")
            )
        self.console.print(variable_table)
        self.console.print()

        routing = Table(title="Statistical Guardrails", border_style="green")
        routing.add_column("Component", style="bold")
        routing.add_column("Configured Method", style="white")
        routing.add_row("Primary test", planner.get("primary_test", "unknown"))
        routing.add_row("Fallback test", planner.get("fallback_test", "unknown"))
        routing.add_row("Multiple testing", planner.get("multiple_testing_correction", "unknown"))
        routing.add_row("Alpha", str(planner.get("alpha", "unknown")))
        self.console.print(routing)

        self.console.print()

    def _print_planner_content(self, lines: list) -> None:
        """Print planner content with appropriate formatting."""
        from rich.text import Text

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Status indicators ([x])
            if line.strip().startswith("[x]"):
                # Use Text object to properly escape [x] tag
                text = Text()
                text.append("[x] ", style="green")
                text.append(line[4:], style="green")
                self.console.print(text)

            # Bullet points (*)
            elif line.strip().startswith("* "):
                self.console.print(f"[dim cyan]{line}[/dim cyan]")

            # Indented items (with -)
            elif line.strip().startswith("  -"):
                self.console.print(f"[dim white]{line}[/dim white]")

            # Critical override notes
            elif "[CRITICAL OVERRIDE]" in line:
                # Use Text object to properly escape the tag
                text = Text()
                idx = line.index("[CRITICAL OVERRIDE]")
                text.append(line[:idx], style="bold red")
                text.append("[CRITICAL OVERRIDE]", style="bold red")
                text.append(line[idx+19:], style="bold red")
                self.console.print(text)

            # Dimension headers
            elif line.strip().startswith("Dimension "):
                self.console.print(f"[bold magenta]{line}[/bold magenta]")

            # Plan status and target
            elif line.strip().startswith("Plan STATUS:") or line.strip().startswith("TARGET HYPOTHESIS"):
                self.console.print(f"[bold]{line}[/bold]")

            # Section dividers (---)
            elif line.strip().startswith("---") and not line.strip().startswith("-----"):
                self.console.print(f"[dim]{line}[/dim]")

            # System messages
            elif line.strip().startswith("[SYSTEM]"):
                # Use Text object to properly escape the tag
                text = Text()
                text.append("[SYSTEM] ", style="bold green")
                text.append(line[8:], style="bold green")
                self.console.print(text)

            # Guardrails applied line
            elif "Guardrails Applied:" in line:
                self.console.print(f"[bold]{line}[/bold]")

            # Regular content
            else:
                self.console.print(line)

    def _print_planner_section(self, title: str, lines: list) -> None:
        """Print a planner section (placeholder for compatibility)."""
        pass

    async def action_4_anchor_cluster_expand(self) -> None:
        """
        Action 4: Anchor-Cluster-Expand sampling
        Show the representative ROI examples retained from the real run.
        """
        examples = self.display_summary.get("interpretation", {}).get("roi_examples", [])
        description = "[Phase 2] Anchor-Cluster-Expand is selecting representative ROIs..."
        await PipelineProgress.show_progress(self.console, description, self.MOCK_AGENT_PROGRESS_SECONDS)

        table = Table(title="Representative ROI Manifest", border_style="cyan")
        table.add_column("ROI", style="bold")
        table.add_column("Classification", style="white")
        table.add_column("Confidence", style="green")
        for example in examples:
            table.add_row(
                example.get("roi_key", "unknown"),
                example.get("classification", "unknown"),
                example.get("confidence", "unknown")
            )
        self.console.print(table)
        self.console.print()

        PipelineProgress.format_completion(
            self.console,
            f"Representative ROI manifest loaded: {len(examples)} examples from real pipeline snapshot"
        )

    async def action_5_high_throughput_interpretation(self) -> None:
        """
        Action 5: High-throughput interpretation
        Display the retained multimodal interpretation examples.
        """
        examples = self.display_summary.get("interpretation", {}).get("roi_examples", [])
        description = "[Phase 2] VisualProfiler, OmicsProfiler, and OmicsInterpreter are preparing ROI summaries..."
        await PipelineProgress.show_progress(self.console, description, self.MOCK_AGENT_PROGRESS_SECONDS)

        for example in examples:
            panel_text = (
                f"[bold]Classification:[/bold] {example.get('classification', 'unknown')}\n\n"
                f"[bold cyan]VisualProfiler:[/bold cyan] {example.get('visual_summary', 'unknown')}\n\n"
                f"[bold magenta]OmicsProfiler:[/bold magenta] {example.get('omics_summary', 'unknown')}\n\n"
                f"[dim]Confidence: {example.get('confidence', 'unknown')}[/dim]"
            )
            self.console.print(Panel(panel_text, title=example.get("roi_key", "ROI"), border_style="blue"))
            self.console.print()

        PipelineProgress.format_completion(
            self.console,
            "Representative interpretation reports are stored in this mock output session"
        )

    async def action_6_semantic_retriever(self) -> None:
        """
        Action 6: SemanticRetriever
        Display focused semantic queries generated in the real run.
        """
        retriever = self.display_summary.get("retriever", {})

        # Show search progress
        await PipelineProgress.show_progress(
            self.console,
            "[Phase 3] SemanticRetriever is drafting focused retrieval queries...",
            self.MOCK_AGENT_PROGRESS_SECONDS
        )

        # Display results
        query_table = Table(title="Semantic Retrieval Queries", border_style="green")
        query_table.add_column("#", style="bold", justify="right")
        query_table.add_column("Query", style="cyan")

        for idx, query in enumerate(retriever.get("queries", []), start=1):
            query_table.add_row(str(idx), query)

        self.console.print(query_table)
        self.console.print(f"[dim]Database target: {retriever.get('database', 'interpretation index')}[/dim]")
        self.console.print()

    async def action_7_hypothesis_validator(self) -> None:
        """
        Action 7: HypothesisValidator
        Print the real snapshot's statistical validation result.
        """
        validator = self.display_summary.get("validator", {})
        await self._show_agent_progress(
            "HypothesisValidator",
            "checking full-registry statistics and FDR results"
        )

        conclusion = validator.get("conclusion", "UNKNOWN")
        color = "green" if conclusion == "VERIFIED" else "red"
        summary_text = (
            f"[bold {color}]Conclusion: {conclusion}[/bold {color}]\n\n"
            f"{validator.get('interpretation', '')}\n\n"
            f"[bold]Evidence:[/bold] {validator.get('significant_findings', 0)} of "
            f"{validator.get('total_findings', 0)} target variables are significant after "
            f"{validator.get('correction_method', 'multiple-testing')} correction.\n"
            f"[bold]Data source:[/bold] {validator.get('data_source', 'unknown')} "
            f"({validator.get('validation_rois', 'unknown')} ROIs)"
        )
        self.console.print(Panel(summary_text, title="Hypothesis Validation", border_style=color))
        self.console.print()

        findings = Table(title="Statistical Findings", border_style="yellow")
        findings.add_column("Feature", style="bold")
        findings.add_column("Status")
        findings.add_column("Effect")
        findings.add_column("Raw p", justify="right")
        findings.add_column("FDR p", justify="right")
        for finding in validator.get("findings", []):
            status = finding.get("status", "unknown")
            status_style = "green" if status == "supported" else "red"
            findings.add_row(
                finding.get("feature", "unknown"),
                f"[{status_style}]{status}[/{status_style}]",
                finding.get("effect_size", "unknown"),
                f"{finding.get('p_value', 0):.4g}",
                f"{finding.get('corrected_p_value', 0):.4g}"
            )

        self.console.print(findings)
        self.console.print()

    async def execute_action(self, action_num: int) -> None:
        """
        Execute a specific action by number.

        Args:
            action_num: Action number (1-7)
        """
        action_methods = {
            1: self.action_1_data_analyst,
            2: self.action_2_literature_reviewer,
            3: self.action_3_planner,
            4: self.action_4_anchor_cluster_expand,
            5: self.action_5_high_throughput_interpretation,
            6: self.action_6_semantic_retriever,
            7: self.action_7_hypothesis_validator,
        }

        action_names = {
            1: "DataAnalyst",
            2: "LiteratureReviewer",
            3: "Planner (PI)",
            4: "Anchor-Cluster-Expand Sampling",
            5: "High-throughput Interpretation",
            6: "SemanticRetriever",
            7: "HypothesisValidator",
        }

        if action_num in action_methods:
            method = action_methods[action_num]

            # Check if method is async
            if asyncio.iscoroutinefunction(method):
                await method()
            else:
                method()
        else:
            self.console.print(f"[bold red]Error:[/bold red] Invalid action number: {action_num}")

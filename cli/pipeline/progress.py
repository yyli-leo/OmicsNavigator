"""
Progress bar utilities for pipeline demo.
Uses Rich progress bars for simulated operations.
"""

import asyncio
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class PipelineProgress:
    """Progress bar utilities for pipeline actions."""

    @staticmethod
    async def show_progress(
        console: Console,
        description: str,
        duration_seconds: float,
        total_steps: int = 100
    ) -> None:
        """
        Display a progress bar with simulated timing.

        Args:
            console: Rich console instance
            description: Description text for the progress bar
            duration_seconds: Total duration to simulate (in seconds)
            total_steps: Number of progress steps (default: 100)
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(description, total=total_steps)

            # Calculate delay per step
            delay_per_step = duration_seconds / total_steps

            while not progress.finished:
                await asyncio.sleep(delay_per_step)
                progress.update(task, advance=1)

    @staticmethod
    def format_phase_header(console: Console, phase_num: int, phase_name: str) -> None:
        """Print a formatted phase header."""
        console.print()
        console.print(f"[bold cyan]{'=' * 70}[/bold cyan]")
        console.print(f"[bold cyan]Phase {phase_num}: {phase_name}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 70}[/bold cyan]")
        console.print()

    @staticmethod
    def format_action_header(console: Console, action_num: int, action_name: str) -> None:
        """Print a formatted action header."""
        console.print(f"[bold yellow]Action {action_num}: {action_name}[/bold yellow]")
        console.print()

    @staticmethod
    def format_completion(console: Console, message: str) -> None:
        """Print a completion message."""
        console.print(f"[green]✓[/green] {message}")
        console.print()

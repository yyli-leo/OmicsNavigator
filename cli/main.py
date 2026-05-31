import asyncio
import os
import warnings
import yaml
from collections import Counter
from pathlib import Path
from typing import Dict, List, Callable, Awaitable, Optional, Any, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
# Use relative import assuming execution context is properly set up,
# or absolute import if running as a package (e.g., from src.cli.agent import ...)
from .agent import SpatialOmicsAgent, AgentResponse
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


def _load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load .env file on module import
_load_env_file()

# Suppress third-party warnings by default (scipy, numpy, statsmodels, etc.)
# Re-enabled only in debug mode via /debug command
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SlashCommandCompleter(Completer):
    """
    Advanced custom completer that provides a dropdown menu with
    command descriptions (meta-information) and dynamic filtering.
    """

    def __init__(self, commands: Dict[str, str]):
        self.commands_dict: Dict[str, str] = commands

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        # Capture exactly what the user has typed before the cursor
        text: str = document.text_before_cursor

        # Only trigger the completion menu if the input starts with a slash
        if not text.startswith('/'):
            return

        # If there's a space, the user has finished typing the command, stop completing
        if " " in text:
            return

        # Extract the current word being typed (e.g., "/" or "/a")
        current_word: str = text.split(" ")[0]

        # Filter and yield matching commands dynamically
        for cmd, description in self.commands_dict.items():
            if cmd.startswith(current_word):
                yield Completion(
                    text=cmd,
                    # Replace the partially typed text with the full command
                    start_position=-len(current_word),
                    # The actual text shown in the left column
                    display=cmd,
                    # The description shown in the right column (like Claude Code)
                    display_meta=description
                )


class AgentCLI:
    """
    Command-Line Interface manager for the Spatial Omics tool.
    Handles the asynchronous REPL and slash command routing.
    """

    def __init__(self, work_dir: Path):
        self.work_dir: Path = work_dir
        self.console: Console = Console()
        self.agent: SpatialOmicsAgent = SpatialOmicsAgent()

        # Load configuration
        self.config: Optional[Dict[str, Any]] = self._load_config()
        self.cli_state: str = "INIT"  # States: INIT, READY, RUNNING

        # Execution mode management
        # Execution mode: read from config or default to mock
        exec_config = self.config.get("system", {}).get("execution", {}) if self.config else {}
        self.execution_mode: str = exec_config.get("default_mode", "mock")  # "mock" or "real"
        self.active_phases: List[int] = [1, 2, 3]  # Default all phases
        self.debug_mode: bool = False  # Debug mode for verbose output

        # Define native slash commands and their descriptions
        self.commands: Dict[str, str] = {
            "/help": "Show available commands",
            "/model": "View model & agent configuration",
            "/data": "View dataset & metadata",
            "/hypothesis": "View hypothesis definitions",
            "/status": "System readiness check",
            "/mock": "Run mock pipeline (fast demo)",
            "/start": "Run the current mode pipeline",
            "/mode": "Switch execution mode (mock/real)",
            "/debug": "Toggle debug output",
            "/clear": "Clear the terminal screen",
            "/exit": "Exit the CLI",
            "/quit": "Exit the CLI",
        }

        # Initialize the interactive auto-completer
        self.completer: SlashCommandCompleter = SlashCommandCompleter(self.commands)

        self.session: PromptSession = PromptSession(completer=self.completer)

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from config.yaml."""
        config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            self.console.print(f"[bold red]Config file not found:[/bold red] {config_path}")
            return None

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.console.print(f"[bold red]Error loading config:[/bold red] {e}")
            return None

    async def run_loop(self) -> None:
        """
        The core asynchronous REPL that maintains the terminal session.
        """
        self._print_banner()

        while True:
            try:
                # Asynchronously wait for user input
                raw_input: str = await self.session.prompt_async("OmicsNavigator> ")
                text: str = raw_input.strip()

                if not text:
                    continue

                # Command Routing Logic
                if text in ("/exit", "/quit"):
                    self.console.print("[dim]Shutting down OmicsNavigator Agent interface...[/dim]")
                    break
                elif text == "/help":
                    self._show_help()
                elif text == "/clear":
                    self.console.clear()
                    self._print_banner()
                elif text == "/model":
                    self._cmd_model()
                elif text == "/data":
                    self._cmd_data()
                elif text == "/hypothesis":
                    self._cmd_hypothesis()
                elif text == "/status":
                    self._cmd_status()
                elif text == "/mock":
                    await self._cmd_mock()
                elif text == "/start":
                    await self._cmd_start()
                elif text.startswith("/mode"):
                    await self._cmd_mode(text[5:].strip())  # Extract arguments after "/mode"
                elif text.startswith("/debug"):
                    await self._cmd_debug(text[6:].strip())
                elif text.startswith("/"):
                    self.console.print(
                        f"[bold red]Unknown command:[/bold red] {text}. Type [bold cyan]/help[/bold cyan] for options.")
                else:
                    # Pass standard inputs to the mock LLM agent
                    response: AgentResponse = self.agent.process_query(text)
                    self._render_response(response)

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully without exiting the application
                continue
            except EOFError:
                # Handle Ctrl+D to exit
                self.console.print("\n[dim]Session terminated by user (EOF).[/dim]")
                break

    def _render_response(self, response: AgentResponse) -> None:
        """
        Translates the strict dataclass schema into rich terminal formatting.
        """
        self.console.print(Markdown(response.content))
        if response.tool_used:
            self.console.print(f"[dim italic]System log: Triggered backend tool -> {response.tool_used}[/dim italic]\n")
        else:
            self.console.print()  # Add a newline for readability

    def _show_help(self) -> None:
        """Renders categorized help with descriptions and examples."""
        groups = [
            ("Configuration", [
                ("/model", "View model & agent configuration", "/model"),
                ("/data", "View dataset & metadata", "/data"),
                ("/hypothesis", "View hypothesis definitions", "/hypothesis"),
                ("/status", "System readiness check", "/status"),
            ]),
            ("Pipeline Control", [
                ("/start", "Run the current mode pipeline", "/start"),
                ("/mock", "Run mock pipeline (fast demo)", "/mock"),
                ("/mode", "Switch execution mode", "/mode real"),
            ]),
            ("Utilities", [
                ("/debug", "Toggle debug output", "/debug on"),
                ("/clear", "Clear the terminal screen", "/clear"),
                ("/exit", "Exit the CLI", "/exit"),
                ("/quit", "Exit the CLI", "/quit"),
            ]),
        ]

        for group_name, cmds in groups:
            self.console.print(f"  [bold]{group_name}[/bold]")
            table = Table(show_header=True, box=None, padding=(0, 2), border_style="dim")
            table.add_column("Command", style="bold cyan", width=12)
            table.add_column("Description", style="white", width=40)
            table.add_column("Example", style="dim", width=16)
            for cmd, desc, example in cmds:
                table.add_row(cmd, desc, f"[dim]{example}[/dim]")
            self.console.print(table)
            self.console.print()

        self.console.print("[dim]Tip: Use /mode mock for the cost-free demo, then /start to begin analysis.[/dim]")
        self.console.print()

    # ==========================================
    # Config-based Command Handlers
    # ==========================================

    def _cmd_model(self) -> None:
        """Display current model configuration including per-agent overrides."""
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            self.console.print("[yellow]Hint:[/yellow] Ensure config.yaml exists in cli/")
            return

        system_config = self.config.get("system", {})

        if not system_config or "default_model" not in system_config:
            self.console.print("[bold yellow]Warning:[/bold yellow] Model configuration not found in config.yaml")
            self.console.print("[dim]Required: system.default_model[/dim]")
            return

        # Default model config
        table = Table(title="Model Configuration", border_style="cyan")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="green")

        table.add_row("Default Model", system_config.get("default_model", "Not set"))
        table.add_row("API Key Env Var", system_config.get("api_key_env_var", "Not set"))
        table.add_row("Proxy", self._format_proxy_config(system_config))
        table.add_row("Log Level", system_config.get("log_level", "Not set"))
        table.add_row("Max Workers", str(system_config.get("max_workers", "Not set")))

        self.console.print(table)

        # Per-agent model overrides
        agents_config = system_config.get("agents", {})
        default_model = system_config.get("default_model", "unknown")

        if agents_config:
            agent_table = Table(title="Agent Model Overrides", border_style="yellow")
            agent_table.add_column("Agent", style="bold yellow")
            agent_table.add_column("Model", style="green")
            agent_table.add_column("Temperature", style="cyan")
            agent_table.add_column("Source", style="dim")

            for agent_name, agent_cfg in agents_config.items():
                model = agent_cfg.get("model", default_model)
                temp = str(agent_cfg.get("temperature", "default"))
                source = "[yellow]override[/yellow]" if "model" in agent_cfg else "default"
                agent_table.add_row(agent_name, model, temp, source)

            self.console.print(agent_table)

        self.console.print()

    def _format_proxy_config(self, system_config: Dict[str, Any]) -> str:
        """Return a compact user-facing proxy summary."""
        proxy_config = system_config.get("proxy", {})
        if not proxy_config.get("enabled", False):
            return "disabled"

        scheme = proxy_config.get("scheme", "http")
        host = proxy_config.get("host", "127.0.0.1")
        port = proxy_config.get("port", "not set")
        return f"{scheme}://{host}:{port}"

    def _cmd_data(self) -> None:
        """
        Display current dataset configuration from config.yaml.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            return

        session_config = self.config.get("session", {})
        metadata_config = self.config.get("metadata", {})

        if not session_config:
            self.console.print("[bold yellow]Warning:[/bold yellow] Dataset configuration not found in config.yaml")
            self.console.print("[dim]Required: session section with data_path[/dim]")
            return

        # Create a table for dataset info
        table = Table(title="Dataset Configuration", border_style="blue")
        table.add_column("Parameter", style="bold blue")
        table.add_column("Value", style="green")

        data_path = session_config.get("data_path", "Not set")
        output_dir = session_config.get("output_dir", "Not set")
        sample_id = session_config.get("sample_id", "Not set")

        # Check if paths exist
        data_path_obj = Path(data_path)
        path_status = "[green]✓ Exists[/green]" if data_path_obj.exists() else "[red]✗ Not found[/red]"

        table.add_row("Sample ID", sample_id)
        table.add_row("Data Path", data_path)
        table.add_row("Path Status", path_status)
        table.add_row("Output Directory", output_dir)

        self.console.print(table)

        # Show metadata if available
        if metadata_config:
            self.console.print("\n[bold]Metadata Context:[/bold]")
            meta_table = Table(show_header=False, box=None, padding=(0, 2))
            meta_table.add_column("Key", style="cyan")
            meta_table.add_column("Value", style="white")

            for key, value in metadata_config.items():
                if key == "clinical_mapping" and isinstance(value, dict):
                    counts = Counter(value.values())
                    dist = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                    meta_table.add_row(
                        key.capitalize(),
                        f"{len(value)} regions mapped\n  [dim]└ {dist}[/dim]"
                    )
                else:
                    meta_table.add_row(key.capitalize(), str(value))

            self.console.print(meta_table)

        self.console.print()

    def _cmd_hypothesis(self) -> None:
        """
        Display current hypothesis configuration from config.yaml.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            return

        hypotheses = self.config.get("hypotheses", [])

        if not hypotheses:
            self.console.print("[bold yellow]Warning:[/bold yellow] No hypotheses defined in config.yaml")
            self.console.print("[dim]Required: hypotheses section with at least one hypothesis[/dim]")
            self.console.print("\n[yellow]Example configuration:[/yellow]")
            example_yaml = """
hypotheses:
  - id: "H1"
    description: "Your hypothesis description here..."
"""
            self.console.print(Markdown(f"```yaml{example_yaml}```"))
            return

        # Create a table for hypotheses
        table = Table(title=f"Defined Hypotheses ({len(hypotheses)})", border_style="magenta")
        table.add_column("ID", style="bold magenta", width=8)
        table.add_column("Description", style="white", width=60)

        for hyp in hypotheses:
            hyp_id = hyp.get("id", "Unknown")
            description = hyp.get("description", "No description")
            table.add_row(hyp_id, description)

        self.console.print(table)
        self.console.print()

    def _get_readiness_checks(self) -> tuple[Dict[str, bool], Dict[str, Any]]:
        """Return readiness checks and reusable context for status/start."""
        checks = {
            "Model": False,
            "Dataset": False,
            "Metadata": False,
            "Hypothesis": False
        }
        context: Dict[str, Any] = {
            "system_config": {},
            "session_config": {},
            "metadata_config": {},
            "hypotheses": [],
            "data_path": ""
        }

        if self.config is None:
            return checks, context

        system_config = self.config.get("system", {})
        session_config = self.config.get("session", {})
        metadata_config = self.config.get("metadata", {})
        hypotheses = self.config.get("hypotheses", [])
        data_path = session_config.get("data_path", "") if session_config else ""

        context.update({
            "system_config": system_config,
            "session_config": session_config,
            "metadata_config": metadata_config,
            "hypotheses": hypotheses,
            "data_path": data_path
        })

        if system_config and "default_model" in system_config:
            checks["Model"] = True

        if data_path and Path(data_path).exists():
            checks["Dataset"] = True

        if metadata_config and all(k in metadata_config for k in ["technology", "tissue"]):
            checks["Metadata"] = True

        if hypotheses:
            checks["Hypothesis"] = True

        return checks, context

    def _set_ready_state(self, checks: Dict[str, bool]) -> bool:
        """Update CLI state from readiness checks."""
        all_ok = all(checks.values())
        self.cli_state = "READY" if all_ok else "INIT"
        return all_ok

    def _cmd_status(self) -> None:
        """
        Diagnostic command to verify the AnalysisContext.
        Displays dataset, metadata keys, hypothesis status, and execution mode.
        """
        if self.config is None:
            self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
            return

        checks, context = self._get_readiness_checks()
        system_config = context["system_config"]
        metadata_config = context["metadata_config"]
        hypotheses = context["hypotheses"]
        data_path = context["data_path"]

        # Build status table
        table = Table(title="AnalysisContext Status", border_style="white")
        table.add_column("Component", style="bold white")
        table.add_column("Status", width=12)
        table.add_column("Details", style="dim")

        for component, status in checks.items():
            status_text = "[green]✓ OK[/green]" if status else "[red]✗ Missing[/red]"
            details = ""

            if component == "Model" and status:
                details = f"Using: {system_config.get('default_model')}"
            elif component == "Model" and not status:
                details = "Set system.default_model in config.yaml"

            elif component == "Dataset" and status:
                details = f"Path: {data_path}"
            elif component == "Dataset" and not status:
                details = "Set session.data_path in config.yaml"

            elif component == "Metadata" and status:
                details = f"{metadata_config.get('technology')} / {metadata_config.get('tissue')}"
            elif component == "Metadata" and not status:
                details = "Set metadata.technology and metadata.tissue"

            elif component == "Hypothesis" and status:
                details = f"{len(hypotheses)} hypothesis(es) defined"
            elif component == "Hypothesis" and not status:
                details = "Add hypotheses to config.yaml"

            table.add_row(component, status_text, details)

        self.console.print(table)

        # Display execution mode and active phases
        mode_table = Table(title="Execution Configuration", border_style="cyan")
        mode_table.add_column("Setting", style="bold cyan")
        mode_table.add_column("Value", style="green")

        mode_text = f"[bold]{self.execution_mode.upper()}[/bold]"
        mode_table.add_row("Execution Mode", mode_text)
        mode_table.add_row("Active Phases", f"[bold]{', '.join(map(str, self.active_phases))}[/bold]")
        mode_table.add_row("Proxy", self._format_proxy_config(system_config))

        self.console.print(mode_table)

        # Agent configuration table
        phases_config = self.config.get("phases", {})
        agents_overrides = self.config.get("system", {}).get("agents", {})
        default_model = self.config.get("system", {}).get("default_model", "unknown")

        if phases_config:
            agent_table = Table(title="Agent Configuration", border_style="yellow")
            agent_table.add_column("Agent", style="bold yellow")
            agent_table.add_column("Phase", style="cyan", width=6)
            agent_table.add_column("Model", style="green")
            agent_table.add_column("Temp.", style="dim")

            for phase_key in sorted(phases_config.keys()):
                phase_cfg = phases_config[phase_key]
                phase_num = ''.join(c for c in phase_key if c.isdigit())
                for agent_name in phase_cfg.get("agents", []):
                    lookup_key = agent_name.lower()
                    override = agents_overrides.get(lookup_key, {})
                    model = override.get("model", default_model)
                    temp = str(override.get("temperature", "default"))
                    is_custom = "model" in override or "temperature" in override
                    marker = "  [yellow]★[/yellow]" if is_custom else ""
                    agent_table.add_row(agent_name, phase_num, model, temp + marker)

            self.console.print(agent_table)
            if agents_overrides:
                self.console.print("[dim]  ★ = custom override[/dim]")

        # Overall status
        if self._set_ready_state(checks):
            self.console.print("\n[bold green]✓ System Status: READY[/bold green]")
            self.console.print("[dim]All components validated. You can proceed with analysis.[/dim]")
        else:
            missing = [k for k, v in checks.items() if not v]
            self.console.print(f"\n[bold yellow]⚠ System Status: INIT[/bold yellow]")
            self.console.print(f"[dim]Missing components: {', '.join(missing)}[/dim]")
            self.console.print("[yellow]Please configure the missing items in config.yaml[/yellow]")

        self.console.print()

    async def _cmd_start(self) -> None:
        """
        Execute the spatial omics analysis pipeline.
        Uses the current execution mode (mock or real).
        Runs a readiness check automatically if the CLI is still in INIT state.
        """
        from .pipeline.orchestrator import PipelineOrchestrator

        # State validation
        if self.cli_state != "READY":
            if self.config is None:
                self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
                return
            checks, _ = self._get_readiness_checks()
            if not self._set_ready_state(checks):
                missing = [k for k, v in checks.items() if not v]
                self.console.print("[bold red]Error:[/bold red] Pipeline execution requires a valid configuration")
                self.console.print(f"[dim]Missing components: {', '.join(missing)}[/dim]")
                self.console.print("[yellow]Hint:[/yellow] Run [bold]/status[/bold] for detailed diagnostics")
                return

        # If trying to run real pipeline, validate prerequisites
        if self.execution_mode == "real":
            if not self._validate_real_mode():
                self.console.print("[yellow]Falling back to mock mode...[/yellow]")
                self.execution_mode = "mock"

        # Create and execute pipeline with mode and phases
        orchestrator = PipelineOrchestrator(
            self.console,
            self.session,
            self.cli_state,
            self.config,
            self.execution_mode,
            self.active_phases,
            self.debug_mode
        )
        await orchestrator.execute_pipeline()

    async def _cmd_mock(self) -> None:
        """
        Execute the mock pipeline.
        Forces mock mode regardless of current execution mode.
        """
        from .pipeline.orchestrator import PipelineOrchestrator

        # State validation
        if self.cli_state != "READY":
            if self.config is None:
                self.console.print("[bold red]Error:[/bold red] Configuration not loaded.")
                return
            checks, _ = self._get_readiness_checks()
            if not self._set_ready_state(checks):
                missing = [k for k, v in checks.items() if not v]
                self.console.print("[bold red]Error:[/bold red] Pipeline execution requires a valid configuration")
                self.console.print(f"[dim]Missing components: {', '.join(missing)}[/dim]")
                self.console.print("[yellow]Hint:[/yellow] Run [bold]/status[/bold] for detailed diagnostics")
                return

        # Force mock mode
        self.console.print("[cyan]Executing mock pipeline (fast demo mode)[/cyan]")
        self.execution_mode = "mock"

        # Create and execute pipeline
        orchestrator = PipelineOrchestrator(
            self.console,
            self.session,
            self.cli_state,
            self.config,
            "mock",  # Force mock mode
            self.active_phases,
            self.debug_mode
        )
        await orchestrator.execute_pipeline()

    async def _cmd_mode(self, args: str) -> None:
        """
        Switch execution mode between mock and real.

        Args:
            args: Mode argument ("mock", "real", or empty to show current)
        """
        if not args:
            exec_config = self.config.get("system", {}).get("execution", {}) if self.config else {}
            mode_color = "green" if self.execution_mode == "real" else "cyan"

            table = Table(title="Execution Mode", border_style=mode_color)
            table.add_column("Setting", style="bold")
            table.add_column("Value", style=mode_color)

            table.add_row("Current Mode", f"[bold]{self.execution_mode.upper()}[/bold]")
            table.add_row("Default Mode", exec_config.get("default_mode", "unknown"))
            table.add_row("Mode Switching", "Allowed" if exec_config.get("allow_mode_switch", True) else "Disabled")

            self.console.print(table)

            if self.execution_mode == "mock":
                self.console.print("[dim]Mock mode uses pre-generated outputs for fast demonstration.[/dim]")
            else:
                self.console.print("[dim]Real mode uses LLM-powered agents for actual analysis.[/dim]")
            self.console.print("[dim]Usage: /mode [mock|real][/dim]")
            return

        mode = args.lower()

        if mode == "mock":
            self.execution_mode = "mock"
            self.console.print("[green]Mode: MOCK[/green] (fast demo)")
            self.console.print("[dim]Use /start to run the mock pipeline[/dim]")

        elif mode == "real":
            # Validate prerequisites
            if not self._validate_real_mode():
                self.console.print("[yellow]Cannot switch to real mode. Please fix the issues above.[/yellow]")
                return

            self.execution_mode = "real"
            self.console.print("[green]Mode: REAL[/green] (LLM-powered)")
            self.console.print("[dim]Use /start to run the real pipeline[/dim]")

        else:
            self.console.print(f"[bold red]Invalid mode:[/bold red] {args}")
            self.console.print("[yellow]Valid options: mock, real[/yellow]")
            self.console.print("[dim]Usage: /mode [mock|real][/dim]")

    async def _cmd_debug(self, args: str) -> None:
        """Toggle debug mode for verbose output (warnings, tracebacks)."""
        if not args:
            # Toggle
            self.debug_mode = not self.debug_mode
        elif args.lower() in ("on", "true", "1"):
            self.debug_mode = True
        elif args.lower() in ("off", "false", "0"):
            self.debug_mode = False
        else:
            self.console.print(f"[bold red]Invalid argument:[/bold red] {args}")
            self.console.print("[yellow]Usage: /debug [on|off][/yellow]")
            return

        if self.debug_mode:
            warnings.filterwarnings("default", category=RuntimeWarning)
            warnings.filterwarnings("default", category=UserWarning)
            self.console.print("[green]Debug mode: ON[/green] (warnings visible)")
        else:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            self.console.print("[cyan]Debug mode: OFF[/cyan] (warnings suppressed)")

    def _validate_real_mode(self) -> bool:
        """
        Check if real mode prerequisites are met.

        Returns:
            True if all prerequisites are satisfied, False otherwise
        """
        import os

        all_valid = True

        # Check API key
        api_key_env_var = self.config.get("system", {}).get("api_key_env_var", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env_var)

        if not api_key:
            self.console.print(f"[red]✗ API key not found:[/red] {api_key_env_var}")
            self.console.print(f"  [dim]Set environment variable: export {api_key_env_var}=your_key[/dim]")
            all_valid = False
        else:
            self.console.print(f"[green]✓ API key found: {api_key_env_var}[/green]")

        system_config = self.config.get("system", {})
        self.console.print(f"[green]✓ Proxy setting: {self._format_proxy_config(system_config)}[/green]")

        # Check LangGraph availability
        try:
            from src.workflows.langgraph_orchestrator import LangGraphOrchestrator
            self.console.print("[green]✓ LangGraph orchestrator available[/green]")
        except ImportError as e:
            self.console.print(f"[red]✗ LangGraph not available: {e}[/red]")
            all_valid = False

        # Check hypothesis configuration
        hypotheses = self.config.get("hypotheses", [])
        if not hypotheses:
            self.console.print("[yellow]⚠ No hypotheses defined in config.yaml[/yellow]")
        else:
            self.console.print(f"[green]✓ {len(hypotheses)} hypothesis(es) defined[/green]")

        # Check data path
        data_path = self.config.get("session", {}).get("data_path")
        if not data_path:
            self.console.print("[yellow]⚠ No data path configured[/yellow]")
        elif not Path(data_path).exists():
            self.console.print(f"[yellow]⚠ Data path does not exist: {data_path}[/yellow]")
        else:
            self.console.print(f"[green]✓ Data path exists: {data_path}[/green]")

        return all_valid

    def _print_banner(self) -> None:
        """
        Renders a Claude-Code style welcome banner with a pixel art logo,
        split-pane layout, and system information.
        """
        # 1. Define the Pixel Art Logo using Unicode blocks and Rich color tags
        # Replace this string with your own generated ASCII/ANSI art
        bot_color: str = "#AEC6CF"  # Light pastel blue
        eye_color: str = "#000000"  # Black

        # Using a multiline f-string to draw the bot
        logo_art: str = (
            f"[{bot_color}]▄▄▄▄▄▄▄ [/]\n"
            f"[{bot_color}]██[{eye_color}]█[{bot_color}]█[{eye_color}]█[{bot_color}]██[/]\n"
            f"[{bot_color}]█▀▀▀▀▀█[/]\n"
            f"[{bot_color}]█     █  [/]"
        )

        # 2. Construct the Left Column Content
        model_name: str = self.agent.current_model
        workspace_path: str = str(self.work_dir.resolve())

        left_content: str = (
            "\n[bold]Welcome back![/bold]\n\n"
            f"{logo_art}\n\n"
            f"[dim]{model_name} · Local Environment\n"
            f"{workspace_path}[/dim]"
        )

        # 3. Construct the Right Column Content
        right_color: str = "#D58B8B"  # Muted red/pink from the screenshot
        right_content: str = (
            f"\n[bold {right_color}]Tips for getting started[/bold {right_color}]\n"
            "Run [bold]/help[/bold] to see available commands or [bold]/model[/bold] to switch backends.\n\n"
            f"[bold {right_color}]Recent activity[/bold {right_color}]\n"
            "[dim]No recent activity found.[/dim]\n"
        )

        # 4. Create a borderless Grid Table for layout
        grid: Table = Table.grid(expand=True, padding=(0, 2))
        # Left column for logo (centered), Right column for text (left-aligned)
        grid.add_column(justify="center", ratio=4)
        grid.add_column(justify="left", ratio=6)

        # Add the content to the grid, adding a vertical separator line conceptually
        # Rich's grid doesn't draw internal lines by default, which matches the UI goal
        grid.add_row(Align.center(left_content), right_content)

        # 5. Wrap everything in a styled Panel
        banner: Panel = Panel(
            grid,
            title=f"[bold {right_color}]OmicsNavigator v0.1.0[/]",
            title_align="left",
            border_style=right_color,
            padding=(1, 2)
        )

        self.console.print(banner)
        self.console.print()  # Add padding before the prompt starts


async def app_entry() -> None:
    """
    Standard entry point for the asyncio event loop.
    """
    # Utilizing Pathlib for robust directory resolution
    base_dir: Path = Path.cwd() / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    cli = AgentCLI(work_dir=base_dir)
    await cli.run_loop()


if __name__ == "__main__":
    asyncio.run(app_entry())

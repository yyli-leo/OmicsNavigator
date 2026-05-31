"""
Prompt Manager for Multi-Agent System.

Centralized prompt template management using Jinja2.
Supports agent-based template organization, shared components,
and backward compatibility during migration.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
import json


class PromptManager:
    """
    Centralized prompt template manager for multi-agent system.

    Features:
    - Jinja2 template loading and rendering
    - Agent-based template organization
    - Shared component support via includes
    - Template caching for performance
    - Fallback to hardcoded prompts during migration
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize PromptManager.

        Args:
            prompts_dir: Root prompts directory (default: src/prompts/)
        """
        self.prompts_dir = prompts_dir or Path(__file__).parent.parent / "prompts"
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._cache: Dict[str, Template] = {}

    def get_system_prompt(
        self,
        agent_name: str,
        use_template: bool = True,
        fallback_prompt: Optional[str] = None
    ) -> str:
        """
        Load system prompt for an agent.

        Args:
            agent_name: Agent identifier (e.g., 'data_analyst', 'visual_profiler')
            use_template: If False, return fallback_prompt directly
            fallback_prompt: Hardcoded fallback if template fails

        Returns:
            System prompt string
        """
        if not use_template:
            if fallback_prompt:
                return fallback_prompt
            raise ValueError("No fallback prompt provided when use_template=False")

        template_path = f"{agent_name}/system.jinja2"
        return self._render_template(template_path, {})

    def get_task_prompt(
        self,
        agent_name: str,
        task_name: str,
        context: Dict[str, Any],
        use_template: bool = True,
        fallback_prompt: Optional[str] = None
    ) -> str:
        """
        Load and render task-specific prompt.

        Args:
            agent_name: Agent identifier
            task_name: Task identifier (e.g., 'initial_analysis', 'analyze_roi')
            context: Variables for template rendering
            use_template: If False, format fallback_prompt with context
            fallback_prompt: Hardcoded fallback string with {placeholders}

        Returns:
            Rendered prompt string
        """
        if not use_template:
            if fallback_prompt:
                try:
                    return fallback_prompt.format(**context)
                except KeyError as e:
                    raise ValueError(
                        f"Missing context variable for fallback prompt: {e}"
                    )
            raise ValueError("No fallback prompt provided when use_template=False")

        template_path = f"{agent_name}/{task_name}.jinja2"
        return self._render_template(template_path, context)

    def _render_template(self, template_path: str, context: Dict[str, Any]) -> str:
        """
        Render template with caching.

        Args:
            template_path: Template path relative to prompts_dir
            context: Template variables

        Returns:
            Rendered template string
        """
        # Load template if not cached
        if template_path not in self._cache:
            try:
                self._cache[template_path] = self.env.get_template(template_path)
            except Exception as e:
                raise PromptLoadError(
                    f"Failed to load template '{template_path}': {e}"
                )

        # Render with context
        try:
            return self._cache[template_path].render(**context)
        except Exception as e:
            raise PromptRenderError(
                f"Failed to render template '{template_path}': {e}"
            )

    def load_shared_component(self, component_name: str) -> str:
        """
        Load shared component template.

        Args:
            component_name: Component name (e.g., 'tissue_context', 'biomarker_info')

        Returns:
            Rendered component string (no variables needed for shared components)
        """
        template_path = f"shared/components/{component_name}.jinja2"
        return self._render_template(template_path, {})

    def load_reference_data(self, data_name: str) -> Dict[str, Any]:
        """
        Load shared reference data (JSON).

        Args:
            data_name: Data file name (without .json)

        Returns:
            Parsed JSON dictionary
        """
        data_path = self.prompts_dir / "shared" / "data" / f"{data_name}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Reference data not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clear_cache(self):
        """Clear template cache. Useful for testing or dynamic template updates."""
        self._cache.clear()


class PromptLoadError(Exception):
    """Raised when template loading fails."""
    pass


class PromptRenderError(Exception):
    """Raised when template rendering fails."""
    pass

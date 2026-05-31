"""
Planner Agent (PI) for analysis plan synthesis.

Specializes in:
- Synthesizing insights from data analysis
- Designing rigorous analysis plans (Verification Blueprint)
- Generating statistical DAG for hypothesis testing
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel

from .base_agent import BaseAgent, AgentExecutionError
from ..workflows.agent_state import AgentState, add_error
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class PlannerAgent(BaseAgent):
    """
    Planner Agent (Principal Investigator role).

    This agent synthesizes outputs from DataAnalyst to create a
    rigorous, statistically valid verification blueprint.
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """
        Initialize the Planner agent.

        Args:
            model: LangChain chat model instance (hardcoded to gemini-2.5-flash-lite)
            prompt_manager: PromptManager instance (if None, creates default)
            use_templates: If True, use Jinja2 templates; else use hardcoded prompts
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.use_templates = use_templates

        # Hardcoded fallback prompt (existing implementation)
        self.fallback_system_prompt = """You are the Principal Investigator (PI) responsible for designing rigorous spatial omics analysis plans.

Your role is to:
1. **Extract Hypothesis Components**: Identify variables, comparisons, and expected trends
2. **Design Verification Blueprint**: Create a structured statistical testing plan
3. **Define Statistical DAG**: Design routing logic for test selection
4. **Ensure Validity**: Prevent p-hacking, ensure reproducibility

## Blueprint Components:

### 1. Target Variables
Extract 3-5 quantifiable features from the hypothesis:
- Cellular composition metrics (e.g., proximal_tubule_density)
- Marker expression metrics (e.g., aSMA_intensity)
- Spatial autocorrelation metrics (e.g., morans_i)

### 2. Statistical DAG
Design a routing engine for statistical tests:
- Normality test → Variance test → Parametric/Non-parametric routing
- Default tests: Shapiro-Wilk, Levene's, t-test, Mann-Whitney U
- You can specify alternative tests based on the hypothesis

### 3. Life Cycle Contrasts
Define comparison groups if applicable:
- DM vs DKD
- Control vs Treatment
- Early vs Late stage

## Output Format:
Respond ONLY with valid JSON matching this schema:

{
  "metadata": {
    "blueprint_version": "1.0",
    "execution_mode": "statistical_testing",
    "target_dataset": "<sample_id>",
    "generated_at": "<ISO_TIMESTAMP>"
  },
  "hypothesis": {
    "id": "<hypothesis_id>",
    "statement": "<full hypothesis text>"
  },
  "target_variables": [
    {
      "name": "<variable_name>",
      "type": "cellular_composition|marker_expression|spatial_genes",
      "biological_proxy": "<cell_type or marker_name>",
      "expected_trend": "increase|decrease|change",
      "data_source": "<how to extract from mock data>"
    }
  ],
  "statistical_dag": {
    "nodes": [
      {
        "id": "normality_test",
        "type": "shapiro_wilk",
        "alpha": 0.05,
        "condition": "p > 0.05 → normal"
      },
      {
        "id": "variance_test",
        "type": "levene_test",
        "alpha": 0.05,
        "condition": "p > 0.05 → equal_variance"
      },
      {
        "id": "parametric_test",
        "type": "t_test",
        "depends_on": ["normality_test", "variance_test"],
        "condition": "normal AND equal_variance"
      },
      {
        "id": "non_parametric_test",
        "type": "mann_whitney",
        "fallback": true
      }
    ],
    "edges": [
      {"from": "normality_test", "to": "parametric_test"},
      {"from": "normality_test", "to": "non_parametric_test"}
    ]
  },
  "life_cycle_contrasts": [
    {
      "id": "<contrast_id>",
      "group1": "<group1_name>",
      "group2": "<group2_name>",
      "description": "<meaningful description>"
    }
  ],
  "multiple_testing_correction": {
    "method": "fdr_bh",
    "alpha": 0.05
  }
}

Be specific about biological proxies and statistical methods. You have full autonomy to design the DAG structure."""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        # Checklist generation prompts (fallback, not template-based)
        self.checklist_system_prompt = """You are a domain expert in spatial omics and kidney pathology. Your task is to generate Background Checklists that guide automated interpretation agents in analyzing kidney tissue ROIs."""

        self.visual_profiler_checklist_prompt = """Based on the following context, generate a VisualProfiler Background Checklist for identifying kidney tissue structures from multiplexed immunofluorescence images.

## Context
**Hypothesis:** {hypothesis_description}
**Dataset:** {dataset_context}
**Literature:** {literature_context}

## Task
Generate a channel-specific visual cue checklist for these tissue structures:
- Glomeruli
- Distal Tubules / Collecting Ducts (DT/CD)
- Proximal Tubules (PT)
- Blood Vessels (Arteries/Arterioles)
- Interstitium

For each structure, specify:
- **Core Signal**: Which channel/color to look for and its pattern (e.g., "Red (Nestin)+ in a reticular pattern")
- **Boundary/Structure**: Key structural features (e.g., Bowman's capsule ring, basement membrane)
- **Key differentiator**: What distinguishes it from similar structures

Channel mapping: Nestin=red, CD227=green, CD183=blue, CD45=yellow, aSMA=cyan, Perlecan=magenta, CollagenIV=gray.

Format the output as a clean markdown document starting with a title "CHECKLIST (Channel-Specific Visual Cues)"."""

        self.omics_profiler_checklist_prompt = """Based on the following context, generate an OmicsProfiler Background Checklist for analyzing cell composition of kidney tissue structures.

## Context
**Hypothesis:** {hypothesis_description}
**Dataset:** {dataset_context}
**Literature:** {literature_context}

## Task
Generate a cell composition checklist for these tissue structures:
- Proximal tubules
- Distal tubules
- Glomeruli
- Blood vessel
- Interstitium

For each structure, specify:
- **Enriched**: Cell types and their key biomarkers that are highly expressed (e.g., "Tubular cells (CD183++/CD227-)")
- **Sparse**: Cell types and biomarkers that are lowly expressed or absent

Use these known biomarkers where relevant: CD183, CD227, CD31, CD196, Nestin, aSMA, CD45, CD68, CD11b.

Format the output as a clean markdown document."""

        self.omics_interpreter_checklist_prompt = """Based on the following context, generate an OmicsInterpreter Background Checklist for cross-modal alignment and conflict resolution.

## Context
**Hypothesis:** {hypothesis_description}
**Dataset:** {dataset_context}
**Literature:** {literature_context}

## Task
Generate conflict resolution rules for the following common interpretation scenarios:

1. **Glomeruli** (High Visual Reliability)
   - Conflict Rule: When to trust visual over omics
   - Alignment: What omics signals should confirm the visual finding

2. **Blood Vessels** (Structure vs. Cell Type Disambiguation)
   - Conflict Rule: How to distinguish blood vessels from other aSMA+ structures
   - Alignment: What structural features must be visually confirmed

3. **Interstitium vs. Proximal Tubules** (High Omics Reliability)
   - Conflict Rule: When visual boundaries are ambiguous, trust omics
   - Alignment: What cell composition pattern confirms interstitium

4. **Proximal Tubules vs. Distal Tubules** (High Omics Reliability)
   - Conflict Rule: How to differentiate PT from DT using omics
   - Alignment: Key biomarker differentiators (CD183++/CD227- for PT, CD227+ for DT)

Format the output as a clean markdown document starting with "BACKGROUND KNOWLEDGE: CROSS-MODAL ALIGNMENT & CONFLICT RESOLUTION"."""

        super().__init__(
            name="Planner",
            model=model,
            tools=[],  # Planner uses reasoning only
            system_prompt=system_prompt,
            temperature=0.2  # Lowest temperature for rigorous planning
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt from template or fallback."""
        return self.prompt_manager.get_system_prompt(
            agent_name="planner",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        return self.system_prompt

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute planning by generating verification blueprint.

        Args:
            state: Current workflow state

        Returns:
            Updated state with analysis_plan containing blueprint
        """
        state["current_action"] = "planning"
        state["current_phase"] = "phase1"

        # Gather context from DataAnalyst
        data_context = self._prepare_data_context(state)

        try:
            display = CLIDisplay.get()

            # Generate blueprint with retry logic
            blueprint = await self._generate_blueprint_with_retry(
                state["hypothesis_id"],
                state["hypothesis_description"],
                data_context
            )

            # Save blueprint to file
            blueprint_path = self._save_blueprint(
                state,
                blueprint
            )

            # Display blueprint summary
            display.show_blueprint(blueprint)

            # Generate natural language summary
            raw_plan = await self._generate_plan_summary(
                blueprint,
                state["hypothesis_description"]
            )

            # Generate Background Checklists for interpretation agents
            display.agent_progress("Planner", "Generating background checklists...")
            checklists = await self._generate_checklists(state)
            saved_paths = self._save_checklists(state, checklists)
            display.agent_done("Planner", "blueprint + checklists saved")

            # Update state
            state["analysis_plan"] = {
                "blueprint_path": blueprint_path,
                "blueprint": blueprint,
                "raw_plan": raw_plan,
                "status": "generated"
            }

            # Set checklist paths in state for downstream agents
            if "visual_profiler" in saved_paths:
                state["visual_profiler_checklist_path"] = saved_paths["visual_profiler"]
            if "omics_profiler" in saved_paths:
                state["omics_profiler_checklist_path"] = saved_paths["omics_profiler"]
            if "omics_interpreter" in saved_paths:
                state["omics_interpreter_checklist_path"] = saved_paths["omics_interpreter"]

            self._log_execution(state, "planning", "completed")

        except Exception as e:
            error_msg = f"Planner execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "planning", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    def _prepare_data_context(self, state: AgentState) -> str:
        """
        Prepare summarized data context from DataAnalyst report.

        Args:
            state: Current workflow state

        Returns:
            Summarized data context (information diet)
        """
        if not state.get("dataset_profile"):
            return "\n**Data Analysis:** Not available"

        profile = state["dataset_profile"]
        report = profile.get("analysis_report", "")
        iterations = profile.get("iteration_count", 0)

        # Extract only key statistics (information diet)
        context = f"""
**Data Analysis Summary (After {iterations} iterations):**
Sample: {state.get('sample_id', 'unknown')}
"""

        # Extract key metrics from report
        if "cells" in report.lower() or "cell" in report.lower():
            context += "\n- Dataset contains cellular composition data"

        if "sparsity" in report.lower():
            context += "\n- Sparsity analysis completed"

        if "composition" in report.lower() or "%" in report:
            context += "\n- Cell type distribution available"

        if "moran" in report.lower() or "spatial" in report.lower():
            context += "\n- Spatially variable genes identified"

        # Add validation history summary
        validation_log = profile.get("validation_log", [])
        if validation_log:
            passed_count = sum(1 for log in validation_log if log.get("passed"))
            context += f"\n- Validation: {passed_count}/{len(validation_log)} checks passed"

        return context

    async def _generate_blueprint_with_retry(
        self,
        hypothesis_id: str,
        hypothesis_description: str,
        data_context: str
    ) -> Dict[str, Any]:
        """
        Generate blueprint with retry logic for invalid JSON.

        Args:
            hypothesis_id: Hypothesis identifier
            hypothesis_description: Full hypothesis text
            data_context: Summarized data analysis context

        Returns:
            Validated blueprint dictionary
        """
        max_retries = 3
        blueprint = None
        last_error = None

        for attempt in range(max_retries):
            # Prepare context for template
            context = {
                "hypothesis_id": hypothesis_id,
                "hypothesis_description": hypothesis_description,
                "data_context": data_context,
                "json_schema": self._get_json_schema_instruction()
            }

            # Prepare fallback prompt
            fallback_prompt = """Design a statistical verification blueprint for this hypothesis:

**Hypothesis ID:** {hypothesis_id}

**Hypothesis:**
{hypothesis_description}

{data_context}

**Task:**
Create a verification blueprint that will be used to statistically test this hypothesis.

**Important:**
1. Respond ONLY with valid JSON (no markdown, no code blocks)
2. Define 3-5 quantifiable target variables
3. Design a statistical DAG with routing logic
4. Specify multiple testing correction method

**JSON Schema:**
{json_schema}

Generate the blueprint now."""

            prompt = self.prompt_manager.get_task_prompt(
                agent_name="planner",
                task_name="generate_blueprint",
                context=context,
                use_template=self.use_templates,
                fallback_prompt=fallback_prompt
            )

            try:
                # Invoke LLM
                response = await self._invoke_llm(self._format_prompt(prompt))

                # Parse JSON response
                blueprint = self._parse_json_response(response)

                # Validate schema
                self._validate_blueprint_schema(blueprint)

                # Success - exit retry loop
                break

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    # Retry with feedback
                    prompt += f"\n\n**Previous attempt failed with error:** {last_error}"
                    prompt += "\n\nPlease fix the JSON and try again."
                else:
                    # Final attempt failed
                    raise Exception(f"Failed to generate valid blueprint after {max_retries} attempts. Last error: {last_error}")

        if blueprint is None:
            raise Exception("Failed to generate blueprint")

        return blueprint

    def _get_json_schema_instruction(self) -> str:
        """Get JSON schema instruction for LLM."""
        return '''The JSON must have this structure:
{
  "metadata": {...},
  "hypothesis": {...},
  "target_variables": [
    {
      "name": "variable_name",
      "type": "cellular_composition|marker_expression|spatial_genes",
      "biological_proxy": "cell_type or marker",
      "expected_trend": "increase|decrease|change",
      "data_source": "how to extract from data"
    }
  ],
  "statistical_dag": {
    "nodes": [...],
    "edges": [...]
  },
  "life_cycle_contrasts": [...],
  "multiple_testing_correction": {...}
}'''

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM.

        Args:
            response: Raw LLM response

        Returns:
            Parsed dictionary

        Raises:
            Exception: If JSON is invalid
        """
        # Clean up response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```'):
            parts = response.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Content between ``` markers
                    response = part
                    # Remove language identifier if present
                    if response.startswith('json'):
                        response = response[4:].strip()
                    break

        # Parse JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON: {e}")

    def _validate_blueprint_schema(self, blueprint: Dict[str, Any]) -> None:
        """
        Validate blueprint schema.

        Args:
            blueprint: Blueprint dictionary to validate

        Raises:
            Exception: If schema is invalid
        """
        # Check required top-level keys
        required_keys = ["metadata", "hypothesis", "target_variables", "statistical_dag"]
        for key in required_keys:
            if key not in blueprint:
                raise Exception(f"Missing required key: {key}")

        # Validate target_variables
        if not isinstance(blueprint["target_variables"], list):
            raise Exception("target_variables must be a list")

        if len(blueprint["target_variables"]) < 2:
            raise Exception("At least 2 target variables required")

        for var in blueprint["target_variables"]:
            for field in ["name", "type", "biological_proxy", "expected_trend"]:
                if field not in var:
                    raise Exception(f"Target variable missing field: {field}")

        # Validate statistical_dag
        if "nodes" not in blueprint["statistical_dag"]:
            raise Exception("statistical_dag must contain 'nodes'")

        # Validate multiple_testing_correction
        if "multiple_testing_correction" not in blueprint:
            raise Exception("Missing multiple_testing_correction")

    def _save_blueprint(
        self,
        state: Dict[str, Any],
        blueprint: Dict[str, Any]
    ) -> str:
        """
        Save blueprint to session directory.

        Args:
            state: Current agent state with session_dir
            blueprint: Blueprint dictionary

        Returns:
            Path to saved blueprint file
        """
        # Get session directory from state
        session_dir = Path(state.get("session_dir", "./outputs"))
        blueprint_path = session_dir / "blueprint.json"

        # Save blueprint
        with open(blueprint_path, 'w', encoding='utf-8') as f:
            json.dump(blueprint, f, indent=2, ensure_ascii=False)

        return str(blueprint_path)

    async def _generate_plan_summary(
        self,
        blueprint: Dict[str, Any],
        hypothesis_description: str
    ) -> str:
        """
        Generate natural language summary of the blueprint.

        Args:
            blueprint: Validated blueprint dictionary
            hypothesis_description: Original hypothesis

        Returns:
            Natural language summary
        """
        # Create a summary from the blueprint
        summary = f"""# Analysis Plan

**Hypothesis:** {hypothesis_description}

## Target Variables
"""
        for var in blueprint["target_variables"]:
            summary += f"\n- **{var['name']}** ({var['type']})"
            summary += f"\n  - Proxy: {var['biological_proxy']}"
            summary += f"\n  - Expected: {var['expected_trend']}"

        summary += f"""

## Statistical Testing Framework
- DAG routing: Normality → Variance → Parametric/Non-parametric
- Multiple testing correction: {blueprint['multiple_testing_correction']['method']}
"""

        if "life_cycle_contrasts" in blueprint and blueprint["life_cycle_contrasts"]:
            summary += "\n\n## Comparison Groups\n"
            for contrast in blueprint["life_cycle_contrasts"]:
                summary += f"- **{contrast['id']}**: {contrast['group1']} vs {contrast['group2']}"
                summary += f"\n  {contrast.get('description', '')}"

        return summary

    def _prepare_literature_context(self, state: AgentState) -> str:
        """
        Prepare summarized literature context from LiteratureReviewer output.

        Args:
            state: Current workflow state

        Returns:
            Summarized literature context string
        """
        lit = state.get("literature_summaries")
        if not lit:
            return "No literature review available."

        # Handle structured literature output
        if isinstance(lit, dict):
            summary = lit.get("executive_summary", lit.get("summary", ""))
            key_findings = lit.get("key_findings", [])
            if summary:
                return summary[:500]
            if key_findings:
                return "\n".join(f"- {f}" for f in key_findings[:5])

        return str(lit)[:500] if lit else "No literature review available."

    async def _generate_checklists(self, state: AgentState) -> Dict[str, str]:
        """
        Generate three Background Checklists for interpretation agents.

        Uses dataset profile, literature summaries, and hypothesis context
        to generate tailored checklists for VisualProfiler, OmicsProfiler,
        and OmicsInterpreter.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with keys: visual_profiler, omics_profiler, omics_interpreter
            Each value is the checklist markdown content.
        """
        hypothesis_description = state.get("hypothesis_description", "")
        dataset_context = self._prepare_data_context(state)
        literature_context = self._prepare_literature_context(state)

        # Common format context for all three checklists
        format_context = {
            "hypothesis_description": hypothesis_description,
            "dataset_context": dataset_context,
            "literature_context": literature_context,
        }

        checklists = {}

        # Generate VisualProfiler checklist
        vp_prompt = self.prompt_manager.get_task_prompt(
            agent_name="planner",
            task_name="generate_visual_profiler_checklist",
            context=format_context,
            use_template=self.use_templates,
            fallback_prompt=self.visual_profiler_checklist_prompt
        )
        vp_response = await self._invoke_llm_with_system(
            vp_prompt,
            self.checklist_system_prompt
        )
        checklists["visual_profiler"] = vp_response

        # Generate OmicsProfiler checklist
        op_prompt = self.prompt_manager.get_task_prompt(
            agent_name="planner",
            task_name="generate_omics_profiler_checklist",
            context=format_context,
            use_template=self.use_templates,
            fallback_prompt=self.omics_profiler_checklist_prompt
        )
        op_response = await self._invoke_llm_with_system(
            op_prompt,
            self.checklist_system_prompt
        )
        checklists["omics_profiler"] = op_response

        # Generate OmicsInterpreter checklist
        oi_prompt = self.prompt_manager.get_task_prompt(
            agent_name="planner",
            task_name="generate_interpreter_checklist",
            context=format_context,
            use_template=self.use_templates,
            fallback_prompt=self.omics_interpreter_checklist_prompt
        )
        oi_response = await self._invoke_llm_with_system(
            oi_prompt,
            self.checklist_system_prompt
        )
        checklists["omics_interpreter"] = oi_response

        return checklists

    def _save_checklists(
        self,
        state: Dict[str, Any],
        checklists: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Save checklists to session directory and update state with paths.

        Args:
            state: Current agent state with session_dir
            checklists: Dictionary of checklist_name -> markdown content

        Returns:
            Dictionary mapping checklist names to file paths
        """
        session_dir = Path(state.get("session_dir", "./outputs"))

        path_map = {
            "visual_profiler": "VisualProfiler_checklist.md",
            "omics_profiler": "OmicsProfiler_checklist.md",
            "omics_interpreter": "OmicsInterpreter_checklist.md",
        }

        saved_paths = {}
        for checklist_name, filename in path_map.items():
            content = checklists.get(checklist_name, "")
            if content:
                filepath = session_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                saved_paths[checklist_name] = str(filepath)

        return saved_paths

    async def _invoke_llm_with_system(
        self,
        user_prompt: str,
        system_prompt: str
    ) -> str:
        """
        Invoke LLM with a specific system prompt.

        Args:
            user_prompt: User prompt text
            system_prompt: System prompt to use

        Returns:
            LLM response text
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = await self.model.ainvoke(messages)
        return self._extract_text(response.content)

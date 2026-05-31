"""
LiteratureReviewer Agent for scientific literature search and summarization.

Specializes in:
- Extracting metadata from DataAnalyst reports (biomarkers, cell types, concepts)
- Building Deep Research queries
- Retrieving comprehensive literature reviews
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent, AgentExecutionError
from ..workflows.agent_state import AgentState, add_error
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class LiteratureReviewerAgent(BaseAgent):
    """
    LiteratureReviewer Agent for scientific literature analysis.

    This agent performs a three-step process:
    1. Extract metadata (biomarkers, cell types, concepts) from DataAnalyst report
    2. Build Deep Research query and retrieve literature report
    3. Structure output with summary and full report
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """
        Initialize the LiteratureReviewer agent.

        Args:
            model: LangChain chat model instance
            prompt_manager: PromptManager instance (if None, creates default)
            use_templates: If True, use Jinja2 templates; else use hardcoded prompts
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.use_templates = use_templates

        # System prompt
        self.fallback_system_prompt = """You are a Scientific Literature Reviewer specializing in spatial omics, kidney disease, and fibrosis research.

Your expertise includes:
1. **Literature Retrieval**: Finding relevant papers for spatial omics and kidney disease
2. **Paper Summarization**: Extracting key findings, methodologies, and insights
3. **Biological Priors**: Identifying established knowledge for hypothesis testing

## Analysis Guidelines:
- Focus on spatial transcriptomics, multiplexed imaging, and kidney disease
- Prioritize recent papers (last 5-10 years) with high impact
- Extract methodological insights applicable to spatial omics
- Identify biological mechanisms related to the hypothesis

## Output Format:
Provide comprehensive literature reviews with:
- Hypothesis deconstruction
- Spatial phenotype analysis
- Biomarker grounding strategies
- Recommendations for spatial analysis"""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        super().__init__(
            name="LiteratureReviewer",
            model=model,
            tools=[],  # No tools for literature review (Deep Research API to be added later)
            system_prompt=system_prompt,
            temperature=0.5  # Medium temperature for synthesis work
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt from template or fallback."""
        return self.prompt_manager.get_system_prompt(
            agent_name="literature_reviewer",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        return self.system_prompt

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute three-step literature review process.

        Workflow:
        1. Extract metadata from DataAnalyst report
        2. Build Deep Research query and get report (MOCKED)
        3. Structure output with summary and full report
        """
        state["current_action"] = "literature_review"
        state["current_phase"] = "phase1"

        try:
            display = CLIDisplay.get()

            # Step 1: Extract metadata from DataAnalyst report
            display.agent_start("LiteratureReviewer", "Extracting metadata from DataAnalyst report...")
            extracted_metadata = await self._extract_metadata(state)
            n_biomarkers = len(extracted_metadata.get('biomarker_list', []))
            n_cell_types = len(extracted_metadata.get('cell_type_list', []))
            n_concepts = len(extracted_metadata.get('key_concepts', []))
            display.agent_progress("LiteratureReviewer", f"Extracted {n_biomarkers} biomarkers, {n_cell_types} cell types, {n_concepts} concepts")

            # Step 2: Build query and get report (MOCKED)
            display.agent_progress("LiteratureReviewer", "Building deep-research query...")
            full_report = await self._build_query_and_get_report(state, extracted_metadata)
            display.agent_done("LiteratureReviewer", f"report ready ({len(full_report)} chars) -> literature_report.md")

            # Step 3: Structure output
            structured_output = self._structure_output(extracted_metadata, full_report, state)

            # Update state
            state["literature_metadata"] = extracted_metadata
            state["literature_summaries"] = structured_output
            state["literature_report_path"] = structured_output["report_path"]

            self._log_execution(state, "literature_review", "completed")

        except Exception as e:
            error_msg = f"LiteratureReviewer execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "literature_review", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    async def _extract_metadata(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract biomarkers, cell types, and key concepts from DataAnalyst report.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with biomarker_list, cell_type_list, key_concepts
        """
        # Get DataAnalyst report
        data_report = state.get("dataset_profile", {}).get(
            "analysis_report",
            "No data analysis available"
        )

        # Prepare context for template
        context = {
            "data_report": data_report,
            "hypothesis_description": state["hypothesis_description"]
        }

        # Fallback prompt
        fallback_prompt = """Extract the following structured information from the DataAnalyst report:

**DataAnalyst Report:**
{data_report}

**Hypothesis:**
{hypothesis_description}

**Your Task:**
Extract:
1. **Biomarker List**: All biomarkers mentioned in the dataset
2. **Cell Type List**: All cell types available in the dataset
3. **Key Concepts**: 5-10 biological concepts from the hypothesis

**Response Format (JSON only):**
```json
{{
  "biomarker_list": ["ACE2", "TFAM", "Clusterin", "aSMA", ...],
  "cell_type_list": ["Proximal Tubules", "Distal Tubules", ...],
  "key_concepts": ["proximal tubule atrophy", "fibrosis progression", ...]
}}
```

Return ONLY valid JSON, no additional text."""

        # Get prompt from template or fallback
        prompt = self.prompt_manager.get_task_prompt(
            agent_name="literature_reviewer",
            task_name="extract_metadata",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt.format(**context)
        )

        # Invoke LLM
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        response_text = self._extract_text(response.content)

        # Parse JSON response
        response_text = response_text.strip()

        # Handle markdown code blocks
        if response_text.startswith('```'):
            # Find the end of the code block
            parts = response_text.split('```')
            for part in parts:
                part = part.strip()
                if not part:  # Skip empty parts
                    continue
                # Remove 'json' prefix if present
                if part.startswith('json'):
                    part = part[4:].strip()
                # If we have non-empty content after cleaning, use it
                if part and not part.startswith('```'):
                    response_text = part
                    break

        # Clean up any remaining markdown
        response_text = response_text.strip()
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()

        try:
            extracted = json.loads(response_text)

            # Validate structure
            required_keys = ["biomarker_list", "cell_type_list", "key_concepts"]
            for key in required_keys:
                if key not in extracted:
                    # Add empty list if missing
                    extracted[key] = []

            return extracted

        except json.JSONDecodeError as e:
            # If JSON parsing fails, return empty structure
            print(f"  ⚠️  Failed to parse JSON response: {e}")
            print(f"  ⚠️  Response was: {response_text[:200]}...")
            return {
                "biomarker_list": [],
                "cell_type_list": [],
                "key_concepts": []
            }

    async def _build_query_and_get_report(
        self,
        state: AgentState,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Build Deep Research query and get literature report.

        For now, this is MOCKED by loading from file.

        Args:
            state: Current workflow state
            metadata: Extracted metadata from Step 1

        Returns:
            Full literature report as string
        """
        # Step 1: Build the query (for future Deep Research API)
        query_path = await self._build_deep_research_query(state, metadata)
        state["literature_query_path"] = str(query_path)

        # Step 2: MOCK - Load from file instead of calling API
        mock_report_path = Path("cli/mockdata/literaturereviewer_report.md")

        if not mock_report_path.exists():
            raise FileNotFoundError(
                f"Mock report not found at {mock_report_path}. "
                "This is expected until Deep Research API is integrated."
            )

        with open(mock_report_path, 'r', encoding='utf-8') as f:
            full_report = f.read()

        return full_report

    async def _build_deep_research_query(
        self,
        state: AgentState,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Build and save Deep Research query to file.

        Args:
            state: Current workflow state
            metadata: Extracted metadata

        Returns:
            Path to saved query file
        """
        # Load template
        template_path = Path("cli/mockdata/LiteratureReviewer_query.md")
        if not template_path.exists():
            raise FileNotFoundError(f"Query template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Extract disease context from key concepts
        disease_context = self._extract_disease_context(metadata.get("key_concepts", []))

        # Fill placeholders
        try:
            query = template.format(
                Domain_Expertise="Spatial Omics and Kidney Disease Research",
                Disease_Context=disease_context,
                Clinical_Hypothesis=state["hypothesis_description"],
                Cell_Types_List=", ".join(metadata.get("cell_type_list", [])),
                Biomarkers_List=", ".join(metadata.get("biomarker_list", [])),
                Key_Structures_or_Events="tubular structures, fibrotic regions",
                Key_Pathological_Concept="spatial coupling of tubular damage and fibrosis",
                Specific_Mechanistic_Question="What is the spatiotemporal relationship between proximal tubule loss and fibrosis marker upregulation?",
                Key_Markers_for_Phenotype=", ".join(metadata.get("biomarker_list", [])[:10]),  # First 10 markers
                Target_Microenvironment="tubulointerstitial compartment",
                Interacting_Cell_Populations="proximal tubules, immune cells, fibroblasts",
                Target_Tissue="Kidney",
                Target_States_for_Markers="tubular stress, fibrosis activation, immune infiltration"
            )
        except KeyError as e:
            # If template has different placeholders, skip formatting
            print(f"  ⚠️  Could not fill all placeholders in template: {e}")
            query = template

        # Save to session directory
        session_dir = Path(state.get("session_dir", "./outputs"))
        session_dir.mkdir(parents=True, exist_ok=True)
        query_path = session_dir / "literature_query.md"

        with open(query_path, 'w', encoding='utf-8') as f:
            f.write(query)

        return query_path

    def _extract_disease_context(self, key_concepts: List[str]) -> str:
        """
        Extract disease context from key concepts.

        Args:
            key_concepts: List of key concepts

        Returns:
            Disease context string
        """
        # Look for disease-related keywords
        disease_keywords = {
            "DKD": "Diabetic Kidney Disease",
            "Diabetic Kidney Disease": "Diabetic Kidney Disease",
            "diabetic nephropathy": "Diabetic Kidney Disease",
            "kidney disease": "Kidney Disease",
            "renal fibrosis": "Renal Fibrosis"
        }

        for concept in key_concepts:
            for keyword, disease in disease_keywords.items():
                if keyword.lower() in concept.lower():
                    return disease

        return "Kidney Disease"  # Default fallback

    def _structure_output(
        self,
        metadata: Dict[str, Any],
        full_report: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Create structured summary of literature review.

        Args:
            metadata: Extracted metadata from Step 1
            full_report: Full literature report
            state: Current workflow state

        Returns:
            Structured output dictionary
        """
        # Save full report to file
        session_dir = Path(state.get("session_dir", "./outputs"))
        session_dir.mkdir(parents=True, exist_ok=True)
        report_path = session_dir / "literature_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)

        # Create structured output
        biomarker_count = len(metadata.get("biomarker_list", []))
        cell_type_count = len(metadata.get("cell_type_list", []))
        concept_count = len(metadata.get("key_concepts", []))

        return {
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                "biomarker_count": biomarker_count,
                "cell_type_count": cell_type_count,
                "concept_count": concept_count
            },
            "extracted_metadata": metadata,
            "summary": f"Full literature review generated via Deep Research with {biomarker_count} biomarkers and {cell_type_count} cell types.",
            "report_path": str(report_path),
            "query_path": state.get("literature_query_path", "")
        }

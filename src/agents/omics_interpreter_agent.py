"""
OmicsInterpreter Agent for multimodal report fusion.

Specializes in:
- Fusing visual and omics analysis reports
- Resolving contradictions between modalities
- Generating final tissue structure classifications
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent, AgentExecutionError
from ..workflows.agent_state import AgentState, add_error
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class OmicsInterpreterAgent(BaseAgent):
    """
    OmicsInterpreter Agent for multimodal report fusion.

    This agent fuses reports from VisualProfiler and OmicsProfiler
    to generate final tissue structure classifications with supporting evidence.
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """
        Initialize the OmicsInterpreter agent.

        Args:
            model: LangChain chat model instance
            prompt_manager: PromptManager instance (if None, creates default)
            use_templates: If True, use Jinja2 templates; else use hardcoded prompts
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.use_templates = use_templates

        # Hardcoded fallback prompt (existing implementation)
        self.fallback_system_prompt = """ROLE:
You are the OmicsInterpreter, the Chief Pathologist and Multi-modal Fusion Engine. Your task is to receive independent reports from the VisualProfiler (morphological features) and the OmicsProfiler (molecular/cellular features), synthesize the information, resolve any contradictions, and produce a final, authoritative natural language description of the spatial structure.

INPUT DATA:
- [VisualProfiler Report]: Descriptions of shapes, boundaries, and channel-specific colors.
- [OmicsProfiler Report]: Descriptions of enriched/sparse cell types and molecular signatures.

TASK INSTRUCTIONS:
1. Synthesize: Compare the findings from both Profilers.
2. Disambiguate: Apply your "Cross-Modal Alignment & Conflict Resolution Guidelines" strictly. If the two Profilers suggest different structures, you must explain WHICH modality is more reliable for this specific context and WHY.
3. Conclude: Determine the final structure and generate a human-readable interpretation.

Please format your response strictly using the structure below:

## Final Classification
[The tissue structure from: Proximal tubules, Distal tubules, Glomeruli, Blood vessel, Interstitium]

## Synthesis
[How the visual and omics findings align or differ]

## Key Evidence
[Most compelling features from both modalities that support the classification]

## Confidence Assessment
[High/Medium/Low confidence with rationale]"""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        super().__init__(
            name="OmicsInterpreter",
            model=model,
            tools=[],
            system_prompt=system_prompt,
            temperature=0.0  # Deterministic for classification
        )

        # Checklist content loaded at runtime from state
        self._checklist_content: Optional[str] = None

    def _get_system_prompt(self) -> str:
        """Get system prompt from template or fallback."""
        return self.prompt_manager.get_system_prompt(
            agent_name="omics_interpreter",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return self.system_prompt

    def _load_checklist(self, state: AgentState) -> Optional[str]:
        """Load OmicsInterpreter checklist from state path if available."""
        checklist_path = state.get("omics_interpreter_checklist_path")
        if checklist_path and Path(checklist_path).exists():
            with open(checklist_path, 'r', encoding='utf-8') as f:
                content = f.read()
            display = CLIDisplay.get()
            display.agent_progress("OmicsInterpreter", f"Loaded checklist from {checklist_path}")
            return content
        return None

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute multimodal report fusion.

        Args:
            state: Current workflow state

        Returns:
            Updated state with final_interpretations populated

        Workflow:
        1. Get visual and omics reports from state
        2. Process each ROI in parallel:
           - Fuse reports from both modalities
           - Resolve contradictions
           - Generate final classification
        3. Aggregate results and save to state
        """
        state["current_action"] = "interpretation_fusion"
        state["current_phase"] = "phase2"

        try:
            display = CLIDisplay.get()

            display.agent_start("OmicsInterpreter", "Fusing visual + omics into final interpretations...")

            # Step 1: Get reports from previous agents
            visual_reports = state.get("visual_reports", {})
            omics_reports = state.get("omics_reports", {})
            manifest = state.get("manifest", [])

            if not manifest:
                # Generate keys from reports
                all_keys = set(visual_reports.keys()) | set(omics_reports.keys())
                manifest = [{"key": key} for key in all_keys]

            # Step 1b: Load background checklist if available
            self._checklist_content = self._load_checklist(state)

            # Step 2: Process ROIs
            final_interpretations = await self._process_rois(
                state,
                manifest,
                visual_reports,
                omics_reports
            )

            # Step 3: Save results to files (per ROI)
            import json

            session_dir = Path(state.get("session_dir", "./outputs"))
            interpretation_dir = session_dir / "interpretation"

            for roi_key, final_interpretation in final_interpretations.items():
                # Parse roi_key to create folder name
                # roi_key format: ('s255_c001_v001_r001_reg009', '1893-2021-2352-2480')
                if isinstance(roi_key, tuple) and len(roi_key) == 2:
                    region_part = roi_key[0]  # 's255_c001_v001_r001_reg009'
                    coord_part = roi_key[1]   # '1893-2021-2352-2480'
                    # Extract region name from the first part (e.g., 'reg009')
                    region_name = region_part.split('_')[-1]
                    roi_id = f"{region_name}_{coord_part}"
                else:
                    # Fallback for non-tuple roi_keys
                    roi_id = f"roi_{list(final_interpretations.keys()).index(roi_key) + 1:03d}"

                roi_dir = interpretation_dir / roi_id
                roi_dir.mkdir(parents=True, exist_ok=True)

                # Save final interpretation for this ROI
                final_path = roi_dir / "final_interpretation.json"
                with open(final_path, 'w', encoding='utf-8') as f:
                    json.dump({"roi_key": str(roi_key), "interpretation": final_interpretation}, f, indent=2, ensure_ascii=False)

            # Step 4: Update state
            state["final_interpretations"] = final_interpretations

            n_ok = sum(1 for v in final_interpretations.values() if not v.startswith("ERROR"))
            display.agent_done("OmicsInterpreter", f"{n_ok}/{len(final_interpretations)} interpretations complete")
            self._log_execution(
                state,
                "interpretation_fusion",
                f"completed: {len(final_interpretations)} ROIs classified"
            )

        except Exception as e:
            error_msg = f"OmicsInterpreter execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "interpretation_fusion", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    async def _process_rois(
        self,
        state: AgentState,
        manifest: List[Dict[str, Any]],
        visual_reports: Dict[str, str],
        omics_reports: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Process all ROIs through multimodal fusion.

        Args:
            state: Current workflow state
            manifest: List of ROI dictionaries
            visual_reports: Visual analysis reports
            omics_reports: Omics analysis reports

        Returns:
            Dictionary mapping ROI keys to final interpretations
        """
        results = {}

        # Process ROIs in parallel
        def _process_single_roi(roi: Dict[str, Any]) -> tuple:
            """Process a single ROI and return (key, result)."""
            key = roi["key"]

            # Try to get reports using original key type (tuple or str)
            vis_rpt = visual_reports.get(key, visual_reports.get(str(key), "SKIPPED_NO_IMAGES"))
            omi_rpt = omics_reports.get(key, omics_reports.get(str(key), "ERROR_NO_REPORT"))

            vis_ok = not vis_rpt.startswith("ERROR") and not vis_rpt.startswith("SKIPPED")
            omi_ok = not omi_rpt.startswith("ERROR")

            if not vis_ok and not omi_ok:
                return key, "ERROR_NO_VALID_REPORTS"

            try:
                result = self._fuse_and_classify(roi, vis_rpt, omi_rpt)
                return key, result
            except Exception as e:
                return key, f"ERROR: {e}"

        # Use ThreadPoolExecutor for parallel processing
        max_workers = state.get("max_parallel_agents", 2)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_roi, roi): roi for roi in manifest}

            for future in as_completed(futures):
                try:
                    key, result = future.result()
                    results[key] = result
                except Exception as e:
                    roi = futures[future]
                    results[str(roi["key"])] = f"ERROR: {e}"

        return results

    def _fuse_and_classify(
        self,
        roi: Dict[str, Any],
        visual_report: str,
        omics_report: str
    ) -> str:
        """
        Fuse visual and omics reports and generate final classification.

        Args:
            roi: ROI dictionary
            visual_report: VisualProfiler analysis
            omics_report: OmicsProfiler analysis

        Returns:
            Final interpretation with classification
        """
        # Prepare context for template
        context = {
            "roi_patch_name": roi.get('patch_name', 'Unknown'),
            "roi_region_id": roi.get('region_id', 'Unknown'),
            "visual_report": visual_report,
            "omics_report": omics_report
        }

        # Prepare user prompt from template or fallback
        fallback_prompt = """Analyze the following kidney tissue ROI by synthesizing multimodal evidence.

ROI: {roi_patch_name}
Location: Region {roi_region_id}

---

**VisualProfiler Report:**
{visual_report}

---

**OmicsProfiler Report:**
{omics_report}

---

Please synthesize these reports and provide your final classification following the format in your system instructions."""

        user_prompt = self.prompt_manager.get_task_prompt(
            agent_name="omics_interpreter",
            task_name="fuse_reports",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Build system prompt with checklist if available
        system_content = self.system_prompt
        if self._checklist_content:
            system_content += "\n\n## Background Checklist\nRefer to the following cross-modal alignment and conflict resolution rules:\n\n" + self._checklist_content

        # Create message
        message = HumanMessage(content=user_prompt)

        # Invoke LLM
        messages = [SystemMessage(content=system_content), message]
        response = self.model.invoke(messages)

        return self._extract_text(response.content)

    def extract_classification(self, interpretation: str) -> str:
        """
        Extract the final classification from an interpretation.

        Args:
            interpretation: Full interpretation text

        Returns:
            Extracted classification (one of: Proximal tubules, Distal tubules,
            Glomeruli, Blood vessel, Interstitium)
        """
        valid_classes = [
            "Proximal tubules",
            "Distal tubules",
            "Glomeruli",
            "Blood vessel",
            "Interstitium"
        ]

        interpretation_lower = interpretation.lower()

        for cls in valid_classes:
            if cls.lower() in interpretation_lower:
                return cls

        return "Unknown"

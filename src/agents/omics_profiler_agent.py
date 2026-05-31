"""
OmicsProfiler Agent for ROI omics analysis.

Specializes in:
- Cell type composition analysis
- Biomarker expression analysis
- Tabular data interpretation
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
from ..utils.roi_descriptions import generate_roi_descriptions
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class OmicsProfilerAgent(BaseAgent):
    """
    OmicsProfiler Agent for ROI omics analysis.

    This agent analyzes cell type composition and biomarker expression
    to provide molecular context for tissue structure identification.
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """
        Initialize the OmicsProfiler agent.

        Args:
            model: LangChain chat model instance
            prompt_manager: PromptManager instance (if None, creates default)
            use_templates: If True, use Jinja2 templates; else use hardcoded prompts
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.use_templates = use_templates

        # Hardcoded fallback prompt (existing implementation)
        self.fallback_system_prompt = """You are a knowledgeable biological research assistant specializing in spatial omics and multiplexed immunofluorescence data analysis. Your task is to help study regions of interest (ROIs) within a tissue sample.

The tissue sample is derived from the following tissue types: Kidney, and it has been collected from patients exhibiting these phenotypes: Healthy, Diabetic Mellitus, Diabetic Kidney Disease.

A series of protein biomarkers were measured in this sample and the ROI, and these biomarkers were used to characterize cells and multi-cellular structures. Your primary responsibility is to annotate ROIs with the provided context and your knowledge on the biomarkers, cell types, tissues, and the phenotypes.

Your responses should be informative, simple, and concise."""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        super().__init__(
            name="OmicsProfiler",
            model=model,
            tools=[],
            system_prompt=system_prompt,
            temperature=0.0  # Deterministic for analysis
        )

        # Checklist content loaded at runtime from state
        self._checklist_content: Optional[str] = None

    def _get_system_prompt(self) -> str:
        """Get system prompt from template or fallback."""
        return self.prompt_manager.get_system_prompt(
            agent_name="omics_profiler",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return self.system_prompt

    def _load_checklist(self, state: AgentState) -> Optional[str]:
        """Load OmicsProfiler checklist from state path if available."""
        checklist_path = state.get("omics_profiler_checklist_path")
        if checklist_path and Path(checklist_path).exists():
            with open(checklist_path, 'r', encoding='utf-8') as f:
                content = f.read()
            display = CLIDisplay.get()
            display.agent_progress("OmicsProfiler", f"Loaded checklist from {checklist_path}")
            return content
        return None

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute omics profiling analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with omics_reports populated

        Workflow:
        1. Get manifest from state (from VisualProfiler)
        2. Generate ROI descriptions from CSV data
        3. Process each ROI in parallel:
           - Load cell composition and biomarker data
           - Call LLM with tabular data description
        4. Aggregate results and save to state
        """
        state["current_action"] = "omics_profiling"
        state["current_phase"] = "phase2"

        try:
            display = CLIDisplay.get()

            display.agent_start("OmicsProfiler", "Generating cell composition profiles...")

            # Step 1: Get manifest from previous agent
            manifest = state.get("manifest")
            if not manifest:
                # Generate manifest if not available
                manifest = await self._prepare_manifest(state)

            # Step 1b: Load background checklist if available
            self._checklist_content = self._load_checklist(state)

            # Step 2: Generate ROI descriptions
            display.agent_progress("OmicsProfiler", "Generating ROI descriptions...")
            all_descriptions = await self._generate_all_descriptions(state, manifest)

            # Step 3: Process ROIs
            omics_reports = await self._process_rois(state, manifest, all_descriptions)

            # Step 4: Save results to files (per ROI)
            import json

            session_dir = Path(state.get("session_dir", "./outputs"))
            interpretation_dir = session_dir / "interpretation"

            for roi_key, omics_report in omics_reports.items():
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
                    roi_id = f"roi_{list(omics_reports.keys()).index(roi_key) + 1:03d}"

                roi_dir = interpretation_dir / roi_id
                roi_dir.mkdir(parents=True, exist_ok=True)

                # Save omics report for this ROI
                omics_report_path = roi_dir / "omics_report.json"
                with open(omics_report_path, 'w', encoding='utf-8') as f:
                    json.dump({"roi_key": str(roi_key), "report": omics_report}, f, indent=2, ensure_ascii=False)

            # Step 5: Update state
            state["omics_reports"] = omics_reports

            n_ok = sum(1 for v in omics_reports.values() if not v.startswith("ERROR"))
            display.agent_done("OmicsProfiler", f"{n_ok}/{len(omics_reports)} descriptions generated")
            self._log_execution(state, "omics_profiling", f"completed: {len(omics_reports)} ROIs analyzed")

        except Exception as e:
            error_msg = f"OmicsProfiler execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "omics_profiling", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    async def _prepare_manifest(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Prepare manifest from registry if not available from previous agent.

        Uses cluster_center_indices to get cluster center ROIs (the actual sampling points).

        Args:
            state: Current workflow state

        Returns:
            List of ROI dictionaries with keys: region_id, patch_name, cell_ids, roi_obj
        """
        import pickle

        # Load from registry (same logic as VisualProfiler)
        data_root = Path(state["data_root"])
        sample_id = state["sample_id"]

        possible_paths = [
            data_root / f"{sample_id}_pivot_ROIs_registry_dev.pkl",
            data_root / f"{sample_id}_pivot_ROIs_registry.pkl",
            data_root / f"{sample_id}_pivot_ROIs.pkl",
        ]

        registry_path = None
        for path in possible_paths:
            if path.exists():
                registry_path = path
                display = CLIDisplay.get()
                display.agent_progress("OmicsProfiler", f"Found registry: {path.name}")
                break

        if not registry_path:
            display = CLIDisplay.get()
            display.agent_progress("OmicsProfiler", "No registry file found")
            return []

        with open(registry_path, 'rb') as f:
            registry = pickle.load(f)

        manifest = []

        # Build manifest using cluster_center_indices
        for region_id, shard in registry.shards.items():
            if shard.cluster_center_indices:
                for (feat_id, spatial_id), roi_indices in shard.cluster_center_indices.items():
                    for roi_idx in roi_indices:
                        roi = shard.rois[roi_idx]
                        manifest.append({
                            "region_id": region_id,
                            "patch_name": roi.patch_name,
                            "cell_ids": roi.cell_ids,
                            "roi_obj": roi,
                            "key": (region_id, roi.patch_name),
                            "feat_id": feat_id,
                            "spatial_id": spatial_id
                        })
            else:
                # Fallback: use all ROIs if no cluster_center_indices
                print(f"  No cluster_center_indices for {region_id}, using all ROIs")
                for roi in shard.rois:
                    manifest.append({
                        "region_id": region_id,
                        "patch_name": roi.patch_name,
                        "cell_ids": roi.cell_ids,
                        "roi_obj": roi,
                        "key": (region_id, roi.patch_name)
                    })

        display = CLIDisplay.get()
        display.agent_progress("OmicsProfiler", f"Generated manifest with {len(manifest)} ROIs from {len(registry.shards)} regions")
        return manifest[:50]  # Process first 50 ROIs

    async def _generate_all_descriptions(
        self,
        state: AgentState,
        manifest: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate descriptions for all ROIs from CSV data.

        Args:
            state: Current workflow state
            manifest: List of ROI dictionaries

        Returns:
            Dictionary mapping ROI keys to description strings
        """
        # Group ROIs by region
        region_targets = {}
        for r in manifest:
            rid = r["region_id"]
            region_targets.setdefault(rid, []).append({
                "cell_ids": r["cell_ids"],
                "patch_name": r["patch_name"],
            })

        # Generate descriptions for each region
        all_descriptions = {}
        for region_id, targets in region_targets.items():
            try:
                descs = generate_roi_descriptions(
                    region_id=region_id,
                    root_dir=state["data_root"],
                    target_rois=targets,
                    include_biomarker=True,
                    include_cell_type=True,
                    include_morphology=False,
                    verbose=False
                )
                # Convert tuple keys to string keys
                for (rid, patch_name), desc in descs.items():
                    all_descriptions[f"{rid}_{patch_name}"] = desc
            except Exception as e:
                print(f"  Failed to generate descriptions for {region_id}: {e}")
                # Add placeholder descriptions for this region
                for target in targets:
                    key = f"{region_id}_{target['patch_name']}"
                    all_descriptions[key] = "Description not available"

        display = CLIDisplay.get()
        display.agent_progress("OmicsProfiler", f"Generated {len(all_descriptions)} descriptions")
        return all_descriptions

    async def _process_rois(
        self,
        state: AgentState,
        manifest: List[Dict[str, Any]],
        all_descriptions: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Process all ROIs through omics profiling.

        Args:
            state: Current workflow state
            manifest: List of ROI dictionaries
            all_descriptions: Pre-generated descriptions

        Returns:
            Dictionary mapping ROI keys to omics analysis reports
        """
        results = {}

        # Process ROIs in parallel
        def _process_single_roi(roi: Dict[str, Any]) -> tuple:
            """Process a single ROI and return (key, result)."""
            key = roi["key"]
            desc_key = f"{roi['region_id']}_{roi['patch_name']}"
            desc = all_descriptions.get(desc_key, "")

            if not desc:
                return key, "ERROR_EMPTY_DESC"

            try:
                result = self._analyze_roi_omics(roi, desc)
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
                    results[roi["key"]] = f"ERROR: {e}"

        return results

    def _analyze_roi_omics(self, roi: Dict[str, Any], description: str) -> str:
        """
        Analyze a single ROI's omics data using LLM.

        Args:
            roi: ROI dictionary
            description: Generated ROI description

        Returns:
            Omics analysis report from LLM
        """
        # Prepare context for template
        context = {
            "roi_patch_name": roi.get('patch_name', 'Unknown'),
            "roi_region_id": roi.get('region_id', 'Unknown'),
            "roi_description": description
        }

        # Prepare user prompt from template or fallback
        fallback_prompt = """Analyze the kidney tissue ROI based on its cellular and molecular composition.

ROI: {roi_patch_name}
Location: Region {roi_region_id}

{roi_description}

Please provide:
1. Tissue structure identification from: Proximal tubules, Distal tubules, Glomeruli, Blood vessel, Interstitium
2. Key cellular features and their significance
3. Biomarker expression patterns and what they suggest
4. Correlation with known kidney tissue morphology

Be concise and focus on the most informative features."""

        user_prompt = self.prompt_manager.get_task_prompt(
            agent_name="omics_profiler",
            task_name="analyze_roi",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Build system prompt with checklist if available
        system_content = self.system_prompt
        if self._checklist_content:
            system_content += "\n\n## Background Checklist\nRefer to the following checklist when analyzing cell composition:\n\n" + self._checklist_content

        # Create message
        message = HumanMessage(content=user_prompt)

        # Invoke LLM
        messages = [SystemMessage(content=system_content), message]
        response = self.model.invoke(messages)

        return self._extract_text(response.content)

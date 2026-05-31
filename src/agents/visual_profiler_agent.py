"""
VisualProfiler Agent for ROI visual analysis.

Specializes in:
- Multiplexed immunofluorescence image analysis
- Tissue structure identification from images
- Bounding box visualization
"""

import io
import sys
import pickle
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from .base_agent import BaseAgent, AgentExecutionError
from ..workflows.agent_state import AgentState, add_error
from ..utils.image_preparation import (
    load_rendered_image_bytes,
    prepare_images_for_llm,
    cleanup_temp_images
)
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class VisualProfilerAgent(BaseAgent):
    """
    VisualProfiler Agent for ROI visual analysis.

    This agent analyzes multiplexed immunofluorescence images to identify
    tissue structures (Proximal tubules, Distal tubules, Glomeruli, Blood vessel, Interstitium).
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """
        Initialize the VisualProfiler agent.

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

A series of protein biomarkers were measured in this sample and the ROI, and these biomarkers were used to characterize cells and multi-cellular structures.

You are given two images of the same section:

Image A (full slide): 7-channel composite with the ROI marked by a red bounding box.

Image B (zoomed ROI): enlarged view of that exact ROI.

Channels and colors (strict): Nestin=red, CD227=green, CD183=blue, CD45=yellow, aSMA=cyan, Perlecan=magenta, CollagenIV=gray.

Task: Using only what is visible inside the red ROI, identify the structure from this fixed set:
[Proximal tubules, Distal tubules, Glomeruli, Blood vessel, Interstitium].

Base your decision on staining intensity/patterns, co-localization, and morphology (e.g., tubular lumen, basement-membrane outlines, tuft-like aggregates, vascular wall). If evidence is weak/ambiguous, include a secondary candidate.

Your responses should be informative, simple, and concise."""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        super().__init__(
            name="VisualProfiler",
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
            agent_name="visual_profiler",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return self.system_prompt

    def _load_checklist(self, state: AgentState) -> Optional[str]:
        """Load VisualProfiler checklist from state path if available."""
        checklist_path = state.get("visual_profiler_checklist_path")
        if checklist_path and Path(checklist_path).exists():
            with open(checklist_path, 'r', encoding='utf-8') as f:
                content = f.read()
            display = CLIDisplay.get()
            display.agent_progress("VisualProfiler", f"Loaded checklist from {checklist_path}")
            return content
        return None

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute visual profiling analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with visual_reports populated

        Workflow:
        1. Load or generate ROI manifest
        2. Pre-load rendered image bytes for all regions
        3. Process each ROI in parallel:
           - Prepare images with bounding boxes
           - Call LLM with multimodal input
           - Clean up temporary files
        4. Aggregate results and save to state
        """
        state["current_action"] = "visual_profiling"
        state["current_phase"] = "phase2"

        try:
            display = CLIDisplay.get()

            display.agent_start("VisualProfiler", "Analyzing ROIs from registry...")

            # Step 1: Load or generate manifest
            manifest = await self._prepare_manifest(state)

            # Step 2: Load registry data
            roi_registry = await self._load_roi_registry(state)
            state["roi_registry"] = roi_registry
            state["manifest"] = manifest

            # Step 2b: Load background checklist if available
            self._checklist_content = self._load_checklist(state)

            # Step 3: Process ROIs
            visual_reports = await self._process_rois(state, manifest, roi_registry)

            # Step 4: Save results to files (per ROI)
            import json

            session_dir = Path(state.get("session_dir", "./outputs"))
            interpretation_dir = session_dir / "interpretation"
            interpretation_dir.mkdir(parents=True, exist_ok=True)

            for roi_key, visual_report in visual_reports.items():
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
                    roi_id = f"roi_{list(visual_reports.keys()).index(roi_key) + 1:03d}"

                roi_dir = interpretation_dir / roi_id
                roi_dir.mkdir(parents=True, exist_ok=True)

                # Save visual report for this ROI
                visual_report_path = roi_dir / "visual_report.json"
                with open(visual_report_path, 'w', encoding='utf-8') as f:
                    json.dump({"roi_key": str(roi_key), "report": visual_report}, f, indent=2, ensure_ascii=False)

            # Step 5: Update state
            state["visual_reports"] = visual_reports

            n_ok = sum(1 for v in visual_reports.values() if not v.startswith("ERROR") and not v.startswith("SKIPPED"))
            display.agent_done("VisualProfiler", f"{n_ok}/{len(visual_reports)} ROIs classified")
            self._log_execution(state, "visual_profiling", f"completed: {len(visual_reports)} ROIs analyzed")

        except Exception as e:
            error_msg = f"VisualProfiler execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "visual_profiling", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    async def _prepare_manifest(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Prepare ROI manifest from registry.

        Uses cluster_center_indices to get cluster center ROIs (the actual sampling points).

        Args:
            state: Current workflow state

        Returns:
            List of ROI dictionaries with keys: region_id, patch_name, cell_ids, roi_obj
        """
        # Check if manifest already exists from previous agent
        if state.get("manifest"):
            return state["manifest"]

        # Load from registry
        registry = await self._load_roi_registry(state)
        manifest = []

        # Build manifest using cluster_center_indices (same as interpretation_module)
        for region_id, shard in registry.shards.items():
            # Use cluster_center_indices to get cluster center ROIs
            # This is the actual sampling - cluster centers represent sampled regions
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

        # Limit to a reasonable number for demo
        display = CLIDisplay.get()
        display.agent_progress("VisualProfiler", f"Generated manifest with {len(manifest)} ROIs from {len(registry.shards)} regions")
        return manifest[:50]  # Process first 50 ROIs

    async def _load_roi_registry(self, state: AgentState) -> Dict[str, Any]:
        """
        Load ROI registry from pickle file.

        Args:
            state: Current workflow state

        Returns:
            SpatialPivotRegistry object (not converted to dict)
        """
        data_root = Path(state["data_root"])
        sample_id = state["sample_id"]

        # Try multiple possible registry paths
        possible_paths = [
            data_root / f"{sample_id}_pivot_ROIs_registry_dev.pkl",  # Dev registry
            data_root / f"{sample_id}_pivot_ROIs_registry.pkl",       # Standard registry
            data_root / f"{sample_id}_pivot_ROIs.pkl",                 # Legacy name
        ]

        registry_path = None
        for path in possible_paths:
            if path.exists():
                registry_path = path
                display = CLIDisplay.get()
                display.agent_progress("VisualProfiler", f"Found registry: {path.name}")
                break

        if not registry_path:
            # Return mock registry for testing
            display = CLIDisplay.get()
            display.agent_progress("VisualProfiler", "No registry file found, using mock data")
            return self._generate_mock_registry_from_real_dirs(data_root)

        with open(registry_path, 'rb') as f:
            registry = pickle.load(f)

        # Return the actual registry object (not converted to dict)
        # This preserves cluster_center_indices which is needed for manifest generation
        return registry

    def _generate_mock_registry_from_real_dirs(self, data_root: Path) -> Dict[str, Any]:
        """Generate mock registry from actual data directories."""
        from interpretation_module.src.data.data_models import (
            RegionOfInterest, RegionShardData, SpatialPivotRegistry
        )
        import numpy as np

        # Get actual region directories
        region_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("s255_c")]

        shards = []
        for region_dir in region_dirs[:3]:  # Limit to first 3 regions
            region_id = region_dir.name

            # Create mock ROIs
            rois = [
                RegionOfInterest(
                    patch_name=f"{region_id}_roi_{i}",
                    center_x=1000 + i * 100,
                    center_y=1000 + i * 100,
                    xyranges=(900 + i * 100, 1100 + i * 100, 900 + i * 100, 1100 + i * 100),
                    cell_ids=list(range(i * 100, (i + 1) * 100))
                )
                for i in range(5)
            ]

            shard = RegionShardData(
                region_id=region_id,
                rois=rois,
                combined_features=np.random.rand(len(rois), 42),
                cluster_assignments=[]
            )
            shards.append(shard)

        registry = SpatialPivotRegistry(dataset_id="mock_from_real")
        for shard in shards:
            registry.add_shard(shard)

        return {
            "dataset_id": registry.dataset_id,
            "shards": {shard.region_id: {
                "region_id": shard.region_id,
                "rois": shard.rois,
                "combined_features": shard.combined_features.tolist(),
            } for shard in shards}
        }

    def _generate_mock_registry(self) -> Dict[str, Any]:
        """Generate mock registry for testing."""
        from interpretation_module.src.data.data_models import (
            RegionOfInterest, RegionShardData, SpatialPivotRegistry
        )
        import numpy as np

        # Create mock ROIs
        rois = [
            RegionOfInterest(
                patch_name=f"roi_{i}",
                center_x=1000 + i * 100,
                center_y=1000 + i * 100,
                xyranges=(900 + i * 100, 1100 + i * 100, 900 + i * 100, 1100 + i * 100),
                cell_ids=list(range(i * 100, (i + 1) * 100))
            )
            for i in range(10)
        ]

        shard = RegionShardData(
            region_id="mock_region",
            rois=rois,
            combined_features=np.random.rand(len(rois), 42),
            cluster_assignments=[]
        )

        registry = SpatialPivotRegistry(dataset_id="mock")
        registry.add_shard(shard)

        return {
            "dataset_id": registry.dataset_id,
            "shards": {shard.region_id: {
                "region_id": shard.region_id,
                "rois": shard.rois,
                "combined_features": shard.combined_features.tolist(),
            } for shard in [shard]}
        }

    async def _process_rois(
        self,
        state: AgentState,
        manifest: List[Dict[str, Any]],
        roi_registry: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Process all ROIs through visual profiling.

        Args:
            state: Current workflow state
            manifest: List of ROI dictionaries
            roi_registry: Loaded ROI registry

        Returns:
            Dictionary mapping ROI keys to visual analysis reports
        """
        results = {}
        data_root = Path(state["data_root"])

        # Get scale_factor from state or use default
        scale_factor = state.get("scale_factor", 0.25)

        # Pre-load rendered image bytes (thread-safe)
        rendered_image_bytes = {}
        display = CLIDisplay.get()
        # roi_registry is now a SpatialPivotRegistry object, not a dict
        for region_id in roi_registry.shards.keys():
            try:
                rendered_image_bytes[region_id] = load_rendered_image_bytes(region_id, data_root)
                display.agent_progress("VisualProfiler", f"Loaded rendered image bytes: {region_id}")
            except FileNotFoundError as e:
                display.agent_progress("VisualProfiler", str(e))
                rendered_image_bytes[region_id] = None

        # Process ROIs in parallel
        def _process_single_roi(roi: Dict[str, Any]) -> tuple:
            """Process a single ROI and return (key, result)."""
            key = roi["key"]
            region_id = roi["region_id"]

            if region_id not in rendered_image_bytes or rendered_image_bytes[region_id] is None:
                return key, "SKIPPED_NO_RENDERED_IMAGE"

            img_data = None
            try:
                # Create fresh Image from bytes in each thread
                rendered_image = Image.open(io.BytesIO(rendered_image_bytes[region_id]))

                img_data = prepare_images_for_llm(
                    rendered_image,
                    roi,
                    scale_factor=scale_factor
                )

                # Prepare multimodal message for LLM
                result = self._analyze_roi_visual(roi, img_data)

                return key, result

            except Exception as e:
                return key, f"ERROR: {e}"

            finally:
                if img_data:
                    cleanup_temp_images(img_data)

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

    def _analyze_roi_visual(self, roi: Dict[str, Any], img_data: Dict[str, Any]) -> str:
        """
        Analyze a single ROI visually using LLM.

        Args:
            roi: ROI dictionary
            img_data: Image data from prepare_images_for_llm

        Returns:
            Visual analysis report from LLM
        """
        # Encode images to base64 for LangChain
        with open(img_data['full_path'], 'rb') as f:
            full_image_base64 = base64.b64encode(f.read()).decode('utf-8')
        with open(img_data['small_path'], 'rb') as f:
            small_image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Prepare context for template
        context = {
            "roi_patch_name": roi.get('patch_name', 'Unknown'),
            "roi_region_id": roi.get('region_id', 'Unknown')
        }

        # Prepare user prompt from template or fallback
        fallback_prompt = """Analyze the kidney tissue ROI shown in these images.

ROI: {roi_patch_name}
Location: Region {roi_region_id}

Please identify the tissue structure from: Proximal tubules, Distal tubules, Glomeruli, Blood vessel, Interstitium.

Consider:
- Staining patterns and intensity
- Co-localization of biomarkers
- Morphological features visible in the zoomed view

Provide your analysis with the identified structure and supporting evidence."""

        user_prompt = self.prompt_manager.get_task_prompt(
            agent_name="visual_profiler",
            task_name="analyze_roi",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Build system prompt with checklist if available
        system_content = self.system_prompt
        if self._checklist_content:
            system_content += "\n\n## Background Checklist\nRefer to the following checklist when identifying tissue structures:\n\n" + self._checklist_content

        # Create multimodal message
        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{full_image_base64}"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{small_image_base64}"}
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ])

        # Invoke LLM
        messages = [SystemMessage(content=system_content), message]
        response = self.model.invoke(messages)

        return self._extract_text(response.content)

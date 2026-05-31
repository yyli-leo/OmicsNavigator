"""
LangGraph Orchestrator for multi-agent workflow.

Manages the stateful multi-agent workflow using LangGraph's StateGraph.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from ..agents import (
    DataAnalystAgent,
    LiteratureReviewerAgent,
    PlannerAgent,
    RetrieverAgent,
    ValidatorAgent,
    VisualProfilerAgent,
    OmicsProfilerAgent,
    OmicsInterpreterAgent,
)
from ..tools.mcp_wrappers import MCPToolWrapper
from ..utils.api_logger import create_api_logger
from ..utils.cli_display import CLIDisplay
from .agent_state import AgentState, add_error


class LangGraphOrchestrator:
    """
    Orchestrates the multi-agent workflow using LangGraph.

    This class manages:
    - LangGraph StateGraph construction
    - Agent initialization with LLM and tools
    - Workflow execution with checkpointing
    - State management between agents
    """

    def __init__(self, config: Dict[str, Any], session_id_prefix: str = "session"):
        """
        Initialize the LangGraph orchestrator.

        Args:
            config: Configuration dictionary with model settings
            session_id_prefix: Prefix for session ID (e.g., "cli_complete_pipeline", "demo")
        """
        self.config = config
        self.checkpoint_path = config.get("langgraph", {}).get(
            "checkpoint_db", "checkpoints.db"
        )

        # Create session directory with prefix
        session_id = f"{session_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_output_dir = config.get("session", {}).get("output_dir", "./outputs")
        self.session_dir = Path(base_output_dir) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "api_calls").mkdir(exist_ok=True)
        (self.session_dir / "interpretation").mkdir(exist_ok=True)

        # Create session metadata
        self._create_session_metadata()

        # Apply proxy configuration before API loggers capture environment metadata.
        self.proxy_url = self._configure_proxy()

        # Create API loggers for each agent (like demo)
        api_log_dir = self.session_dir / "api_calls"
        default_model = config.get("system", {}).get("default_model", "unknown")
        self.api_loggers = self._create_api_loggers(str(api_log_dir), default_model)

        # Create models for each agent with their logger
        self.models = self._create_models()

        # Use MemorySaver for state persistence (in-memory)
        # For production, consider using a persistent checkpoint solution
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _create_api_loggers(self, api_log_dir: str, default_model: str = "unknown") -> Dict[str, Any]:
        """Create API loggers for all agents."""
        agent_names = [
            "data_analyst",
            "literature_reviewer",
            "planner",
            "visual_profiler",
            "omics_profiler",
            "omics_interpreter",
            "retriever",
            "validator"
        ]

        # Resolve per-agent model names from config overrides
        agent_overrides = self.config.get("system", {}).get("agents", {})

        loggers = {}
        for agent_name in agent_names:
            model = agent_overrides.get(agent_name, {}).get("model", default_model)
            loggers[agent_name] = create_api_logger(
                log_dir=api_log_dir,
                agent_name=agent_name,
                model_name=model
            )

        return loggers

    def _configure_proxy(self) -> Optional[str]:
        """
        Apply config-driven proxy settings for live API calls.

        The config is authoritative: proxy.enabled=false removes proxy variables
        from this process so real mode does not silently inherit a local proxy.
        """
        proxy_config = self.config.get("system", {}).get("proxy", {})
        proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")

        if not proxy_config.get("enabled", False):
            for key in proxy_keys:
                os.environ.pop(key, None)
            return None

        scheme = str(proxy_config.get("scheme", "http") or "http").strip()
        host = str(proxy_config.get("host", "127.0.0.1") or "127.0.0.1").strip()
        port = proxy_config.get("port")

        if not host or port in (None, ""):
            raise ValueError(
                "Proxy is enabled, but system.proxy.host or system.proxy.port is missing."
            )

        proxy_url = f"{scheme}://{host}:{port}"
        for key in proxy_keys:
            os.environ[key] = proxy_url

        return proxy_url

    def _create_models(self) -> Dict[str, ChatGoogleGenerativeAI]:
        """Create LLM models for each agent with their respective logger."""
        # Check API key
        api_key_env_var = self.config.get("system", {}).get("api_key_env_var", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"{api_key_env_var} environment variable not set. "
                "Please set it before running the workflow."
            )

        model_name = self.config.get("system", {}).get("default_model", "gemini-2.5-flash-lite")
        temperature = 0.7
        agent_overrides = self.config.get("system", {}).get("agents", {})

        models = {}
        for agent_name, logger in self.api_loggers.items():
            override = agent_overrides.get(agent_name, {})
            models[agent_name] = ChatGoogleGenerativeAI(
                model=override.get("model", model_name),
                api_key=api_key,
                temperature=override.get("temperature", temperature),
                callbacks=[logger]
            )

        return models

    def _create_session_metadata(self):
        """Create session metadata file."""
        metadata = {
            "session_id": self.session_dir.name,
            "start_time": datetime.now().isoformat(),
            "config": self.config
        }
        metadata_path = self.session_dir / "session_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _load_pivot_rois_from_registry(self, registry_path: str) -> Optional[Dict]:
        """
        Build pivot_rois structure from SpatialPivotRegistry.

        This is consistent with demo_analysis_pipeline.build_pivot_rois_from_registry()

        Args:
            registry_path: Path to the registry pickle file

        Returns:
            Dict with structure: {region_id: (sampled_rois, roi_features, cluster_labels, rep_rois)}
            or None if loading fails
        """
        try:
            import pickle
            from ..core.data_models import SpatialPivotRegistry

            with open(registry_path, "rb") as f:
                registry: SpatialPivotRegistry = pickle.load(f)

            pivot_rois = {}
            for region_id, shard_data in registry.shards.items():
                # Convert RegionOfInterest objects to dicts for consistency
                sampled_rois = []
                for roi in shard_data.rois:
                    sampled_rois.append({
                        "patch_name": roi.patch_name,
                        "region_id": region_id,
                        "center_x": float(roi.center_x),
                        "center_y": float(roi.center_y),
                        "xyrange": roi.xyranges,
                        "cell_ids": roi.cell_ids
                    })

                roi_features = shard_data.combined_features
                cluster_labels = shard_data.cluster_assignments

                # Build representative ROIs from cluster_center_indices
                rep_rois = {}
                for cluster_key, indices in shard_data.cluster_center_indices.items():
                    rep_rois[cluster_key] = [shard_data.rois[i] for i in indices]

                pivot_rois[region_id] = (sampled_rois, roi_features, cluster_labels, rep_rois)

            return pivot_rois

        except Exception as e:
            print(f"  Failed to load pivot_rois from registry: {e}")
            return None

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph.

        Returns:
            Compiled StateGraph ready for execution
        """
        # Initialize MCP tools
        mcp_tools = MCPToolWrapper()

        # Initialize all agents with their respective models
        data_analyst = DataAnalystAgent(self.models["data_analyst"], mcp_tools)
        literature_reviewer = LiteratureReviewerAgent(self.models["literature_reviewer"])
        planner = PlannerAgent(self.models["planner"])
        # Interpretation Module Agents
        visual_profiler = VisualProfilerAgent(self.models["visual_profiler"])
        omics_profiler = OmicsProfilerAgent(self.models["omics_profiler"])
        omics_interpreter = OmicsInterpreterAgent(self.models["omics_interpreter"])
        retriever = RetrieverAgent(self.models["retriever"])
        validator = ValidatorAgent(self.models["validator"])

        # Define the workflow
        workflow = StateGraph(AgentState)

        # Add nodes (agents)
        workflow.add_node("data_analyst", data_analyst.execute)
        workflow.add_node("literature_reviewer", literature_reviewer.execute)
        workflow.add_node("planner", planner.execute)
        workflow.add_node("visual_profiler", visual_profiler.execute)
        workflow.add_node("omics_profiler", omics_profiler.execute)
        workflow.add_node("omics_interpreter", omics_interpreter.execute)
        workflow.add_node("retriever", retriever.execute)
        workflow.add_node("validator", validator.execute)

        # Define edges for workflow flow

        # Entry point: Start with both data_analyst and literature_reviewer in parallel
        # We use a conditional router to start both agents
        from langgraph.constants import START, END

        # Set entry point to data_analyst, which will trigger literature_reviewer via edge
        workflow.set_entry_point("data_analyst")

        # Phase 1: Parallel execution - data_analyst triggers literature_reviewer
        workflow.add_edge("data_analyst", "literature_reviewer")

        # Both agents complete, then go to planner
        workflow.add_edge("literature_reviewer", "planner")

        # Phase 2: NEW Interpretation Module workflow (replacing old Sampler → Interpreter)
        # Planner → VisualProfiler → OmicsProfiler → OmicsInterpreter
        workflow.add_edge("planner", "visual_profiler")
        workflow.add_edge("visual_profiler", "omics_profiler")
        workflow.add_edge("omics_profiler", "omics_interpreter")

        # Phase 3: Sequential execution
        # OmicsInterpreter → Retriever
        workflow.add_edge("omics_interpreter", "retriever")
        # Retriever → Validator
        workflow.add_edge("retriever", "validator")
        # Validator → END
        workflow.add_edge("validator", END)

        # Compile with checkpointer for state persistence
        app = workflow.compile(checkpointer=self.checkpointer)

        return app

    async def execute(
        self,
        hypothesis_id: str,
        hypothesis_description: str,
        data_root: str,
        sample_id: str,
        blueprint_path: Optional[str] = None
    ) -> AgentState:
        """
        Execute the complete multi-agent workflow.

        Args:
            hypothesis_id: Identifier for the hypothesis
            hypothesis_description: Full hypothesis text
            data_root: Path to data directory
            sample_id: Sample/dataset identifier
            blueprint_path: Optional path to JSON blueprint

        Returns:
            Final AgentState with all results
        """
        # ========================================================================
        # Analysis Phase Setup
        # ========================================================================
        pivot_rois = None
        final_interpretations = {}

        display = CLIDisplay.get()

        # Load pivot_rois for spatial expansion
        display.agent_progress("System", "Loading pivot_rois for spatial expansion...")
        registry_path = Path(data_root) / f"{sample_id}_pivot_ROIs_registry_dev.pkl"

        if registry_path.exists():
            try:
                pivot_rois = self._load_pivot_rois_from_registry(str(registry_path))
                if pivot_rois:
                    display.agent_progress("System", f"Loaded pivot_rois from registry: {len(pivot_rois)} regions")
            except Exception as e:
                display.agent_progress("System", f"Failed to load pivot_rois from registry: {e}")

        if not pivot_rois:
            display.agent_progress("System", "No pivot_rois available, spatial expansion will be limited")

        # Load final_interpretations for FAISS index
        mockdata_dir = Path("cli/mockdata/interpretation_reports")
        mock_fi_path = mockdata_dir / "final_interpretations.pkl"

        if mock_fi_path.exists() and mock_fi_path.stat().st_size > 0:
            try:
                import pickle
                with open(mock_fi_path, "rb") as f:
                    final_interpretations = pickle.load(f)
                display.agent_progress("System", f"Loaded {len(final_interpretations)} ROIs from {mock_fi_path}")
            except Exception as e:
                display.agent_progress("System", f"Failed to load final_interpretations: {e}")

        # Initialize state as dict (TypedDict)
        initial_state: AgentState = {
            "hypothesis_id": hypothesis_id,
            "hypothesis_description": hypothesis_description,
            "data_root": data_root,
            "sample_id": sample_id,
            "blueprint_path": blueprint_path,
            "session_dir": str(self.session_dir),  # Add session directory to state
            "model_name": self.config.get("system", {}).get("default_model", "gemini-2.5-flash-lite"),
            "agent_models": {name: m.model for name, m in self.models.items()},
            "max_parallel_agents": self.config.get("system", {}).get("max_workers", 2),
            "current_phase": "init",
            "errors": [],
            "execution_log": [],
            # Enable spatial expansion for Retriever
            "enable_spatial_expansion": True,
            # Analysis phase data
            "pivot_rois": pivot_rois,
            "final_interpretations": final_interpretations,
            # VisualProfiler image scaling factor
            "scale_factor": self.config.get("phases", {}).get("phase2_interpretation", {}).get("scale_factor", 0.25),
            # Clinical phenotype mapping (region_id → phenotype)
            "clinical_mapping": self.config.get("metadata", {}).get("clinical_mapping")
        }

        # Configure thread ID for checkpointing
        config = {
            "configurable": {
                "thread_id": f"omicsnav_{hypothesis_id}_{sample_id}"
            }
        }

        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state, config)

            # Update phase marker
            result["current_phase"] = "complete"

            return result

        except Exception as e:
            # Log error to state
            add_error(initial_state, f"Workflow execution failed: {str(e)}")
            initial_state["current_phase"] = "failed"
            return initial_state

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow graph.

        Returns:
            Dictionary with graph structure info
        """
        return {
            "nodes": [
                "data_analyst",
                "literature_reviewer",
                "planner",
                "visual_profiler",
                "omics_profiler",
                "omics_interpreter",
                "retriever",
                "validator"
            ],
            "phases": {
                "phase1": ["data_analyst", "literature_reviewer", "planner"],
                "phase2": ["visual_profiler", "omics_profiler", "omics_interpreter"],
                "phase3": ["retriever", "validator"]
            },
            "checkpoint_db": self.checkpoint_path
        }

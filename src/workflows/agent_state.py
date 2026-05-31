"""
AgentState definition for LangGraph workflow.

Defines the state object that flows through all agents in the workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class AgentState(TypedDict):
    """
    Centralized state for LangGraph multi-agent workflow.

    This state object flows through all agents and accumulates
    results from each phase of the pipeline.

    Attributes:
        hypothesis_id: Identifier for the hypothesis being tested
        hypothesis_description: Full text description of the hypothesis
        data_root: Root directory containing the dataset
        sample_id: Sample/dataset identifier
        blueprint_path: Optional path to JSON blueprint file

        Phase 1 Outputs (Parallel):
        dataset_profile: Results from DataAnalyst with validation and iteration info:
            - analysis_report: Final validated markdown report (str)
            - iteration_count: Number of iterations performed (int)
            - validation_log: List of validation results (List[Dict])
            - report_path: Path to saved report file (str)
        literature_summaries: Literature review from LiteratureReviewer
        analysis_plan: Analysis plan from Planner with blueprint:
            - blueprint_path: Path to Verification_Blueprint.json (str)
            - raw_plan: Natural language summary (str)
            - blueprint: Structured blueprint dictionary (Dict)

        Phase 2 Outputs (Sequential):
        roi_registry_path: Path to ROI registry file
        roi_count: Number of ROIs generated
        interpretation_reports_path: Path to interpretation reports
        # Interpretation Module Outputs (NEW):
        manifest: List of ROI dictionaries generated from analysis_plan
        visual_reports: Dictionary mapping ROI keys to visual analysis reports
        omics_reports: Dictionary mapping ROI keys to omics analysis reports
        final_interpretations: Dictionary mapping ROI keys to final tissue classifications
        roi_registry: Loaded SpatialPivotRegistry with ROI data

        Phase 3 Outputs (Sequential):
        semantic_search_results: FAISS search results
        validation_results: Statistical validation results:
            - conclusion: VERIFIED|FALSIFIED|INCONCLUSIVE (str)
            - confidence: 0-1 score (float)
            - key_findings: List of test results (List[Dict])
            - raw_statistics: Full statistical output (Dict)
            - execution_log: Execution steps (List[str])

        Execution Metadata:
        current_phase: Current phase identifier
        current_action: Current action identifier
        errors: List of error messages
        execution_log: List of execution log entries

        Configuration:
        model_name: LLM model name being used
        max_parallel_agents: Maximum number of parallel agents
    """

    # Input/Configuration
    hypothesis_id: str
    hypothesis_description: str
    data_root: str
    sample_id: str
    blueprint_path: NotRequired[Optional[str]]

    # Phase 1: Planning Module Outputs (Parallel)
    dataset_profile: NotRequired[Optional[Dict[str, Any]]]
    literature_summaries: NotRequired[Optional[Dict[str, Any]]]  # Changed: List → Dict for structured output
    analysis_plan: NotRequired[Optional[Dict[str, Any]]]

    # LiteratureReviewer specific outputs
    literature_metadata: NotRequired[Optional[Dict[str, Any]]]  # Extracted biomarkers, cell types, concepts
    literature_query_path: NotRequired[Optional[str]]  # Path to generated Deep Research query
    literature_report_path: NotRequired[Optional[str]]  # Path to full literature report

    # Planner checklist outputs (passed to Interpretation Module agents)
    visual_profiler_checklist_path: NotRequired[Optional[str]]  # Path to VisualProfiler checklist
    omics_profiler_checklist_path: NotRequired[Optional[str]]  # Path to OmicsProfiler checklist
    omics_interpreter_checklist_path: NotRequired[Optional[str]]  # Path to OmicsInterpreter checklist

    # Phase 2: Interpretation Module Outputs (Sequential)
    roi_registry_path: NotRequired[Optional[str]]
    roi_count: NotRequired[Optional[int]]
    interpretation_reports_path: NotRequired[Optional[str]]
    # NEW: Interpretation Module Agent Outputs
    manifest: NotRequired[Optional[List[Dict[str, Any]]]]  # List of ROI dictionaries
    visual_reports: NotRequired[Optional[Dict[str, str]]]  # ROI key → visual analysis
    omics_reports: NotRequired[Optional[Dict[str, str]]]  # ROI key → omics analysis
    final_interpretations: NotRequired[Optional[Dict[str, str]]]  # ROI key → classification
    roi_registry: NotRequired[Optional[Dict]]  # Loaded SpatialPivotRegistry
    pivot_rois: NotRequired[Optional[Dict]]  # ROI features for spatial expansion: {region_id: (sampled_rois, roi_features, cluster_labels, rep_rois)}

    # Phase 3: Analysis Module Outputs (Sequential)
    semantic_search_results: NotRequired[Optional[Dict[str, Any]]]
    validation_results: NotRequired[Optional[Dict[str, Any]]]
    # Spatial expansion results (optional)
    spatial_expansion_results: NotRequired[Optional[Dict[str, Any]]]  # Results from spatial expansion: {num_hits, num_total, threshold, results}

    # Execution Metadata
    current_phase: NotRequired[str]
    current_action: NotRequired[str]
    errors: NotRequired[List[str]]
    execution_log: NotRequired[List[Dict[str, str]]]

    # Configuration
    model_name: NotRequired[str]
    max_parallel_agents: NotRequired[int]
    # Spatial expansion configuration (optional)
    enable_spatial_expansion: NotRequired[Optional[bool]]  # Enable spatial expansion in RetrieverAgent
    spatial_expansion_threshold: NotRequired[Optional[float]]  # Quantile threshold for distance normalization (default: 0.333)
    spatial_expansion_filter: NotRequired[Optional[str]]  # Optional comma-separated filter keywords

    # Session Management
    session_dir: NotRequired[Optional[str]]  # Path to current session directory

    # Clinical phenotype mapping (region_id → phenotype)
    clinical_mapping: NotRequired[Optional[Dict[str, str]]]

    # Validator report output
    validation_report: NotRequired[Optional[str]]  # Generated validation report markdown


# Helper functions for state manipulation (since TypedDict can't have methods)
def add_error(state: AgentState, error: str) -> None:
    """Add an error to the state's error list."""
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(error)


def add_log(state: AgentState, agent: str, action: str, status: str, details: str = "") -> None:
    """Add an execution log entry."""
    from datetime import datetime

    if "execution_log" not in state:
        state["execution_log"] = []

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "status": status,
        "details": details[:500]  # Truncate for memory
    }
    state["execution_log"].append(log_entry)


def has_errors(state: AgentState) -> bool:
    """Check if state has any errors."""
    return len(state.get("errors", [])) > 0


def get_phase_output(state: AgentState, phase: int) -> Dict[str, Any]:
    """
    Get outputs for a specific phase.

    Args:
        state: The AgentState instance
        phase: Phase number (1, 2, or 3)

    Returns:
        Dictionary containing phase outputs
    """
    if phase == 1:
        return {
            "dataset_profile": state.get("dataset_profile"),
            "literature_summaries": state.get("literature_summaries"),
            "analysis_plan": state.get("analysis_plan")
        }
    elif phase == 2:
        return {
            "roi_registry_path": state.get("roi_registry_path"),
            "roi_count": state.get("roi_count"),
            "interpretation_reports_path": state.get("interpretation_reports_path"),
            "manifest": state.get("manifest"),
            "visual_reports": state.get("visual_reports"),
            "omics_reports": state.get("omics_reports"),
            "final_interpretations": state.get("final_interpretations"),
            "roi_registry": state.get("roi_registry")
        }
    elif phase == 3:
        return {
            "semantic_search_results": state.get("semantic_search_results"),
            "validation_results": state.get("validation_results")
        }
    else:
        return {}

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class AgentResponse:
    """
    Standardized Data Transfer Object (DTO) for agent outputs.
    """
    content: str
    tool_used: Optional[str] = None
    status: str = "success"


class SpatialOmicsAgent:
    """
    Mock agent class to handle spatial omics queries and maintain session state.
    """

    def __init__(self, name: str = "OmicsNavigator Core"):
        self.name: str = name
        # State management for the current active LLM
        self.current_model: str = "Gemini-2.5-pro"

    def get_model_info(self) -> AgentResponse:
        """
        Retrieves the currently configured LLM model.
        """
        return AgentResponse(
            content=f"**Current Model Configuration:** `{self.current_model}`",
            tool_used="SystemConfig"
        )

    def process_query(self, query: str) -> AgentResponse:
        """
        Simulates the processing pipeline of a user query.
        Eventually, this will interface with actual LLMs via LangChain or direct APIs.
        """
        query_lower: str = query.lower()

        if "seurat" in query_lower:
            return AgentResponse(
                content="**Executing Spatial Analysis:** Running `FindSpatiallyVariableFeatures`...\n\n- Identified **150** spatially variable features.\n- P-values mapped to metadata.",
                tool_used="SeuratIntegrationTool"
            )

        return AgentResponse(
            content=f"[{self.name}] Acknowledged query: '{query}'.\n\n*Waiting for specific omics instructions...*"
        )

    def get_pipeline_action_data(self, action_id: str) -> str:
        """
        Return mock data content for pipeline actions.

        Args:
            action_id: Action identifier (e.g., "action_1", "action_2")

        Returns:
            Content from the corresponding mock data file, or empty string if not found
        """
        # Get the mockdata directory
        current_file = Path(__file__)
        mock_data_dir = current_file.parent / "mockdata"

        # Mapping of action IDs to mock data files
        action_files = {
            "action_1": "H1_DataAnalyst.txt",
            "action_2": "H1_LiteratureReviewer.txt",
            "action_3": "H1_Planner.txt",
            "action_7": "H1_HypothesisValidator.txt"
        }

        if action_id in action_files:
            file_path = mock_data_dir / action_files[action_id]
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

        return ""  # For actions without mock files (4, 5, 6) or not found
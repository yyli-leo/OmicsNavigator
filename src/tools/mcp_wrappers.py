"""
MCP Tool Wrapper for LangChain integration.

Wraps FastMCP tools as LangChain tools for use with agents.
"""

from typing import Dict, List, Optional
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field


# Pydantic schemas for tool inputs
class InspectSchemaInput(BaseModel):
    """Input schema for inspect_data_schema tool."""
    data_root: str = Field(description="Root directory containing the sample")
    sample_id: str = Field(description="Sample/dataset identifier")


class AnalyzeSampleInput(BaseModel):
    """Input schema for analyze_spatial_sample tool."""
    data_root: str = Field(description="Root directory containing the sample")
    sample_id: str = Field(description="Sample/dataset identifier")


class ExecuteScriptInput(BaseModel):
    """Input schema for execute_analysis_script tool."""
    script_content: str = Field(description="Python code to execute in sandbox")


class MCPToolWrapper:
    """
    Wraps MCP tools as LangChain tools.

    For now, this provides mock implementations that simulate tool behavior.
    Real MCP tool integration can be added later.
    """

    def __init__(self):
        """Initialize the MCP tool wrapper."""
        self._tools: Dict[str, BaseTool] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize tools as LangChain tools (mock implementations for now)."""

        # Tool 1: inspect_data_schema (mock)
        def mock_inspect_schema(data_root: str, sample_id: str) -> str:
            return f"""Schema inspection for {sample_id}:
Columns: cell_id, x, y, cell_type, marker_1, marker_2, ..., marker_42
Shape: (137654 cells, 45 features)
Cell types: 11 unique types
Markers: 42 spatial omics markers
"""

        self._tools["inspect_schema"] = StructuredTool.from_function(
            func=mock_inspect_schema,
            name="inspect_data_schema",
            description=(
                "Inspect the CSV schema of a spatial omics dataset. "
                "Returns columns, first rows, and data types. "
                "Use this to understand the structure of a new dataset."
            ),
            args_schema=InspectSchemaInput
        )

        # Tool 2: analyze_spatial_sample (mock)
        def mock_analyze_sample(data_root: str, sample_id: str) -> str:
            return f"""=== Data Profile Report for Sample: {sample_id} ===
Total Regions Found: 17

**Global Metadata:**
- Total Cell Count: N = 137,654
- Available Biomarkers: 42 markers (ACE2, C1QC, C3a, C3aR, C3d, C4d, C5aR, C9, CD107a, CD11b, CD11c, CD141, CD183, CD196, CD21, CD227, CD25, CD31, CD35, CD38, CD45, CD46, CD55, CD68, Clusterin, CollagenIV, DAPI, EpCAM, FoxP3, GranzymeB, ICOS, MASP2, Nestin, PD1, Perlecan, RORgammaT, SC5b9, SPP1, TFAM, VWF, aSMA, bCatenin1)
- Available Cell Types: 11 types (Proximal Tubules, Distal Tubules, Endothelial cells, Basement membrane, VSMCs, Immune cells, Macrophages, Low expressing, Endothelial (CD31+/CD196+), Nestin+ cells, Myeloid cells)

--- 1. Cell Type Composition (Global Aggregated) ---
Cell Type                                | Count    | %
------------------------------------------------------------
Proximal Tubules                         | 48,910   | 35.53
Distal Tubules                           | 26,057   | 18.93
Endothelial cells                        | 12,909   | 9.38
Basement membrane                        | 11,642   | 8.46
VSMCs                                    | 10,557   | 7.67
Immune cells                             | 8,271    | 6.01
Macrophages                              | 5,953    | 4.32
Low expressing                           | 5,512    | 4.00
Endothelial (CD31+/CD196+)               | 4,787    | 3.48
Nestin+ cells                            | 1,924    | 1.40
Myeloid cells                            | 1,132    | 0.82

--- 2. Molecular Dynamic Range & Sparsity ---
Marker          | Sparsity   | Min        | 25th       | Median     | 75th       | 90th       | 95th       | 99th       | Max
--------------------------------------------------------------------------------------------------------------------
CollagenIV      | 0.12%      | 0.00       | 45.23      | 156.78     | 234.56     | 312.45     | 389.12     | 523.67     | 789.45
aSMA            | 0.08%      | 0.00       | 23.45      | 123.34     | 198.76     | 267.89     | 345.67     | 456.78     | 612.34
SPP1            | 0.15%      | 0.00       | 12.34      | 89.56      | 145.23     | 201.45     | 267.89     | 356.78     | 489.12
Nestin          | 0.22%      | 0.00       | 8.90       | 67.45      | 112.34     | 167.89     | 223.45     | 301.23     | 412.56
CD227           | 0.18%      | 0.00       | 15.67      | 78.90      | 134.56     | 189.23     | 245.67     | 334.12     | 445.78

--- 3. Top 15 Highly Expressed Markers (Global Mean) ---
Marker              | Mean Intensity
------------------------------------
CollagenIV          | 156.78
aSMA                | 123.34
SPP1                | 89.56
Nestin              | 67.45
CD227               | 78.90

[Result] Report generation complete.
"""

        self._tools["analyze_sample"] = StructuredTool.from_function(
            func=mock_analyze_sample,
            name="analyze_spatial_sample",
            description=(
                "Perform full statistical profile including sparsity, "
                "cell composition, and spatial density. "
                "Use this to get comprehensive analysis of a spatial omics sample."
            ),
            args_schema=AnalyzeSampleInput
        )

        # Tool 3: execute_analysis_script (mock)
        def mock_execute_script(script_content: str) -> str:
            return "Script executed successfully (mock mode). Output: Analysis complete."

        self._tools["execute_script"] = StructuredTool.from_function(
            func=mock_execute_script,
            name="execute_analysis_script",
            description=(
                "Execute custom Python analysis script in secure sandbox. "
                "60 second timeout. No file deletion or writes to /data allowed. "
                "Use this for custom analyses that require code execution."
            ),
            args_schema=ExecuteScriptInput
        )

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.

        Args:
            name: Tool identifier (e.g., "inspect_schema", "analyze_sample")

        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all available MCP tools.

        Returns:
            List of all tool instances
        """
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool exists.

        Args:
            name: Tool identifier

        Returns:
            True if tool exists, False otherwise
        """
        return name in self._tools

    def list_tools(self) -> List[str]:
        """
        List all available tool names.

        Returns:
            List of tool identifiers
        """
        return list(self._tools.keys())

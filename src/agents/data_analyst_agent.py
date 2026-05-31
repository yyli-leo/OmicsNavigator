"""
DataAnalyst Agent for spatial omics dataset analysis.

Specializes in:
- Dataset profiling and structure understanding
- Sparsity analysis
- Cellular composition analysis
- Spatially variable gene identification
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.mcp_wrappers import MCPToolWrapper
from ..workflows.agent_state import AgentState
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class DataAnalystAgent(BaseAgent):
    """
    DataAnalyst Agent for spatial omics dataset analysis.

    This agent analyzes spatial omics datasets to provide:
    - Data structure and schema information
    - Sparsity metrics (zero density, non-zero statistics)
    - Cellular composition (cell type distributions)
    - Spatially variable genes (Moran's I analysis)
    """

    def __init__(
        self,
        model: BaseChatModel,
        mcp_tools: MCPToolWrapper,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """
        Initialize the DataAnalyst agent.

        Args:
            model: LangChain chat model instance
            mcp_tools: MCP tool wrapper for accessing analysis tools
            prompt_manager: PromptManager instance (if None, creates default)
            use_templates: If True, use Jinja2 templates; else use hardcoded prompts
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.use_templates = use_templates
        self.mcp_tools = mcp_tools

        # Hardcoded fallback prompt (existing implementation)
        self.fallback_system_prompt = """You are a Spatial Omics Data Analyst specializing in analyzing multiplexed imaging fluorescence (MIF) datasets.

Your expertise includes:
1. **Dataset Profiling**: Understanding data structure, dimensions, and quality
2. **Sparsity Analysis**: Computing expression matrix zero density and statistics
3. **Cellular Composition**: Determining cell type distributions across regions
4. **Spatially Variable Genes**: Identifying spatially autocorrelated genes using Moran's I
5. **Quality Validation**: Ensuring analysis completeness and statistical validity

## Available Tools:
- **inspect_data_schema**: Examine the CSV structure, columns, and data types
- **analyze_spatial_sample**: Get comprehensive statistical profile (sparsity, composition, spatial genes)
- **execute_analysis_script**: Run custom Python code for additional analysis (60s timeout)

## Analysis Workflow:
1. **Initial Analysis**: Use inspect_data_schema and analyze_spatial_sample to gather data
2. **Self-Validation**: Review your analysis for completeness and quality
3. **Iterate if Needed**: If analysis is incomplete, use execute_analysis_script to gather missing information
4. **Final Report**: Provide comprehensive, validated analysis

## Quality Standards:
A complete analysis must include:
- Dataset dimensions and structure (cell count, features, cell types)
- Sparsity metrics (zero density, non-zero statistics)
- Cellular composition table with counts and percentages
- Molecular Dynamic Range: Percentile distributions for key biomarkers. Do NOT include Sparsity (%) column in the report table.
- Biological insights relevant to the hypothesis
- Actionable recommendations for downstream analysis

## Output Format:
Provide analysis results in markdown format with:
- Clear section headers
- Tables for composition and spatial genes
- Key statistics (means, standard deviations, p-values)
- Biological interpretations
- Recommendations

Be thorough but concise. Prioritize actionable insights for hypothesis testing."""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        # Get available MCP tools
        tools = []
        if mcp_tools.has_tool("inspect_schema"):
            tools.append(mcp_tools.get_tool("inspect_schema"))
        if mcp_tools.has_tool("analyze_sample"):
            tools.append(mcp_tools.get_tool("analyze_sample"))
        if mcp_tools.has_tool("execute_script"):
            tools.append(mcp_tools.get_tool("execute_script"))

        super().__init__(
            name="DataAnalyst",
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for analytical precision
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt from template or fallback."""
        return self.prompt_manager.get_system_prompt(
            agent_name="data_analyst",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return self.system_prompt

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute dataset analysis with validation and iteration.

        Args:
            state: Current workflow state (TypedDict)

        Returns:
            Updated state with validated dataset_profile

        Workflow:
        1. Phase 1: Initial Analysis - Call tools and generate initial report
        2. Phase 2: Validation Loop - LLM validates report quality
        3. Phase 3: Iterative Improvement - Execute custom code if needed
        4. Phase 4: Finalization - Save report and update state
        """
        state["current_action"] = "dataset_analysis"
        state["current_phase"] = "phase1"

        try:
            # Phase 1: Generate initial analysis
            display = CLIDisplay.get()
            display.agent_start("DataAnalyst", "Analyzing dataset with MCP tools...")
            current_report = await self._generate_initial_analysis(state)
            display.agent_progress("DataAnalyst", f"Initial report generated ({len(current_report)} chars)")
            iteration_count = 0
            validation_log = []
            max_iterations = 3

            # Phase 2 & 3: Validation and Iteration Loop
            while iteration_count < max_iterations:
                # Validate current report
                validation_result = await self._validate_report(current_report, state)
                validation_log.append({
                    "iteration": iteration_count,
                    "passed": validation_result["passed"],
                    "feedback": validation_result["feedback"]
                })

                if validation_result["passed"]:
                    # Validation passed, proceed to finalization
                    display.agent_progress("DataAnalyst", f"Validation round {iteration_count+1}: PASSED")
                    break
                else:
                    feedback_preview = validation_result["feedback"][:120].replace("\n", " ")
                    display.agent_progress("DataAnalyst", f"Validation round {iteration_count+1}: FAILED")
                    display.agent_progress("DataAnalyst", f"Reason: {feedback_preview}...")
                    if iteration_count < max_iterations - 1:
                        display.agent_progress("DataAnalyst", "Running improved analysis script...")

                # Validation failed, iterate if not at max
                if iteration_count < max_iterations - 1:
                    # Generate improvement plan
                    improvement = await self._plan_improvement(
                        current_report,
                        validation_result["feedback"],
                        state
                    )

                    # Execute custom analysis
                    new_insights = await self._execute_custom_analysis(improvement)

                    # Update report with new insights
                    if new_insights and new_insights.strip():
                        current_report = await self._update_report(
                            current_report,
                            new_insights
                        )
                        display.agent_progress("DataAnalyst", f"Report updated ({len(current_report)} chars)")

                iteration_count += 1

            # Phase 4: Finalization
            return await self._finalize_analysis(
                state,
                current_report,
                iteration_count,
                validation_log
            )

        except Exception as e:
            from ..workflows.agent_state import add_error
            error_msg = f"DataAnalyst execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "dataset_analysis", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

    # ========================================================================
    # Phase 1: Initial Analysis Methods
    # ========================================================================

    async def _generate_initial_analysis(self, state: AgentState) -> str:
        """
        Generate initial analysis report by calling MCP tools.

        Args:
            state: Current workflow state

        Returns:
            Initial analysis report as markdown string
        """
        # Prepare context for template
        context = {
            "data_root": state["data_root"],
            "sample_id": state["sample_id"]
        }

        # Prepare fallback prompt
        fallback_prompt = """Analyze the spatial omics dataset with the following details:

**Data Location:**
- Data Root: {data_root}
- Sample ID: {sample_id}

**Analysis Tasks:**
1. Inspect the data schema to understand the structure
2. Compute percentile distributions for biomarkers
3. Analyze cellular composition with absolute counts and relative percentages
4. Generate molecular dynamic range report

**Expected Output:**
Provide a factual analysis report with:
- **Global Metadata**: Total cell count, available biomarkers, available cell types
- **Cellular Composition**: Table with cell types showing BOTH absolute counts AND percentages
- **Molecular Dynamic Range**: Percentile distributions (Min, 25th, Median, 75th, 90th, 95th, 99th, Max) for key biomarkers. Do NOT include Sparsity (%) column in the report table.

Focus on presenting the data accurately and clearly. Do not include biological interpretations, hypotheses, or recommendations.

Use the available tools (inspect_data_schema, analyze_spatial_sample) to perform this analysis."""

        # Build analysis prompt from template or fallback
        prompt = self.prompt_manager.get_task_prompt(
            agent_name="data_analyst",
            task_name="initial_analysis",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Use the model with tools directly
        if self.tools:
            # Bind tools to model
            model_with_tools = self.model.bind_tools(list(self.tools.values()))

            # Create messages
            from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            # Loop to handle multi-turn tool calling
            # Some models (e.g., gemini-3.1-flash-lite-preview) make sequential
            # tool calls rather than batching, so we need to keep handling them.
            max_tool_rounds = 10
            for _ in range(max_tool_rounds):
                response = await model_with_tools.ainvoke(messages)
                messages.append(response)

                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Execute all tool calls from this response
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                        tool_id = tool_call['id']

                        if tool_name in self.tools:
                            tool = self.tools[tool_name]
                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Error executing {tool_name}: {str(e)}"

                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id
                            )
                            messages.append(tool_message)
                    # Continue loop to let model process tool results
                    continue
                else:
                    # No tool calls — model returned a text response
                    return self._extract_text(response.content)

            # Max rounds reached, return whatever we have
            return self._extract_text(response.content)
        else:
            # No tools, just use model directly
            messages = self._format_prompt(prompt)
            return await self._invoke_llm(messages)

    # ========================================================================
    # Phase 2: Validation Methods
    # ========================================================================

    async def _validate_report(self, report: str, state: AgentState) -> Dict[str, Any]:
        """
        Use LLM to validate analysis report quality.

        Args:
            report: Current analysis report
            state: Current workflow state

        Returns:
            Dictionary with validation results:
            - passed: bool indicating if validation passed
            - feedback: str with specific feedback
            - missing_sections: list of missing sections (if any)
            - issues: list of specific issues found (if any)
        """
        # Prepare context for template
        context = {
            "analysis_report": report
        }

        # Prepare fallback prompt
        fallback_prompt = """You are a Quality Assurance specialist for spatial omics analysis.

Review the following analysis report and determine if it meets quality standards.

**Analysis Report:**
{analysis_report}

**Validation Criteria:**
1. **Completeness**: Are all required sections present?
   - Dataset overview with dimensions
   - Sparsity analysis with metrics
   - Cellular composition table
   - Spatially variable genes with statistics

2. **Statistical Validity**: Are statistics reported correctly?
   - Means with standard deviations
   - Percentages sum to 100%
   - P-values included for spatial genes

3. **Biological Insights**: Are there meaningful interpretations?
   - Hypothesis-relevant findings
   - Connection to kidney disease/fibrosis biology
   - Actionable insights

4. **Actionability**: Can this report inform experimental design?

**Response Format:**
Respond in JSON format:
{{
    "passed": true/false,
    "feedback": "Specific feedback on what's missing or needs improvement",
    "missing_sections": ["list", "of", "missing", "sections"],
    "issues": ["specific", "issues", "found"]
}}

Be thorough but reasonable. Minor formatting issues should not fail validation."""

        validation_prompt = self.prompt_manager.get_task_prompt(
            agent_name="data_analyst",
            task_name="validation",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        response = await self._invoke_llm(self._format_prompt(validation_prompt))

        # Parse JSON response
        import json
        try:
            # Try to extract JSON from response
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Remove code blocks if present
                response_clean = response_clean.split('```')[1]
                if response_clean.startswith('json'):
                    response_clean = response_clean[4:]
            return json.loads(response_clean.strip())
        except:
            # Fallback if LLM doesn't return valid JSON
            passed = "PASS" in response.upper() or "VALID" in response.upper() or "COMPLETE" in response.upper()
            return {
                "passed": passed,
                "feedback": response,
                "missing_sections": [],
                "issues": []
            }

    # ========================================================================
    # Phase 3: Iterative Improvement Methods
    # ========================================================================

    async def _plan_improvement(self, report: str, feedback: str, state: AgentState) -> str:
        """
        Generate Python code to address validation feedback.

        Args:
            report: Current analysis report
            feedback: Validation feedback
            state: Current workflow state

        Returns:
            Python code string to execute
        """
        # Prepare context for template
        context = {
            "current_report": report,
            "validation_feedback": feedback,
            "data_root": state["data_root"],
            "sample_id": state["sample_id"]
        }

        # Prepare fallback prompt
        fallback_prompt = """You need to improve the spatial omics analysis based on validation feedback.

**Current Report:**
{current_report}

**Validation Feedback:**
{validation_feedback}

**Available Data:**
- Data Root: {data_root}
- Sample ID: {sample_id}

**Task:**
Generate Python code to address the missing analysis. The code should:
1. Load the dataset from {data_root}/{sample_id}
2. Perform the missing analysis identified in the feedback
3. Print results in a clear, readable format

**Constraints:**
- Execution timeout is 60 seconds
- No file deletion or writes to /data allowed
- Use pandas for data manipulation
- Print results using print() statements

Provide ONLY the Python code, no explanations."""

        improvement_prompt = self.prompt_manager.get_task_prompt(
            agent_name="data_analyst",
            task_name="improvement",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Check if execute_script tool is available
        if "execute_script" not in self.tools:
            # No custom script tool, return empty
            return ""

        # Use LLM with tool to generate and execute code
        model_with_tools = self.model.bind_tools([self.tools["execute_script"]])

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=improvement_prompt)
        ]

        response = await model_with_tools.ainvoke(messages)

        # If LLM decides to use the tool, extract the code
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'execute_analysis_script':
                    # Return the code content
                    return tool_call['args'].get('script_content', '')

            # Model made tool calls but none were execute_analysis_script;
            # handle them and try again
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_id = tool_call['id']
                tool_name = tool_call['name']
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    try:
                        tool_result = tool.invoke(tool_call['args'])
                    except Exception as e:
                        tool_result = f"Error executing {tool_name}: {str(e)}"
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    ))

            response = await model_with_tools.ainvoke(messages)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] == 'execute_analysis_script':
                        return tool_call['args'].get('script_content', '')

        # Fallback: return the LLM's text response as code
        return self._extract_text(response.content)

    async def _execute_custom_analysis(self, code: str) -> str:
        """
        Execute custom analysis code.

        Args:
            code: Python code to execute

        Returns:
            Execution output/results
        """
        if not code or not code.strip():
            return ""

        if "execute_script" not in self.tools:
            return "Custom analysis tool not available"

        tool = self.tools["execute_script"]
        try:
            return str(tool.invoke({"script_content": code}))
        except Exception as e:
            return f"Error executing custom analysis: {str(e)}"

    async def _update_report(self, current_report: str, new_insights: str) -> str:
        """
        Update report with new insights from custom analysis.

        Args:
            current_report: Current analysis report
            new_insights: New insights from custom analysis

        Returns:
            Updated complete report
        """
        if not new_insights or not new_insights.strip():
            return current_report

        update_prompt = f"""Update the analysis report with new insights from additional analysis.

**Current Report:**
{current_report}

**New Analysis Results:**
{new_insights}

**Instructions:**
1. Integrate the new insights seamlessly into the existing report
2. Maintain the report structure and formatting
3. Update relevant sections with new findings
4. Ensure consistency across the report
5. Add a note: "[Updated with additional analysis]" to modified sections

Provide the COMPLETE updated report."""

        return await self._invoke_llm(self._format_prompt(update_prompt))

    # ========================================================================
    # Phase 4: Finalization Methods
    # ========================================================================

    async def _finalize_analysis(
        self,
        state: AgentState,
        report: str,
        iteration_count: int,
        validation_log: list
    ) -> AgentState:
        """
        Save final report and update state.

        Args:
            state: Current workflow state
            report: Final validated report
            iteration_count: Number of iterations performed
            validation_log: Validation history

        Returns:
            Updated state with dataset_profile populated
        """
        from datetime import datetime
        from pathlib import Path

        # Get session directory from state
        session_dir = Path(state.get("session_dir", "./outputs"))
        report_path = session_dir / "analysis_report.md"

        # Save report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Spatial Omics Analysis Report\n\n")
            f.write(f"**Sample:** {state['sample_id']}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Iterations:** {iteration_count}\n\n")
            f.write("---\n\n")
            f.write(report)

        # Update state
        state["dataset_profile"] = {
            "analysis_report": report,
            "iteration_count": iteration_count,
            "validation_log": validation_log,
            "report_path": str(report_path)
        }

        display = CLIDisplay.get()
        display.agent_done("DataAnalyst", f"report saved -> {report_path.name}")
        self._log_execution(state, "dataset_analysis", f"completed after {iteration_count} iterations")

        return state

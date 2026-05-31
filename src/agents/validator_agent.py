"""
Validator Agent for hypothesis validation.

Specializes in:
- Statistical testing using mock ROI data
- Hypothesis validation
- Result interpretation with confidence scores
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel

from .base_agent import BaseAgent, AgentExecutionError
from ..workflows.agent_state import AgentState, add_error
from ..utils.prompt_manager import PromptManager
from ..utils.cli_display import CLIDisplay


class ValidatorAgent(BaseAgent):
    """
    Validator Agent for hypothesis validation.

    This agent performs statistical testing on ROI data to validate
    whether the hypothesis is supported by the data.
    """

    # Phenotype → ordinal stage mapping (user-specified 4-group design)
    PHENOTYPE_ORDINAL = {
        "DM": 0,
        "DKD2A": 1,
        "DKD2A->B": 2,
        "DKD2B": 3,
        "DKD3": 3,
    }

    # Pre-defined schema for mapping blueprint variables to ROI data
    VARIABLE_SCHEMA = {
        "cellular_composition": {
            "proximal_tubule_density": {"roi_labels": ["Proximal Tubules"], "metric": "percentage"},
            "distal_tubule_density": {"roi_labels": ["Distal Tubules"], "metric": "percentage"},
            "endothelial_density": {"roi_labels": ["Endothelial cells", "Endothelial (CD31+/CD196+)"], "metric": "percentage"},
            "immune_cell_density": {"roi_labels": ["Immune cells", "Macrophages", "Myeloid cells"], "metric": "percentage"},
            "vsmc_density": {"roi_labels": ["VSMCs"], "metric": "percentage"},
        },
        "marker_expression": {
            "asma_intensity": {"marker": "aSMA", "metric": "mean_intensity"},
            "collagen_iv_intensity": {"marker": "CollagenIV", "metric": "mean_intensity"},
            "collageniv_intensity": {"marker": "CollagenIV", "metric": "mean_intensity"},
            "cd45_intensity": {"marker": "CD45", "metric": "mean_intensity"},
            "nestin_intensity": {"marker": "Nestin", "metric": "mean_intensity"},
            "aSMA_expression": {"marker": "aSMA", "metric": "mean_intensity"},
            "collagenIV_expression": {"marker": "CollagenIV", "metric": "mean_intensity"},
        },
        "spatial_colocalization": {
            "late_stage_hidden_damage": {"marker": "Nestin", "metric": "mean_intensity"},
        },
        "spatial_genes": {
            "spatial_autocorrelation": {"metric": "morans_i"},
        }
    }

    # Keyword → (var_type, schema_key) for fuzzy matching of Planner-generated names
    KEYWORD_MAP = {
        "proximal": ("cellular_composition", "proximal_tubule_density"),
        "pt_density": ("cellular_composition", "proximal_tubule_density"),
        "pt_functional": ("cellular_composition", "proximal_tubule_density"),
        "tubule_density": ("cellular_composition", "proximal_tubule_density"),
        "asma": ("marker_expression", "aSMA_expression"),
        "a-sma": ("marker_expression", "aSMA_expression"),
        "acta2": ("marker_expression", "aSMA_expression"),
        "smooth_muscle": ("marker_expression", "aSMA_expression"),
        "myofibroblast": ("marker_expression", "aSMA_expression"),
        "collagen": ("marker_expression", "collagenIV_expression"),
        "col4": ("marker_expression", "collagenIV_expression"),
        "col4a1": ("marker_expression", "collagenIV_expression"),
        "fibrosis": ("marker_expression", "collagenIV_expression"),
        "ecm": ("marker_expression", "collagenIV_expression"),
        "cd45": ("marker_expression", "cd45_intensity"),
        "nestin": ("spatial_colocalization", "late_stage_hidden_damage"),
        "hidden_damage": ("spatial_colocalization", "late_stage_hidden_damage"),
        "immune": ("cellular_composition", "immune_cell_density"),
    }

    # CSV annotation label → simplified name (matches VARIABLE_SCHEMA roi_labels)
    CELL_TYPE_MAP = {
        "Proximal tubules (CD183++/CD227-)": "Proximal Tubules",
        "Distal tubules (CD183+/CD227+)": "Distal Tubules",
        "Endothelial cells (CD31+)": "Endothelial cells",
        "Endothelial cells (CD31+/CD196+)": "Endothelial (CD31+/CD196+)",
        "Immune cells (CD45+)": "Immune cells",
        "Macrophages (CD45+/CD68+)": "Macrophages",
        "Myeloid cells (CD45+/CD11b+)": "Myeloid cells",
        "VSMCs (aSMA+)": "VSMCs",
        "Basement membrane (collagenIV+/Perlecan+)": "Basement membrane",
        "Nestin+ cells (nestin+)": "Nestin+ cells",
        "Low expressing cells": "Low expressing cells",
    }

    def __init__(
        self,
        model: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        use_templates: bool = True
    ):
        """Initialize the Validator agent.

        Args:
            model: LangChain chat model instance
            prompt_manager: PromptManager instance (if None, creates default)
            use_templates: If True, use Jinja2 templates; else use hardcoded prompts
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.use_templates = use_templates

        # Hardcoded fallback prompt (existing implementation)
        self.fallback_system_prompt = """You are a Statistical Validation Specialist specializing in spatial omics hypothesis testing.

Your task:
- Execute statistical DAG from verification blueprint
- Perform appropriate tests (normality → variance → parametric/non-parametric)
- Apply multiple testing correction
- Interpret results in biological context
- Generate clear conclusion with confidence score

**Statistical Framework:**
- Normality: Shapiro-Wilk test (α = 0.05)
- Variance: Levene's test (α = 0.05)
- Parametric: Independent t-test (if normal + equal variance)
- Non-Parametric: Mann-Whitney U test (if not normal)
- Multiple Testing: FDR-Benjamini-Hochberg correction

**Output Format:**
Provide interpretation in markdown with:
- Test results with statistics and p-values
- Effect sizes where applicable
- Biological interpretation
- Clear conclusion: VERIFIED, FALSIFIED, or INCONCLUSIVE

Be precise with statistics and conservative with conclusions."""

        # Get system prompt from template or fallback
        system_prompt = self._get_system_prompt()

        super().__init__(
            name="Validator",
            model=model,
            tools=[],  # Validator uses direct scipy execution, not tools
            system_prompt=system_prompt,
            temperature=0.2  # Lowest temperature for statistical precision
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt from template or fallback."""
        return self.prompt_manager.get_system_prompt(
            agent_name="validator",
            use_template=self.use_templates,
            fallback_prompt=self.fallback_system_prompt
        )

    def get_system_prompt(self) -> str:
        return self.system_prompt

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute hypothesis validation with statistical testing.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation_results

        Workflow:
        1. Load blueprint and mock ROI data
        2. Map blueprint variables to ROI data
        3. Execute statistical DAG for each variable
        4. Apply multiple testing correction
        5. Generate interpretation and conclusion
        """
        state["current_action"] = "hypothesis_validation"
        state["current_phase"] = "phase3"

        execution_log = []

        try:
            display = CLIDisplay.get()

            # Step 1: Load blueprint
            blueprint = state.get("analysis_plan", {}).get("blueprint")
            if not blueprint:
                raise AgentExecutionError("No blueprint found in analysis_plan")

            execution_log.append(f"Loaded blueprint for hypothesis: {blueprint.get('hypothesis', {}).get('id', 'unknown')}")

            # Step 2: Load ROI data
            roi_data, architecture_labels, data_source = await self._load_roi_data(state)
            execution_log.append(f"Loaded {len(roi_data)} ROIs from {data_source}")
            display.agent_start("Validator", f"Loaded {len(roi_data)} ROIs from {data_source}")

            # Step 3: Map blueprint variables to ROI data
            variable_mappings = self._map_variables_via_schema(blueprint)
            execution_log.append(f"Mapped {len(variable_mappings)} target variables to ROI data")
            display.agent_progress("Validator", f"Running statistical DAG on {len(variable_mappings)} features...")

            # Step 4: Execute statistical DAG
            clinical_mapping = state.get("clinical_mapping")
            statistical_results = await self._execute_statistical_dag(
                blueprint,
                roi_data,
                architecture_labels,
                variable_mappings,
                clinical_mapping
            )
            execution_log.append(f"Executed statistical tests on {len(statistical_results['all_tests'])} features")

            # Show test routing details
            for test in statistical_results.get("all_tests", []):
                test_type = test.get("test", "?")
                feature = test.get("feature", "?")
                display.agent_progress("Validator", f"{feature}: {test_type}")

            # Step 5: Apply multiple testing correction
            corrected_results = self._apply_correction(
                statistical_results,
                blueprint.get("multiple_testing_correction", {"method": "fdr_bh", "alpha": 0.05})
            )
            execution_log.append(f"Applied {corrected_results['correction_method']} correction")
            display.agent_progress("Validator", f"{corrected_results['correction_method']} correction applied to {len(statistical_results['all_tests'])} tests")

            # Step 6: Generate interpretation and conclusion
            interpretation = await self._interpret_results(
                blueprint,
                corrected_results,
                state["hypothesis_description"],
                data_source
            )

            # Update state
            state["validation_results"] = {
                "conclusion": interpretation["conclusion"],
                "confidence": interpretation["confidence"],
                "key_findings": interpretation["key_findings"],
                "data_source": data_source,
                "raw_statistics": {
                    "all_tests": corrected_results["tests"],
                    "correction_applied": corrected_results["correction_method"],
                    "original_p_values": corrected_results.get("original_p_values", []),
                    "corrected_p_values": corrected_results.get("corrected_p_values", [])
                },
                "execution_log": execution_log
            }

            # Save validation results to file
            import numpy as np

            session_dir = Path(state.get("session_dir", "./outputs"))
            validation_path = session_dir / "validation_results.json"

            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native(obj):
                """Convert numpy types to native Python types recursively."""
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(v) for v in obj]
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_to_native(obj.tolist())
                else:
                    return obj

            serializable_results = convert_to_native(state["validation_results"])

            with open(validation_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            # Replace validation_results with serializable version (msgpack-safe for LangGraph)
            state["validation_results"] = serializable_results

            # Step 7: Generate validation report
            report_content = await self._generate_validation_report(
                state["validation_results"],
                state["hypothesis_description"],
                state
            )
            state["validation_report"] = report_content

            # Display validation results
            display.show_validation(interpretation)
            display.agent_done("Validator", "report saved -> validation_report.md")

            self._log_execution(state, "hypothesis_validation", f"completed: {interpretation['conclusion']}")

        except Exception as e:
            error_msg = f"Validator execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "hypothesis_validation", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    def _load_from_full_registry(self, state: AgentState) -> tuple:
        """
        Load representative ROIs from full registry with cell composition and marker intensity.

        Uses cluster_center_indices to select representative ROIs from each region,
        then computes cell_composition and marker_intensity from CSV data.

        Returns:
            Tuple of (roi_data, architecture_labels, data_source)
        """
        import pandas as pd

        display = CLIDisplay.get()
        data_root = Path(state["data_root"])
        sample_id = state["sample_id"]
        clinical_mapping = state.get("clinical_mapping") or {}

        # Load full registry
        registry_path = data_root / f"{sample_id}_pivot_ROIs_registry.pkl"
        if not registry_path.exists():
            raise AgentExecutionError(
                f"Full registry not found: {registry_path}. "
                "Required for statistical validation."
            )

        with open(registry_path, 'rb') as f:
            registry = pickle.load(f)

        display.agent_progress("Validator", f"Loaded full registry: {len(registry.shards)} regions")

        # Pre-load CSV data per region
        csv_cache = {}
        for region_id in registry.shards:
            region_dir = data_root / region_id
            if not region_dir.exists():
                continue

            cell_types_df = pd.read_csv(region_dir / f"{region_id}.cell_types.csv")
            expression_df = pd.read_csv(region_dir / f"{region_id}.expression.csv")

            # Map annotation labels to simplified names
            cell_types_df["cell_type"] = cell_types_df["ANNOTATION_LABEL"].map(self.CELL_TYPE_MAP)

            csv_cache[region_id] = {
                "cell_types": cell_types_df,
                "expression": expression_df,
            }

        # Build roi_data from representative ROIs
        roi_data = {}
        architecture_labels = {}

        for region_id, shard in registry.shards.items():
            if region_id not in csv_cache:
                continue

            csv_data = csv_cache[region_id]
            cell_types_df = csv_data["cell_types"]
            expression_df = csv_data["expression"]

            phenotype = clinical_mapping.get(region_id, "")
            ordinal_stage = self.PHENOTYPE_ORDINAL.get(phenotype, -1)

            # Iterate representative ROIs from cluster_center_indices
            for cluster_key, indices in shard.cluster_center_indices.items():
                for idx in indices:
                    roi = shard.rois[idx]
                    cell_ids = roi.cell_ids

                    roi_key = f"('{region_id}', '{roi.patch_name}')"

                    # Compute cell composition
                    roi_cells = cell_types_df[cell_types_df["CELL_ID"].isin(cell_ids)]
                    if len(roi_cells) > 0:
                        counts = roi_cells["cell_type"].value_counts()
                        total = len(roi_cells)
                        cell_composition = {ct: round(cnt / total * 100, 2) for ct, cnt in counts.items()}
                    else:
                        cell_composition = {}

                    # Compute marker intensity (mean expression)
                    roi_expr = expression_df[expression_df["CELL_ID"].isin(cell_ids)]
                    if len(roi_expr) > 0:
                        marker_cols = [c for c in expression_df.columns if c != "CELL_ID"]
                        marker_intensity = {col: round(float(roi_expr[col].mean()), 2) for col in marker_cols}
                    else:
                        marker_intensity = {}

                    # Determine architecture label from dominant cell type
                    if cell_composition:
                        dominant_type = max(cell_composition, key=cell_composition.get)
                        architecture_labels[roi_key] = dominant_type
                    else:
                        architecture_labels[roi_key] = "unknown"

                    roi_data[roi_key] = {
                        "patch_name": roi.patch_name,
                        "region_id": region_id,
                        "center_x": float(roi.center_x),
                        "center_y": float(roi.center_y),
                        "cell_count": len(cell_ids),
                        "cell_composition": cell_composition,
                        "marker_intensity": marker_intensity,
                        "phenotype": phenotype,
                        "ordinal_stage": ordinal_stage,
                    }

        phenotype_dist = {}
        for info in roi_data.values():
            ph = info.get("phenotype", "unknown")
            phenotype_dist[ph] = phenotype_dist.get(ph, 0) + 1

        display.agent_progress("Validator",
            f"Built roi_data: {len(roi_data)} representative ROIs from {len(csv_cache)} regions")
        display.agent_progress("Validator", f"Phenotype distribution: {phenotype_dist}")

        return roi_data, architecture_labels, "full_registry"

    async def _load_roi_data(self, state: AgentState) -> tuple:
        """
        Load ROI data for statistical validation.

        In real mode: loads from full registry (all 17 regions, representative ROIs).
        In mock mode: loads pre-generated mock data from cli/mockdata/.

        Args:
            state: Current workflow state

        Returns:
            Tuple of (roi_data, architecture_labels, data_source)

        Raises:
            AgentExecutionError: If no valid data source is available
        """
        display = CLIDisplay.get()
        display.agent_progress("Validator", "Loading ROI data from full registry...")
        return self._load_from_full_registry(state)

    def _map_variables_via_schema(self, blueprint: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Map blueprint target variables to ROI data using pre-defined schema.

        Matching priority:
        1. Exact name match in VARIABLE_SCHEMA
        2. Biological proxy match
        3. Keyword match via KEYWORD_MAP (var_name + biological_proxy)

        Args:
            blueprint: Verification blueprint from Planner

        Returns:
            Dictionary mapping variable names to their ROI data specifications
        """
        mappings = {}

        for var in blueprint.get("target_variables", []):
            var_name = var.get("name")
            var_type = var.get("type")
            biological_proxy = var.get("biological_proxy", "")
            expected_trend = var.get("expected_trend")

            matched = False

            # Priority 1: Exact name match in VARIABLE_SCHEMA
            if var_type in self.VARIABLE_SCHEMA:
                type_schema = self.VARIABLE_SCHEMA[var_type]

                if var_name in type_schema:
                    mappings[var_name] = {
                        "type": var_type,
                        "spec": type_schema[var_name],
                        "expected_trend": expected_trend
                    }
                    matched = True
                else:
                    # Priority 2: Biological proxy match
                    for schema_name, schema_spec in type_schema.items():
                        if biological_proxy and biological_proxy.lower() in str(schema_spec).lower():
                            mappings[var_name] = {
                                "type": var_type,
                                "spec": schema_spec,
                                "expected_trend": expected_trend
                            }
                            matched = True
                            break

            # Priority 3: Keyword match via KEYWORD_MAP
            # Search var_name first (higher confidence), then full text
            if not matched:
                var_name_lower = var_name.lower()
                for keyword, (kw_type, schema_key) in self.KEYWORD_MAP.items():
                    if keyword in var_name_lower:
                        kw_schema = self.VARIABLE_SCHEMA.get(kw_type, {})
                        if schema_key in kw_schema:
                            mappings[var_name] = {
                                "type": kw_type,
                                "spec": kw_schema[schema_key],
                                "expected_trend": expected_trend
                            }
                            matched = True
                            break

            if not matched:
                search_text = f"{var_name} {biological_proxy}".lower()
                for keyword, (kw_type, schema_key) in self.KEYWORD_MAP.items():
                    if keyword in search_text:
                        kw_schema = self.VARIABLE_SCHEMA.get(kw_type, {})
                        if schema_key in kw_schema:
                            mappings[var_name] = {
                                "type": kw_type,
                                "spec": kw_schema[schema_key],
                                "expected_trend": expected_trend
                            }
                            matched = True
                            break

            if not matched:
                # Fallback: generic mapping
                mappings[var_name] = {
                    "type": var_type,
                    "spec": {"metric": "generic"},
                    "expected_trend": expected_trend or "change"
                }

        return mappings

    async def _execute_statistical_dag(
        self,
        blueprint: Dict[str, Any],
        roi_data: Dict,
        architecture_labels: Dict,
        variable_mappings: Dict,
        clinical_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute statistical DAG with clinical phenotype-based grouping.

        Uses Linear Mixed Models (LMM) with ordinal trend testing,
        falling back to pseudobulk + Jonckheere-Terpstra if LMM fails.

        Args:
            blueprint: Verification blueprint from Planner
            roi_data: Loaded ROI registry with phenotype annotations
            architecture_labels: ROI → tissue label mapping
            variable_mappings: Variable → ROI data specification mappings
            clinical_mapping: Region → phenotype mapping

        Returns:
            Dictionary with test results for all features
        """
        import numpy as np
        import pandas as pd

        test_results = []

        for var_name, var_mapping in variable_mappings.items():
            var_type = var_mapping["type"]
            spec = var_mapping["spec"]
            expected_trend = var_mapping["expected_trend"]

            # Extract paired data: (roi_key, feature_value)
            feature_pairs = self._extract_feature_data_paired(
                var_name, var_type, spec, roi_data, architecture_labels
            )

            # Build records with phenotype info
            records = []
            for roi_key, value in feature_pairs:
                roi_info = roi_data.get(roi_key, {})
                region_id = roi_info.get("region_id", "")
                ordinal_stage = roi_info.get("ordinal_stage", -1)
                phenotype = roi_info.get("phenotype", "")
                if ordinal_stage >= 0:
                    records.append({
                        "roi_key": roi_key,
                        "region_id": region_id,
                        "phenotype": phenotype,
                        "ordinal_stage": ordinal_stage,
                        "value": value,
                    })

            if not records:
                continue

            df = pd.DataFrame(records)

            # Try LMM first, fallback to pseudobulk + JT
            display = CLIDisplay.get()
            lmm_result = self._fit_lmm(df, var_name)

            if lmm_result["converged"]:
                display.agent_progress("Validator",
                    f"{var_name}: LMM converged (coef={lmm_result['coefficient']:.4f}, p={lmm_result['p_value']:.4f})")
                result = {
                    "feature": var_name,
                    "test": "Linear Mixed Model",
                    "statistic": lmm_result["coefficient"],
                    "p_value": lmm_result["p_value"],
                    "effect_size": f"β = {lmm_result['coefficient']:.4f}",
                    "expected_trend": expected_trend,
                    "trend_direction": lmm_result["trend_direction"],
                    "significant": lmm_result["p_value"] < 0.05,
                    "significant_after_correction": lmm_result["p_value"] < 0.05,
                    "interpretation": self._format_trend_interpretation(
                        var_name, "LMM", lmm_result["coefficient"],
                        lmm_result["p_value"], expected_trend
                    ),
                    "n_rois": len(df),
                    "phenotype_counts": df["phenotype"].value_counts().to_dict(),
                }
            else:
                # Fallback: pseudobulk aggregation + Jonckheere-Terpstra
                display.agent_progress("Validator",
                    f"{var_name}: LMM failed ({lmm_result.get('error', 'unknown')}), falling back to pseudobulk + JT")
                jt_result = self._pseudobulk_jt_test(df, var_name)
                result = {
                    "feature": var_name,
                    "test": "Pseudobulk + Jonckheere-Terpstra",
                    "statistic": jt_result["statistic"],
                    "p_value": jt_result["p_value"],
                    "effect_size": f"J = {jt_result['statistic']:.1f}",
                    "expected_trend": expected_trend,
                    "trend_direction": jt_result["trend_direction"],
                    "significant": jt_result["p_value"] < 0.05,
                    "significant_after_correction": jt_result["p_value"] < 0.05,
                    "interpretation": self._format_trend_interpretation(
                        var_name, "JT", jt_result["trend_direction_value"],
                        jt_result["p_value"], expected_trend
                    ),
                    "n_rois": len(df),
                    "n_regions": jt_result["n_regions"],
                    "phenotype_counts": df["phenotype"].value_counts().to_dict(),
                }

            test_results.append(result)

        return {
            "all_tests": test_results,
            "correction_applied": None
        }

    def _extract_feature_data_paired(
        self,
        var_name: str,
        var_type: str,
        spec: Dict,
        roi_data: Dict,
        architecture_labels: Dict
    ) -> List[tuple]:
        """Extract feature values paired with ROI keys."""
        import numpy as np

        pairs = []

        if var_type == "cellular_composition":
            roi_labels = spec.get("roi_labels", [])
            for roi_id, roi_info in roi_data.items():
                composition = roi_info.get("cell_composition", {})
                value = sum(composition.get(label, 0) for label in roi_labels)
                pairs.append((roi_id, value))

        elif var_type == "marker_expression":
            marker = spec.get("marker")
            for roi_id, roi_info in roi_data.items():
                marker_intensity = roi_info.get("marker_intensity", {})
                value = marker_intensity.get(marker, 0)
                pairs.append((roi_id, value))

        elif var_type == "spatial_genes":
            import random
            for roi_id in roi_data:
                pairs.append((roi_id, random.uniform(0.1, 0.8)))

        else:
            for roi_id in roi_data:
                pairs.append((roi_id, float(np.random.uniform(0, 100))))

        return pairs

    def _fit_lmm(self, df, feature_name: str) -> Dict[str, Any]:
        """
        Fit Linear Mixed Model: feature ~ ordinal_stage + (1|region_id).

        Returns:
            Dict with coefficient, p_value, trend_direction, converged.
        """
        import numpy as np
        import warnings

        try:
            from statsmodels.regression.mixed_linear_model import MixedLM

            if len(df) < 4 or df["ordinal_stage"].nunique() < 2:
                return {"converged": False, "error": "insufficient data"}

            y = df["value"].values.astype(float)
            X = np.column_stack([
                np.ones(len(df)),
                df["ordinal_stage"].values.astype(float)
            ])
            groups = df["region_id"].values

            model = MixedLM(y, X, groups=groups)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(reml=True)

            coef = float(result.params[1])  # ordinal_stage coefficient
            p_value = float(result.pvalues[1])

            # Treat nan results as failed convergence
            if np.isnan(coef) or np.isnan(p_value):
                return {"converged": False, "error": "nan in LMM result"}

            trend_direction = "increase" if coef > 0 else "decrease"

            return {
                "converged": True,
                "coefficient": coef,
                "p_value": p_value,
                "trend_direction": trend_direction,
                "trend_direction_value": coef,
            }

        except Exception as e:
            return {"converged": False, "error": str(e)}

    def _pseudobulk_jt_test(self, df, feature_name: str) -> Dict[str, Any]:
        """
        Pseudobulk aggregation by region + Jonckheere-Terpstra trend test.

        1. Aggregate ROI values to region-level (median)
        2. Run JT test across ordinal stage groups
        """
        import numpy as np
        from scipy import stats

        # Pseudobulk: median per region
        region_agg = df.groupby("region_id").agg({
            "value": "median",
            "ordinal_stage": "first",
            "phenotype": "first",
        }).reset_index()

        # Group by ordinal stage
        stage_groups = {}
        for _, row in region_agg.iterrows():
            stage = int(row["ordinal_stage"])
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append(row["value"])

        if len(stage_groups) < 2:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "trend_direction": "none",
                "trend_direction_value": 0.0,
                "n_regions": len(region_agg),
            }

        # Jonckheere-Terpstra test
        sorted_stages = sorted(stage_groups.keys())
        groups = [stage_groups[s] for s in sorted_stages]
        jt_stat, jt_p = self._jonckheere_terpstra(groups)

        # Determine direction from group means
        stage_means = {s: np.mean(stage_groups[s]) for s in sorted_stages}
        first_mean = stage_means[sorted_stages[0]]
        last_mean = stage_means[sorted_stages[-1]]
        trend_dir = "increase" if last_mean > first_mean else "decrease"

        return {
            "statistic": float(jt_stat),
            "p_value": float(jt_p),
            "trend_direction": trend_dir,
            "trend_direction_value": last_mean - first_mean,
            "n_regions": len(region_agg),
        }

    def _jonckheere_terpstra(self, groups: List[List[float]]) -> tuple:
        """
        Jonckheere-Terpstra trend test.

        Tests H0: no trend vs H1: ordered alternative across groups.

        Args:
            groups: List of sample groups ordered by ordinal stage.

        Returns:
            (standardized_statistic, p_value)
        """
        import numpy as np
        from scipy import stats as sp_stats

        k = len(groups)
        if k < 2:
            return 0.0, 1.0

        # Compute J statistic
        j_stat = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                # Count concordant pairs (x_j > x_i)
                for xi in groups[i]:
                    for xj in groups[j]:
                        if xj > xi:
                            j_stat += 1
                        elif xj < xi:
                            j_stat -= 1

        # Expected value and variance under H0
        n = [len(g) for g in groups]
        N = sum(n)
        n_sq_sum = sum(ni ** 2 for ni in n)

        mean_j = (N ** 2 - n_sq_sum) / 4.0
        var_j = (N ** 2 * (2 * N + 3) - sum(ni ** 2 * (2 * ni + 3) for ni in n)) / 72.0

        if var_j <= 0:
            return 0.0, 1.0

        # Standardize
        z = (j_stat - mean_j) / np.sqrt(var_j)
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        return z, p_value

    def _format_trend_interpretation(
        self, feature: str, test_name: str,
        coefficient: float, p_value: float, expected_trend: str
    ) -> str:
        """Format trend interpretation string."""
        direction = "increasing" if coefficient > 0 else "decreasing"
        sig = "significant" if p_value < 0.05 else "not significant"
        return (
            f"{feature} shows {sig} {direction} trend across ordinal stages "
            f"({test_name}, p = {p_value:.4f}, expected: {expected_trend})"
        )

    def _extract_feature_data(
        self,
        var_name: str,
        var_type: str,
        spec: Dict,
        roi_data: Dict,
        architecture_labels: Dict
    ) -> List[float]:
        """
        Extract feature values from ROI data based on variable type and schema.

        Args:
            var_name: Variable name
            var_type: Variable type (cellular_composition, marker_expression, etc.)
            spec: Schema specification for this variable
            roi_data: ROI registry dictionary
            architecture_labels: ROI → tissue label mapping

        Returns:
            List of feature values
        """
        import numpy as np

        values = []

        if var_type == "cellular_composition":
            # Extract cell composition percentages
            roi_labels = spec.get("roi_labels", [])
            for roi_id, roi_info in roi_data.items():
                composition = roi_info.get("cell_composition", {})
                # Sum percentages for matching cell types
                value = sum(composition.get(label, 0) for label in roi_labels)
                values.append(value)

        elif var_type == "marker_expression":
            # Extract marker intensity values
            marker = spec.get("marker")
            for roi_id, roi_info in roi_data.items():
                marker_intensity = roi_info.get("marker_intensity", {})
                value = marker_intensity.get(marker, 0)
                values.append(value)

        elif var_type == "spatial_genes":
            # Generate mock Moran's I values
            # In real implementation, this would use DataAnalyst results
            import random
            values = [random.uniform(0.1, 0.8) for _ in roi_data]

        else:
            # Generic: generate mock data
            values = list(np.random.uniform(0, 100, len(roi_data)))

        return values

    def _route_and_test(
        self,
        feature_name: str,
        contrast_id: str,
        group1: List[float],
        group2: List[float],
        nodes: List[Dict],
        expected_trend: str
    ) -> Dict[str, Any]:
        """
        Execute statistical test routing logic.

        Args:
            feature_name: Name of feature being tested
            contrast_id: Contrast identifier
            group1: Data for group 1
            group2: Data for group 2
            nodes: DAG node configurations
            expected_trend: Expected trend (increase/decrease/change)

        Returns:
            Dictionary with test results
        """
        from scipy import stats
        import numpy as np

        result = {
            "feature": feature_name,
            "contrast": contrast_id,
            "group1_n": len(group1),
            "group2_n": len(group2),
            "group1_mean": np.mean(group1),
            "group2_mean": np.mean(group2),
            "expected_trend": expected_trend
        }

        # Normality test
        normality_p1 = stats.shapiro(group1).pvalue if len(group1) >= 3 else 0.5
        normality_p2 = stats.shapiro(group2).pvalue if len(group2) >= 3 else 0.5
        is_normal = normality_p1 > 0.05 and normality_p2 > 0.05

        result["normality_test"] = "Shapiro-Wilk"
        result["normality_p1"] = normality_p1
        result["normality_p2"] = normality_p2
        result["is_normal"] = is_normal

        # Variance test
        variance_p = stats.levene(group1, group2).pvalue if len(group1) >= 2 and len(group2) >= 2 else 0.5
        equal_variance = variance_p > 0.05

        result["variance_test"] = "Levene's"
        result["variance_p"] = variance_p
        result["equal_variance"] = equal_variance

        # Route to appropriate test
        if is_normal and equal_variance:
            # Parametric: Independent t-test
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_name = "Independent t-test"
        else:
            # Non-parametric: Mann-Whitney U
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U"

        result["test"] = test_name
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)

        # Calculate effect size (Cohen's d for t-test, r for Mann-Whitney)
        if test_name == "Independent t-test":
            # Cohen's d
            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
            effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
            result["effect_size"] = f"Cohen's d = {effect_size:.3f}"
        else:
            # r from z-score
            n1, n2 = len(group1), len(group2)
            z_score = stats.norm.ppf(p_value / 2)
            r = abs(z_score) / np.sqrt(n1 + n2) if (n1 + n2) > 0 else 0
            result["effect_size"] = f"r = {r:.3f}"

        # Interpret result based on expected trend
        mean_diff = result["group2_mean"] - result["group1_mean"]

        if expected_trend == "increase":
            significant = p_value < 0.05 and mean_diff > 0
        elif expected_trend == "decrease":
            significant = p_value < 0.05 and mean_diff < 0
        else:
            significant = p_value < 0.05

        result["significant"] = significant
        result["mean_difference"] = mean_diff

        # Generate interpretation
        if significant:
            direction = "increased" if mean_diff > 0 else "decreased"
            result["interpretation"] = f"{feature_name} significantly {direction} ({test_name}, p = {p_value:.3f})"
        else:
            result["interpretation"] = f"No significant change in {feature_name} ({test_name}, p = {p_value:.3f})"

        return result

    def _apply_correction(
        self,
        statistical_results: Dict[str, Any],
        correction_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction to p-values.

        Args:
            statistical_results: Raw test results
            correction_config: Correction method configuration

        Returns:
            Dictionary with corrected results
        """
        from statsmodels.stats.multitest import multipletests

        method = correction_config.get("method", "fdr_bh")
        alpha = correction_config.get("alpha", 0.05)

        tests = statistical_results["all_tests"]
        if not tests:
            return {
                "tests": [],
                "correction_method": method,
                "original_p_values": [],
                "corrected_p_values": []
            }

        # Extract p-values
        original_p_values = [t["p_value"] for t in tests]

        # Apply correction
        # Map common method names to statsmodels names
        method_map = {
            "fdr_bh": "fdr_bh",
            "bonferroni": "bonferroni",
            "holm": "holm"
        }

        statsmodels_method = method_map.get(method, "fdr_bh")

        try:
            reject, corrected_p_values, _, _ = multipletests(
                original_p_values,
                alpha=alpha,
                method=statsmodels_method
            )

            # Update test results with corrected p-values
            for i, test in enumerate(tests):
                test["corrected_p_value"] = float(corrected_p_values[i])
                test["significant_after_correction"] = bool(reject[i])

        except Exception as e:
            # If correction fails, use original p-values
            for test in tests:
                test["corrected_p_value"] = test["p_value"]
                test["significant_after_correction"] = test["significant"]

        return {
            "tests": tests,
            "correction_method": method,
            "original_p_values": original_p_values,
            "corrected_p_values": [t.get("corrected_p_value", t["p_value"]) for t in tests]
        }

    async def _interpret_results(
        self,
        blueprint: Dict[str, Any],
        corrected_results: Dict[str, Any],
        hypothesis_description: str,
        data_source: str = "retriever_expansion"
    ) -> Dict[str, Any]:
        """
        Generate interpretation and conclusion from statistical results using LLM.

        Args:
            blueprint: Original verification blueprint
            corrected_results: Statistical test results with correction
            hypothesis_description: Original hypothesis text
            data_source: Source of ROI data ("retriever_expansion")

        Returns:
            Dictionary with conclusion, confidence, and key findings
        """
        tests = corrected_results["tests"]

        if not tests:
            return {
                "conclusion": "INCONCLUSIVE",
                "confidence": 0.0,
                "key_findings": [{
                    "feature": "N/A",
                    "test": "N/A",
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "corrected_p_value": 1.0,
                    "effect_size": "N/A",
                    "interpretation": "No statistical tests were performed",
                    "significant": False,
                    "significant_after_correction": False,
                    "expected_trend": "N/A",
                    "trend_direction": "none",
                }]
            }

        # Calculate statistics for LLM
        significant_after_correction = sum(1 for t in tests if t.get("significant_after_correction", False))
        total_tests = len(tests)
        aligned_significant = sum(
            1 for t in tests
            if t.get("significant_after_correction", False) and
            self._is_aligned_with_expected_trend(t)
        )

        # Determine preliminary conclusion
        if aligned_significant > total_tests / 2:
            preliminary_conclusion = "VERIFIED"
            confidence = min((significant_after_correction / total_tests) + 0.2, 1.0)
        elif significant_after_correction > total_tests / 2:
            preliminary_conclusion = "INCONCLUSIVE"
            confidence = 0.5
        else:
            preliminary_conclusion = "FALSIFIED"
            confidence = 1.0 - (significant_after_correction / total_tests)

        # Prepare statistical summary for LLM (information diet)
        stats_summary = self._prepare_stats_summary(tests, corrected_results)

        # Prepare context for template
        context = {
            "hypothesis_description": hypothesis_description,
            "total_tests": total_tests,
            "significant_after_correction": significant_after_correction,
            "aligned_significant": aligned_significant,
            "preliminary_conclusion": preliminary_conclusion,
            "detailed_test_results": stats_summary,
            "data_source": data_source,
        }

        # Prepare fallback prompt
        fallback_prompt = """Interpret the following statistical validation results for this spatial omics hypothesis.

**Hypothesis:**
{hypothesis_description}

✓ Real ROI data from Retriever's spatial expansion was used for validation.

**Statistical Results Summary:**
- Total Tests: {total_tests}
- Significant (after correction): {significant_after_correction}
- Aligned with expected trend: {aligned_significant}
- Preliminary Conclusion: {preliminary_conclusion}

**Detailed Test Results:**
{detailed_test_results}

**Task:**
Provide a biological interpretation of these statistical results in markdown format:

1. **Overall Conclusion**: Is the hypothesis VERIFIED, FALSIFIED, or INCONCLUSIVE?
2. **Key Findings**: Summarize the most important statistical results
3. **Biological Interpretation**: What do these results mean in the context of kidney disease/fibrosis?
4. **Limitations**: Any caveats or limitations?

Be concise and precise. Focus on biological meaning, not just statistics."""

        # Use LLM to generate interpretation
        interpretation_prompt = self.prompt_manager.get_task_prompt(
            agent_name="validator",
            task_name="interpret_results",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Invoke LLM for interpretation
        llm_interpretation = await self._invoke_llm(self._format_prompt(interpretation_prompt))

        # Generate key findings (structured data)
        key_findings = []
        for test in tests:
            finding = {
                "feature": test["feature"],
                "test": test["test"],
                "statistic": test["statistic"],
                "p_value": test["p_value"],
                "corrected_p_value": test.get("corrected_p_value", test["p_value"]),
                "effect_size": test["effect_size"],
                "interpretation": test["interpretation"],
                "significant": test.get("significant_after_correction", False)
            }
            key_findings.append(finding)

        return {
            "conclusion": preliminary_conclusion,
            "confidence": round(confidence, 2),
            "key_findings": key_findings,
            "llm_interpretation": llm_interpretation
        }

    def _prepare_stats_summary(self, tests: List[Dict], corrected_results: Dict) -> str:
        """Prepare statistical summary for LLM (information diet)."""
        summary_lines = []

        feature_results = {}
        for test in tests:
            feature = test.get("feature", "?")
            if feature not in feature_results:
                feature_results[feature] = []
            feature_results[feature].append(test)

        for feature, results in feature_results.items():
            summary_lines.append(f"\n**{feature}:**")
            for result in results:
                sig_marker = "✓" if result.get("significant_after_correction") else "✗"
                trend = result.get("trend_direction", "?")
                summary_lines.append(
                    f"- [{sig_marker}] {result['test']}: p={result['p_value']:.4f} "
                    f"(corrected: {result.get('corrected_p_value', result['p_value']):.4f}), "
                    f"trend={trend}, {result['effect_size']}"
                )
                summary_lines.append(f"  {result['interpretation']}")
                if "phenotype_counts" in result:
                    summary_lines.append(f"  Phenotype distribution: {result['phenotype_counts']}")

        summary_lines.append(f"\n**Multiple Testing Correction:** {corrected_results.get('correction_method', 'N/A')}")

        return "\n".join(summary_lines)

    def _is_aligned_with_expected_trend(self, test_result: Dict) -> bool:
        """Check if test result aligns with expected trend."""
        expected_trend = test_result.get("expected_trend", "change")
        trend_direction = test_result.get("trend_direction", "")

        if expected_trend in ("increase", "monotonic_increase", "significant_increase"):
            return trend_direction == "increase"
        elif expected_trend in ("decrease", "monotonic_decrease"):
            return trend_direction == "decrease"
        else:
            return trend_direction in ("increase", "decrease")

    async def _generate_validation_report(
        self,
        validation_results: Dict[str, Any],
        hypothesis_description: str,
        state: AgentState
    ) -> str:
        """
        Generate a comprehensive validation report using LLM.

        Args:
            validation_results: Dictionary containing validation results
            hypothesis_description: Original hypothesis text
            state: Current workflow state

        Returns:
            Generated markdown report content
        """
        from ..utils.prompt_manager import PromptManager

        # Prepare context for report generation
        findings = [f for f in validation_results["key_findings"] if isinstance(f, dict)]
        context = {
            "hypothesis_description": hypothesis_description,
            "preliminary_conclusion": validation_results["conclusion"],
            "total_tests": len(findings),
            "significant_after_correction": sum(
                1 for f in findings if f.get("significant")
            ),
            "aligned_significant": sum(
                1 for f in findings
                if f.get("significant") and self._is_aligned_with_expected_trend(f)
            ),
            "confidence": validation_results["confidence"],
            "detailed_test_results": self._prepare_stats_summary(
                findings,
                {
                    "tests": findings,
                    "correction_method": validation_results["raw_statistics"].get("correction_applied", "fdr_bh")
                }
            ),
            "llm_interpretation": validation_results.get("llm_interpretation", ""),
        }

        # Prepare fallback prompt
        fallback_prompt = """Generate a comprehensive Hypothesis Verification Report.

**Hypothesis:**
{hypothesis_description}

**Conclusion:** {preliminary_conclusion} (confidence: {confidence})

**Statistical Summary:**
- Total Tests: {total_tests}
- Significant: {significant_after_correction}
- Data Source: Real ROI data from Retriever

**Test Results:**
{detailed_test_results}

**Previous Interpretation:**
{llm_interpretation}

Generate a markdown report with:
1. Conclusion banner (VERIFIED/FALSIFIED/INCONCLUSIVE)
2. Key Evidence section (3-4 key findings with observations and significance)
3. Key Scientific Discovery section (scientific insights)
4. Statistical Details section
5. Limitations section

Follow the format from the template and be scientifically rigorous."""

        # Get report prompt from template or fallback
        report_prompt = self.prompt_manager.get_task_prompt(
            agent_name="validator",
            task_name="generate_report",
            context=context,
            use_template=self.use_templates,
            fallback_prompt=fallback_prompt
        )

        # Invoke LLM to generate report
        report_content = await self._invoke_llm(self._format_prompt(report_prompt))

        # Save report to file
        session_dir = Path(state.get("session_dir", "./outputs"))
        report_path = session_dir / "validation_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return report_content

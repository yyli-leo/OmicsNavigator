"""
Retriever Agent for semantic search via FAISS.

Specializes in:
- FAISS vector database search
- LLM-driven query generation from hypothesis
- Semantic retrieval of ROI interpretations
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
from scipy.spatial.distance import cdist

from .base_agent import BaseAgent, AgentExecutionError
from ..workflows.agent_state import AgentState, add_error
from ..tools.interpretation_vectorstore import InterpretationVectorStore
from ..utils.cli_display import CLIDisplay


# Prompt for generating search queries from hypothesis
QUERY_GENERATION_SYSTEM_PROMPT = """You are a semantic search assistant for a spatial omics ROI database.
Your task is to generate 2-3 focused search queries to retrieve relevant ROI interpretations from a vector database.

The database contains final interpretations of tissue Regions of Interest (ROIs), each classified into types like:
Proximal tubules, Distal tubules, Glomeruli, Blood vessel, Interstitium

Each interpretation includes cell type composition, biomarker expression, and multi-modal reasoning.

Generate queries that target specific tissue structures, pathological conditions, or biomarker patterns mentioned in the hypothesis."""

QUERY_GENERATION_USER_PROMPT = """Generate 2-3 focused search queries for the following hypothesis:

**Hypothesis:** {hypothesis}

**Response Format (JSON only):**
```json
{{
  "queries": [
    "search query 1",
    "search query 2",
    "search query 3"
  ]
}}
```

Return ONLY valid JSON, no additional text."""


class RetrieverAgent(BaseAgent):
    """
    Retriever Agent for semantic search.

    Uses LLM to analyze the hypothesis and generate search queries,
    then searches a FAISS vector database of ROI final_interpretations
    to find semantically relevant ROIs.
    """

    def __init__(
        self,
        model: BaseChatModel,
        index_dir: Optional[str] = None,
        top_k: int = 5,
        enable_spatial_expansion: bool = False,
        expansion_threshold: float = 0.333
    ):
        """Initialize the Retriever agent."""
        self.index_dir = Path(index_dir) if index_dir else Path("./outputs/faiss_index")
        self.top_k = top_k
        self.enable_spatial_expansion = enable_spatial_expansion
        self.expansion_threshold = expansion_threshold

        system_prompt = """You are a Semantic Retrieval Specialist.

Your task:
- Analyze hypotheses to determine search strategies
- Search FAISS vector database with generated keywords
- Retrieve matching ROIs based on semantic similarity
- Rank results by relevance score

The search uses gemini-embedding-001 embeddings for semantic matching."""

        super().__init__(
            name="Retriever",
            model=model,
            tools=[],
            system_prompt=system_prompt,
            temperature=0.3
        )

    def get_system_prompt(self) -> str:
        return self.system_prompt

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute semantic search against the FAISS index.

        Reads:
            state["hypothesis_description"] - used to generate search queries
            state["session_dir"] - used to locate the index

        Writes:
            state["semantic_search_results"] - search results dict
        """
        state["current_action"] = "semantic_search"
        state["current_phase"] = "phase3"

        try:
            display = CLIDisplay.get()

            # Step 1: Locate index
            index_dir = self._resolve_index_dir(state)
            vs = InterpretationVectorStore(index_dir=index_dir)

            # Build from state if index doesn't exist
            if not vs.index_exists:
                final_interpretations = state.get("final_interpretations", {})
                if final_interpretations:
                    display.agent_progress("Retriever", "No cached index found. Building from state...")
                    vs.build_index(final_interpretations)
                else:
                    raise FileNotFoundError(
                        f"No FAISS index found at {index_dir} and no "
                        "final_interpretations in state to build one."
                    )

            # Step 2: Generate search queries via LLM
            hypothesis = state.get("hypothesis_description", "")
            display.agent_start("Retriever", "Generating search queries from hypothesis...")
            queries = await self._generate_queries(hypothesis)
            display.agent_progress("Retriever", f"Generated {len(queries)} queries")

            # Step 3: Search FAISS for each query
            all_results = []
            seen_keys = set()

            for query in queries:
                results = vs.search(query, top_k=self.top_k)
                for r in results:
                    if r["roi_key"] not in seen_keys:
                        seen_keys.add(r["roi_key"])
                        r["query"] = query
                        all_results.append(r)

            # Sort by score (lower = more similar for L2 distance)
            all_results.sort(key=lambda x: x["score"])

            # Limit to top_k unique results
            top_results = all_results[:self.top_k * 2]

            display.agent_progress("Retriever", f"FAISS search: found {len(top_results)} unique ROIs")

            # Step 3: Optional filtering
            if state.get("spatial_expansion_filter"):
                filter_keywords = state["spatial_expansion_filter"].split(",")
                top_results = self._filter_results(top_results, filter_keywords)
                display.agent_progress("Retriever", f"After filtering: {len(top_results)} ROIs")

            # Step 4: Spatial expansion (if enabled)
            expansion_enabled = self.enable_spatial_expansion or state.get("enable_spatial_expansion")

            if expansion_enabled:
                display.agent_progress("Retriever", "Performing spatial expansion...")
                expanded_results = await self._spatial_expand(top_results, state)
                state["spatial_expansion_results"] = expanded_results

                if "error" not in expanded_results:
                    display.agent_progress("Retriever", f"Expanded from {expanded_results['num_hits']} hits to {expanded_results['num_total']} ROIs (threshold: {expanded_results['threshold']:.3f})")
                else:
                    display.agent_progress("Retriever", f"Spatial expansion failed: {expanded_results['error']}")

            # Step 5: Write results to state
            state["semantic_search_results"] = {
                "query": hypothesis,
                "generated_queries": queries,
                "matched_rois": len(top_results),
                "top_k": self.top_k,
                "database": f"FAISS index ({vs.get_stats().get('num_documents', '?')} docs)",
                "results": top_results
            }

            display.agent_done("Retriever", "search complete")
            self._log_execution(
                state,
                "semantic_search",
                f"completed: {len(top_results)} ROIs matched for {len(queries)} queries"
            )

        except Exception as e:
            error_msg = f"RetrieverAgent execution failed: {str(e)}"
            add_error(state, error_msg)
            self._log_execution(state, "semantic_search", f"failed: {str(e)}")
            raise AgentExecutionError(error_msg) from e

        return state

    def _resolve_index_dir(self, state: AgentState) -> Path:
        """Resolve the FAISS index directory from state or default."""
        session_dir = state.get("session_dir")
        if session_dir:
            candidate = Path(session_dir) / "faiss_index"
            if candidate.exists():
                return candidate
        return self.index_dir

    async def _generate_queries(self, hypothesis: str) -> List[str]:
        """
        Use LLM to generate focused search queries from hypothesis.

        Args:
            hypothesis: The hypothesis text

        Returns:
            List of search query strings
        """
        if not hypothesis:
            return ["Proximal tubules", "Fibrosis markers"]

        prompt = QUERY_GENERATION_USER_PROMPT.format(hypothesis=hypothesis)
        messages = [
            SystemMessage(content=QUERY_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = await self.model.ainvoke(messages)
        response_text = self._extract_text(response.content).strip()

        # Parse JSON response
        if response_text.startswith('```'):
            parts = response_text.split('```')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith('json'):
                    part = part[4:].strip()
                if part and not part.startswith('```'):
                    response_text = part
                    break

        response_text = response_text.strip()
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()

        try:
            parsed = json.loads(response_text)
            queries = parsed.get("queries", [])
            if not queries:
                return [hypothesis]
            return queries
        except json.JSONDecodeError:
            # Fallback: use hypothesis directly
            return [hypothesis]

    def _filter_results(self, results: List[Dict], keywords: List[str]) -> List[Dict]:
        """
        Filter results by keywords in interpretation text.

        Args:
            results: List of search results with 'text' field
            keywords: List of keywords to filter by

        Returns:
            Filtered list of results
        """
        filtered = []
        keywords_lower = [k.lower().strip() for k in keywords]
        for r in results:
            text = r.get("text", "").lower()
            if any(kw in text for kw in keywords_lower):
                filtered.append(r)
        return filtered

    async def _spatial_expand(
        self,
        semantic_hits: List[Dict],
        state: AgentState
    ) -> Dict:
        """
        Perform spatial expansion from semantic hits.

        Algorithm:
        1. Load pivot_rois from state
        2. Extract roi_features array
        3. Find indices of hit ROIs in sampled_rois
        4. Calculate Euclidean distances from hits to all ROIs
        5. Normalize distances using quantile threshold
        6. Return all ROIs ranked by distance

        Args:
            semantic_hits: List of semantic search results with 'roi_key' field
            state: AgentState containing pivot_rois

        Returns:
            Dictionary with expansion results:
                - num_hits: Number of semantic hits found in pivot_rois
                - num_total: Total number of ROIs
                - threshold: Quantile threshold used
                - results: List of ROI dicts with distance scores
`                - error: Error message if failed
        """
        # Get pivot_rois from state
        pivot_rois = state.get("pivot_rois")
        if not pivot_rois:
            return {"error": "No pivot_rois in state for spatial expansion"}

        # Use first region_id (or iterate all regions)
        region_id = list(pivot_rois.keys())[0]
        sampled_rois, roi_features, cluster_labels, rep_rois = pivot_rois[region_id]

        # Extract hit ROI keys from FAISS results
        # FAISS returns keys in format "(region_id, coordinates)" from _roi_key_to_str()
        hit_keys = [r["roi_key"] for r in semantic_hits]

        # Create a mapping from various ROI key formats to standardized format
        # Handle both "(region_id, coordinates)" and "region_id-coordinates" formats
        normalized_hit_keys = {}
        for key in hit_keys:
            # Try to parse the FAISS format: "(region_id, coordinates)"
            if key.startswith('(') and ')' in key:
                # Extract region_id and coordinates from "(region_id, coordinates)"
                parts = key[1:key.rindex(')')].split(',', 1)
                if len(parts) == 2:
                    region, coords = parts
                    normalized_hit_keys[key] = (region.strip(), coords.strip())
                    # Also add alternative format variants for compatibility
                    normalized_hit_keys[f"{region.strip()}-{coords.strip()}"] = (region.strip(), coords.strip())
            else:
                # Handle other formats (e.g., "region_id-coordinates")
                if '-' in key:
                    parts = key.split('-', 1)
                    if len(parts) == 2:
                        normalized_hit_keys[key] = (parts[0].strip(), parts[1].strip())

        # Find indices of hit ROIs in sampled_rois
        hit_indices = []
        for idx, roi in enumerate(sampled_rois):
            # Build ROI key from the sampled_rois data
            # Match the format expected by FAISS: (region_id, coordinates)
            roi_region = roi.get('region_id', 'unknown')
            roi_coords = roi.get('coordinates', roi.get('patch_name', ''))

            # Try multiple key formats for matching
            possible_keys = [
                f"({roi_region}, {roi_coords})",  # FAISS format from _roi_key_to_str
                f"{roi_region}-{roi_coords}",      # Alternative format
            ]

            for possible_key in possible_keys:
                if possible_key in hit_keys:
                    hit_indices.append(idx)
                    break

        if not hit_indices:
            return {"error": f"No matching ROIs found in pivot_rois. FAISS returned {len(hit_keys)} keys with format like: {hit_keys[0] if hit_keys else 'N/A'}"}

        # Calculate distances from hits to all ROIs
        hit_features = roi_features[hit_indices]
        dist_to_hits = cdist(hit_features, roi_features, metric='Euclidean')
        dist_to_hits = np.median(dist_to_hits, axis=0)

        # Normalize distances using quantile threshold
        threshold = state.get("spatial_expansion_threshold", self.expansion_threshold)
        dist_to_hits = dist_to_hits - np.min(dist_to_hits)
        quantile_val = np.quantile(dist_to_hits, threshold)
        if quantile_val > 0:
            dist_to_hits = np.clip(dist_to_hits / quantile_val, 0, 1)
        else:
            dist_to_hits = np.clip(dist_to_hits, 0, 1)

        # Create ranked results
        expanded = []
        for idx, (roi, dist) in enumerate(zip(sampled_rois, dist_to_hits)):
            # Use FAISS format for consistency: (region_id, coordinates)
            roi_region = roi.get('region_id', 'unknown')
            roi_coords = roi.get('coordinates', roi.get('patch_name', ''))
            roi_key = f"({roi_region}, {roi_coords})"

            expanded.append({
                "roi_key": roi_key,
                "distance": float(dist),
                "rank": idx,
                "roi_metadata": roi
            })

        # Sort by distance (lower = closer to hits)
        expanded.sort(key=lambda x: x["distance"])

        return {
            "num_hits": len(hit_indices),
            "num_total": len(sampled_rois),
            "threshold": threshold,
            "results": expanded
        }

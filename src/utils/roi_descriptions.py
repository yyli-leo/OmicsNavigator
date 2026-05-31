"""
ROI description generation utilities for OmicsProfiler agent.

This module provides a self-contained implementation for generating ROI (Region of Interest)
descriptions from spatial omics data, fully reproducing the logic from patchsum library.

Key Features:
- Biomarker expression description with z-scores and percentiles
- Cell type composition with enrichment factors and signature biomarkers
- Self-contained implementation (no external custom functions required)

Adapted from interpretation_module/roi_description_generator.py
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy.stats import percentileofscore

import numpy as np
import pandas as pd


# =============================================================================
# Helper Functions (from patchsum/utils.py)
# =============================================================================

def calculate_signal_average(feature_df: pd.DataFrame, cell_ids: List[int]) -> pd.Series:
    """Calculate average signal intensity for a set of cells."""
    sub_feat = feature_df[feature_df['CELL_ID'].isin(cell_ids)]
    signal_average = sub_feat.mean(0)
    return signal_average


def batch_zscores(ref_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    """Calculate z-scores for query values against reference values."""
    # Make sure everything is 2D: N by C
    ref_values = np.array(ref_values)
    if len(ref_values.shape) == 1:
        ref_values = ref_values.reshape((-1, 1))
    query_values = np.array(query_values)
    if len(query_values.shape) == 1:
        query_values = query_values.reshape((-1, 1))

    ref_means = ref_values.mean(0, keepdims=True)
    ref_stds = ref_values.std(0, keepdims=True)
    z_scores = (query_values - ref_means) / (ref_stds + 1e-8)
    return z_scores


def batch_percentiles(ref_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    """Calculate percentiles for query values against reference values."""
    # Make sure everything is 2D: N by C
    ref_values = np.array(ref_values)
    if len(ref_values.shape) == 1:
        ref_values = ref_values.reshape((-1, 1))
    query_values = np.array(query_values)
    if len(query_values.shape) == 1:
        query_values = query_values.reshape((-1, 1))

    percentiles = [percentileofscore(ref_values[:, i], query_values[:, i])
                   for i in range(ref_values.shape[1])]
    percentiles = np.stack(percentiles, 1)
    return percentiles


def get_reference_patch_list(
    cell_seg_df: pd.DataFrame,
    patch_sizes: List[int] = [128],
    n_cells_min: int = 5,
    n_cells_max: int = 50,
    count: int = 1000,
    seed: int = 123
) -> List[List[int]]:
    """
    Generate reference patches using sliding window sampling.

    This creates a list of cell ID subsets for comparison when generating
    ROI descriptions.
    """
    if seed:
        np.random.seed(seed)

    patch_list = []
    stride_ratio = 0.5

    xmin, xmax = cell_seg_df['X'].min(), cell_seg_df['X'].max()
    ymin, ymax = cell_seg_df['Y'].min(), cell_seg_df['Y'].max()

    # Generate patches for each patch size
    for patch_size in patch_sizes:
        stride = int(patch_size * stride_ratio)

        # Generate multiple starting positions
        for _ in range(10):  # Try 10 different starting positions
            x_start = np.random.randint(xmin - stride, xmin + stride)
            y_start = np.random.randint(ymin - stride, ymin + stride)

            x_steps = (xmax - x_start) // stride + 1
            y_steps = (ymax - y_start) // stride + 1

            for i_x in range(x_steps):
                for i_y in range(y_steps):
                    x0, x1 = x_start + i_x * stride, x_start + i_x * stride + patch_size
                    y0, y1 = y_start + i_y * stride, y_start + i_y * stride + patch_size

                    # Get cells within this ROI
                    mask = (
                        (cell_seg_df['X'] >= x0) & (cell_seg_df['X'] <= x1) &
                        (cell_seg_df['Y'] >= y0) & (cell_seg_df['Y'] <= y1)
                    )
                    cids = cell_seg_df[mask]['CELL_ID'].tolist()

                    if len(cids) < n_cells_min:
                        continue
                    if len(cids) > n_cells_max:
                        cids = np.random.choice(cids, n_cells_max, replace=False).tolist()

                    patch_list.append(cids)

                    if len(patch_list) >= count * 2:
                        break
                if len(patch_list) >= count * 2:
                    break
            if len(patch_list) >= count * 2:
                break

    # Randomly sample to get exactly `count` patches
    if len(patch_list) > count:
        indices = np.random.choice(len(patch_list), count, replace=False)
        patch_list = [patch_list[i] for i in indices]

    return patch_list


# =============================================================================
# Biomarker Description Functions (from patchsum/processed_bm_info.py)
# =============================================================================

def get_cell_biomarker_expression_df(
    region_id: str,
    root_dir: Union[str, Path],
    channels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load and preprocess cell biomarker expression data.

    Applies arcsinh transformation and normalization as in the original code.
    """
    root_dir = Path(root_dir)
    table_file_path = root_dir / region_id / f'{region_id}.expression.csv'

    if not table_file_path.exists():
        raise FileNotFoundError(f'Cell biomarker expression table not found: {table_file_path}')

    cell_bm_df = pd.read_csv(table_file_path)
    cell_bm_df = cell_bm_df.set_index('CELL_ID')

    if channels is not None and len(channels) > 0:
        available_channels = [c for c in channels if c in cell_bm_df.columns]
        cell_bm_df = cell_bm_df[available_channels]

    # Apply arcsinh transformation and normalization (as in original)
    cell_bm_df = np.arcsinh(cell_bm_df / (5 * np.quantile(cell_bm_df, 0.2, axis=0) + 1e-5))
    cell_bm_df = cell_bm_df / cell_bm_df.std(0)
    cell_bm_df = cell_bm_df.reset_index()

    return cell_bm_df


def generate_cell_bm_description_for_target_patches(
    cell_id_target_subsets: List[List[int]],
    cell_id_reference_subsets: List[List[int]],
    cell_bm_df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    return_raw: bool = False,
    z_threshold: float = 1.5,
    percentile_threshold: float = 95,
    always_include_topk: int = 2
) -> List[Union[Dict[str, Dict[str, float]], str]]:
    """
    Generate biomarker expression description for target ROIs.

    Compares each ROI against reference patches to calculate z-scores
    and percentiles, then categorizes biomarkers as highly/lowly/moderately
    expressed.
    """
    # Ensure target_subsets is a list of lists
    if isinstance(cell_id_target_subsets[0], int):
        cell_id_target_subsets = [cell_id_target_subsets]

    # Determine which channels to use
    if channels is None or len(channels) == 0:
        channels = [bm for bm in cell_bm_df.columns if bm != 'CELL_ID']
    else:
        channels = sorted(set(channels) & set(cell_bm_df.columns))

    def worker_fn(cell_ids):
        signal_average = calculate_signal_average(cell_bm_df, cell_ids)
        return [signal_average[bm] for bm in channels]

    # Calculate reference and target features
    reference_feats = np.array([worker_fn(cids) for cids in cell_id_reference_subsets])
    target_feats = np.array([worker_fn(cids) for cids in cell_id_target_subsets])

    # Calculate z-scores and percentiles
    target_summaries = [{} for _ in range(len(cell_id_target_subsets))]

    for bm_i, bm in enumerate(channels):
        # Filter out NaN values from reference
        valid_mask = ~np.isnan(reference_feats[:, bm_i])
        reference_vals = reference_feats[valid_mask, bm_i]
        valid_target_mask = ~np.isnan(target_feats[:, bm_i])

        if len(reference_vals) == 0:
            continue

        # For each target, calculate z-score and percentile
        for target_j in range(len(cell_id_target_subsets)):
            if valid_target_mask[target_j]:
                target_val = target_feats[target_j:target_j+1, bm_i]
                zs = batch_zscores(reference_vals.reshape(-1, 1), target_val.reshape(1, 1))
                pers = batch_percentiles(reference_vals.reshape(-1, 1), target_val.reshape(1, 1))

                if zs[0, 0] == zs[0, 0]:  # Check for NaN
                    target_summaries[target_j][bm] = {
                        "z": float(zs[0, 0]),
                        "percentile": float(pers[0, 0])
                    }

    if return_raw:
        return target_summaries
    else:
        return [bm_summary_to_text(ts, z_threshold, percentile_threshold, always_include_topk)
                for ts in target_summaries]


def bm_summary_to_text(
    target_summary: Dict[str, Dict[str, float]],
    z_threshold: float = 1.5,
    percentile_threshold: float = 95,
    always_include_topk: int = 2,
    include_high: bool = True,
    include_low: bool = True,
    include_others: bool = True
) -> str:
    """Convert biomarker summary statistics to natural language text."""
    highly_expressed = []
    lowly_expressed = []
    others = []

    # Sort biomarkers by z-score
    sorted_bms = sorted(target_summary.keys(), key=lambda x: -target_summary[x]['z'])

    for bm_i, biomarker in enumerate(sorted_bms):
        stats = target_summary[biomarker]
        z_score = stats['z']
        percentile = stats['percentile']

        if z_score > z_threshold or percentile > percentile_threshold or bm_i < always_include_topk:
            highly_expressed.append((biomarker, z_score, percentile))
        elif z_score < -z_threshold or percentile < 100 - percentile_threshold:
            lowly_expressed.append((biomarker, z_score, percentile))
        else:
            others.append((biomarker, z_score, percentile))

    # Generate text description
    summary = []

    if highly_expressed and include_high:
        summary.append("**Highly Expressed Biomarkers**:\nThese biomarkers are overexpressed:")
        for biomarker, z, p in sorted(highly_expressed, key=lambda x: x[1], reverse=True):
            summary.append(f"- **{biomarker}**: z-score = {z:.1f}, percentile = {p:.0f}%")

    if lowly_expressed and include_low:
        summary.append("\n**Lowly Expressed Biomarkers**:\nThese biomarkers are underexpressed:")
        for biomarker, z, p in sorted(lowly_expressed, key=lambda x: x[1]):
            summary.append(f"- **{biomarker}**: z-score = {z:.1f}, percentile = {p:.0f}%")

    if others and include_others:
        summary.append("\n**Moderately Expressed Biomarkers**:\n" +
                       "These biomarkers do not show strong trends and have moderate expression levels:")
        summary.append(", ".join([f"**{biomarker}**" for biomarker, z, p in others]))

    return "\n".join(summary)


# =============================================================================
# Cell Type Description Functions (from patchsum/cell_type_info.py)
# =============================================================================

def get_cell_type_df(
    region_id: str,
    root_dir: Union[str, Path]
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Load cell type dataframe for a region."""
    root_dir = Path(root_dir)

    # Try different possible filenames
    possible_filenames = [
        f"{region_id}.cell_types.csv",
        f"{region_id}.cell_type.csv",
    ]

    cell_type_path = None
    for filename in possible_filenames:
        path = root_dir / region_id / filename
        if path.exists():
            cell_type_path = path
            break

    if cell_type_path is None:
        # Try in the region directory directly
        for filename in possible_filenames:
            path = root_dir / filename
            if path.exists():
                cell_type_path = path
                break

    if cell_type_path is None:
        raise FileNotFoundError(f'Cell type file not found for {region_id}')

    cell_type_df = pd.read_csv(cell_type_path)

    # Normalize column names
    if 'ANNOTATION_LABEL' in cell_type_df.columns:
        cell_type_df.columns = ['CELL_ID', 'VALUE', 'CELL_TYPE']
    elif 'CELL_TYPE' not in cell_type_df.columns:
        if 'VALUE' in cell_type_df.columns:
            cell_type_df = cell_type_df.rename(columns={'VALUE': 'CELL_TYPE'})
        else:
            raise ValueError(f"Cannot find cell type column in {cell_type_path}")

    # Create mapping
    if 'VALUE' in cell_type_df.columns:
        cell_type_mapping = dict(zip(cell_type_df['VALUE'], cell_type_df['CELL_TYPE']))
    else:
        cell_type_mapping = {}

    return cell_type_df, cell_type_mapping


def get_signature_biomarkers(
    cell_types_df: pd.DataFrame,
    cell_bm_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """
    Get signature biomarkers for each cell type.

    A biomarker is considered "signature" for a cell type if its normalized
    expression in that cell type is > 0.8.
    """
    cts = set(cell_types_df['CELL_TYPE'])
    biomarkers = [bm for bm in cell_bm_df.columns if bm != 'CELL_ID']

    # Calculate biomarker average by cell type
    bm_levels = {}
    for ct in cts:
        cell_ids = cell_types_df[cell_types_df['CELL_TYPE'] == ct]['CELL_ID']
        bms = dict(cell_bm_df[cell_bm_df['CELL_ID'].isin(cell_ids)][biomarkers].mean())
        bm_levels[ct] = bms

    # Re-normalize biomarker averages across cell types
    for bm in biomarkers:
        bm_level_by_ct = [bm_levels[ct][bm] for ct in cts]
        bm_min, bm_max = min(bm_level_by_ct), max(bm_level_by_ct)
        bm_range = bm_max - bm_min
        if bm_range > 0:
            for ct in cts:
                bm_levels[ct][bm] = (bm_levels[ct][bm] - bm_min) / bm_range

    # Pick signature biomarkers for each cell type (threshold > 0.8)
    signature_biomarkers = {}
    for ct in cts:
        signature_bms = []
        for bm, val in bm_levels[ct].items():
            if val > 0.8:
                signature_bms.append(bm)
        signature_biomarkers[ct] = signature_bms

    return signature_biomarkers


def calculate_cell_type_composition(
    cell_types_df: pd.DataFrame,
    cell_ids: List[int]
) -> Dict[str, float]:
    """Calculate cell type composition given a list of cell IDs."""
    sub_ct_df = cell_types_df[cell_types_df['CELL_ID'].isin(cell_ids)]
    ct_count = dict(Counter(sub_ct_df['CELL_TYPE']))
    comp = {k: v / len(cell_ids) for k, v in ct_count.items()}
    return comp


def generate_cell_type_description_for_target_patches(
    cell_id_target_subsets: List[List[int]],
    cell_types_df: pd.DataFrame,
    cell_bm_df: pd.DataFrame,
    n_key_cell_types: int = 3,
    return_raw: bool = False
) -> List[Union[Dict[str, Tuple[float, float, List[str]]], str]]:
    """
    Generate cell type composition description for target ROIs.

    Calculates composition percentages, enrichment factors (compared to region average),
    and signature biomarkers for each cell type.
    """
    # Ensure target_subsets is a list of lists
    if isinstance(cell_id_target_subsets[0], int):
        cell_id_target_subsets = [cell_id_target_subsets]

    # Get signature biomarkers for each cell type
    signature_biomarkers = get_signature_biomarkers(cell_types_df, cell_bm_df)

    # Calculate average composition across the entire region
    avg_comp = calculate_cell_type_composition(cell_types_df, list(cell_types_df['CELL_ID']))

    target_summaries = []
    for cell_ids in cell_id_target_subsets:
        roi_comp = calculate_cell_type_composition(cell_types_df, cell_ids)
        enrichment = {ct: roi_comp[ct] / (avg_comp[ct] + 1e-5) for ct in roi_comp}

        target_summary = {}
        for ct in roi_comp:
            ct_comp = roi_comp[ct] * 100  # Convert to percentage
            enr = enrichment[ct]
            sig_bms = signature_biomarkers.get(ct, [])
            target_summary[ct] = (ct_comp, enr, sig_bms)
        target_summaries.append(target_summary)

    if return_raw:
        return target_summaries
    else:
        return [ct_composition_summary_to_text(ts, n_key_cell_types)
                for ts in target_summaries]


def ct_composition_summary_to_text(
    target_summary: Dict[str, Tuple[float, float, List[str]]],
    n_key_cell_types: int = 3
) -> str:
    """Convert cell type composition summary to natural language text."""
    cts = list(target_summary.keys())

    # First, select by composition percentage
    key_cts = sorted(cts, key=lambda x: target_summary[x][0], reverse=True)[:n_key_cell_types]
    key_cts = [ct for ct in key_cts if target_summary[ct][0] > 5]  # At least 5% composition

    # Then add highly enriched cell types
    for ct in sorted(cts, key=lambda x: target_summary[x][1], reverse=True)[:n_key_cell_types]:
        if ct not in key_cts and target_summary[ct][1] > 1.5 and target_summary[ct][0] > 5:
            key_cts.append(ct)

    summary = ["**Major Cell Types**:"]
    for ct_i, ct in enumerate(key_cts):
        ct_comp, enr, sig_bms = target_summary[ct]
        line = f'{ct_i + 1}. **{ct}**:\n'
        line += f'Cells of type "{ct}" make up **{ct_comp:.0f}%** of the total composition. '
        line += f"They are enriched by **{enr:.1f}** compared to region average."
        if len(sig_bms) > 0:
            sig_bms_str = ','.join([f'**{bm}**' for bm in sig_bms])
            line += f" This cell type is characterized by the biomarker(s): {sig_bms_str}"
        summary.append(line)
        summary.append("")

    return "\n".join(summary)


# =============================================================================
# Cell Segmentation Data Loading
# =============================================================================

def get_cell_segmentation_df(region_id: str, root_dir: Union[str, Path]) -> pd.DataFrame:
    """Load cell segmentation dataframe with cell coordinates."""
    root_dir = Path(root_dir)
    seg_path = root_dir / region_id / f'{region_id}.cell_data.csv'

    if not seg_path.exists():
        raise FileNotFoundError(f'Cell segmentation file not found: {seg_path}')

    cell_seg_df = pd.read_csv(seg_path)
    return cell_seg_df


# =============================================================================
# Main ROI Description Generation Function
# =============================================================================

def generate_roi_descriptions(
    region_id: str,
    root_dir: Union[str, Path],
    target_rois: List[Dict[str, Any]],
    include_biomarker: bool = True,
    include_cell_type: bool = True,
    include_morphology: bool = False,
    cell_type_rename: Optional[Dict[str, str]] = None,
    biomarkers: Optional[List[str]] = None,
    n_reference_patches: int = 1000,
    z_threshold: float = 1.5,
    percentile_threshold: float = 95,
    n_key_cell_types: int = 3,
    verbose: bool = False
) -> Dict[Tuple[str, str], str]:
    """
    Generate ROI descriptions for a region.

    This is the main function that reproduces the exact logic from s255_manual.ipynb.
    It generates biomarker and cell type descriptions matching the reference format.

    Args:
        region_id: Region identifier (e.g., 's255_c001_v001_r001_reg002')
        root_dir: Root directory containing the region data
        target_rois: List of target ROI dicts with 'cell_ids' and 'patch_name' keys
        include_biomarker: Whether to include biomarker expression description
        include_cell_type: Whether to include cell type composition description
        include_morphology: (Unused, for compatibility)
        cell_type_rename: Optional mapping for renaming cell types
        biomarkers: Optional list of biomarker channels to include
        n_reference_patches: Number of reference patches for comparison
        z_threshold: Z-score threshold for biomarker categorization
        percentile_threshold: Percentile threshold for biomarker categorization
        n_key_cell_types: Number of top cell types to report
        verbose: Show progress messages

    Returns:
        Dictionary mapping (region_id, patch_name) to description text
    """
    root_dir = Path(root_dir)

    if verbose:
        print(f"Loading data for region {region_id}...")

    # Load cell data
    cell_seg_df = get_cell_segmentation_df(region_id, root_dir)
    cell_bm_df = get_cell_biomarker_expression_df(region_id, root_dir, biomarkers)
    cell_types_df, cell_type_mapping = get_cell_type_df(region_id, root_dir)

    # Apply cell type renaming if provided
    if cell_type_rename:
        cell_types_df['CELL_TYPE'] = cell_types_df['CELL_TYPE'].map(
            lambda x: cell_type_rename.get(x, x)
        )

    if verbose:
        print("Generating reference patches...")

    # Generate reference patches
    reference_patches = get_reference_patch_list(
        cell_seg_df,
        patch_sizes=[128],
        n_cells_min=5,
        n_cells_max=50,
        count=n_reference_patches,
        seed=123
    )

    # Extract target ROI cell IDs
    target_patches = [roi['cell_ids'] for roi in target_rois]

    if verbose:
        print("Generating descriptions...")

    descriptions = {}

    for i, roi in enumerate(target_rois):
        patch_cells = [roi['cell_ids']] if not isinstance(roi['cell_ids'], list) else [roi['cell_ids']]
        patch_name = roi['patch_name']

        summary_parts = []

        # Generate biomarker summary
        if include_biomarker:
            bm_summaries = generate_cell_bm_description_for_target_patches(
                patch_cells, reference_patches,
                cell_bm_df,
                channels=biomarkers,
                return_raw=False,
                z_threshold=z_threshold,
                percentile_threshold=percentile_threshold,
                always_include_topk=2
            )
            summary_parts.append(bm_summaries[0])

        # Generate cell type summary
        if include_cell_type:
            ct_summaries = generate_cell_type_description_for_target_patches(
                patch_cells,
                cell_types_df,
                cell_bm_df,
                n_key_cell_types=n_key_cell_types,
                return_raw=False
            )
            summary_parts.append(ct_summaries[0])

        # Combine summaries
        description = "\n\n".join(summary_parts)
        descriptions[(region_id, patch_name)] = description

    if verbose:
        print(f"Generated {len(descriptions)} ROI descriptions")

    return descriptions


# =============================================================================
# Backward Compatibility Functions (keep existing API)
# =============================================================================

def load_cell_data(
    region_id: str,
    root_dir: Union[str, Path],
    cell_ids: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cell type and expression data for a set of cells.

    Args:
        region_id: Region identifier (e.g., "s255_c001_v001_r001_reg023")
        root_dir: Root directory containing the data (e.g., "./data/s255")
        cell_ids: List of cell IDs to load

    Returns:
        Tuple of (cell_types_df, expression_df)
    """
    root_dir = Path(root_dir)

    # Load cell type data
    cell_types_df, _ = get_cell_type_df(region_id, root_dir)
    filtered_cell_types = cell_types_df[cell_types_df['CELL_ID'].isin(cell_ids)]

    # Load expression data
    try:
        expression_df = get_cell_biomarker_expression_df(region_id, root_dir)
        filtered_expression = expression_df[expression_df['CELL_ID'].isin(cell_ids)]
    except FileNotFoundError:
        filtered_expression = None

    return filtered_cell_types, filtered_expression


def summarize_composition(cell_types_df: pd.DataFrame, top_n: int = 3) -> str:
    """
    Summarize cell type composition from a DataFrame.

    Args:
        cell_types_df: DataFrame with CELL_TYPE column
        top_n: Number of top cell types to include

    Returns:
        Formatted string with cell type composition
    """
    if cell_types_df.empty:
        return "No cell data available"

    cell_type_counts = cell_types_df['CELL_TYPE'].value_counts(normalize=True)
    top_types = cell_type_counts.head(top_n)

    return "Cell type composition: " + ", ".join(
        [f"{label} ({pct:.1%})" for label, pct in zip(top_types.index, top_types.values)]
    )


def summarize_biomarkers(expression_df: pd.DataFrame, top_n: int = 5) -> str:
    """
    Summarize biomarker expression from a DataFrame.

    Args:
        expression_df: DataFrame with numeric biomarker columns
        top_n: Number of top biomarkers to include

    Returns:
        Formatted string with highly expressed biomarkers
    """
    if expression_df is None or expression_df.empty:
        return "No expression data available"

    # Get numeric columns (excluding CELL_ID if present)
    numeric_cols = expression_df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return "No numeric biomarker data available"

    # Calculate mean expression per biomarker
    top_bm = expression_df[numeric_cols].mean().nlargest(top_n)

    return "Highly expressed biomarkers: " + ", ".join([f"{col}" for col in top_bm.index])

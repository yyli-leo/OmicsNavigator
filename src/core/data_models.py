import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from pathlib import Path

@dataclass
class RegionOfInterest:
    """
    Represents a single physical Region of Interest (ROI) bounding box and its enclosed cells.
    """
    patch_name: str
    center_x: float
    center_y: float
    xyranges: Tuple[float, float, float, float]  # (x0, x1, y0, y1)
    cell_ids: List[int]

    @property
    def num_cells(self) -> int:
        return len(self.cell_ids)


@dataclass
class ClusterIdentity:
    """
    Encapsulates the two-stage clustering assignment for robust serialization.
    """
    feature_cluster_id: int
    spatial_subcluster_id: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.feature_cluster_id, self.spatial_subcluster_id)


@dataclass
class RegionShardData:
    """
    Encapsulates all multi-modal features, physical boundaries, and clustering
    metadata for a specific independent spatial region (FOV).
    """
    region_id: str
    rois: List[RegionOfInterest]
    combined_features: np.ndarray  # Shape: (N_ROIs, M_Features)
    cluster_assignments: List[ClusterIdentity]

    # Critical Fix: Store ONLY integer indices pointing to `self.rois`
    # instead of duplicating the entire ROI objects to prevent memory bloat.
    cluster_center_indices: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Runtime validation to strictly enforce data alignment constraints.
        Prevents silent length mismatches between parallel arrays.
        """
        n_rois = len(self.rois)
        if self.combined_features.shape[0] != n_rois:
            raise ValueError(f"Feature matrix row count ({self.combined_features.shape[0]}) "
                             f"must match ROI list length ({n_rois}).")
        if len(self.cluster_assignments) != n_rois:
            raise ValueError(f"Cluster assignment list length ({len(self.cluster_assignments)}) "
                             f"must match ROI list length ({n_rois}).")

    def get_representative_rois(self, feat_id: int, spatial_id: int) -> List[RegionOfInterest]:
        """
        Helper method to dynamically retrieve the actual ROI objects for a cluster center.
        """
        target_key = (feat_id, spatial_id)
        if target_key not in self.cluster_center_indices:
            return []

        indices = self.cluster_center_indices[target_key]
        return [self.rois[i] for i in indices]


@dataclass
class SpatialPivotRegistry:
    """
    The top-level root data structure replacing `s255_pivot_ROIs`.
    """
    dataset_id: str = "MIF_s255"
    shards: Dict[str, RegionShardData] = field(default_factory=dict)

    def add_shard(self, shard_data: RegionShardData):
        self.shards[shard_data.region_id] = shard_data
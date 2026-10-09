"""Small reproducible trajectory-clustering demonstration.

This module is intentionally separate from the original published research
scripts. It provides an import-safe, deterministic example using the same
multivariate time-series clustering family used in the study.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans


DEFAULT_FEATURES = ("longitude", "latitude", "altitude")


def load_trajectory_table(csv_path: str | Path) -> pd.DataFrame:
    """Load the demonstration trajectory table and validate required columns."""
    df = pd.read_csv(csv_path)
    required = {"track_id", "time_step", *DEFAULT_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.sort_values(["track_id", "time_step"]).reset_index(drop=True)


def trajectory_table_to_array(
    df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
) -> tuple[np.ndarray, list[str]]:
    """Convert long-form trajectory rows to [track, time, feature] form."""
    grouped = list(df.groupby("track_id", sort=True))
    if not grouped:
        raise ValueError("No trajectories were found.")

    lengths = {len(group) for _, group in grouped}
    if len(lengths) != 1:
        raise ValueError("All demonstration trajectories must have equal length.")

    track_ids = [str(track_id) for track_id, _ in grouped]
    data = np.stack(
        [group.loc[:, feature_columns].to_numpy(dtype=float) for _, group in grouped]
    )
    return data, track_ids


def cluster_trajectories(
    data: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster multivariate trajectories with DTW TimeSeriesKMeans."""
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=20,
        n_init=2,
        random_state=random_state,
    )
    labels = model.fit_predict(data)
    return labels, model.cluster_centers_, float(model.inertia_)


def run_demo(csv_path: str | Path) -> dict:
    """Run the complete demonstration and return a compact result summary."""
    df = load_trajectory_table(csv_path)
    data, track_ids = trajectory_table_to_array(df)
    labels, centers, inertia = cluster_trajectories(data)

    return {
        "track_ids": track_ids,
        "labels": labels.tolist(),
        "data_shape": list(data.shape),
        "centers_shape": list(centers.shape),
        "inertia": inertia,
    }

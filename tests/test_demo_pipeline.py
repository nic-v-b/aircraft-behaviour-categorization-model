from pathlib import Path

from aircraft_behaviour.demo_pipeline import (
    cluster_trajectories,
    load_trajectory_table,
    trajectory_table_to_array,
)


DEMO_CSV = Path(__file__).resolve().parents[1] / "examples" / "synthetic_trajectories.csv"


def test_synthetic_demo_has_expected_shape():
    df = load_trajectory_table(DEMO_CSV)
    data, track_ids = trajectory_table_to_array(df)

    assert track_ids == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert data.shape == (6, 6, 3)


def test_dtw_clustering_recovers_two_trajectory_families():
    df = load_trajectory_table(DEMO_CSV)
    data, _track_ids = trajectory_table_to_array(df)
    labels, centers, inertia = cluster_trajectories(data, n_clusters=2, random_state=42)

    # Cluster labels themselves are arbitrary; the partition is what matters.
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert centers.shape == (2, 6, 3)
    assert inertia >= 0

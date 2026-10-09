"""Run the small reproducible trajectory-clustering example."""

from __future__ import annotations

from pathlib import Path

from aircraft_behaviour.demo_pipeline import run_demo


def main() -> None:
    csv_path = Path(__file__).with_name("synthetic_trajectories.csv")
    summary = run_demo(csv_path)

    print("Synthetic trajectory clustering demo")
    print("-----------------------------------")
    print(f"tracks: {summary['track_ids']}")
    print(f"labels: {summary['labels']}")
    print(f"data shape: {summary['data_shape']}")
    print(f"centers shape: {summary['centers_shape']}")
    print(f"inertia: {summary['inertia']:.6f}")


if __name__ == "__main__":
    main()

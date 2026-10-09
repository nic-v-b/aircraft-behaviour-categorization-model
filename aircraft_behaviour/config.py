"""Portable configuration for study-specific input and output paths.

The original research scripts used absolute Windows paths from the author's
development workstation. This module keeps those paths out of source code and
allows each machine to provide its own locations through environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _directory_from_env(variable: str, default_relative: str) -> str:
    """Return an absolute directory path ending with the platform separator."""
    raw_value = os.getenv(variable)
    path = Path(raw_value).expanduser() if raw_value else REPO_ROOT / default_relative
    return str(path.resolve()) + os.sep


def _file_from_env(variable: str, default_relative: str) -> str:
    """Return an absolute file path from an environment variable or repo-relative default."""
    raw_value = os.getenv(variable)
    path = Path(raw_value).expanduser() if raw_value else REPO_ROOT / default_relative
    return str(path.resolve())


@dataclass(frozen=True)
class ResearchPaths:
    """File-system locations used by the original research workflow."""

    raw_adsb_dir: str
    paper_dir: str
    aircraft_registry_file: str
    jat_data_dir: str
    flight_tracks_dir: str
    results_dir: str
    visuals_dir: str

    @classmethod
    def from_env(cls) -> "ResearchPaths":
        """Build a configuration from environment variables."""
        return cls(
            raw_adsb_dir=_directory_from_env("ABC_RAW_ADSB_DIR", "data/raw_adsb"),
            paper_dir=_directory_from_env("ABC_PAPER_DIR", "data/paper"),
            aircraft_registry_file=_file_from_env(
                "ABC_AIRCRAFT_REGISTRY_FILE",
                "data/aircraft_registry/processed_data_2021.csv",
            ),
            jat_data_dir=_directory_from_env("ABC_JAT_DATA_DIR", "data/jat"),
            flight_tracks_dir=_directory_from_env(
                "ABC_FLIGHT_TRACKS_DIR",
                "data/paper/flight tracks",
            ),
            results_dir=_directory_from_env("ABC_RESULTS_DIR", "results"),
            visuals_dir=_directory_from_env("ABC_VISUALS_DIR", "results/visuals"),
        )

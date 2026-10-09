from pathlib import Path

from aircraft_behaviour.config import REPO_ROOT, ResearchPaths


def test_default_paths_are_repository_relative(monkeypatch):
    for variable in (
        "ABC_RAW_ADSB_DIR",
        "ABC_PAPER_DIR",
        "ABC_AIRCRAFT_REGISTRY_FILE",
        "ABC_JAT_DATA_DIR",
        "ABC_FLIGHT_TRACKS_DIR",
        "ABC_RESULTS_DIR",
        "ABC_VISUALS_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)

    paths = ResearchPaths.from_env()

    assert Path(paths.raw_adsb_dir).resolve() == (REPO_ROOT / "data/raw_adsb").resolve()
    assert Path(paths.results_dir).resolve() == (REPO_ROOT / "results").resolve()
    assert Path(paths.aircraft_registry_file).resolve() == (
        REPO_ROOT / "data/aircraft_registry/processed_data_2021.csv"
    ).resolve()


def test_environment_overrides(monkeypatch, tmp_path):
    raw_dir = tmp_path / "adsb"
    results_dir = tmp_path / "results"
    registry_file = tmp_path / "registry.csv"

    monkeypatch.setenv("ABC_RAW_ADSB_DIR", str(raw_dir))
    monkeypatch.setenv("ABC_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("ABC_AIRCRAFT_REGISTRY_FILE", str(registry_file))

    paths = ResearchPaths.from_env()

    assert Path(paths.raw_adsb_dir).resolve() == raw_dir.resolve()
    assert Path(paths.results_dir).resolve() == results_dir.resolve()
    assert Path(paths.aircraft_registry_file).resolve() == registry_file.resolve()

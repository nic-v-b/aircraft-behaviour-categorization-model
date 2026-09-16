# Aircraft Behaviour Categorization Using Machine Learning

[![Python CI](https://github.com/nic-v-b/aircraft-behaviour-categorization-model/actions/workflows/python-syntax.yml/badge.svg)](https://github.com/nic-v-b/aircraft-behaviour-categorization-model/actions/workflows/python-syntax.yml)

Research implementation accompanying the peer-reviewed article:

**N. Vincent-Boulay and C. Marsden, "Aircraft Categorization Approach Using Machine Learning to Analyze Aircraft Behavior," _Journal of Air Transportation_, 32(4), 218–229, 2024.**  
DOI: https://doi.org/10.2514/1.D0398

## Overview

This project investigates automated, behavior-based aircraft categorization from real-world ADS-B trajectory data. The workflow converts aircraft observations into comparable trajectories, applies unsupervised time-series clustering, and generates quantitative and visual outputs for interpretation.

The work was developed as part of a broader research program on machine learning and geospatial modeling for complex airspace environments.

## What this project demonstrates

- Processing of real-world ADS-B trajectory data
- Spatiotemporal trajectory preparation and smoothing
- Multivariate time-series representation of aircraft behavior
- Unsupervised clustering of aircraft trajectories
- Cluster interpretation and validation
- Reproducible generation of research figures and tabular outputs
- Application of machine learning to an aerospace surveillance / situational-awareness problem

## Workflow

1. **Trajectory processing** — `trajectory_processing.py`
   - cleans and filters ADS-B observations
   - associates aircraft metadata
   - derives trajectory-level quantities
   - segments and prepares aircraft tracks for analysis

2. **Clustering** — `clustering_algorithm.py`
   - loads processed trajectory time series
   - constructs multivariate trajectory representations
   - applies time-series clustering
   - evaluates and stores clustering outputs

3. **Result interpretation and visualization** — `generate_results_visuals.py`
   - loads clustering outputs
   - creates trajectory/centroid plots
   - generates cluster statistics and interpretation figures

## Repository structure

```text
.
├── README.md
├── trajectory_processing.py
├── clustering_algorithm.py
├── generate_results_visuals.py
├── aircraft_behaviour/
│   ├── config.py
│   ├── demo_pipeline.py
│   ├── geo_utils.py
│   ├── platform_utils.py
│   └── time_utils.py
├── examples/
│   ├── run_demo.py
│   └── synthetic_trajectories.csv
├── tests/
├── sample results/
├── nb clusters 4 silhouette plot.png
├── nb clusters 4 feature importance plot.png
├── requirements.txt
├── requirements-ci.txt
├── RESEARCH_CODE_NOTES.md
├── .env.example
└── .github/workflows/python-syntax.yml
```

## Example outputs

### Silhouette analysis

![Silhouette analysis](nb%20clusters%204%20silhouette%20plot.png)

### Feature interpretation

![Feature importance](nb%20clusters%204%20feature%20importance%20plot.png)

Representative CSV outputs are available in the `sample results/` directory.

## Quick reproducible demo

A small synthetic dataset is included so the core multivariate time-series clustering concept can be exercised without the original ADS-B dataset:

```bash
pip install -r requirements-ci.txt
python examples/run_demo.py
```

The demo builds equal-length longitude/latitude/altitude trajectories and clusters them with `TimeSeriesKMeans(metric="dtw")`, matching the clustering family used in the research implementation. GitHub Actions runs this demonstration and its regression tests automatically.

## Setup

Create a Python environment and install the direct dependencies:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
```

### Optional Mapbox configuration

Some Plotly map visualizations can use a Mapbox access token. The repository no longer stores a personal token in source code.

Set the environment variable before running a script that requires Mapbox:

**Windows PowerShell**

```powershell
$env:MAPBOX_TOKEN="pk.your_token_here"
```

**macOS / Linux**

```bash
export MAPBOX_TOKEN="pk.your_token_here"
```

See `.env.example` for the expected variable name.

### Portable research-data paths

The original research implementation referenced absolute paths on the development workstation. Those locations are now handled by `aircraft_behaviour/config.py`.

You can point the scripts to external copies of the study data with:

- `ABC_RAW_ADSB_DIR`
- `ABC_PAPER_DIR`
- `ABC_AIRCRAFT_REGISTRY_FILE`
- `ABC_JAT_DATA_DIR`
- `ABC_FLIGHT_TRACKS_DIR`
- `ABC_RESULTS_DIR`
- `ABC_VISUALS_DIR`

If they are not set, repository-relative defaults under `data/` and `results/` are used.

## Data and reproducibility

The original study used real-world ADS-B observations together with aircraft-registration metadata. The full source dataset is not redistributed in this repository.

The repository preserves the research scripts used for the published study. Study-specific input/output paths are now handled through portable environment-based configuration. Sample outputs are included so that the structure of the analysis results can be inspected without the original dataset.

The three original research scripts intentionally remain recognizable as the published experimental implementation rather than being rewritten wholesale into a production application. New reusable code is kept in `aircraft_behaviour/`, and the self-contained example in `examples/run_demo.py` uses a conventional `main()` entry point. See [RESEARCH_CODE_NOTES.md](RESEARCH_CODE_NOTES.md) for the preservation rationale.

For the study design, preprocessing methodology, features, clustering formulation, and interpretation of results, see the published paper linked above.

## Software-engineering status

This repository began as research code developed to support the published analysis. It is being incrementally curated for clearer documentation, portability, testing, and reproducibility while preserving the scientific behavior of the original implementation.

Shared configuration, time-conversion, geospatial, and platform helpers have been extracted into reusable modules. GitHub Actions now performs automated syntax checks, runs `pytest` unit/regression tests, and executes the deterministic synthetic trajectory-clustering demo on every push and pull request.

`requirements-ci.txt` records the dependency ranges used for the automated test environment. The historical research scripts are retained separately from this test harness so that portfolio-oriented engineering improvements do not obscure the published workflow.

## Citation

If you use this work, please cite:

```text
N. Vincent-Boulay and C. Marsden,
"Aircraft Categorization Approach Using Machine Learning to Analyze Aircraft Behavior,"
Journal of Air Transportation, vol. 32, no. 4, pp. 218–229, 2024.
https://doi.org/10.2514/1.D0398
```

A machine-readable citation is also provided in `CITATION.cff`.

## Author

**Nicolas Vincent-Boulay**  
Aerospace engineering PhD candidate and machine-learning researcher

- LinkedIn: https://www.linkedin.com/in/nicolas-vincent-boulay/
- Google Scholar: https://scholar.google.ca/citations?user=5syjcmIAAAAJ&hl=en

# Aircraft Behaviour Categorization Using Machine Learning

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
├── sample results/
├── nb clusters 4 silhouette plot.png
├── nb clusters 4 feature importance plot.png
├── requirements.txt
├── .env.example
└── .github/workflows/python-syntax.yml
```

## Example outputs

### Silhouette analysis

![Silhouette analysis](nb%20clusters%204%20silhouette%20plot.png)

### Feature interpretation

![Feature importance](nb%20clusters%204%20feature%20importance%20plot.png)

Representative CSV outputs are available in the `sample results/` directory.

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

## Data and reproducibility

The original study used real-world ADS-B observations together with aircraft-registration metadata. The full source dataset is not redistributed in this repository.

The repository currently preserves the research scripts used for the published study. Some study-specific input/output paths from the original research environment remain in the scripts and will be migrated to portable configuration in a follow-up refactor. Sample outputs are included so that the structure of the analysis results can be inspected without the original dataset.

For the study design, preprocessing methodology, features, clustering formulation, and interpretation of results, see the published paper linked above.

## Software-engineering status

This repository began as research code developed to support the published analysis. It is being incrementally curated for clearer documentation, portability, testing, and reproducibility while preserving the scientific behavior of the original implementation.

A lightweight GitHub Actions workflow currently performs automated Python syntax checks on pushes and pull requests. Functional/unit tests will be added as the research scripts are modularized.

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

# Research-code preservation notes

This repository contains the scripts used during the published aircraft-behaviour
categorization study. They are intentionally preserved as research scripts rather
than rewritten wholesale into a production application.

## Why the original scripts remain script-style

The original files contain the experimental sequence, parameter exploration,
plot-generation logic, and intermediate analysis choices used during the study.
Moving thousands of lines into a new application architecture purely for cosmetic
reasons would make it harder to trace the public repository back to the published
workflow.

Accordingly:

- the original three research scripts remain recognizable and close to the study implementation;
- portability issues such as hard-coded workstation paths and Windows-only completion sounds are removed;
- clearly reusable helpers are extracted into `aircraft_behaviour/`;
- new code is written as import-safe modules with explicit entry points;
- automated tests focus on reusable functions and a small deterministic demonstration;
- larger scientific refactors should only be made when they improve reproducibility or maintainability without obscuring the published workflow.

## Recommended entry point for a quick evaluation

For a small, self-contained demonstration that does not require the original ADS-B
dataset, run:

```bash
python examples/run_demo.py
```

This demonstration uses synthetic multivariate trajectories and the same
`TimeSeriesKMeans` / DTW clustering family used in the research code.

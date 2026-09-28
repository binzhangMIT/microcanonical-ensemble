# Statistical Mechanics with a 2-D Lennard-Jones System

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GITHUB_USERNAME/statmech-lj-ensemble/HEAD?labpath=Statistical_Mechanics_Ensemble_Project.ipynb)

This repository is a computational experiment for a statistical mechanics
course. It uses one microscopic model—a two-dimensional Lennard-Jones particle
system—to explore microstates, macrostates, time averages, probability
distributions, ensemble averages, the microcanonical ensemble, fluctuations,
and sampling.

## Launch it

The Binder badge above assumes that the repository will be hosted as
`GITHUB_USERNAME/statmech-lj-ensemble`. Replace `GITHUB_USERNAME` with the
GitHub account or organization that owns the repository, and change the
repository name in the URL if needed. Binder will open the notebook directly
in JupyterLab; no local installation or special hardware is required.

## Contents

- `Statistical_Mechanics_Ensemble_Project.ipynb` — the student-facing experiment
- `lj_simulation.py` — the hidden-in-practice simulation machinery
- `requirements.txt` — the minimal Binder environment

The simulation uses reduced Lennard-Jones units, periodic boundaries, a
truncated-and-shifted potential, velocity-Verlet integration, and NVE
production dynamics. The default settings are deliberately modest for a
CPU-only Binder session. The notebook includes optional, clearly labeled
extensions for longer trajectories and finite-time sampling.

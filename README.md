# ruscs

**Rumor Spreading Complex Networks Simulations**

A Python package for simulating rumor spreading on networks using the Maki-Thompson (MT) model.

![](docs/images/ruscs_init.png)

## Overview

`ruscs` provides tools for modeling and analyzing how rumors spread. The package implements the Maki-Thompson model with support for quasi-stationary dynamics, localized spreading, and choice-based propagation mechanisms on arbitrary network topologies.

## Features

- **Multiple spreading models**: Implements Maki-Thompson model using the Gillespie algorithm
- **Quasi-stationary dynamics**: Support for QS approximations for efficient simulation
- **Network flexibility**: Works with arbitrary network topologies via NetworkX
- **Localized spreading**: Model localized rumor propagation on networks
- **Choice-based dynamics**: Simulate selective information sharing
- **Easy configuration**: YAML-based configuration for experiments
- **Logging and utilities**: Built-in logging and helper functions for analysis

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/evarifa/ruscs.git
cd ruscs
```

### Requirements

- Python >= 3.8
- numpy
- networkx
- pyyaml
- statsmodels
- scikit-learn

## Quick Start

### Running a Simple Simulation

```python
import numpy as np
from ruscs.mt_density import mt_density

# Basic Network Initialization
neighbors, kmax = initialize_network_RRG_list(500, 4)

# Run simulation
result = mt_density(
    rate = 0.3,
    alpha = 0.5,
    delta = 0.1,
    maxiter = 1000,
    neighborz = neighbors,
    kmax = kmax,
)
```

## Code Structure

### Main Functions

- **`initialize_network_RRG_list(N, k)`**: Create a graph
- **`mt_process_update(...)`**: Core MT model update step
- **`mt_process_update_pulse(...)`**: Update step with pulse dynamics
- **`mt_density(...)`**: Run MT simple simulation
- **`mt_density_dynamic_qs(...)`**: MT with quasi-stationary dynamics
- **`mt_density_dynamic_qs_pulse_choice(...)`**: QS with added spreaders based on choice
- **`mt_density_dynamic_qs_pulse_loc(...)`**: QS with added spreaders based on localization

### Utilities

- **`init_logging(...)`**: Configure logging for simulations

### Scripts
- **`Sim_MT_density_pulse_localized.py`**: Run MT density simulation with localized new spreaders
- **`Sim_MT_lifetime_pulse_nearseed.py`**: Run MT lifetime simulation with near-seed pulse

## Examples

Check the `notebooks/` directory for Jupyter notebooks demonstrating:
- MT density dynamics analysis (`MT_density_QS.ipynb`)
- Activity pattern analysis (`analyze_activity.ipynb`)

## Citation

If you use `ruscs` in your research, please cite:

```bibtex
@article{2bh8-p6rx,
  title = {Virality detection and control strategies in rumor models},
  author = {Rifà, Eva and Vicens, Julian and Cozzo, Emanuele},
  journal = {Phys. Rev. Res.},
  pages = {--},
  year = {2026},
  month = {Feb},
  publisher = {American Physical Society},
  doi = {10.1103/2bh8-p6rx},
  url = {https://link.aps.org/doi/10.1103/2bh8-p6rx}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

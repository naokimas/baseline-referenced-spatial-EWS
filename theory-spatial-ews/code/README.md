# `code/`

The scripts behind Fig. 1 of the manuscript. The full documentation of this repository is in the
[top-level README](../README.md). This file is only a map of the folder.

| File | Role |
| --- | --- |
| `generate_equilibria.py` | **Stage 1.** Runs the ten simulations (5 networks × 2 control parameters) and writes `data/equilibria/`. Needs `numba`. A few minutes. |
| `plot_ews_examples.py` | **Stage 2.** Computes the five spatial EWSs from the stage-1 snapshots and writes Fig. 1 to `figures/fig_ews_examples.{png,pdf}`, creating that directory. Runs in seconds; needs no simulation. |
| `common/model_parameters.py` | The constants of the coupled double-well dynamics, Eq. (2), and the value at which the non-swept control parameter is held. Numbers only, no code. |
| `common/sde_simulator.py` | The Euler–Maruyama integrator. Every state in `data/equilibria/` came out of `solve_in_range()` here. |
| `common/home_range.py` | The home range: the contiguous control values before the first node leaves the upper state. Imported by both stages, so the definition exists once. |

Because the output of stage 1 is stored in `data/equilibria/`, running stage 2 alone reproduces
Fig. 1 exactly.

Both scripts read the adjacency matrices from `data/networks/`, and the four empirical ones are not
redistributed here; see "Data availability" in the [top-level README](../README.md) before running
either stage.

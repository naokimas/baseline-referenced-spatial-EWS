# Code and data for Fig. 1 of "A theory of spatial early warning signals for tipping points on complex networks"

This repository contains everything behind **Fig. 1**, the only figure produced by numerical
simulation in the manuscript

> Naoki Masuda,
> **"A theory of spatial early warning signals for tipping points on complex networks"**,
> arXiv:2608.15476 (2026). <https://arxiv.org/abs/2608.15476>

The figure shows five classical spatial early warning signals
(EWSs) for the coupled double-well dynamics on five networks. Section S2 of
the Supplementary Material of the paper describes the same simulations in prose.

The code is an adaptation of that used in our prior paper whose code is stored and explained in the parent folder of this one.

## Data availability

Fig. 1 uses five networks: one synthetic and four empirical. The empirical networks are not ours.
They were collected and published by other researchers, and are distributed by public repositories
under those owners' terms. Therefore, we do not redistribute them here, so the four corresponding
adjacency-matrix files are absent from `data/networks/`, which contains only the square lattice that
we generated ourselves.

**What is included.** All ten equilibrium runs, i.e., the entire output of stage 1, are in
`data/equilibria/`, for the empirical networks as well as for the lattice. These are our own
simulation output, and they are the numbers behind every curve of Fig. 1.

**Where to get the four empirical networks.** We obtained them from one of two public collections,
KONECT (<http://konect.cc/networks/>) and Netzschleuder (<https://networks.skewed.de/>). Section S2
of the Supplementary Material of the paper describes each network and cites the original study that
collected the data. The `data/networks/README.md` of the parent repository lists the same networks
with the preprocessing we applied (largest connected component; multiple edges, self-loops, edge
directions, and weights removed), the file format the scripts expect (a `scipy.sparse` CSR matrix
written with `scipy.sparse.save_npz`: `float64`, symmetric, binary, zero diagonal), and a snippet
that builds such a file from an edge list.

**Consequence for running the code.** Save each downloaded network under the file name given in the
table below, in `data/networks/`. Until then, both stages stop with a `FileNotFoundError` at the
first network they cannot find: stage 2 loads every network in turn, because Moran's $I$ needs the
adjacency matrix, so the lattice panels alone cannot be drawn. Nothing else has to be adjusted; with
the four files in place, both stages run exactly as described below.

## Quick start

The pipeline has two stages:

```bash
python3 code/generate_equilibria.py   # data/simulation_parameters.csv, data/networks/  ->  data/equilibria/
python3 code/plot_ews_examples.py     # data/equilibria/, data/networks/                ->  figures/fig_ews_examples.{png,pdf}
```

The output of stage 1 is available on this repository. So, once the four empirical networks are in
place (see above), stage 2 alone reproduces Fig. 1 and takes a few seconds; it writes the figure
into `figures/`, creating that directory if it does not exist.
Stage 1 regenerates the simulation data from scratch and takes a few minutes, dominated by the US
power grid, the largest of the five networks; it accepts an optional substring argument
(`python3 code/generate_equilibria.py lattice`) to rerun only some of the ten runs.

Requirements: Python 3.9 or later, `numpy`, `scipy`, `matplotlib`, and `numba` (the last needed by
stage 1 only). The committed output was produced with Python 3.9.6, numpy 2.0.2, scipy 1.13.1,
matplotlib 3.9.4, and numba 0.60.0. `ROOT` in every script is the repository root and all paths are
resolved relative to it, so the scripts can be run from any working directory.

## Layout

```
code/
  common/
    model_parameters.py   constants of the coupled double-well dynamics; no simulation code
    sde_simulator.py      numba Euler-Maruyama integrator; produced all of data/equilibria/
    home_range.py         the home range: the control values before the first node tips
  generate_equilibria.py  stage 1
  plot_ews_examples.py    stage 2
data/
  networks/lattice.npz             the square lattice (scipy sparse, undirected, unweighted); the
                                   4 empirical adjacency matrices are NOT included -- see above
  simulation_parameters.csv        one row per run: simulation range [far, near] and time step
  equilibria/<net>-<cparam>.npy    100 x N equilibrated node states, one row per control value
  equilibria/<net>-<cparam>.json   metadata: control values, home-range length, RNG seed, etc.
  equilibria/manifest.csv          the ten runs in one table
README.md                          this file
code/README.md                     a map of code/, pointing back here
```

Stage 2 creates the `figures/` directory and writes `fig_ews_examples.png` and
`fig_ews_examples.pdf` into it. Those two files are the figure as it appears in the manuscript and
are deliberately **not** stored in this repository: they are outputs, fully determined by the
committed data and by `code/plot_ews_examples.py`.

Every run uses the coupled double-well dynamics and sweeps its control parameter downward, so a run
is identified by the network and the control parameter alone: `<net>-<cparam>`, with `cparam` being
`D` or `u`. There are 5 networks × 2 control parameters = 10 runs.

## What is simulated

Stage 1 accepts a run only if the noise-free system remains in the upper state at the far end of
the range, no state diverges, and at least 30 of the 100 control values precede the tipping point.
These are the three checks reported in Section S2.

## How to check the code against the manuscript

| Manuscript | Code |
| --- | --- |
| Eq. (2), the coupled double-well dynamics | `common/sde_simulator.py`, the `drift` line of `_run_doublewell()`; constants in `common/model_parameters.py` |
| Eq. (17), $V$ | `plot_ews_examples.py`, `ews_curve(..., "var")` (`np.var(..., ddof=1)`, i.e., the $1/(N-1)$ normalization) |
| Eq. (18), CV | `ews_curve(..., "cv")` |
| Eq. (19), $g_1$ | `ews_curve(..., "skew")`; the figure plots $g_1' = -g_1$, hence the minus sign |
| Eq. (20), $g_2$ | `ews_curve(..., "kurt")` (`fisher=False`, so no $-3$ shift) |
| Eq. (21), $I_{\mathrm M}$ | `moran()` in `plot_ews_examples.py` |
| S2, numerical method | `generate_equilibria.py`; ranges and $\Delta t$ in `data/simulation_parameters.csv` |
| S2, home range | `common/home_range.py`; the resulting lengths are the `home_len` column of `data/equilibria/manifest.csv` |
| S2, networks | `data/networks/`; see the table below |

## Networks

All five are used in both rows of Fig. 1. Each empirical network is the largest connected component
of the original data, treated as undirected and unweighted, with multiple edges and self-loops
removed.

The $N$ and $\lvert E\rvert$ below are the values after this preprocessing, and the file name is the
code name that `data/simulation_parameters.csv`, the equilibrium file names, and the scripts all use
to refer to a network.

| File | Included? | Network | $N$ | $\lvert E\rvert$ | Source |
| --- | --- | --- | ---: | ---: | --- |
| `lattice.npz` | yes | Square lattice, $10\times10$, periodic boundaries, 4 neighbors per node | 100 | 200 | generated by us |
| `montreal.npz` | no | Montreal street gangs | 29 | 75 | Descormiers & Morselli, *Int. Crim. Justice Rev.* **21**, 297 (2011) |
| `jazz.npz` | no | Jazz musician collaborations | 198 | 2742 | Gleiser & Danon, *Adv. Complex Syst.* **6**, 565 (2003) |
| `metabolic.npz` | no | *C. elegans* metabolic network | 453 | 2025 | Jeong et al., *Nature* **407**, 651 (2000) |
| `powergrid.npz` | no | US Western States power grid | 4941 | 6594 | Watts & Strogatz, *Nature* **393**, 440 (1998) |

# Code and data for Fig. 1 of "A theory of spatial early warning signals for tipping points on complex networks"

Naoki Masuda (University of Michigan; Kobe University), 2026.

This repository contains everything behind **Fig. 1**, the only figure produced by numerical
simulation in the manuscript. The figure shows the five classical spatial early warning signals
(EWSs) — the spatial variance $V$, coefficient of variation CV, sign-adjusted skewness $g_1' = -g_1$,
kurtosis $g_2$, and Moran's $I_{\mathrm M}$, i.e., Eqs. (17)–(21) of the manuscript — for the coupled
double-well dynamics, Eq. (2), on five networks, as the system is driven toward a tipping point by
a descending coupling strength $D$ (top row) or a descending stress $u$ (bottom row). Section S2 of
the Supplementary Material describes the same simulations in prose; this README is its
computational counterpart.

The rest of the manuscript is analytical, so nothing else here is needed to reproduce the paper.

## Quick start

The pipeline has two stages:

```bash
python3 code/generate_equilibria.py   # data/simulation_parameters.csv, data/networks/  ->  data/equilibria/
python3 code/plot_ews_examples.py     # data/equilibria/, data/networks/                ->  figures/fig_ews_examples.{png,pdf}
```

The output of stage 1 is committed, so **stage 2 alone reproduces Fig. 1** and takes a few seconds;
it writes the figure into `figures/`, creating that directory if it does not exist.
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
  networks/<net>.npz               5 adjacency matrices (scipy sparse, undirected, unweighted)
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

The model is Eq. (2) of the manuscript,

$$\mathrm{d}x_i = \Big[-(x_i-r_1)(x_i-r_2)(x_i-r_3) + D\sum_j A_{ij}x_j + u\Big]\mathrm{d}t + \sigma\,\mathrm{d}W_i,$$

with $r_1 = 1$, $r_2 = 3$, $r_3 = 5$, and noise standard deviation $\sigma = 0.1$
(`common/model_parameters.py`). When $D$ is swept, $u = -5$ is fixed; when $u$ is swept,
$D = 0.05$ is fixed.

The protocol, matching Section S2, is: take 100 equally spaced values of the control parameter
spanning the simulation range in `data/simulation_parameters.csv`; at each value integrate the
dynamics by Euler–Maruyama with time step $\Delta t$ from the uniform initial condition $x_i(0)=5$
to the final time $T = 50$; and record the single snapshot $\{x_1(T),\dots,x_N(T)\}$. The 100
control values of a run are 100 **independent** simulations, not one continuous sweep, so each
snapshot is an independently equilibrated sample. The EWSs are then computed from each snapshot
over the home range only, i.e., the contiguous run of control values before the first one at which
any node has left the upper state ($x_i > r_2 = 3$); see `common/home_range.py`.

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

The home-range lengths quoted in Section S2 (88, 91, 79, 78, 82 under descending $D$ and 95, 81,
81, 79, 78 under descending $u$) are exactly the `home_len` column of `manifest.csv`.

## Networks

All five are used in both rows of Fig. 1. Each empirical network is the largest connected component
of the original data, treated as undirected and unweighted, with multiple edges and self-loops
removed.

| File | Network | $N$ | $\lvert E\rvert$ | Source |
| --- | --- | ---: | ---: | --- |
| `lattice.npz` | Square lattice, $10\times10$, periodic boundaries, 4 neighbors per node | 100 | 200 | synthetic |
| `montreal.npz` | Montreal street gangs | 29 | 75 | Descormiers & Morselli, *Int. Crim. Justice Rev.* **21**, 297 (2011) |
| `jazz.npz` | Jazz musician collaborations | 198 | 2742 | Gleiser & Danon, *Adv. Complex Syst.* **6**, 565 (2003) |
| `metabolic.npz` | *C. elegans* metabolic network | 453 | 2025 | Jeong et al., *Nature* **407**, 651 (2000) |
| `powergrid.npz` | US Western States power grid | 4941 | 6594 | Watts & Strogatz, *Nature* **393**, 440 (1998) |

### Why these five

1. **The same five networks appear in both rows**, so that a column isolates the effect of the
   control parameter: (a) and (f) are the same network under $D$ and under $u$, and so on.
2. **The square lattice is (a)/(f).** Being vertex-transitive, it is the reference case against
   which the heterogeneous networks are read, and it is the case covered by Theorem 5.1.

The remaining four (ordered ascending in $N$) were chosen from the 40 networks of the companion
numerical study so as to make the behavior of the five EWSs as diverse as possible. They were
picked by an explicit criterion rather than by eye: for every network, a 20-dimensional feature
vector was formed from Kendall's tau of each EWS against the control value, over the whole home
range and over each of its halves, for both control parameters; the four networks were then chosen
to maximize the minimum pairwise Euclidean distance in that space, with the lattice fixed in the
set and the candidate pool restricted to the empirical networks that have a citable primary
reference. The half-range taus are what let the criterion distinguish flat and non-monotonic curves
from monotone ones.

The result covers monotone increase, monotone decrease, noise-dominated flatness, non-monotonic
curves, and sign reversals of the same EWS between the $D$ and the $u$ column — which is the point
Fig. 1 makes.

## Reproducibility of the noise

Each run draws its noise from its own block of RNG seeds, derived from the run's identity with
`zlib.crc32` (not the built-in `hash()`, which is salted per process), and each of the 100 control
values within a run uses its own seed inside that block; see the docstrings of
`generate_equilibria.py` and `common/sde_simulator.py`. The noise is therefore independent across
networks, control parameters, and control values, and identical from machine to machine. Every run
records its base seed in its `.json` and in `manifest.csv`.

Two caveats on exact bit-level reproduction:

* **Stage 2 is exact.** Run on the committed data, `plot_ews_examples.py` reproduces the PNG of the
  figure published with the manuscript byte for byte (the PDF differs only in its embedded creation
  date, which matplotlib stamps at write time).
* **Stage 1 is exact only up to floating-point rounding.** The numba kernel is compiled with
  `fastmath=True`, so a different numba/LLVM build may reorder floating-point operations. Rerunning
  stage 1 with the recorded seeds reproduces the committed states to within an absolute difference
  of about $10^{-4}$ (states of order 5, after up to 50,000 time steps), which leaves all ten
  home-range lengths unchanged and the EWS curves visually identical. Within one installation the
  result is bit-identical from run to run.

## Relation to the companion numerical study

The protocol above follows the companion numerical study (Bandara, Yu and Masuda, cited as
`Bandara2026arxiv` in the manuscript), and parts of the code here are adapted from that study's
code. The data in this repository are nevertheless our own: they were generated by the code here,
with seeds independent of that study's, so the individual curves differ from those of its Fig. 1
even for the networks that the two figures have in common. Only a single dynamical system, a single
sweep direction, and five networks are involved here, whereas that study covers four dynamical
systems, ten combinations of control parameter and direction, and 40 networks.

## Citing

If you use this code or these data, please cite the manuscript:

> N. Masuda. A theory of spatial early warning signals for tipping points on complex networks
> (2026).

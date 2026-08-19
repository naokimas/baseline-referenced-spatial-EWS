# Networks

The study uses 40 networks: 34 empirical and 6 synthetic.

## The 34 empirical networks are not distributed here

The empirical network data are not our data. We therefore do not redistribute them in this repository.

**Where to get them.** We obtained all 34 from one of two public collections:

- **KONECT** — <http://konect.cc/networks/>
- **Netzschleuder** — <https://networks.skewed.de/>

Table S1 of the Supplementary Material of the paper gives, for every network, a description of what its
nodes and edges represent, together with the citation of the original study that collected
the data.

**Preprocessing we applied.** For each downloaded network we
1. extracted the largest connected component,
2. removed multiple edges and self-loops, and
3. discarded edge directions and weights,

giving an undirected, unweighted, simple graph. The $N$ (the number of nodes) and $M$ (the number of edges) in the table below are
the values after this preprocessing.

**File format.** Each `<network>.npz` is a SciPy sparse adjacency matrix written with
`scipy.sparse.save_npz`, in CSR format, `float64`, symmetric, binary (entries 0 or 1),
with a zero diagonal. To rebuild one from an edge list:

```python
import numpy as np, scipy.sparse as sp
# edges: array of shape (M, 2), 0-based node indices of the preprocessed graph
i, j = edges[:, 0], edges[:, 1]
N = int(edges.max()) + 1
A = sp.csr_matrix((np.ones(2 * len(edges)),
                   (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(N, N))
A.data[:] = 1.0                      # binary
A.setdiag(0); A.eliminate_zeros()    # no self-loops
sp.save_npz("data/networks/<network>.npz", A.tocsr())
```

The file name must be the code name in the first column below: the simulation scripts,
`data/simulation_parameters.csv`, and the equilibrium data file names all refer to a
network by that code name.

| file (not included) | name in Table S1 | $N$ | $M$ |
|---|---|---:|---:|
| `montreal.npz` | Montreal | 29 | 75 |
| `chesapeake.npz` | Chesapeake | 39 | 170 |
| `windsurfers.npz` | Windsurfer | 43 | 336 |
| `contigusa.npz` | Geographic | 49 | 107 |
| `catlins.npz` | Catlins | 59 | 110 |
| `dolphin.npz` | Dolphin | 62 | 159 |
| `train_terrorists.npz` | Terrorist | 64 | 243 |
| `iceland.npz` | Contact | 75 | 114 |
| `drug.npz` | Drug interaction | 75 | 181 |
| `canton.npz` | Canton | 109 | 717 |
| `genefusion.npz` | Gene fusion | 110 | 124 |
| `adjnoun.npz` | Word | 112 | 425 |
| `football.npz` | Football | 115 | 613 |
| `physician_trust.npz` | Physician | 117 | 465 |
| `students.npz` | Student | 141 | 256 |
| `protein.npz` | Protein | 161 | 209 |
| `email_company.npz` | Email | 167 | 3250 |
| `ug_village.npz` | Village | 187 | 431 |
| `jazz.npz` | Jazz player | 198 | 2742 |
| `flamingo.npz` | Flamingo software | 228 | 491 |
| `ecoli.npz` | E. coli | 328 | 456 |
| `london_transport.npz` | Transportation | 369 | 430 |
| `netsci.npz` | Coauthorship | 379 | 914 |
| `wiki_ht.npz` | Wikipedia user | 404 | 734 |
| `proximity.npz` | Proximity | 410 | 2765 |
| `metabolic.npz` | C. elegans metabolic | 453 | 2025 |
| `gap_junction_herm.npz` | C. elegans neuronal | 460 | 1432 |
| `yeast.npz` | S. cerevisiae | 664 | 1065 |
| `SITC.npz` | Product | 774 | 1779 |
| `jung_c.npz` | JUNG software | 879 | 2047 |
| `urv_email.npz` | U. Rovira i Virgili | 1133 | 5451 |
| `powergrid.npz` | US power grid | 4941 | 6594 |
| `routeviews.npz` | Route views | 6474 | 12572 |
| `erdos.npz` | Erdős collaboration | 6927 | 11850 |

## The 6 synthetic networks are included

We generated these synthetic networks ourselves, so they are committed here. Section S1 of the Supplementary
Material of the paper gives the generative model and its parameters
for each. The five random-graph
models were generated with `igraph`, except HK, which used NetworkX; the lattice is a
$10 \times 10$ square lattice with periodic boundary conditions.

Because the instances below are single random realisations, regenerating them with a different seed would give different
networks and hence different numbers from the ones in the paper.

| file (included) | name in Table S1 | $N$ | $M$ |
|---|---|---:|---:|
| `gkk.npz` | GKK | 96 | 300 |
| `barabasialbert.npz` | BA | 100 | 197 |
| `erdosrenyi.npz` | ER | 100 | 249 |
| `hk100.npz` | HK | 100 | 196 |
| `lattice.npz` | Lattice | 100 | 200 |
| `smallworld.npz` | WS | 100 | 400 |
## What still runs without the empirical networks

`data/results/*.csv` — every numerical result reported in the paper — is committed, and it
already aggregates over all 40 networks. Figures 3–6 and Figs S1–S4 are drawn from those
CSVs alone and therefore reproduce exactly as published. See "Data availability" and
"What you can reproduce" in the top-level `../../README.md`.

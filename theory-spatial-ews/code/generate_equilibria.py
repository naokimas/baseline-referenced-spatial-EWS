"""Stage 1: generate the equilibrium data behind Fig. 1 of the manuscript.

The protocol is the one described in Section S2 of the Supplementary Material: 100 equally
spaced values of the control parameter, one independent Euler-Maruyama simulation per value
from the uniform upper initial condition to the final time T = 50, and one recorded snapshot
per value.  Concretely, for each of the five networks and each of the two control parameters
(D and u), both swept downward:

  * load the adjacency matrix from data/networks/<net>.npz,
  * read the simulation range [far, near] and the time step from data/simulation_parameters.csv,
  * simulate the 100 control values with Euler-Maruyama (the numba kernel of
    code/common/sde_simulator.py; sigma = 0.1 is the noise standard deviation; u = -5 when D is
    swept, D = 0.05 when u is swept),
  * check that the run is usable, i.e. finite everywhere and leaving a well-populated home range,
  * save the 100 equilibrated snapshots and the metadata.

Output: data/equilibria/<net>-<cparam>.npy (100 x N) + <net>-<cparam>.json, and
        data/equilibria/manifest.csv, which summarizes the ten runs in one table.

Run:  python3 code/generate_equilibria.py [substr]     # all ten runs, or those whose label
                                                       # contains <substr>
The largest network (US power grid, N = 4941, dt = 0.001) dominates the running time.
"""
import os, sys, json, csv, time, zlib
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "common"))
import sde_simulator as F                  # the Euler-Maruyama integrator (numba kernel)
from model_parameters import PARAMS, FIXED  # model constants only; no simulation code
from home_range import home_range

NETDIR = os.path.join(ROOT, "data", "networks")
CSVIN = os.path.join(ROOT, "data", "simulation_parameters.csv")
OUT = os.path.join(ROOT, "data", "equilibria")
LMAX, T = 100, 50.0        # number of control values per run; integration time per control value
HOME_MIN = 30              # a run is accepted only if at least this many control values precede the tip

# Each run gets its own block of RNG seeds, so that the noise driving one network is independent of
# that driving every other network and control parameter. Control value l of a run uses base + l
# (see sde_simulator.py), so a run occupies the block [base, base + LMAX) and the stride must exceed
# LMAX. The seed is derived with crc32 rather than the built-in hash(), which is salted per process;
# crc32 makes each run's noise reproducible on any machine. Each run records its seed in its .json.
SEED_STRIDE = 1000
SEED_SLOTS = 4_000_000
SEED_KEY = "spatial-ews-theory"    # this study's own seeds, independent of any other study's
# The noise streams are thus identical on any machine. The resulting states are reproduced only up
# to floating-point rounding, because the numba kernel is compiled with fastmath=True and a
# different numba/LLVM build may reorder floating-point operations; see the README. Regenerating
# the committed data reproduces it to within about 1e-4 in the node states and leaves all ten
# home-range lengths, and hence Fig. 1, unchanged.


def run_seed(net, cparam):
    """Base RNG seed for one run, derived from its identity."""
    return 1 + SEED_STRIDE * (zlib.crc32(f"{SEED_KEY}:{net}-{cparam}".encode()) % SEED_SLOTS)


def run_one(net, cparam, far, near, dt, verbose=True):
    """Simulate one (network, control parameter) pair and write its .npy and .json."""
    seed = run_seed(net, cparam)
    A = sp.load_npz(os.path.join(NETDIR, net + ".npz")).tocsr(); N = A.shape[0]
    cvals = np.linspace(far, near, LMAX)

    # The far end of the range must lie inside the stable region, i.e. the noise-free system must
    # stay in the upper state there. Otherwise the run would start past the transition and have no
    # home range at all.
    x_far = F.solve_in_range(A, [cvals[0]], cparam, dt, seed=seed, T=T, sigma=0.0)[0]
    if not (np.all(np.isfinite(x_far)) and x_far.min() > PARAMS['basin']):
        raise RuntimeError(f"{net}-{cparam}: the far end {cvals[0]:g} is not in the stable region")

    M = F.solve_in_range(A, cvals, cparam, dt, seed=seed, T=T)
    nonfinite = int((~np.isfinite(M)).sum())
    hl = len(home_range(M))
    if nonfinite:
        raise RuntimeError(f"{net}-{cparam}: {nonfinite} non-finite states; reduce deltaT")
    if not (HOME_MIN <= hl <= LMAX - 2):
        raise RuntimeError(f"{net}-{cparam}: home range of {hl} of {LMAX} control values is "
                           f"outside [{HOME_MIN}, {LMAX - 2}]; re-place the range in "
                           f"data/simulation_parameters.csv")

    label = f"{net}-{cparam}"
    meta = dict(label=label, model="doublewell", network=net, cparam=cparam, direction="down",
                N=N, sigma=PARAMS['sigma'], fixed_param=FIXED[cparam], dt=dt, T=T,
                far=far, near=near, basin=PARAMS['basin'], home_len=hl, nonfinite=nonfinite,
                seed=seed, cparam_vals=list(np.round(cvals, 8)))
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, label + ".npy"), M.astype(np.float64))
    json.dump(meta, open(os.path.join(OUT, label + ".json"), "w"))
    if verbose:
        print(f"  {label:16s} N={N:<5d} dt={dt:<6g} home={hl:<3d} tip at {cvals[hl]:.4g}", flush=True)
    return meta


MANIFEST_COLS = ["label", "network", "cparam", "N", "dt", "far", "near", "home_len", "seed"]

if __name__ == "__main__":
    rows = list(csv.DictReader(open(CSVIN)))
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = [r for r in rows if filt in f"{r['network']}-{r['cparam']}"]
    # smallest network first: the cheap runs warm up the JIT compilation
    todo.sort(key=lambda r: sp.load_npz(os.path.join(NETDIR, r["network"] + ".npz")).shape[0])
    seeds = {(r["network"], r["cparam"]): run_seed(r["network"], r["cparam"]) for r in rows}
    assert len(set(seeds.values())) == len(seeds), "RNG seed collision between runs"
    assert min(np.diff(sorted(seeds.values()))) >= LMAX, "RNG seed blocks overlap"

    print(f"Running {len(todo)} of {len(rows)} simulations -> {OUT}", flush=True)
    t0 = time.time(); metas = []
    for r in todo:
        metas.append(run_one(r["network"], r["cparam"], float(r["far"]), float(r["near"]),
                             float(r["deltaT"])))
    # The manifest is rebuilt by scanning the output directory, so that it stays complete when only
    # a subset of the runs is regenerated through the optional substring argument.
    written = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT))
               if f.endswith(".json")]
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(MANIFEST_COLS)
        for m in sorted(written, key=lambda m: m["label"]):
            w.writerow([m[k] for k in MANIFEST_COLS])
    print(f"Done in {time.time()-t0:.0f}s.", flush=True)

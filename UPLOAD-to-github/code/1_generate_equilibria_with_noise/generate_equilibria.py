"""Driver: generate the equilibrium data for all conditions with a TIPPING GUARANTEE.

For each (model, network, control-param, direction):
  * load the adjacency matrix from data/networks/<net>.npz,
  * read the simulation range [far, near] and the time step dt from data/simulation_parameters.csv,
  * simulate 100 control values (Euler-Maruyama via the numba kernels of
    code/common/sde_simulator.py; sigma = standard deviation; double-well sigma=0.1; u=-5 for
    descending-D double-well/mutualistic),
  * GUARANTEE a usable transition: the home range (the control values for which all nodes are still
    in their initial state) must be a prefix of at least HOME_MIN = 30 of the 100 control values and
    must end inside the range; otherwise the range is retuned and, if necessary, re-placed from the
    observed tip until that holds (see valid() and _refine_range()).

Numerical policy:
  * N < 1000 (36 smaller networks): use the dt and range given in the CSV; retune the range only
    if no transition is present.
  * N >= 1000 (4 large networks): CSV dt=0.01 is numerically unstable (explicit-Euler overshoot
    -> NaN or spurious collapse). Pick dt by convergence at the stiffest (far) end, set far to
    the minimal value holding the initial state, locate the first tip, and place the range so the
    tip sits at ~85% (home range ~85 of 100 control values).

Output: data/equilibria_with_noise/<model>-<net>-<cparam>-<dir>.npy (100 x N) + .json metadata,
        and data/equilibria_with_noise/_manifest.csv.

Run:  python3 code/1_generate_equilibria_with_noise/generate_equilibria.py [substr]
      # all 400 cases, or only those whose label contains <substr>
"""
import os, sys, json, csv, time, glob, zlib
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "code", "common"))
import sde_simulator as F                        # the Euler-Maruyama integrator (numba kernels)
from model_parameters import PARAMS, fixed_value  # model constants only; no simulation code

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(os.path.dirname(HERE))
NETDIR = os.path.join(ROOT, "data", "networks")
CSV    = os.path.join(ROOT, "data", "simulation_parameters.csv")
OUT    = os.path.join(ROOT, "data", "equilibria_with_noise")
LMAX, T, BIG = 100, 50.0, 1000
DT_FLOOR, CONV_TOL = 1.0e-4, 0.03

CONDITIONS = [("doublewell","D","up"),("doublewell","D","down"),("doublewell","u","up"),
              ("doublewell","u","down"),("SIS","D","up"),("SIS","D","down"),
              ("mutualistic","D","down"),("mutualistic","u","down"),
              ("genereg","D","down"),("genereg","u","down")]

def home_len(M, basin, direction):
    in_init = (M.max(1) < basin) if direction == "up" else (M.min(1) > basin)
    bad = np.where(~in_init)[0]
    return int(bad[0]) if bad.size else len(M)

HOME_MIN = 30          # a range is only accepted if at least this many control values precede the tip
HOME_TARGET = 0.85     # target position of the tipping point within the range
REFINE_MAX = 4         # re-placement attempts driven by the observed (noisy) tip

def valid(M, basin, direction):
    """Is this range usable? It must be finite everywhere and leave a well-populated home range.

    At least HOME_MIN of the LMAX control values must precede the tipping point. The EWSs are all
    estimated from the states at those control values, so this bound sets how much data each of the
    400 simulation conditions contributes, and keeps that amount comparable across conditions
    irrespective of where the transition happens to fall for a given noise realisation.
    """
    if not np.all(np.isfinite(M)):
        return False, home_len(M, basin, direction)
    hl = home_len(M, basin, direction)
    return (HOME_MIN <= hl <= LMAX - 2), hl

def _refine_range(far, near, dt, cparam, sweep, basin, direction, sigma, stable):
    """Position the range so that the tipping point of the NOISY system sits at HOME_TARGET of it.

    _place_range() works from the noise-free tipping point, which is cheap to locate. The stochastic
    system tips somewhat earlier than the deterministic one, and markedly so where the bifurcation is
    compressed into a narrow window of the control parameter, as it is for the dense hub networks.
    This routine therefore sweeps WITH noise, reads off the control value at which the system actually
    tipped, and rescales the range so that value falls at HOME_TARGET:

        near = far + (c_tip - far) / HOME_TARGET

    which places the observed tip at that fraction while leaving `far` untouched -- necessary because
    `far` may be pinned at D = 0. One or two iterations suffice: the tip location in the control
    parameter is stable across sweeps even though its index within the range is not.

    Returns (far, near, cvals, M, ok, home_len).
    """
    cvals = np.linspace(far, near, LMAX); M = sweep(cvals, dt, sigma)
    ok, hl = valid(M, basin, direction)
    for _ in range(REFINE_MAX):
        if ok:
            break
        toward = np.sign(near - far) or -1.0
        if hl >= LMAX - 1:                      # no transition in range: reach further past `far`
            near = far + toward * 1.5 * abs(near - far)
        else:
            gap = cvals[hl] - far               # signed distance from the far end to the observed tip
            if abs(gap) < 1e-12:                # tips at the very first control value: back `far` off
                far = far - toward * 0.5 * abs(near - far)
            else:
                near = far + gap / HOME_TARGET
        if cparam == "D":                       # the coupling strength cannot be negative
            far, near = max(far, 0.0), max(near, 0.0)
        g = 0
        while not stable(far, dt) and g < 20:   # keep the far end inside the stable region
            far = far - toward * 0.10 * abs(near - far)
            if cparam == "D" and far < 0:
                far = 0.0; break
            g += 1
        cvals = np.linspace(far, near, LMAX); M = sweep(cvals, dt, sigma)
        ok, hl = valid(M, basin, direction)
    return far, near, cvals, M, ok, hl

SEED_STRIDE = 1000          # must exceed LMAX, so that the seed blocks of two runs cannot overlap
SEED_SLOTS = 4_000_000      # number of distinct blocks available

def run_seed(model, net, cparam, direction):
    """Base RNG seed for one simulation condition, derived from its identity.

    Inside the integrator, control value l of a run uses `base + l` (see sde_simulator.py), so a run
    occupies the seed block [base, base + LMAX). The seeding therefore has to satisfy two properties:

      * each simulation condition gets its own block, so that the dynamical noise driving one network
        is statistically independent of that driving every other network, model and sweep direction.
        The 40 networks entering a condition average are then independent replicates.
      * blocks do not overlap, hence the stride SEED_STRIDE = 1000 > LMAX = 100 control values. Within
        a run the per-control-value seeds also differ, so the equilibria at different control values
        are independent too -- the assumption underlying the surrogate null of the sup-F test.

    The seed is derived with crc32 rather than the built-in hash(), which is salted per process; crc32
    makes each run's noise reproducible on any machine. The driver asserts that the 400 blocks in use
    are distinct and non-overlapping, and each run records its seed in the .json metadata.
    """
    key = f"{model}-{net}-{cparam}-{direction}".encode()
    return 1 + SEED_STRIDE * (zlib.crc32(key) % SEED_SLOTS)

def run_condition(model, net, cparam, direction, rows, seed=None, verbose=True):
    if seed is None:
        seed = run_seed(model, net, cparam, direction)
    P = PARAMS[model]; basin = P["basin"]
    fixed = fixed_value(model, cparam, direction)
    xinit = P["xlow"] if direction == "up" else P["xhigh"]
    A = sp.load_npz(os.path.join(NETDIR, net + ".npz")).tocsr(); N = A.shape[0]
    r = rows.get((model, cparam, direction, net))
    if r is None:
        return dict(label=f"{model}-{net}-{cparam}-{direction}", status="NO_CSV_ROW")
    far, near, dt = float(r["rng.far"]), float(r["rng.near"]), float(r["deltaT"])
    # The coupling strength D is non-negative, so the range read from the CSV is clamped at 0 here,
    # before anything else uses it. (The stress u is unrestricted in sign and is swept negative.)
    # _place_range() applies the same bound when it retunes, so no negative D can enter from either route.
    if cparam == "D":
        far, near = max(far, 0.0), max(near, 0.0)
    toward = np.sign(near - far) or -1.0
    span = abs(near - far) or 1.0

    def sweep(cvals, dt_, sigma):
        return F.solve_in_range(model, A, np.atleast_1d(cvals), cparam, fixed, xinit, dt_, T=T, sigma=sigma, seed=seed)
    def in_init(M):
        return (M.max(1) < basin) if direction == "up" else (M.min(1) > basin)
    def stable(c, dt_):
        M = sweep([c], dt_, 0.0)
        return bool(np.all(np.isfinite(M)) and in_init(M)[0])

    retuned = False
    if N < BIG:
        cvals = np.linspace(far, near, LMAX); M = sweep(cvals, dt, P["sigma"])
        ok, hl = valid(M, basin, direction)
        if not ok:
            retuned = True
            far, near = _place_range(far, dt, toward, cparam, stable, span, sweep, basin, direction)
            far, near, cvals, M, ok, hl = _refine_range(far, near, dt, cparam, sweep, basin,
                                                        direction, P["sigma"], stable)
    else:
        retuned = True
        dt = min(dt, 0.001)
        far = _ensure_stable(far, dt, toward, cparam, stable, span)
        dt = _converged_dt(far, dt, sweep)
        far = _ensure_stable(far, dt, toward, cparam, stable, span)
        far, near = _place_range(far, dt, toward, cparam, stable, span, sweep, basin, direction)
        far, near, cvals, M, ok, hl = _refine_range(far, near, dt, cparam, sweep, basin,
                                                    direction, P["sigma"], stable)

    label = f"{model}-{net}-{cparam}-{direction}"
    nbad = int((~np.isfinite(M)).sum())
    status = "OK" if ok else ("DIVERGED" if nbad else "NO_TRANSITION")
    meta = dict(label=label, model=model, network=net, cparam=cparam, direction=direction,
                sigma=P["sigma"], fixed_param=fixed, dt=dt, far=far, near=near, N=N,
                basin=basin, home_len=hl, status=status, retuned=retuned, nonfinite=nbad,
                seed=seed,                      # recorded so each run's noise can be reproduced exactly
                bparam_vals=list(np.round(cvals, 8)))
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, label + ".npy"), M.astype(np.float64))
    json.dump(meta, open(os.path.join(OUT, label + ".json"), "w"))
    if verbose:
        print(f"  {label:40s} N={N:<5d} dt={dt:<7g} home={hl:<3d} {'RT ' if retuned else '   '}{status}", flush=True)
    return meta

def _ensure_stable(far, dt, toward, cparam, stable, span):
    g = 0
    while not stable(far, dt) and g < 60:
        far -= toward * span * 0.25
        if cparam == "D" and far < 0:
            far = 0.0; break
        g += 1
    return far

def _first_tip(far, span, toward, dt, cparam, sweep, basin, direction):
    """Scan NOISE-FREE (sigma=0) from `far` toward the transition and return the first control value at
    which some node has left its initial state. Used only to place the simulation range.

    The scan widens (K = 1, 2, 4, 8 times the nominal span) until a transition is found, and never
    proposes a negative D.
    """
    for K in (1.0, 2.0, 4.0, 8.0):
        grid = far + toward * np.linspace(0, K * span, 120)[1:]
        if cparam == "D":
            grid = grid[grid >= 0]
        if grid.size == 0:
            continue
        M = sweep(grid, dt, 0.0)
        in_i = (M.max(1) < basin) if direction == "up" else (M.min(1) > basin)
        # A control value with any non-finite state counts as NOT being in the initial state, so the
        # scan stops there and the range is placed before it. This keeps the range inside the region
        # where the explicit-Euler integration is numerically well behaved. (All 400 committed runs are
        # finite throughout; each records nonfinite = 0 in its metadata.)
        in_i = in_i & np.all(np.isfinite(M), axis=1)
        bad = np.where(~in_i)[0]
        if bad.size:
            return grid[bad[0]]
    return None

def _place_range(far, dt, toward, cparam, stable, span, sweep, basin, direction):
    # locate the first tip c* by scanning from a stable anchor toward `near`
    anchor = _ensure_stable(far, dt, toward, cparam, stable, span)
    cstar = _first_tip(anchor, span, toward, dt, cparam, sweep, basin, direction)
    if cstar is None:
        return anchor, anchor
    # place c* at ~85% of a sensibly-wide range so the home range is well populated and `far`
    # sits DEEP in the stable region (not on the stability boundary)
    W = max(span, abs(cstar - far), 0.3)
    new_far  = cstar - toward * 0.85 * W
    new_near = cstar + toward * 0.15 * W
    # The coupling strength D is a non-negative quantity, so neither end of the range may go below 0.
    # (The stress u is unrestricted in sign and is deliberately swept negative.)
    if cparam == "D":
        new_far = max(new_far, 0.0)
        new_near = max(new_near, 0.0)
    g = 0
    while not stable(new_far, dt) and g < 20:    # ensure the (deep) far end is stable
        new_far -= toward * 0.10 * W
        if cparam == "D" and new_far < 0:        # keep the clamp above; do not let it be undone here
            new_far = 0.0; break
        g += 1
    return new_far, new_near

def _converged_dt(c, dt, sweep):
    x = sweep([c], dt, 0.0)[0]
    while dt > DT_FLOOR:
        x2 = sweep([c], dt / 2, 0.0)[0]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(x2)):
            rel = np.max(np.abs(x - x2) / (np.abs(x2) + 1e-9))
            if rel < CONV_TOL:
                break
        dt /= 2; x = x2
    return max(dt, DT_FLOOR)

def load_rows():
    return {(r["dynamics"], r["cparam"], r["direction"], r["network"]): r
            for r in csv.DictReader(open(CSV))}

def all_labels():
    nets = sorted(f[:-4] for f in os.listdir(NETDIR) if f.endswith(".npz"))
    return [(m, net, cp, d) for net in nets for (m, cp, d) in CONDITIONS]

MANIFEST_COLS = ["label","model","network","cparam","direction","N","dt","far","near","home_len","retuned","status"]

def write_manifest():
    """Write _manifest.csv, one row per simulation condition, by scanning every .json in OUT.

    The manifest summarizes all 400 runs in a single table: network size, time step, control-parameter
    range, home-range length and status. Scanning the output directory means it stays complete when
    only a subset of conditions is regenerated via the optional substring argument.
    """
    metas = []
    for jf in glob.glob(os.path.join(OUT, "*.json")):
        metas.append(json.load(open(jf)))
    metas.sort(key=lambda m: m["label"])
    with open(os.path.join(OUT, "_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(MANIFEST_COLS)
        for m in metas:
            w.writerow([m.get(k, "") for k in MANIFEST_COLS])
    return len(metas)

if __name__ == "__main__":
    rows = load_rows()
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = [t for t in all_labels() if filt in f"{t[0]}-{t[1]}-{t[2]}-{t[3]}"]
    # small networks first (fast; warms JIT), then large
    todo.sort(key=lambda t: sp.load_npz(os.path.join(NETDIR, t[1] + ".npz")).shape[0])
    # Every condition must occupy its own, non-overlapping block of RNG seeds.
    seeds = {t: run_seed(*t[:4]) for t in all_labels()}
    assert len(set(seeds.values())) == len(seeds), "RNG seed collision between simulation conditions"
    assert min(abs(a - b) for i, a in enumerate(sorted(seeds.values()))
               for b in sorted(seeds.values())[i + 1:i + 2]) >= LMAX, "RNG seed blocks overlap"
    print(f"Running {len(todo)} conditions -> {OUT}", flush=True)
    t0 = time.time(); results = []
    for t in todo:
        results.append(run_condition(*t, rows))
    os.makedirs(OUT, exist_ok=True)
    write_manifest()
    bad = [r for r in results if r.get("status") != "OK"]
    print(f"\nDone in {time.time()-t0:.0f}s. OK={len(results)-len(bad)}  problems={len(bad)}", flush=True)
    for r in bad:
        print("  PROBLEM:", r["label"], r.get("status"), flush=True)

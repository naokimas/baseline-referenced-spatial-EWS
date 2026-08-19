"""Euler-Maruyama integrator for the four networked SDE models (numba-JIT, CSR adjacency).

THIS module performed every stochastic simulation behind the paper: all 400 files of
`data/equilibria_with_noise/` were produced by `solve_in_range()` below, called from
`code/1_generate_equilibria_with_noise/generate_equilibria.py`. The model constants come from
`model_parameters.py`, which contains no code of its own.

There is one JIT-compiled kernel per model rather than a single generic one, because numba
specialises best on a fully concrete loop body; the four kernels are otherwise identical in
structure. Each kernel:

  * takes the adjacency in CSR form (`indptr`, `indices`), so the neighbour sum over j of A_ij x_j
    is the inner `for k in range(indptr[i], indptr[i+1])` loop -- an explicit sparse mat-vec that
    numba compiles to tight machine code;
  * runs the L control values in parallel (`prange`), each an INDEPENDENT simulation of the whole
    network from the same initial state, giving one row of the returned (L x N) matrix;
  * advances x <- x + drift*dt + sigma*sqrt(dt)*Z, with sigma the noise STANDARD DEVIATION;
  * writes into a scratch vector `xn` and swaps, so that every node is updated from the state at
    the *previous* time step (a synchronous update), not from a partially updated state;
  * clamps negatives to zero for the three models whose state is nonnegative (see `floor` in
    `model_parameters.py`); the double-well kernel has no such clamp.

`np.random.seed(seed + l)` inside the prange loop gives each control value l its own noise stream.
This is required for correctness under `parallel=True`: numba gives each thread its own random
state, so seeding per iteration makes the result independent of how iterations are scheduled
across threads, and hence reproducible.
"""
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from model_parameters import PARAMS

@njit(parallel=True, cache=True, fastmath=True)
def _run_doublewell(indptr, indices, cvals, is_D, fixedD, fixedu, xinit, dt, nsteps, sigma,
                    r1, r2, r3, seed):
    L = len(cvals); N = len(indptr) - 1
    out = np.empty((L, N)); sq = sigma * np.sqrt(dt)
    for l in prange(L):
        np.random.seed(seed + l)
        D = cvals[l] if is_D else fixedD
        u = fixedu if is_D else cvals[l]
        x = np.full(N, xinit); xn = np.empty(N)
        for _ in range(nsteps):
            for i in range(N):
                c = 0.0
                for k in range(indptr[i], indptr[i+1]):
                    c += x[indices[k]]
                drift = -(x[i]-r1)*(x[i]-r2)*(x[i]-r3) + D*c + u
                xn[i] = x[i] + drift*dt + (sq*np.random.standard_normal() if sigma > 0 else 0.0)
            x, xn = xn, x
        out[l] = x
    return out

@njit(parallel=True, cache=True, fastmath=True)
def _run_SIS(indptr, indices, cvals, is_D, fixedD, fixedu, xinit, dt, nsteps, sigma, mu, seed):
    L = len(cvals); N = len(indptr) - 1
    out = np.empty((L, N)); sq = sigma * np.sqrt(dt)
    for l in prange(L):
        np.random.seed(seed + l)
        D = cvals[l] if is_D else fixedD
        x = np.full(N, xinit); xn = np.empty(N)
        for _ in range(nsteps):
            for i in range(N):
                c = 0.0
                for k in range(indptr[i], indptr[i+1]):
                    c += x[indices[k]]
                drift = -mu*x[i] + D*(1.0 - x[i])*c
                v = x[i] + drift*dt + (sq*np.random.standard_normal() if sigma > 0 else 0.0)
                xn[i] = v if v > 0.0 else 0.0
            x, xn = xn, x
        out[l] = x
    return out

@njit(parallel=True, cache=True, fastmath=True)
def _run_genereg(indptr, indices, cvals, is_D, fixedD, fixedu, xinit, dt, nsteps, sigma,
                 B, f, h, seed):
    L = len(cvals); N = len(indptr) - 1
    out = np.empty((L, N)); sq = sigma * np.sqrt(dt)
    for l in prange(L):
        np.random.seed(seed + l)
        D = cvals[l] if is_D else fixedD
        u = fixedu if is_D else cvals[l]
        x = np.full(N, xinit); xn = np.empty(N)
        for _ in range(nsteps):
            for i in range(N):
                c = 0.0
                for k in range(indptr[i], indptr[i+1]):
                    xj = x[indices[k]]
                    c += xj**h / (1.0 + xj**h)
                drift = -B*x[i]**f + D*c + u
                v = x[i] + drift*dt + (sq*np.random.standard_normal() if sigma > 0 else 0.0)
                xn[i] = v if v > 0.0 else 0.0
            x, xn = xn, x
        out[l] = x
    return out

@njit(parallel=True, cache=True, fastmath=True)
def _run_mutualistic(indptr, indices, cvals, is_D, fixedD, fixedu, xinit, dt, nsteps, sigma,
                     B, K, C, Dt, E, H, seed):
    L = len(cvals); N = len(indptr) - 1
    out = np.empty((L, N)); sq = sigma * np.sqrt(dt)
    for l in prange(L):
        np.random.seed(seed + l)
        D = cvals[l] if is_D else fixedD
        u = fixedu if is_D else cvals[l]
        x = np.full(N, xinit); xn = np.empty(N)
        for _ in range(nsteps):
            for i in range(N):
                xi = x[i]; s = 0.0
                for k in range(indptr[i], indptr[i+1]):
                    xj = x[indices[k]]
                    s += xj / (Dt + E*xi + H*xj)
                drift = B + xi*(1.0 - xi/K)*((xi/C) - 1.0) + D*xi*s + u
                v = xi + drift*dt + (sq*np.random.standard_normal() if sigma > 0 else 0.0)
                xn[i] = v if v > 0.0 else 0.0
            x, xn = xn, x
        out[l] = x
    return out

def solve_in_range(model, A, cvals, cparam, fixed, xinit, dt, seed, T=50.0, sigma=None):
    """Integrate `model` on network `A` for every control value in `cvals`; return an (L x N) matrix.

    Row l is the state of all N nodes after integrating for time T from the uniform initial
    condition x_i = `xinit`, with the control parameter held at `cvals[l]` throughout. The L rows
    are therefore L independent simulations, not a single sweep: nothing is carried over from one
    control value to the next, so the noise in row l is independent of the noise in row l+1.

    Arguments
      model   one of 'doublewell', 'mutualistic', 'SIS', 'genereg'
      A       adjacency matrix (any scipy sparse format; converted to CSR here)
      cvals   the L values of the control parameter to simulate
      cparam  'D' or 'u' -- which parameter `cvals` refers to
      fixed   value held by the other parameter (see `fixed_value()` in model_parameters.py)
      xinit   uniform initial state, `xlow` for ascending and `xhigh` for descending runs
      dt      Euler-Maruyama time step
      seed    base seed; control value l uses seed + l (see the module docstring). Required, so that
              the caller states explicitly which noise realisation it wants.
      T       integration time; T/dt steps are taken. T=50 is long enough for the state to settle
              at every control value in the home range (the equilibria are what the EWSs are
              computed from), and is the value used for all of the published data.
      sigma   noise standard deviation; None means "use the model's own value" from PARAMS.
              Pass sigma=0 for a deterministic (noise-free) run.
    """
    P = PARAMS[model]
    sigma = P['sigma'] if sigma is None else sigma
    A = sp.csr_matrix(A)
    indptr = A.indptr.astype(np.int64); indices = A.indices.astype(np.int64)
    cvals = np.asarray(cvals, float)
    is_D = (cparam == 'D')
    fD = fixed if not is_D else 0.0
    fu = fixed if is_D else 0.0
    n = int(round(T / dt))
    if model == 'doublewell':
        return _run_doublewell(indptr, indices, cvals, is_D, fD, fu, float(xinit), dt, n, sigma,
                               P['r1'], P['r2'], P['r3'], seed)
    if model == 'SIS':
        return _run_SIS(indptr, indices, cvals, is_D, fD, fu, float(xinit), dt, n, sigma, P['mu'], seed)
    if model == 'genereg':
        return _run_genereg(indptr, indices, cvals, is_D, fD, fu, float(xinit), dt, n, sigma,
                            P['B'], P['f'], P['h'], seed)
    if model == 'mutualistic':
        return _run_mutualistic(indptr, indices, cvals, is_D, fD, fu, float(xinit), dt, n, sigma,
                               P['B'], P['K'], P['C'], P['Dt'], P['E'], P['H'], seed)
    raise ValueError(model)

"""Euler-Maruyama integrator for the coupled double-well dynamics on a network (numba-JIT, CSR).

This module performs every stochastic simulation behind Fig. 1: all of `data/equilibria/` is
produced by `solve_in_range()` below, called from `code/generate_equilibria.py`. The model
constants come from `model_parameters.py`, which contains no code of its own.

The JIT-compiled kernel:

  * takes the adjacency matrix in CSR form (`indptr`, `indices`), so the neighbor sum
    sum_j A_ij x_j is the inner `for k in range(indptr[i], indptr[i+1])` loop, which numba compiles
    to an explicit sparse matrix-vector product;
  * runs the L control values in parallel (`prange`), each an INDEPENDENT simulation of the whole
    network from the same initial state, giving one row of the returned (L x N) matrix;
  * advances x <- x + drift*dt + sigma*sqrt(dt)*Z, with sigma the noise standard deviation;
  * writes into a scratch vector `xn` and swaps, so that every node is updated from the state at
    the PREVIOUS time step (a synchronous update), not from a partially updated state.

`np.random.seed(seed + l)` inside the prange loop gives each control value l its own noise stream.
This is required for correctness under `parallel=True`: numba gives each thread its own random
state, so seeding per iteration makes the result independent of how iterations are scheduled across
threads, and hence reproducible.
"""
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from model_parameters import PARAMS


@njit(parallel=True, cache=True, fastmath=True)
def _run_doublewell(indptr, indices, cvals, is_D, fixed_D, fixed_u, xinit, dt, nsteps, sigma,
                    r1, r2, r3, seed):
    L = len(cvals); N = len(indptr) - 1
    out = np.empty((L, N)); sq = sigma * np.sqrt(dt)
    for l in prange(L):
        np.random.seed(seed + l)
        D = cvals[l] if is_D else fixed_D
        u = fixed_u if is_D else cvals[l]
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


def solve_in_range(A, cvals, cparam, dt, seed, T=50.0, sigma=None):
    """Integrate the double-well dynamics on network `A` for every control value; return (L x N).

    Row l is the state of all N nodes after integrating for time T from the uniform initial
    condition x_i = PARAMS['xhigh'], with the control parameter held at `cvals[l]` throughout. The
    L rows are therefore L independent simulations, not a single sweep: nothing is carried over
    from one control value to the next, so the noise in row l is independent of that in row l+1.

    Arguments
      A       adjacency matrix (any scipy sparse format; converted to CSR here)
      cvals   the L values of the control parameter to simulate
      cparam  'D' or 'u'; the other parameter is held at model_parameters.FIXED[cparam]
      dt      Euler-Maruyama time step
      seed    base seed; control value l uses seed + l (see the module docstring). Required, so
              that the caller states explicitly which noise realization it wants.
      T       integration time; T/dt steps are taken. T = 50 is long enough for the state to settle
              at every control value of the home range, which is what the EWSs are computed from.
      sigma   noise standard deviation; None means "use PARAMS['sigma']". Pass sigma=0 for a
              deterministic run, which `generate_equilibria.py` uses to check that the far end of
              the simulation range lies inside the stable region.
    """
    from model_parameters import FIXED
    sigma = PARAMS['sigma'] if sigma is None else sigma
    A = sp.csr_matrix(A)
    indptr = A.indptr.astype(np.int64); indices = A.indices.astype(np.int64)
    cvals = np.asarray(cvals, float)
    is_D = (cparam == 'D')
    fixed = FIXED[cparam]
    return _run_doublewell(indptr, indices, cvals, is_D,
                           0.0 if is_D else fixed, fixed if is_D else 0.0,
                           float(PARAMS['xhigh']), dt, int(round(T / dt)), sigma,
                           PARAMS['r1'], PARAMS['r2'], PARAMS['r3'], seed)

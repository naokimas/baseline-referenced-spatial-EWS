"""Constants of the coupled double-well dynamics used in Fig. 1 of the manuscript.

This module holds only numbers; it performs no simulation. The integrator that produces the
equilibrium data is `sde_simulator.py`, which imports `PARAMS` from here, so the constants used by
the simulation and by the analysis cannot drift apart.

The dynamics is Eq. (2) of the manuscript,

    dx_i = [ -(x_i-r1)(x_i-r2)(x_i-r3) + D sum_j A_ij x_j + u ] dt + sigma dW_i,

with r1 < r2 < r3, so that an isolated noise-free node has stable equilibria at x_i = r1 and
x_i = r3 and an unstable one at x_i = r2. Here `sigma` is the STANDARD DEVIATION of the noise
increment, i.e. the Euler-Maruyama step is

    x_i <- x_i + drift * dt + sigma * sqrt(dt) * Z,     Z ~ N(0, 1) per node and per time step.

Every run in this repository is DESCENDING: all nodes start in the upper state x_i = `xhigh`, and
the control parameter (D or u) is decreased until the upper state collapses. A node counts as still
being in the upper state while x_i > `basin` = r2, which is what defines the home range; see
`home_range.py`.
"""

PARAMS = dict(
    r1=1.0,        # lower stable state of an isolated noise-free node
    r2=3.0,        # unstable state in between; also the upper/lower classification threshold
    r3=5.0,        # upper stable state of an isolated noise-free node
    sigma=0.1,     # noise standard deviation (see the convention above)
    xhigh=5.0,     # uniform initial condition of every (descending) run
    basin=3.0,     # = r2: node i is in the upper state iff x_i > basin
)

# Value at which the parameter that is NOT swept is held.
#   Sweeping D: the stress is fixed at u = -5. The negative stress is required here: it destabilizes
#               the upper equilibrium, so that decreasing D produces a downward transition.
#   Sweeping u: the coupling strength is fixed at D = 0.05.
FIXED = dict(D=-5.0, u=0.05)

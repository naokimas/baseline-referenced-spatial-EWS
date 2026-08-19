"""The home range of a simulation: the control values before the first node tips.

Every analysis in this repository computes its early warning signals over the home range, so the
definition lives here once and is imported everywhere rather than repeated per script.

DEFINITION (as in the manuscript). Sweeping the control parameter gives states x_i at control values
c_0, c_1, ..., c_99, ordered from far from the transition toward it. A node counts as still being in
its initial state while x_i < basin (ascending runs) or x_i > basin (descending runs), with `basin`
taken from model_parameters.PARAMS. The home range is the CONTIGUOUS run of control values from c_0
up to but excluding the first one at which ANY node has left its initial state:

    home range = {0, 1, ..., hl-1},   hl = the smallest index at which some node has tipped.

The range is contiguous by construction: it stops at the first tipping and does not resume. This
matters because, once the transition has begun, noise can carry a node back inside its initial basin
at a later control value. Such a value lies past the tipping point and is excluded, so every early
warning signal is computed on pre-tipping data only, and a sequential alarm can never be raised at a
control value beyond the tipping point.
"""
import numpy as np
from model_parameters import PARAMS


def home_range(M, model, direction):
    """Return the home-range control-value indices as an array [0, 1, ..., hl-1].

    M         (L x N) matrix of equilibrium states: row l is all N nodes at control value l.
    model     one of 'doublewell', 'mutualistic', 'SIS', 'genereg' (selects the basin threshold).
    direction 'up' for ascending sweeps, 'down' for descending ones.

    A row also counts as tipped if it contains any non-finite value, so that a diverged simulation
    truncates the home range rather than propagating NaN into the statistics. (For the committed
    data this guard never triggers: no equilibrium file contains a non-finite value.)
    """
    basin = PARAMS[model]['basin']
    in_initial_state = np.all(np.isfinite(M), 1) & ((M.max(1) < basin) if direction == "up"
                                                    else (M.min(1) > basin))
    tipped = np.where(~in_initial_state)[0]
    hl = int(tipped[0]) if tipped.size else len(M)
    return np.arange(hl)

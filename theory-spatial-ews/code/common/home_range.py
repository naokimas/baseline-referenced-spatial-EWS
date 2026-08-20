"""The home range of a simulation: the control values before the first node tips.

Every early warning signal (EWS) in this repository is computed over the home range, so the
definition lives here once and is imported by both the generator and the plotting script rather
than being repeated.

DEFINITION (as in Section S2 of the Supplementary Material). Sweeping the control parameter downward gives
states x_i at control values c_0, c_1, ..., c_99, ordered from far from the transition toward it.
A node counts as still being in its initial (upper) state while x_i > basin = r2. The home range is
the CONTIGUOUS run of control values from c_0 up to but excluding the first one at which ANY node
has left the upper state:

    home range = {0, 1, ..., hl-1},   hl = the smallest index at which some node has tipped.

The range is contiguous by construction: it stops at the first tipping and does not resume. This
matters because, once the transition has begun, noise can carry a node back inside the upper basin
at a later control value. Such a value lies past the tipping point and is excluded, so every EWS is
computed on pre-tipping data only.
"""
import numpy as np
from model_parameters import PARAMS


def home_range(M):
    """Return the home-range control-value indices as an array [0, 1, ..., hl-1].

    M is the (L x N) matrix of equilibrium states: row l holds all N nodes at control value l.

    A row also counts as tipped if it contains any non-finite value, so that a diverged simulation
    truncates the home range rather than propagating NaN into the statistics. (For the committed
    data this guard never triggers: no equilibrium file contains a non-finite value.)
    """
    in_upper_state = np.all(np.isfinite(M), 1) & (M.min(1) > PARAMS['basin'])
    tipped = np.where(~in_upper_state)[0]
    hl = int(tipped[0]) if tipped.size else len(M)
    return np.arange(hl)

"""Stage 2: draw Fig. 1 of the manuscript from the data written by ``generate_equilibria.py``.

The figure has two rows of five panels.  Top row (a)-(e): descending coupling strength ``D``.
Bottom row (f)-(j): descending stress ``u``.  The two rows use the SAME five networks in the same
order, so that a column isolates the effect of the control parameter, and the square lattice, the
only vertex-transitive network of the five, comes first.

Each panel shows the five spatial early warning signals (EWSs) of Eqs. (17)-(21) of the manuscript,
computed from the single equilibrated snapshot recorded at each control value and plotted over the
home range only (see ``common/home_range.py``).  The horizontal axis runs from far from the
transition (left) toward it (right); the dotted vertical line marks the first control value at
which some node has left the upper state, i.e., the first value outside the home range.  Each curve
is min-max normalized to [0, 1] within its own panel, because the five EWSs have incomparable
units; the figure is therefore about the SHAPE of each curve, not its level.

No simulation is run here: this script only reads ``data/equilibria/`` and ``data/networks/`` and
is cheap to rerun.  Output: ``figures/fig_ews_examples.png`` and ``figures/fig_ews_examples.pdf``
(the latter is what the manuscript includes).  The two figure files are outputs and are not stored
in the repository, so this script creates the ``figures/`` directory itself.

Run:  python3 code/plot_ews_examples.py
"""
import os
import sys
import json

import numpy as np
import scipy.sparse as sp
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")           # no display needed; must precede the pyplot import
import matplotlib.pyplot as plt

# All paths are resolved relative to the repository root, so the script runs from any directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "common"))
from home_range import home_range   # noqa: E402  (the sys.path line above must come first)

RES = os.path.join(ROOT, "data", "equilibria")     # input: snapshots + metadata (stage 1 output)
NET = os.path.join(ROOT, "data", "networks")       # input: adjacency matrices (Moran's I only)
OUT = os.path.join(ROOT, "figures", "fig_ews_examples.png")   # the .pdf is written alongside it

# ----------------------------------------------------------------------------------------------
# The five EWSs, in plotting (and legend) order, with the colors and labels used in the figure.
# The keys are the internal names accepted by ews_curve() below.
# ----------------------------------------------------------------------------------------------
EWS = ["var", "cv", "skew", "kurt", "moranI"]
COLORS = {"var": "#1f77b4", "cv": "#ff7f0e", "skew": "#2ca02c",
          "kurt": "#d62728", "moranI": "#9467bd"}
LABELS = {"var": "$V$", "cv": "CV", "skew": "$g_1'$", "kurt": "$g_2$", "moranI": "$I_M$"}

# The five networks and their display names.  The lattice is deliberately first; the other four are
# in ascending order of N (29, 198, 453, 4941).  See the README for how the four were selected.
NETS = ["lattice", "montreal", "jazz", "metabolic", "powergrid"]
NAME = {"lattice": "Square lattice", "montreal": "Montreal", "jazz": "Jazz",
        "metabolic": r"$C.\ elegans$ metabolic", "powergrid": "US power grid"}

# Panel order: the five networks under descending D first (row 1), then under descending u (row 2).
PANELS = [("D", n) for n in NETS] + [("u", n) for n in NETS]
LET = "abcdefghij"               # panel letters (a)-(j), in the same order as PANELS


def moran(x, A):
    """Moran's I of the snapshot ``x`` on the network with adjacency matrix ``A``.

    This is Eq. (21) of the manuscript,

        I_M = (N / W) * sum_ij A_ij (x_i - xbar)(x_j - xbar) / sum_i (x_i - xbar)^2,

    with W = sum_ij A_ij, which for the undirected unweighted networks used here equals twice the
    number of edges.  The double sum in the numerator is evaluated as the quadratic form
    ``xc @ (A @ xc)`` with ``xc`` the centered snapshot, which is exactly the same quantity and is
    what makes the largest network (N = 4941) cheap.  Returns NaN in the degenerate cases W = 0
    (empty network) and constant ``x`` (zero denominator); neither occurs for the committed data.
    """
    W = A.sum()
    xc = x - x.mean()
    den = (xc * xc).sum()
    return np.nan if (W == 0 or den == 0) else (len(x) / W) * (xc @ (A @ xc)) / den


def ews_curve(M, rows, A, name):
    """Evaluate one EWS at every control value in ``rows``; return the curve as a 1-D array.

    ``M`` is the (L x N) matrix of equilibrated states, so ``M[r]`` is the single snapshot
    x = (x_1, ..., x_N) recorded at control value ``r``; ``rows`` is the home range; ``A`` is used
    by Moran's I only.  The five statistics follow Eqs. (17)-(21) of the manuscript exactly:

      var     V   = sample variance with the 1/(N-1) normalization (``ddof=1``), Eq. (17).
      cv      CV  = sqrt(V) / xbar, Eq. (18).  Every snapshot here has xbar > 0 (all nodes are in
                    the upper state x_i > 3 over the home range), so the absolute value in the
                    definition CV = sqrt(V)/|xbar| is immaterial; the guard covers xbar = 0 only.
      skew    g_1'= -g_1 with g_1 = mu_3 / mu_2^{3/2}, Eq. (19).  ``bias=True`` selects exactly this
                    ratio of the 1/N centered moments mu_k rather than a sample-size-corrected
                    variant.  The SIGN FLIP is the plotting convention stated in the caption of
                    Fig. 1: both rows sweep the control parameter downward, and plotting g_1' = -g_1
                    makes an increase mean "closer to the tipping point" for all five curves.
      kurt    g_2 = mu_4 / mu_2^2, Eq. (20).  ``fisher=False`` keeps the raw ratio (no -3 shift),
                    and ``bias=True`` again selects the 1/N moments.
      moranI  I_M , Eq. (21); see moran() above.
    """
    out = []
    for r in rows:
        x = M[r]
        if name == "var":
            v = np.var(x, ddof=1)
        elif name == "cv":
            v = np.std(x, ddof=1) / x.mean() if x.mean() != 0 else np.nan
        elif name == "skew":
            v = -skew(x, bias=True)
        elif name == "kurt":
            v = kurtosis(x, fisher=False, bias=True)
        else:
            v = moran(x, A)
        out.append(v)
    return np.array(out, float)


def norm01(y):
    """Min-max normalize a curve to [0, 1], as stated in the caption of Fig. 1.

    The five EWSs are not comparable in magnitude, so each curve is rescaled linearly within its
    own panel: its smallest value is mapped to 0 and its largest to 1.  A curve that is constant
    (or whose range is not finite) would make the map ill-defined and is drawn as the flat line 0.5
    instead; this fallback is never triggered by the committed data.
    """
    y = np.asarray(y, float)
    m = np.nanmin(y)
    Mx = np.nanmax(y)
    return np.full_like(y, 0.5) if not np.isfinite(Mx - m) or (Mx - m) < 1e-12 else (y - m) / (Mx - m)


# ----------------------------------------------------------------------------------------------
# The figure: one panel per (control parameter, network) pair, in the order given by PANELS.
# ----------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(15.5, 7.0))
axes = axes.ravel()

for k, (cp, net) in enumerate(PANELS):
    ax = axes[k]

    # Load the run: metadata (.json) and the (100 x N) matrix of snapshots (.npy).
    jf = os.path.join(RES, f"{net}-{cp}.json")
    m = json.load(open(jf))
    M = np.load(jf[:-5] + ".npy")

    cpv = np.array(m["cparam_vals"])   # the 100 control values, ordered far -> near the transition
    hl = m["home_len"]                 # length of the home range, as recorded by stage 1
    h = home_range(M)                  # the home-range indices, recomputed from the data itself
    A = sp.load_npz(os.path.join(NET, net + ".npz")).tocsr()
    xx = cpv[h]                        # horizontal coordinate: the control values of the home range

    for e in EWS:
        ax.plot(xx, norm01(ews_curve(M, h, A, e)), color=COLORS[e], lw=1.2)

    # Dotted line at the first control value outside the home range, i.e., where the first node has
    # left the upper state.  ``hl < len(cpv)`` unless no tipping occurred in the range at all,
    # which stage 1 rejects; the fallback only keeps the plotting code total.
    tip = cpv[hl] if hl < len(cpv) else cpv[-1]
    ax.axvline(tip, color="black", ls=":", lw=1.4)

    # Axes: the horizontal range is the home range plus a small margin past the tipping line, and
    # only its two ends are ticked, because the interesting information is the shape of the curves.
    far = cpv[h[0]]
    ax.set_xlim(far, tip + 0.06 * (tip - far))
    ax.set_ylim(-0.025, 1.025)
    ax.set_title(f"({LET[k]}) {NAME[net]}", fontsize=15, pad=4, loc="left")
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=13, length=3)
    ax.set_xticks([far, tip])
    ax.set_xticklabels([f"{far:.3g}", f"{tip:.3g}"])
    ax.set_xlabel("$D$" if k < 5 else "$u$", fontsize=17)
    if k in (0, 5):                    # leftmost panel of each row
        ax.set_ylabel("EWS", fontsize=17)

# One shared legend for all ten panels, below the figure.
handles = [plt.Line2D([], [], color=COLORS[e], lw=2.5, label=LABELS[e]) for e in EWS]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=17, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0.01, 0.04, 1, 1])   # leave room at the bottom for the legend

os.makedirs(os.path.dirname(OUT), exist_ok=True)   # figures/ is an output directory, not committed
for path in (OUT, OUT[:-4] + ".pdf"):
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
plt.close(fig)
print("wrote", os.path.relpath(OUT, ROOT))

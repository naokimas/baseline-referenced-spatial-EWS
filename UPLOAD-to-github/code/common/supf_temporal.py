"""Applying the sup-F steepening test to the SINGLE-NODE TEMPORAL variance.

The statistical machinery itself is in supf.py and is shared with the spatial analysis in
compute_sentinel_and_supf.py, so that Fig. 5(b,c) and SI Figs. S3-S4 compare the temporal and
spatial EWSs using exactly the same test rather than two similar ones. This module only adds the
temporal bookkeeping: turning recorded samples into a variance series and averaging over nodes.

Imported by compute_temporal_stats.py.
"""
import numpy as np
from supf import M0, ALPHA, prefix, supf_window, Pcut, steepen, calibrate, nonseq_p  # noqa: F401


def series_success(x, y, n, Smat, c_alpha):
    """Run both sup-F tests on one EWS series y observed over the n home-range control values x.

    Returns (non-sequential success, sequential success), each 1 or 0; the sequential entry is nan
    when the home range is too short for the growing-window procedure to run at all.

    A series that is constant or contains a non-finite value cannot steepen, so it scores 0 outright.
    """
    if not np.all(np.isfinite(y)) or np.nanstd(y) == 0:
        return 0, (np.nan)
    P = prefix(x, y)
    F, b1, b2 = supf_window(Pcut(P, n), n)
    ns = 1 if (np.isfinite(F) and nonseq_p(F, n, Smat) < ALPHA and steepen(b1, b2)) else 0
    ss = np.nan
    if n >= M0 + 1:
        thr = c_alpha[n]; al = 0
        for ell in range(M0, n + 1):          # reveal the home range one control value at a time
            Fe, e1, e2 = supf_window(Pcut(P, ell), ell)
            if Fe > thr and steepen(e1, e2):  # first window that both crosses the boundary and accelerates
                al = 1; break
        ss = al
    return ns, ss


def case_success(samp, x, n, Lgrid, Smat, c_alpha):
    """Success rates for one (condition, network) case, as a function of the sample size L'.

    samp has shape (L, number of nodes, n): L temporal samples of each recorded node at each of the n
    home-range control values. For each L' in Lgrid we use only the first L' samples, take the
    temporal variance of each node at each control value -- that variance series is the single-node
    temporal EWS -- test it, and average the outcome over the recorded nodes.

    Returns (non-sequential rates, sequential rates), one entry per L' in Lgrid.
    """
    Lmax, nsn, _ = samp.shape
    nsr = np.full(len(Lgrid), np.nan); ssr = np.full(len(Lgrid), np.nan)
    for gi, Lp in enumerate(Lgrid):
        nsv = []; ssv = []
        for j in range(nsn):
            s = samp[:Lp, j, :]                       # (L', n) samples of node j over the home range
            y = s.var(axis=0, ddof=1)                 # temporal variance at each control value
            a, b = series_success(x, y, n, Smat, c_alpha)
            nsv.append(a)
            if np.isfinite(b): ssv.append(b)
        nsr[gi] = np.mean(nsv) if nsv else np.nan
        ssr[gi] = np.mean(ssv) if ssv else np.nan
    return nsr, ssr

"""The sup-F test for a structural change at an unknown breakpoint, used for every steepening
result in the paper.

An early warning signal that merely rises is weaker evidence than one that rises ever faster: an
acceleration is what theory predicts as a tipping point is approached, and Kendall's tau cannot tell
a linear rise from an accelerating one. This module supplies that test, in two forms:

  NON-SEQUENTIAL  one look at the complete home range. Appropriate for asking, in hindsight, whether
                  a usable precursor existed at all.
  SEQUENTIAL      the home range is revealed one control value at a time and we alarm at the first
                  window showing a significant acceleration. This is what a real monitoring
                  programme could do, and it is the version that yields a lead time.

Both are used by compute_sentinel_and_supf.py for the spatial EWSs (Figs. 4 and 6) and, through
supf_temporal.py, for the single-node temporal variance (Fig. 5 and SI Figs. S3-S4), so that the
spatial and temporal comparisons are made with exactly the same test.

METHOD. At each candidate breakpoint, fit the EWS series y against the control parameter x with TWO
SEPARATE straight lines, one before the breakpoint and one after it; both the intercept and the slope
may differ between the segments, and the fit is NOT constrained to be continuous there (hence 4 free
parameters, and the ell-4 denominator degrees of freedom in supf_window). Comparing that fit to a
single straight line gives the F statistic for a common intercept and slope -- the classical Chow
(1960) statistic for a break at a prescribed point. The sup-F statistic is the supremum of that F
over all admissible breakpoints (Quandt 1960; Andrews 1993). Its null distribution is not the usual
F distribution, because the breakpoint is chosen to maximize the statistic, so we obtain the null by
Monte-Carlo simulation instead (see calibrate).

NAMING. Andrews (1993) also defines sup-Wald, sup-LM and sup-LR forms of this test. In a Gaussian
linear regression they are strictly increasing functions of the same F for a fixed number of points,
so they select the same breakpoint and, under the Monte-Carlo calibration used here, return exactly
the same p-value. What this module computes is the F (Wald) form, so we call it sup-F throughout;
earlier versions of the manuscript and of this code called it sup-LM.

EFFICIENCY. Evaluating a breakpoint naively costs O(n). We instead precompute cumulative sums of
x, y, x^2, xy and y^2 once per series (prefix), after which the least-squares fit of any segment is
available in O(1) from differences of those sums (seg), and all breakpoints are evaluated at once.
"""
import numpy as np

M0 = 15        # shortest window the sequential test considers; a slope change needs enough points
ALPHA = 0.05   # significance level, for the non-sequential p-value and the sequential boundary alike
B = 2000       # number of Monte-Carlo surrogate series used to build the null distribution


def prefix(x, y):
    """Cumulative sums of x, y, x*x, x*y and y*y, each with a leading 0 so that seg() can subtract."""
    cx = np.concatenate([[0], np.cumsum(x)]); cy = np.concatenate([[0], np.cumsum(y)])
    cxx = np.concatenate([[0], np.cumsum(x * x)]); cxy = np.concatenate([[0], np.cumsum(x * y)])
    cyy = np.concatenate([[0], np.cumsum(y * y)])
    return cx, cy, cxx, cxy, cyy


def seg(P, a, b):
    """Least-squares fit of y on x over each half-open segment [a[i], b[i]), vectorised over i.

    Returns (rss, slope): residual sum of squares and fitted slope per segment, each in O(1) from the
    cumulative sums. Segments whose x values are (numerically) constant get slope 0, since their
    slope is undefined.
    """
    cx, cy, cxx, cxy, cyy = P; n = (b - a).astype(float)
    Sx = cx[b] - cx[a]; Sy = cy[b] - cy[a]
    Sxx = cxx[b] - cxx[a]; Sxy = cxy[b] - cxy[a]; Syy = cyy[b] - cyy[a]
    Sxxc = Sxx - Sx * Sx / n; Sxyc = Sxy - Sx * Sy / n; Syyc = Syy - Sy * Sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rss = np.maximum(Syyc - np.where(Sxxc > 1e-15, Sxyc * Sxyc / Sxxc, 0.0), 0.0)
        slope = np.where(Sxxc > 1e-15, Sxyc / Sxxc, 0.0)
    return rss, slope


def supf_window(P, ell):
    """sup-F over all breakpoints, using the first `ell` points. Returns (F, slope_before, slope_after).

    Breakpoints are trimmed so each segment keeps at least 15% of the points (and at least 4): a
    segment with very few points has an unstable slope and would inflate F. The two returned slopes
    belong to the maximizing breakpoint and are what steepen() inspects.
    """
    ms = max(4, int(np.ceil(0.15 * ell))); ks = np.arange(ms, ell - ms + 1)
    if len(ks) == 0 or ell - 4 <= 0:
        return -np.inf, 0.0, 0.0
    rss_r, _ = seg(P, np.array([0]), np.array([ell]))          # one-segment (restricted) fit
    a0 = np.zeros(len(ks), int)
    rss1, b1 = seg(P, a0, ks)                                   # before each candidate breakpoint
    rss2, b2 = seg(P, ks, np.full(len(ks), ell))                # after it
    rss_u = rss1 + rss2                                         # two-segment (unrestricted) fit
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(rss_u > 0, ((rss_r[0] - rss_u) / 2.0) / (rss_u / (ell - 4)), 0.0)
    i = int(np.argmax(F))
    return float(F[i]), float(b1[i]), float(b2[i])


def Pcut(P, ell):
    """Restrict precomputed prefix sums to the first `ell` points."""
    return tuple(c[:ell + 1] for c in P)


def steepen(b1, b2):
    """Is the detected slope change an ACCELERATION, given the slopes before (b1) and after (b2)?

    A significant sup-F only says the regression coefficients (intercept, slope, or both) changed
    somewhere; it does not say the EWS is rising ever faster. Following the definition in the Methods, we declare a positive steepening when the fitted
    slope is non-negative before the breakpoint and strictly larger after it. (Together these imply
    b2 > 0, so the later segment is always a genuine rise.)
    """
    return (b1 >= 0) and (b2 > b1)


def calibrate(Lmax, seed=0):
    """Monte-Carlo null distribution of sup-F, computed once and reused for every series.

    Returns (Smat, c_alpha).

    Smat[b, ell] is the sup-F of surrogate series b -- pure noise against the same equally spaced x --
    restricted to its first `ell` points. Ranking an observed F against column n gives the
    non-sequential p-value (nonseq_p).

    c_alpha[L] is the boundary for the SEQUENTIAL test. Testing every growing window against a fixed
    threshold would raise the false-alarm rate far above ALPHA, because many windows are examined, and
    a Bonferroni correction would be wrong here since nested windows are strongly dependent. Instead,
    for each surrogate we take the RUNNING MAXIMUM of its sup-F over all windows up to L, and set
    c_alpha[L] to the (1-ALPHA) quantile of that running maximum. A surrogate crosses this boundary
    anywhere in the first L windows with probability ALPHA, so the false-alarm probability of the
    whole sequential procedure -- not of each window separately -- is controlled at ALPHA.
    """
    rng = np.random.default_rng(seed); x = np.arange(Lmax, dtype=float)
    Smat = np.zeros((B, Lmax + 1))
    for bb in range(B):
        P = prefix(x, rng.standard_normal(Lmax))
        for ell in range(8, Lmax + 1):
            Smat[bb, ell] = supf_window(Pcut(P, ell), ell)[0]
    rm = np.maximum.accumulate(np.where(np.arange(Lmax + 1) >= M0, Smat, -np.inf), axis=1)
    c_alpha = np.full(Lmax + 1, np.inf)
    for L in range(M0, Lmax + 1):
        c_alpha[L] = np.quantile(rm[:, L], 1 - ALPHA)
    return Smat, c_alpha


def nonseq_p(F, n, Smat):
    """Monte-Carlo p-value: rank of the observed F among the B surrogate values of the same length.
    The +1 in numerator and denominator counts the observation itself, which keeps the test valid --
    the p-value can never come out as exactly 0 -- with a finite number of surrogates."""
    return (1 + int(np.sum(Smat[:, n] >= F))) / (B + 1)

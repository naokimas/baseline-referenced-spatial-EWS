"""Constants of the four dynamical models, and the value of the non-control parameter.

This module holds *only* numbers -- it performs no simulation. The Euler-Maruyama integrator that
actually produced every equilibrium in `data/equilibria_with_noise/` is `sde_simulator.py`, which
imports `PARAMS` from here. Keeping the constants in one place means the simulator, the temporal
sampler and the driver script cannot drift apart.

Every value below is the one stated in the "Dynamics" subsection of the Methods of the manuscript;
the equation each model refers to is given with it. In all four models the noise is additive and
`sigma` is the STANDARD DEVIATION of the noise increment, i.e. the integrator advances

    x_i <- x_i + f_i(x) * dt + sigma * sqrt(dt) * Z,      Z ~ N(0, 1) drawn independently per node
                                                          and per time step,

which is the Euler-Maruyama discretisation of  dx_i = f_i(x) dt + sigma dxi_i.
"""

# Keys used by every model:
#   sigma  noise standard deviation (see the convention above)
#   xlow   initial state x_i(0) used in ASCENDING simulations (every node starts in its lower state)
#   xhigh  initial state x_i(0) used in DESCENDING simulations (every node starts in its upper state)
#   floor  if True, any negative x_i is reset to 0 after each time step, because a negative value is
#          nonphysical for an abundance, a probability, or an expression level (Methods: "To prevent
#          nonphysical values, we set any negative value of x_i to zero as soon as it appears in this
#          and the following two models"). The double-well state is not sign-constrained, so False.
#   basin  threshold that classifies node i as still being in its INITIAL state. A node is in its
#          lower state if x_i < basin, and in its upper state otherwise. It is what defines the home
#          range: the control values before any node has left its initial state.
PARAMS = {
    # Coupled double-well, Eq. (doublewell) of the manuscript:
    #   dx_i = [ -(x_i-r1)(x_i-r2)(x_i-r3) + D sum_j A_ij x_j + u ] dt + sigma dxi_i
    # r1 and r3 are the stable lower/upper states of an isolated node, r2 the unstable one between
    # them, which is therefore also the classification threshold (basin = r2 = 3).
    'doublewell':  dict(r1=1., r2=3., r3=5.,                     sigma=0.1,   xlow=1.,    xhigh=5.,    floor=False, basin=3.),

    # Mutualistic species interaction (Gao et al. 2016), Eq. (mutualistic):
    #   dx_i = [ B + x_i(1-x_i/K)(x_i/C-1) + D sum_j A_ij x_i x_j/(Dt + E x_i + H x_j) + u ] dt + sigma dxi_i
    # x_i is an abundance; B immigration, K carrying capacity, C Allee threshold, and Dt (= D-tilde
    # in the manuscript), E, H modulate the interaction term. Nodes start at x_i = 6 (established
    # population) and count as upper while x_i > C, hence basin = C = 1.
    'mutualistic': dict(B=0.1, K=5., C=1., Dt=5., E=0.9, H=0.1,  sigma=0.001, xlow=0.,    xhigh=6.,    floor=True,  basin=1.),

    # SIS epidemic dynamics, Eq. (sis):
    #   dx_i = [ -mu x_i + D sum_j A_ij (1-x_i) x_j ] dt + sigma dxi_i
    # x_i is the probability that node i is infectious; mu the recovery rate, D the infection rate
    # (the control parameter -- this model has no u). Ascending runs start near the disease-free
    # state (x_i = 0.001), descending runs at high prevalence (x_i = 0.999). Following MacLaren et
    # al. (2025), a node counts as lower iff x_i < 5*sigma, hence basin = 5*0.001 = 0.005.
    'SIS':         dict(mu=1.,                                   sigma=0.001, xlow=0.001, xhigh=0.999, floor=True,  basin=0.005),

    # Gene-regulatory dynamics (Gao et al. 2016), Eq. (genereg):
    #   dx_i = [ -B x_i^f + D sum_j A_ij x_j^h/(1+x_j^h) + u ] dt + sigma dxi_i
    # x_i is an expression level; -B x_i^f is self-degradation and h the Hill coefficient. Runs
    # start near the active state (x_i = 2) to study its loss of resilience; basin = 5*sigma as above.
    'genereg':     dict(B=1., f=1., h=2.,                        sigma=0.001, xlow=0.,    xhigh=2.,    floor=True,  basin=0.005),
}


def fixed_value(model, cparam, direction):
    """Value at which the *other* parameter is held while `cparam` is swept.

    Each simulation condition varies one control parameter, D (coupling) or u (uniform stress), and
    keeps the other fixed; these are the values stated in the Methods.

      * Sweeping D: u = 0, except for the DESCENDING double-well and mutualistic runs, where u = -5.
        The negative stress is required there: it destabilises the upper equilibrium so that a
        downward transition actually occurs as D decreases. (The SIS model has no u; the 0.0
        returned for it is unused, since its drift does not contain a u term.)
      * Sweeping u: D = 0.05 for the double-well and mutualistic models, and D = 1 for the
        gene-regulatory model. The SIS model is never swept in u.
    """
    if cparam == 'D':
        if model in ('doublewell', 'mutualistic') and direction == 'down':
            return -5.0
        return 0.0
    return {'doublewell': 0.05, 'mutualistic': 0.05, 'genereg': 1.0}[model]

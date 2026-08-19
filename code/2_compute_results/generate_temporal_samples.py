"""Single-node TEMPORAL samples, for the temporal-versus-spatial comparison of Fig. 5 and SI Figs S2-S4.

For each of the 400 (simulation condition x network) cases this script restarts the dynamics from the
already-equilibrated snapshot in data/equilibria_with_noise/ at every home-range control value, runs a
one-skip-interval burn-in, and then records L=200 states of 10 randomly chosen nodes, spaced by the
sampling interval of Masuda et al., Nat. Commun. 2024 (1 time unit for the double-well, SIS and
gene-regulatory models; 0.1 for the mutualistic model, whose dynamics are faster).

It is VECTORISED over control values (the state X is an N x Lc matrix and the coupling is one sparse
product A@X) and MULTIPROCESSED across the 400 cases.

INPUT:  data/equilibria_with_noise/*.{npy,json}, data/networks/*.npz.
OUTPUT: raw samples in data/temporal_samples/ (regenerable, NOT committed; consumed by
        compute_temporal_stats.py) and the supporting table data/results/temporal_results.csv, which
        holds tau' of the single-node temporal variance per network (see README; no figure uses it)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import sys,json,glob,csv,time
import numpy as np
from scipy.stats import kendalltau
from multiprocessing import get_context
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
SDIR=os.path.join(ROOT,"data","temporal_samples"); os.makedirs(SDIR,exist_ok=True)   # raw single-node samples (regenerable; not committed)
sys.path.insert(0,os.path.join(ROOT,"code","common"))
from require_data import require_dir, load_network
from model_parameters import PARAMS   # model constants (sigma etc.); this script has its own integrator, see simulate() below
RES=os.path.join(ROOT,"data","equilibria_with_noise"); NET=os.path.join(ROOT,"data","networks")
require_dir(RES,"code/1_generate_equilibria_with_noise/generate_equilibria.py")
L=200; NSEL=10; SKIP_TU={"doublewell":1.0,"SIS":1.0,"genereg":1.0,"mutualistic":0.1}
FLOOR={"SIS","genereg","mutualistic"}
CONDS=[("doublewell","D","up"),("doublewell","D","down"),("doublewell","u","up"),("doublewell","u","down"),
 ("mutualistic","D","down"),("mutualistic","u","down"),("SIS","D","up"),("SIS","D","down"),
 ("genereg","D","down"),("genereg","u","down")]
CKEY={("doublewell","D","up"):"DW $D$ asc",("doublewell","D","down"):"DW $D$ desc",("doublewell","u","up"):"DW $u$ asc",
 ("doublewell","u","down"):"DW $u$ desc",("mutualistic","D","down"):"MUT $D$ desc",("mutualistic","u","down"):"MUT $u$ desc",
 ("SIS","D","up"):"SIS $D$ asc",("SIS","D","down"):"SIS $D$ desc",("genereg","D","down"):"GR $D$ desc",("genereg","u","down"):"GR $u$ desc"}

def simulate(model,A,cvals,cparam,fixed,X0,dt,n_burn,n_skip,sel,sigma,seed):
    """Euler-Maruyama integration that RECORDS a time series, returning samples of shape (L, ns, Lc).

    X0: (Lc,N) equilibrated snapshot, i.e. one row per home-range control value. sel: indices of the
    ns nodes to record. The state is carried as X of shape (N, Lc), so all Lc control values advance
    together and the coupling sum is the single sparse product A@X.

    This is a second implementation of the same four drift equations as code/common/sde_simulator.py,
    kept separate because the two jobs differ: sde_simulator returns only the FINAL state of a run
    started from a uniform scalar initial condition, whereas here we start from a given per-node
    snapshot and must sample the trajectory every n_skip steps. It is written with numpy/scipy rather
    than numba because the recorded state is a matrix and the per-node numba loop is far slower than
    one sparse mat-mat product on the large hub networks. The model constants are imported from
    model_parameters.py, so the numbers in the two integrators cannot diverge.
    """
    P=PARAMS[model]; N=A.shape[0]; Lc=len(cvals); ns=len(sel); sq=sigma*np.sqrt(dt)
    is_D=(cparam=="D"); D=(cvals if is_D else np.full(Lc,fixed)); u=(np.full(Lc,fixed) if is_D else cvals)
    Dr=D[None,:]; ur=u[None,:]; X=np.ascontiguousarray(X0.T)          # (N,Lc)
    rng=np.random.default_rng(seed); out=np.empty((L,ns,Lc)); total=n_burn+(L-1)*n_skip
    floor=model in FLOOR
    if model=="mutualistic":
        rows,cols=A.nonzero(); B=P['B'];K=P['K'];Cal=P['C'];Dt=P['Dt'];E=P['E'];H=P['H']
    for step in range(total):
        if model=="doublewell":
            C=A@X; drift=-(X-P['r1'])*(X-P['r2'])*(X-P['r3'])+Dr*C+ur
        elif model=="SIS":
            C=A@X; drift=-P['mu']*X+Dr*(1.0-X)*C
        elif model=="genereg":
            Xh=X**P['h']; C=A@(Xh/(Xh+1.0)); drift=-P['B']*X**P['f']+Dr*C+ur
        else:
            xi=X[rows]; xj=X[cols]; contrib=xj/(Dt+E*xi+H*xj)
            S=np.zeros((N,Lc)); np.add.at(S,rows,contrib)
            drift=B+X*(1.0-X/K)*((X/Cal)-1.0)+Dr*X*S+ur
        X=X+drift*dt+sq*rng.standard_normal((N,Lc))
        if floor: np.maximum(X,0.0,out=X)
        t=step-n_burn+1
        if step==n_burn-1: out[0]=X[sel]
        elif t>0 and t%n_skip==0: out[t//n_skip]=X[sel]
    return out

def work(task):
    ci,ni,mod,cp,dr,net=task; P=PARAMS[mod]
    m=json.load(open(os.path.join(RES,f"{mod}-{net}-{cp}-{dr}.json")))
    dt=m["dt"]; N=m["N"]; hl=m["home_len"]; cpv=np.array(m["bparam_vals"]); fixed=m["fixed_param"]
    hidx=np.arange(hl)      # the whole home range; see code/common/home_range.py for the definition
    ch=cpv[hidx]
    n_skip=int(round(SKIP_TU[mod]/dt)); n_burn=n_skip
    M0=np.load(os.path.join(RES,f"{mod}-{net}-{cp}-{dr}.npy"))[hidx]     # (Lc,N) equilibrated snapshot at home pts
    A=load_network(NET,net).tocsr().astype(float)
    rng=np.random.default_rng(1000*ci+ni); sel=np.sort(rng.choice(N,size=min(NSEL,N),replace=False))
    samp=simulate(mod,A,ch,cp,fixed,M0,dt,n_burn,n_skip,sel,P["sigma"],seed=1000*ci+ni+7)  # (L,ns,Lc)
    np.save(os.path.join(SDIR,f"{mod}-{cp}-{dr}-{net}.npy"),samp.astype(np.float32))          # for sup-F
    xor=(ch if dr=="up" else -ch).astype(float); ns=len(sel); tau_acc=np.zeros(L+1)
    for j in range(ns):
        s=samp[:,j,:]                                                   # (L, hl)
        cs1=np.cumsum(s,axis=0); cs2=np.cumsum(s*s,axis=0)
        for mm in range(2,L+1):
            vh=(cs2[mm-1]-cs1[mm-1]**2/mm)/(mm-1); g=np.isfinite(vh)
            tau_acc[mm]+=kendalltau(xor[g],vh[g]).correlation if g.sum()>=4 and np.std(vh[g])>0 else np.nan
    ck=CKEY[(mod,cp,dr)]
    return [[ck,net,N,mm,round(tau_acc[mm]/ns,5)] for mm in range(2,L+1)]

def main():
    byc={}
    for jf in glob.glob(os.path.join(RES,"*.json")):
        m=json.load(open(jf)); byc[(m["model"],m["cparam"],m["direction"],m["network"])]=m["N"]
    tasks=[]
    for ci,(mod,cp,dr) in enumerate(CONDS):
        nets=sorted([n for (a,b,c,n) in byc if (a,b,c)==(mod,cp,dr)], key=lambda n:byc[(mod,cp,dr,n)])
        for ni,net in enumerate(nets): tasks.append((ci,ni,mod,cp,dr,net))
    out=os.path.join(ROOT,"data","results","temporal_results.csv")
    corder={CKEY[c]:i for i,c in enumerate(CONDS)}          # canonical condition order, as in the figures
    def dump(rows):
        # Sorted into the canonical condition order used by the figures, which also makes the file
        # byte-reproducible: the worker pool returns cases in completion order, which varies per run.
        rows=sorted(rows,key=lambda r:(corder[r[0]],r[1],r[3]))
        with open(out,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["cond","net","N","Lprime","tau_avg10"]); w.writerows(rows)
    print("cases:",len(tasks),flush=True); t0=time.time(); rows=[]; done=0
    ctx=get_context("spawn")               # spawn (not fork) -> avoids macOS Accelerate fork crashes
    with ctx.Pool(6) as pool:
        for r in pool.imap_unordered(work,tasks):
            rows+=r; done+=1
            if done%10==0 or done==len(tasks):
                dump(rows); print("  %d/%d  %.0fs"%(done,len(tasks),time.time()-t0),flush=True)
    print("rows:",len(rows),"total %.0fs"%(time.time()-t0))

if __name__=="__main__":
    main()

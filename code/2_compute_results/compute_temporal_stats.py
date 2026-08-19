"""Per-(network,node) statistics from saved samples: for each condition and L', collect the 400
(40 networks x 10 nodes) values of tau', non-seq sup-F success, seq sup-F success, and output the
INPUT: data/temporal_samples/*.npy (from generate_temporal_samples.py).  OUTPUT: data/results/perpair_tau.csv, perpair_supf.csv.
MEAN and STD over the 400 pairs (for mean +/- 1 std bands). tau' on L'=2..200; sup-F on a geom grid."""
import os,sys,json,csv,numpy as np
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
from multiprocessing import get_context
from scipy.stats import kendalltau
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,os.path.join(ROOT,"code","common"))   # must precede the import below
import supf_temporal as S
from require_data import require_dir
RES=os.path.join(ROOT,"data","equilibria_with_noise"); SDIR=os.path.join(ROOT,"data","temporal_samples")
require_dir(RES,"code/1_generate_equilibria_with_noise/generate_equilibria.py"); require_dir(SDIR,"code/2_compute_results/generate_temporal_samples.py")
CONDS=[("doublewell","D","up"),("doublewell","D","down"),("doublewell","u","up"),("doublewell","u","down"),
 ("mutualistic","D","down"),("mutualistic","u","down"),("SIS","D","up"),("SIS","D","down"),
 ("genereg","D","down"),("genereg","u","down")]
CKEY={("doublewell","D","up"):"DW $D$ asc",("doublewell","D","down"):"DW $D$ desc",("doublewell","u","up"):"DW $u$ asc",
 ("doublewell","u","down"):"DW $u$ desc",("mutualistic","D","down"):"MUT $D$ desc",("mutualistic","u","down"):"MUT $u$ desc",
 ("SIS","D","up"):"SIS $D$ asc",("SIS","D","down"):"SIS $D$ desc",("genereg","D","down"):"GR $D$ desc",("genereg","u","down"):"GR $u$ desc"}
# Every performance measure is evaluated at every number of temporal samples L' = 2, 3, ..., 200, as
# stated in the Methods. tau' and the two sup-F success fractions therefore share one grid.
LTAU=list(range(2,201)); LSUP=LTAU
_SM=None;_CA=None
def _init(sm,ca):
    global _SM,_CA; _SM=sm;_CA=ca
def work(task):
    mod,cp,dr,net=task; p=os.path.join(SDIR,f"{mod}-{cp}-{dr}-{net}.npy")
    if not os.path.exists(p): return None
    samp=np.load(p).astype(float)                     # (L,ns,Lc)
    m=json.load(open(os.path.join(RES,f"{mod}-{net}-{cp}-{dr}.json"))); hl=m["home_len"]; cpv=np.array(m["bparam_vals"])
    hidx=np.arange(hl); ch=cpv[hidx]; x=(ch if dr=="up" else -ch).astype(float); n=len(ch)
    L,ns,_=samp.shape
    tau=np.full((ns,len(LTAU)),np.nan); nsr=np.full((ns,len(LSUP)),np.nan); ssr=np.full((ns,len(LSUP)),np.nan)
    for j in range(ns):
        s=samp[:,j,:]; cs1=np.cumsum(s,0); cs2=np.cumsum(s*s,0)
        for gi,Lp in enumerate(LTAU):
            vh=(cs2[Lp-1]-cs1[Lp-1]**2/Lp)/(Lp-1); g=np.isfinite(vh)
            if g.sum()>=4 and np.std(vh[g])>0: tau[j,gi]=kendalltau(x[g],vh[g]).correlation
        for gi,Lp in enumerate(LSUP):
            y=s[:Lp].var(0,ddof=1); a,b=S.series_success(x,y,n,_SM,_CA); nsr[j,gi]=a
            if np.isfinite(b): ssr[j,gi]=b
    return (CKEY[(mod,cp,dr)],tau,nsr,ssr)
def main():
    import glob
    tasks=[]
    for (mod,cp,dr) in CONDS:
        for f in glob.glob(SDIR+f"/{mod}-{cp}-{dr}-*.npy"):
            net=os.path.basename(f)[len(f"{mod}-{cp}-{dr}-"):-4]; tasks.append((mod,cp,dr,net))
    if not tasks:
        # The raw samples are regenerable and therefore not committed (see README), so this script is
        # a no-op until generate_temporal_samples.py has produced them. It returns before opening the
        # output files, leaving the committed perpair_*.csv in place.
        print("No sample files in %s.\nRun generate_temporal_samples.py first; the existing "
              "perpair_*.csv have been left untouched."%SDIR); return
    Sm,Ca=S.calibrate(100); print("calibrated; cases",len(tasks),flush=True)
    agg={CKEY[c]:{"tau":[],"ns":[],"ss":[]} for c in CONDS}
    ctx=get_context("spawn")
    with ctx.Pool(6,initializer=_init,initargs=(Sm,Ca)) as pool:
        d=0
        for r in pool.imap_unordered(work,tasks):
            d+=1
            if r is None: continue
            ck,tau,nsr,ssr=r; agg[ck]["tau"].append(tau); agg[ck]["ns"].append(nsr); agg[ck]["ss"].append(ssr)
            if d%40==0: print(" ",d,flush=True)
    with open(os.path.join(ROOT,"data","results","perpair_tau.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["cond","Lprime","mean","std"])
        for c in CONDS:
            A=np.vstack(agg[CKEY[c]]["tau"])   # (400, len LTAU)
            for gi,Lp in enumerate(LTAU):
                col=A[:,gi]; col=col[np.isfinite(col)]
                w.writerow([CKEY[c],Lp,round(float(np.mean(col)),5),round(float(np.std(col)),5)])
    with open(os.path.join(ROOT,"data","results","perpair_supf.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["cond","Lprime","ns_mean","ns_std","sq_mean","sq_std"])
        for c in CONDS:
            NS=np.vstack(agg[CKEY[c]]["ns"]); SQ=np.vstack(agg[CKEY[c]]["ss"])
            for gi,Lp in enumerate(LSUP):
                a=NS[:,gi]; a=a[np.isfinite(a)]; b=SQ[:,gi]; b=b[np.isfinite(b)]
                w.writerow([CKEY[c],int(Lp),round(float(np.mean(a)),5),round(float(np.std(a)),5),
                            round(float(np.mean(b)),5) if len(b) else "",round(float(np.std(b)),5) if len(b) else ""])
    print("wrote perpair_tau.csv, perpair_supf.csv")
if __name__=="__main__": main()

"""Sentinel-node analysis and the sup-F steepening tests. Produces Fig. 4 and Fig. 6.

For every simulation condition, network, sentinel-selection rule, sentinel fraction and EWS, this
script computes four numbers and averages each over the 40 networks:

  tau'          sign-adjusted Kendall correlation of the EWS against the control parameter
  non-seq sup-F   does the EWS show a significant POSITIVE STEEPENING over the whole home range?
  seq sup-F       does it steepen significantly at some point while the home range is revealed
                   one control value at a time, i.e. before the tipping point is reached?
  lead          if the sequential test alarms, how much of the home range is still left at that moment

Fig. 4 is the all-node case, i.e. the rows with fraction = 1.0, for which the selection rule is
irrelevant. Fig. 6 uses all fractions and all eight rules.

INPUT:  data/equilibria_with_noise/*.{npy,json}
OUTPUT: data/results/sentinel_results.csv
"""
import os, json, glob, csv, time, zlib
import numpy as np
from scipy.stats import kendalltau
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
import sys; sys.path.insert(0,os.path.join(ROOT,"code","common"))
from home_range import home_range
# The sup-F machinery lives in one place and is shared with the temporal analysis, so that the
# spatial and temporal comparisons of Fig. 5 use an identical test. See common/supf.py.
from supf import prefix, supf_window, Pcut, steepen, calibrate, nonseq_p, M0, ALPHA
from require_data import require_dir
RES=os.path.join(ROOT,"data","equilibria_with_noise")
require_dir(RES,"code/1_generate_equilibria_with_noise/generate_equilibria.py")
ELL=5        # the baseline b_i is the mean of x_i over the first ELL control values of the home range
EPS=1e-9     # guards division by a baseline that is essentially zero
RREP=5       # independent draws averaged over, for the one selection rule that is random
FRACS=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]
EWS=["V","CV","Vdelta","Vrel","CVrel"]   # original V and CV, plus the three baseline-referenced variants
METHODS=["random","largeb","smallb","dirb","halfb","equidist","smallcv","largecv"]
CONDS=[("doublewell","D","up"),("doublewell","D","down"),("doublewell","u","up"),("doublewell","u","down"),
 ("SIS","D","up"),("SIS","D","down"),("mutualistic","D","down"),("mutualistic","u","down"),
 ("genereg","D","down"),("genereg","u","down")]
CLAB={("doublewell","D","up"):"DW $D$ asc",("doublewell","D","down"):"DW $D$ desc",
 ("doublewell","u","up"):"DW $u$ asc",("doublewell","u","down"):"DW $u$ desc",("SIS","D","up"):"SIS $D$ asc",
 ("SIS","D","down"):"SIS $D$ desc",("mutualistic","D","down"):"MUT $D$ desc",("mutualistic","u","down"):"MUT $u$ desc",
 ("genereg","D","down"):"GR $D$ desc",("genereg","u","down"):"GR $u$ desc"}

def ews_series(Msub,bsub,e):
    """The EWS at every home-range control value, from the sentinel nodes only.

    Msub is (number of control values) x (number of sentinel nodes) and bsub their baselines b_i.
    V and CV are the classical spatial variance and coefficient of variation of the raw states;
    Vdelta, Vrel and CVrel are the same statistics computed on the baseline-referenced states
    x_i - b_i and x_i / b_i respectively.
    """
    if e=="V": return Msub.var(axis=1,ddof=1)
    if e=="CV":
        m=Msub.mean(axis=1); return np.where(np.abs(m)>1e-12,Msub.std(axis=1,ddof=1)/m,np.nan)
    if e=="Vdelta": return (Msub-bsub).var(axis=1,ddof=1)
    R=Msub/bsub
    if e=="Vrel": return R.var(axis=1,ddof=1)
    m=R.mean(axis=1); return np.where(np.abs(m)>1e-12,R.std(axis=1,ddof=1)/m,np.nan)

def select(method,N,nsel,b,cv,dr,rng):
    """Indices of the `nsel` sentinel nodes chosen by one of the eight rules.

    Every rule uses only the EARLY part of the home range -- the baseline b_i and the early-home-range
    coefficient of variation cv_i, both computed from the first ELL control values -- so no rule peeks
    at data near the tipping point, which would be unavailable to a real monitoring programme.

      random    uniformly at random                 largeb / smallb  largest / smallest baseline b_i
      dirb      largest b_i when ascending, smallest when descending, i.e. the nodes furthest from the
                state they will tip into                             halfb   half largest, half smallest b_i
      equidist  evenly spaced in the RANK of b_i, so the baseline range is covered uniformly
      smallcv / largecv  smallest / largest early-home-range CV of x_i
    """
    ob=np.argsort(b); ocv=np.argsort(cv)
    if method=="random": return rng.choice(N,nsel,replace=False)
    if method=="largeb": return ob[::-1][:nsel]
    if method=="smallb": return ob[:nsel]
    if method=="dirb": return (ob[::-1] if dr=="up" else ob)[:nsel]
    if method=="halfb":
        nt=int(np.ceil(nsel/2)); nb=nsel-nt; return np.concatenate([ob[::-1][:nt],ob[:nb]])
    if method=="equidist":
        idx=np.unique(np.linspace(0,N-1,nsel).round().astype(int)); return ob[idx]
    if method=="smallcv": return ocv[:nsel]
    return ocv[::-1][:nsel]

def main():
    t0=time.time()
    data={}
    for jf in glob.glob(os.path.join(RES,"*.json")):
        m=json.load(open(jf)); k=(m["model"],m["cparam"],m["direction"])
        if k not in CLAB: continue
        M=np.load(jf[:-5]+".npy"); cpv=np.array(m["bparam_vals"]); hl=m["home_len"]; dr=m["direction"]
        h=home_range(M,m["model"],dr); data.setdefault(k,[]).append((m["network"],M,cpv,hl,h,dr))
    Lmax=max(len(s[4]) for v in data.values() for s in v)
    print("Lmax=%d calibrating..."%Lmax,flush=True); Smat,c_alpha=calibrate(Lmax)
    print("calibrated %.0fs"%(time.time()-t0),flush=True)
    # accumulate per (method,frac,cond,ews): lists over nets
    acc={}
    for k in CONDS:
        for (net,M,cpv,hl,h,dr) in data[k]:
            n=len(h); N=M.shape[1]
            if n<8: continue
            Mh=M[h]                                  # states over the home range: n control values x N nodes
            # Per-node baseline and early-home-range CV, both from the first ELL control values only.
            b=np.nanmean(Mh[:ELL],0); b=np.where(np.abs(b)<EPS,EPS,b)
            with np.errstate(divide="ignore",invalid="ignore"):
                cv=np.nanstd(Mh[:ELL],0,ddof=1)/np.where(np.abs(b)<EPS,EPS,b)
            cv=np.where(np.isfinite(cv),cv,0.0)
            # Orient the control parameter so that it always INCREASES toward the tipping point; then a
            # positive tau' or a positive steepening means the same thing in ascending and descending runs.
            x=cpv[h].astype(float) if dr=="up" else (-cpv[h]).astype(float)
            # Endpoints used to express the lead as a fraction of the home range: the far end, and the
            # first control value at which a node has tipped.
            far=cpv[h[0]]; tip=cpv[hl] if hl<len(cpv) else cpv[-1]
            for method in METHODS:
                # One draw sequence per (network, rule), derived with crc32 rather than the built-in
                # hash(), which is salted per process; this keeps the random-selection rule reproducible.
                rng=np.random.default_rng(zlib.crc32(f"{net}|{method}".encode())%(2**32))
                for f in FRACS:
                    nsel=min(N,max(3,int(round(f*N))))
                    reps=RREP if method=="random" else 1
                    for e in EWS:
                        tps=[]; nss=[]; sss=[]; lds=[]
                        for rep in range(reps):
                            S=select(method,N,nsel,b,cv,dr,rng if method=="random" else np.random.default_rng(0))
                            Msub=Mh[:,S]; bsub=b[S]
                            y=ews_series(Msub,bsub,e)
                            if not np.all(np.isfinite(y)) or np.nanstd(y)==0: continue
                            tps.append(kendalltau(x,y).correlation)
                            P=prefix(x,y)
                            # NON-SEQUENTIAL test: one look at the entire home range.
                            F,b1,b2=supf_window(Pcut(P,n),n)
                            ns=1 if (np.isfinite(F) and nonseq_p(F,n,Smat)<ALPHA and steepen(b1,b2)) else 0
                            nss.append(ns)
                            # SEQUENTIAL test: reveal the home range one control value at a time and
                            # alarm at the first window that crosses the running-max boundary c_alpha.
                            if n>=M0+1:
                                thr=c_alpha[n]; al=None
                                for ell in range(M0,n+1):
                                    Fe,e1,e2=supf_window(Pcut(P,ell),ell)
                                    if Fe>thr and steepen(e1,e2): al=ell; break
                                sss.append(1 if al else 0)
                                # Lead = the fraction of the far-to-tipping interval still ahead when the
                                # alarm fires. The alarming window ends at home-range point al-1, which
                                # lies strictly before the tipping point, so the lead is always positive.
                                if al: lds.append(1-(cpv[h[al-1]]-far)/(tip-far))
                        # Two-level averaging: collapse the repeated random draws to one value per
                        # network here, then average over the 40 networks when writing the CSV. The
                        # lead list holds only the networks that actually alarmed, which is why the
                        # published lead is an average over those networks alone.
                        key=(method,f,k,e)
                        a=acc.setdefault(key,{"tp":[],"ns":[],"ss":[],"ld":[]})
                        if tps: a["tp"].append(np.nanmean(tps))
                        if nss: a["ns"].append(np.mean(nss))
                        if sss: a["ss"].append(np.mean(sss))
                        if lds: a["ld"].append(np.mean(lds))
    with open(os.path.join(ROOT,"data","results","sentinel_results.csv"),"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["method","fraction","condition","ews","mean_tau","mean_nonseq","mean_seq","mean_lead","n_nets"])
        for (method,f,k,e),a in acc.items():
            w.writerow([method,f,CLAB[k],e,
                round(np.mean(a["tp"]),4) if a["tp"] else "",
                round(np.mean(a["ns"]),4) if a["ns"] else "",
                round(np.mean(a["ss"]),4) if a["ss"] else "",
                round(np.mean(a["ld"]),4) if a["ld"] else "", len(a["tp"])])
    print("done %.0fs, %d keys"%(time.time()-t0,len(acc)),flush=True)
if __name__=="__main__": main()

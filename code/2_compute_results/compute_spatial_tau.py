"""Kendall's tau' of the five spatial EWSs, in each of their three variants, for all 40 networks and
all 10 simulation conditions. This produces the numbers behind the heatmaps of Fig. 3 and SI Fig. S1,
and the spatial reference lines of Fig. 5. All nodes are used (no sentinel selection).

INPUT:  data/equilibria_with_noise/*.{npy,json}, data/networks/*.npz.
OUTPUT: data/results/results_full.csv, results_half.csv.

At one control value, let x = (x_i) be the vector of node states and b_i the per-node baseline, i.e.
the mean of x_i over the first ELL=5 control values of the home range. The three variants of the node
vector are

  raw   : y_i = x_i             the EWS as classically defined
  delta : y_i = x_i - b_i       additive baseline referencing
  rel   : y_i = x_i / b_i       multiplicative (fold-change) baseline referencing

On each y we compute all five spatial EWSs: variance, coefficient of variation, skewness g1 (sign
adjusted for descending conditions), kurtosis g2, and Moran's I over the full network adjacency.
Sweeping the control value gives one series per (EWS, variant); tau' is the sign-adjusted Kendall
correlation of that series against the control parameter, so that tau' > 0 always means "the EWS
moves in the direction that signals an approaching transition".

Two windows are reported, in two files of identical layout:
  results_full.csv  tau' over the whole home range
  results_half.csv  tau' over its second half only (index above the median)
Each row is one (condition, EWS, variant) cell, averaged over the 40 networks.
"""
import os, json, glob, csv
import numpy as np
from scipy.stats import skew, kurtosis, kendalltau

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
import sys; sys.path.insert(0,os.path.join(ROOT,"code","common"))
from home_range import home_range
from require_data import require_dir, load_network
RES=os.path.join(ROOT,"data","equilibria_with_noise"); NET=os.path.join(ROOT,"data","networks")
OUT=os.path.join(ROOT,"data","results"); os.makedirs(OUT,exist_ok=True)
require_dir(RES,"code/1_generate_equilibria_with_noise/generate_equilibria.py")
ELL=5; EPS=1e-9
EWS=["var","CV","g1","g2","IM"]; VAR=["raw","delta","rel"]
CONDS=[("doublewell","D","up","DW $D$ asc"),("doublewell","D","down","DW $D$ desc"),
 ("doublewell","u","up","DW $u$ asc"),("doublewell","u","down","DW $u$ desc"),
 ("SIS","D","up","SIS $D$ asc"),("SIS","D","down","SIS $D$ desc"),
 ("mutualistic","D","down","MUT $D$ desc"),("mutualistic","u","down","MUT $u$ desc"),
 ("genereg","D","down","GR $D$ desc"),("genereg","u","down","GR $u$ desc")]

def moran(y,A):
    W=A.sum();
    if W==0: return np.nan
    yc=y-y.mean(); den=(yc*yc).sum(); return np.nan if den==0 else (len(y)/W)*(yc@(A@yc))/den
def ews_on(y,A,name,d):
    if name=="var": return np.var(y,ddof=1)
    if name=="CV":  m=y.mean(); return np.std(y,ddof=1)/m if abs(m)>1e-12 else np.nan
    if name=="g1":  s=skew(y,bias=True); return -s if d=="down" else s
    if name=="g2":  return kurtosis(y,fisher=False,bias=True)
    if name=="IM":  return moran(y,A)
def transforms(x,b):
    """The node vector in its three variants (see the module docstring)."""
    return {"raw":x,"delta":x-b,"rel":x/b}
def tau(series,cp,d):
    s=np.array(series,float); g=np.isfinite(s)&np.isfinite(cp)
    if g.sum()<3: return np.nan
    t=kendalltau(cp[g],s[g]).correlation
    return np.nan if t is None or np.isnan(t) else (-t if d=="down" else t)

byc={}
for jf in glob.glob(os.path.join(RES,"*.json")):
    m=json.load(open(jf)); byc[(m["model"],m["cparam"],m["direction"],m["network"])]=jf
Ac={}
full={}; half={}
for (mod,cp,dr,lab) in CONDS:
    nets=sorted([n for (a,b,c,n) in byc if (a,b,c)==(mod,cp,dr)])
    for v in VAR:
        for e in EWS: full[(lab,v,e)]=[]; half[(lab,v,e)]=[]
    for net in nets:
        jf=byc[(mod,cp,dr,net)]; m=json.load(open(jf)); M=np.load(jf[:-5]+".npy"); cpv=np.array(m["bparam_vals"])
        h=home_range(M,mod,dr)
        if len(h)<ELL: continue
        if net not in Ac: Ac[net]=load_network(NET,net).tocsr()
        A=Ac[net]
        b=np.nanmean(M[h[:ELL]],0); b=np.where(np.abs(b)<EPS,EPS,b)
        ser={(v,e):[] for v in VAR for e in EWS}
        for r in h:
            x=M[r]; T=transforms(x,b)
            for v in VAR:
                y=T[v]
                for e in EWS: ser[(v,e)].append(ews_on(y,A,e,dr))
        mid=np.floor(np.median(h)); hm=h>mid
        for v in VAR:
            for e in EWS:
                s=ser[(v,e)]
                full[(lab,v,e)].append(tau(s,cpv[h],dr))
                half[(lab,v,e)].append(tau(np.array(s)[hm],cpv[h][hm],dr))

def summarize(D,fn):
    with open(os.path.join(OUT,fn),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["condition","ews","variant","mean_tau","std_tau","frac_good_gt0.5","n"])
        for (_,_,_,lab) in CONDS:
            for e in EWS:
                for v in VAR:
                    a=np.array(D[(lab,v,e)]); a=a[np.isfinite(a)]
                    w.writerow([lab,e,v,round(np.mean(a),3),round(np.std(a),3),round(np.mean(a>0.5),2),len(a)])
summarize(full,"results_full.csv"); summarize(half,"results_half.csv")
print("wrote results_full.csv and results_half.csv to data/results/")

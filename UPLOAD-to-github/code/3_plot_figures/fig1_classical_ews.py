"""Manuscript Fig. 1: the five classical spatial EWSs are unreliable on heterogeneous networks.
Two rows x five panels. Top row (a)-(e): double-well, coupling D, descending, five networks (ascending by N).
Bottom row (f)-(j): double-well, stress u, descending, five networks (ascending by N). Each panel shows the
five classical spatial EWS (V, CV, sign-adjusted skewness g1, kurtosis g2, Moran's I) over the home range,
each min-max normalized to [0,1]; x runs far->tipping; dotted line = first tipping.
Output: figures/fig_existing_methods.png
"""
import os, json, numpy as np
from scipy.stats import skew, kurtosis
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
import sys; sys.path.insert(0,os.path.join(ROOT,"code","common"))
from home_range import home_range
from require_data import require_dir, load_network
RES=os.path.join(ROOT,"data","equilibria_with_noise"); NET=os.path.join(ROOT,"data","networks")
OUT=os.path.join(ROOT,"figures","fig_existing_methods.png")
# The figure files themselves are not distributed with this repository; create their
# directories on demand so that a fresh clone can write into them.
os.makedirs(os.path.dirname(OUT),exist_ok=True)
require_dir(RES,"code/1_generate_equilibria_with_noise/generate_equilibria.py")
EWS=["var","cv","skew","kurt","moranI"]
COLORS={"var":"#1f77b4","cv":"#ff7f0e","skew":"#2ca02c","kurt":"#d62728","moranI":"#9467bd"}
LABELS={"var":"$V$","cv":"CV","skew":"$g_1'$","kurt":"$g_2$","moranI":"$I_M$"}
NAME={"montreal":"Montreal","smallworld":"Watts–Strogatz","canton":"Canton","football":"Football",
      "erdos":"Erdős collaboration","contigusa":"Geographic","jazz":"Jazz","london_transport":"Transportation",
      "metabolic":r"$C.\ elegans$ metabolic","gap_junction_herm":r"$C.\ elegans$ neuronal"}
# reordered ascending by N within each row
ROW1=("doublewell","D","down"); ROW2=("doublewell","u","down")
PANELS=[(ROW1,"montreal"),(ROW1,"smallworld"),(ROW1,"canton"),(ROW1,"football"),(ROW1,"erdos"),
        (ROW2,"contigusa"),(ROW2,"jazz"),(ROW2,"london_transport"),(ROW2,"metabolic"),(ROW2,"gap_junction_herm")]
LET="abcdefghij"

def moran(x,A):
    W=A.sum(); xc=x-x.mean(); den=(xc*xc).sum()
    return np.nan if (W==0 or den==0) else (len(x)/W)*(xc@(A@xc))/den
def ews_curve(M,rows,A,name,d):
    out=[]
    for r in rows:
        x=M[r]
        if name=="var": v=np.var(x,ddof=1)
        elif name=="cv": v=np.std(x,ddof=1)/x.mean() if x.mean()!=0 else np.nan
        elif name=="skew": v=skew(x,bias=True); v=-v if d=="down" else v
        elif name=="kurt": v=kurtosis(x,fisher=False,bias=True)
        else: v=moran(x,A)
        out.append(v)
    return np.array(out,float)
def norm01(y):
    y=np.asarray(y,float); m=np.nanmin(y); Mx=np.nanmax(y)
    return np.full_like(y,0.5) if not np.isfinite(Mx-m) or (Mx-m)<1e-12 else (y-m)/(Mx-m)

fig,axes=plt.subplots(2,5,figsize=(15.5,7.0)); axes=axes.ravel()
for k,((mod,cp,dr),net) in enumerate(PANELS):
    ax=axes[k]
    jf=os.path.join(RES,f"{mod}-{net}-{cp}-{dr}.json"); m=json.load(open(jf)); M=np.load(jf[:-5]+".npy")
    cpv=np.array(m["bparam_vals"]); hl=m["home_len"]; h=home_range(M,mod,dr)
    A=load_network(NET,net).tocsr(); xx=cpv[h]
    for e in EWS:
        ax.plot(xx, norm01(ews_curve(M,h,A,e,dr)), color=COLORS[e], lw=1.2)
    tip=cpv[hl] if hl<len(cpv) else cpv[-1]; ax.axvline(tip,color="black",ls=":",lw=1.4)
    far=cpv[h[0]]; ax.set_xlim(far,tip+0.06*(tip-far)); ax.set_ylim(-0.025,1.025)
    ax.set_title(f"({LET[k]}) {NAME[net]}",fontsize=15,pad=4,loc="left")
    ax.set_yticks([0,1]); ax.tick_params(labelsize=13,length=3); ax.set_xticks([far,tip])
    ax.set_xticklabels([f"{far:.3g}",f"{tip:.3g}"])
    ax.set_xlabel("$D$" if k<5 else "$u$",fontsize=17)
    if k in (0,5): ax.set_ylabel("EWS",fontsize=17)
handles=[plt.Line2D([],[],color=COLORS[e],lw=2.5,label=LABELS[e]) for e in EWS]
fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=17,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0.01,0.04,1,1])
fig.savefig(OUT,dpi=200,bbox_inches="tight",pad_inches=0); plt.close(fig)
print("wrote",os.path.relpath(OUT,ROOT))

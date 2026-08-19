"""Figure: the two baseline-referenced variants of the spatial variance, cosmetically matched to Fig 1.
Same 10 panels/order as fig_existing_methods.png. Each panel shows V (blue, identical to Fig 1), V_Delta
(green), V_rel (red), each min-max normalized to [0,1]. Output: figures/fig_variance_variants.png
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
import sys; sys.path.insert(0,os.path.join(ROOT,"code","common"))
from home_range import home_range
from require_data import require_dir
RES=os.path.join(ROOT,"data","equilibria_with_noise")
OUT=os.path.join(ROOT,"figures","fig_variance_variants.png")
# The figure files themselves are not distributed with this repository; create their
# directories on demand so that a fresh clone can write into them.
os.makedirs(os.path.dirname(OUT),exist_ok=True)
require_dir(RES,"code/1_generate_equilibria_with_noise/generate_equilibria.py")
ELL=5; EPS=1e-9
NAME={"montreal":"Montreal","smallworld":"Watts–Strogatz","canton":"Canton","football":"Football",
      "erdos":"Erdős collaboration","contigusa":"Geographic","jazz":"Jazz","london_transport":"Transportation",
      "metabolic":r"$C.\ elegans$ metabolic","gap_junction_herm":r"$C.\ elegans$ neuronal"}
ROW1=("doublewell","D","down"); ROW2=("doublewell","u","down")
PANELS=[(ROW1,"montreal"),(ROW1,"smallworld"),(ROW1,"canton"),(ROW1,"football"),(ROW1,"erdos"),
        (ROW2,"contigusa"),(ROW2,"jazz"),(ROW2,"london_transport"),(ROW2,"metabolic"),(ROW2,"gap_junction_herm")]
LET="abcdefghij"
COL={"orig":"#1f77b4","delta":"#2ca02c","rel":"#d62728"}
LAB={"orig":"$V$ (original)","delta":"$V_\\Delta$","rel":"$V_{\\rm rel}$"}
def norm01(y):
    y=np.asarray(y,float); m=np.nanmin(y); Mx=np.nanmax(y)
    return np.full_like(y,0.5) if not np.isfinite(Mx-m) or (Mx-m)<1e-12 else (y-m)/(Mx-m)
fig,axes=plt.subplots(2,5,figsize=(15.5,7.0)); axes=axes.ravel()
for k,((mod,cp,dr),net) in enumerate(PANELS):
    ax=axes[k]
    jf=os.path.join(RES,f"{mod}-{net}-{cp}-{dr}.json"); m=json.load(open(jf)); M=np.load(jf[:-5]+".npy")
    cpv=np.array(m["bparam_vals"]); hl=m["home_len"]; h=home_range(M,mod,dr); xx=cpv[h]
    b=np.nanmean(M[h[:ELL]],0); b=np.where(np.abs(b)<EPS,EPS,b)
    Vo=[np.var(M[r],ddof=1) for r in h]; Vd=[np.var(M[r]-b,ddof=1) for r in h]; Vr=[np.var(M[r]/b,ddof=1) for r in h]
    ax.plot(xx,norm01(Vo),color=COL["orig"],lw=1.2)
    ax.plot(xx,norm01(Vd),color=COL["delta"],lw=1.2)
    ax.plot(xx,norm01(Vr),color=COL["rel"],lw=1.2)
    tip=cpv[hl] if hl<len(cpv) else cpv[-1]; ax.axvline(tip,color="black",ls=":",lw=1.4)
    far=cpv[h[0]]; ax.set_xlim(far,tip+0.06*(tip-far)); ax.set_ylim(-0.025,1.025)
    ax.set_title(f"({LET[k]}) {NAME[net]}",fontsize=15,pad=4,loc="left")
    ax.set_yticks([0,1]); ax.tick_params(labelsize=13,length=3); ax.set_xticks([far,tip])
    ax.set_xticklabels([f"{far:.3g}",f"{tip:.3g}"])
    ax.set_xlabel("$D$" if k<5 else "$u$",fontsize=17)
    if k in (0,5): ax.set_ylabel("EWS",fontsize=17)
handles=[plt.Line2D([],[],color=COL[v],lw=2.5,label=LAB[v]) for v in ["orig","delta","rel"]]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=17,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=[0.01,0.04,1,1])
fig.savefig(OUT,dpi=200,bbox_inches="tight",pad_inches=0); plt.close(fig)
print("wrote",os.path.relpath(OUT,ROOT))

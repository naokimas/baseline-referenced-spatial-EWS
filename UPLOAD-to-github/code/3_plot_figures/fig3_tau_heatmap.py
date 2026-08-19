"""Heatmap tables of the mean tau' over the 40 networks: 10 simulation conditions (rows) x
15 columns (the 5 classical spatial EWS, each in 3 variants: original / baseline-difference (Delta)
/ baseline-ratio (rel)), read from the CSVs written by compute_spatial_tau.py.
  full home range   -> figures/fig_tau_heatmap_full.png             (main text Fig. 3)
  second half only  -> Supplementary Material/fig_tau_heatmap_half.png  (SI Fig. S1)
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
R2=os.path.join(ROOT,"data","results")
# The figure files themselves are not distributed with this repository; create their
# directories on demand so that a fresh clone can write into them.
for _d in ("figures","Supplementary Material"): os.makedirs(os.path.join(ROOT,_d),exist_ok=True)
cmapM=LinearSegmentedColormap.from_list("rwg",["#8B0000","#e74c3c","#f5a89a","white","#a8d5b0","#1a8c3e","#0a5225"])
CKEY=["DW $D$ asc","DW $D$ desc","DW $u$ asc","DW $u$ desc","MUT $D$ desc","MUT $u$ desc",
      "SIS $D$ asc","SIS $D$ desc","GR $D$ desc","GR $u$ desc"]                 # keys in the csv (canonical order)
CONDS=["DW, $D$, asc","DW, $D$, desc","DW, $u$, asc","DW, $u$, desc","MUT, $D$, desc","MUT, $u$, desc",
       "SIS, $D$, asc","SIS, $D$, desc","GR, $D$, desc","GR, $u$, desc"]        # display labels
EWS=["var","CV","g1","g2","IM"]; EWLAB=["$V$","CV","$g_1'$","$g_2$","$I_{\\mathrm{M}}$"]
VAR3=["raw","delta","rel"]; VLAB={"raw":"orig","delta":"$\\Delta$","rel":"rel"}

def load(fn):
    D={}
    for r in csv.DictReader(open(fn)): D[(r["condition"],r["ews"],r["variant"])]=float(r["mean_tau"])
    return D
def heat(D,out):
    cols=[(e,v) for e in EWS for v in VAR3]
    Mt=np.array([[D.get((c,e,v),np.nan) for (e,v) in cols] for c in CKEY])
    fig,ax=plt.subplots(figsize=(15,6.6)); im=ax.imshow(Mt,cmap=cmapM,vmin=-1,vmax=1,aspect="auto")
    for i in range(len(CONDS)):
        for j in range(len(cols)):
            v=Mt[i,j]
            if np.isfinite(v): ax.text(j,i,f"${v:.2f}$",ha="center",va="center",fontsize=13.3,
                                       color="white" if abs(v)>0.55 else "black")
    ax.set_yticks(range(len(CONDS))); ax.set_yticklabels(CONDS,fontsize=13.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([VLAB[v] for (e,v) in cols],fontsize=13.5)
    for g in range(3,len(cols),3): ax.axvline(g-0.5,color="white",lw=2.8)
    sec=ax.secondary_xaxis("top"); sec.set_xticks([1+3*k for k in range(5)]); sec.set_xticklabels(EWLAB,fontsize=19.5)
    sec.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True); ax.set_yticks(np.arange(-.5,len(CONDS),1),minor=True)
    ax.grid(which="minor",color="white",lw=.6); ax.tick_params(which="minor",length=0)
    cb=fig.colorbar(im,ax=ax,fraction=0.022,pad=0.01); cb.set_label("average $\\tau'$",fontsize=15); cb.ax.tick_params(labelsize=12)
    cb.set_ticks([-1,-0.5,0,0.5,1]); cb.set_ticklabels(["$-1$","$-0.5$","0","0.5","1"])
    fig.tight_layout(); fig.savefig(out,dpi=180,bbox_inches="tight",pad_inches=0); plt.close(fig)
    print("wrote",os.path.relpath(out,ROOT))

heat(load(os.path.join(R2,"results_full.csv")), os.path.join(ROOT,"figures","fig_tau_heatmap_full.png"))
heat(load(os.path.join(R2,"results_half.csv")), os.path.join(ROOT,"Supplementary Material","fig_tau_heatmap_half.png"))

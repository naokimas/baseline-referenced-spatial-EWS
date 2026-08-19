"""Combined sup-F figure (panels a,b,c), all 5 EWS (original V, CV + variants), from the all-nodes
(fraction=1.0) sentinel results. (a) non-sequential detection fraction; (b) sequential detection
fraction; (c) sequential lead. Shared row labels left of (a); single colorbar right of (c); range
[0,1]. Output: figures/fig_supf.png (manuscript Fig. 4)"""
import os, csv, copy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
SRC=os.path.join(ROOT,"data","results","sentinel_results.csv")
OUT=os.path.join(ROOT,"figures","fig_supf.png")
# The figure files themselves are not distributed with this repository; create their
# directories on demand so that a fresh clone can write into them.
os.makedirs(os.path.dirname(OUT),exist_ok=True)
EWS=["V","CV","Vdelta","Vrel","CVrel"]; ELAB=["$V$","CV","$V_\\Delta$","$V_{\\rm rel}$","CV$_{\\rm rel}$"]
CKEY=["DW $D$ asc","DW $D$ desc","DW $u$ asc","DW $u$ desc","MUT $D$ desc","MUT $u$ desc",
      "SIS $D$ asc","SIS $D$ desc","GR $D$ desc","GR $u$ desc"]
CLAB=["DW, $D$, asc","DW, $D$, desc","DW, $u$, asc","DW, $u$, desc","MUT, $D$, desc","MUT, $u$, desc",
      "SIS, $D$, asc","SIS, $D$, desc","GR, $D$, desc","GR, $u$, desc"]
D={}
for r in csv.DictReader(open(SRC)):
    if abs(float(r["fraction"])-1.0)<1e-9:
        D[(r["condition"],r["ews"])]=r
def mat(col):
    return np.array([[float(D[(c,e)][col]) if (c,e) in D and D[(c,e)][col]!="" else np.nan for e in EWS] for c in CKEY])
A=mat("mean_nonseq"); Bm=mat("mean_seq"); C=mat("mean_lead")
# The lead is an average over the networks that raised a valid alarm, so it is undefined when an EWS
# alarms in none of the 40 networks. Such cells are drawn in gray and labeled N/A, which distinguishes
# "no alarm, hence no lead" from a genuine lead of 0.
CMAP=copy.copy(plt.cm.Greens); CMAP.set_bad("#e6e6e6")
fig,axs=plt.subplots(1,3,figsize=(13.5,5.6))
titles=["(a) Non-sequential test","(b) Sequential test","(c) Lead in sequential test"]
for ax,Mt,tt,idx in zip(axs,[A,Bm,C],titles,range(3)):
    im=ax.imshow(Mt,cmap=CMAP,vmin=0,vmax=1,aspect="auto")
    for i in range(len(CKEY)):
        for j in range(len(EWS)):
            v=Mt[i,j]
            if np.isfinite(v): ax.text(j,i,f"${v:.2f}$",ha="center",va="center",fontsize=11,color="white" if v>0.6 else "black")
            else: ax.text(j,i,"N/A",ha="center",va="center",fontsize=10,color="#555555")
    ax.set_xticks(range(len(EWS))); ax.set_xticklabels(ELAB,fontsize=12)
    ax.set_yticks(range(len(CKEY))); ax.set_yticklabels(CLAB if idx==0 else [],fontsize=11)
    ax.set_xticks(np.arange(-.5,len(EWS),1),minor=True); ax.set_yticks(np.arange(-.5,len(CKEY),1),minor=True)
    ax.grid(which="minor",color="white",lw=.7); ax.tick_params(which="minor",length=0)
    ax.set_title(tt,fontsize=15.6,loc="left",pad=4)
fig.subplots_adjust(left=0.13,right=0.9,wspace=0.08)
cax=fig.add_axes([0.915,0.15,0.015,0.7]); cb=fig.colorbar(im,cax=cax); cb.set_ticks([0,0.5,1]); cb.set_ticklabels(["0","0.5","1"])
fig.savefig(OUT,dpi=180,bbox_inches="tight",pad_inches=0); plt.close(fig)
print("wrote",os.path.relpath(OUT,ROOT))

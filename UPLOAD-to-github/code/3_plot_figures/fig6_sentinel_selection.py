"""Sentinel-node figure (24 panels split over 2 PNGs forming one figure via \\ContinuedFloat). Rows =
sentinel selection methods; columns = three performance measures (tau', non-sequential detection,
sequential detection), each averaged over the 40 networks and the 10 simulation conditions. Five EWS
lines per panel; original V, CV dotted; baseline-referenced variants solid. x = sentinel fraction (%),
data from 5%. Each panel is labeled (a1)..(h3) at its top-left.
Output: figures/fig_sentinel_p1.png, fig_sentinel_p2.png (manuscript Fig. 6)"""
import os, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
SRC=os.path.join(ROOT,"data","results","sentinel_results.csv")
# The figure files themselves are not distributed with this repository; create their
# directories on demand so that a fresh clone can write into them.
os.makedirs(os.path.join(ROOT,"figures"),exist_ok=True)
FRACS=[5,10,20,30,40,50,60,70,80,90,100]
EWS=["V","CV","Vdelta","Vrel","CVrel"]; ELAB={"V":"$V$","CV":"CV","Vdelta":"$V_\\Delta$","Vrel":"$V_{\\rm rel}$","CVrel":"CV$_{\\rm rel}$"}
STYLE={"V":("#1f77b4",":"),"CV":("#ff7f0e",":"),"Vdelta":("#2ca02c","-"),"Vrel":("#d62728","-"),"CVrel":("#9467bd","-")}
METHODS=[("random","random"),("largeb","largest $b_i$"),("smallb","smallest $b_i$"),
 ("dirb","direction-dependent $b_i$"),("halfb","half-half $b_i$"),
 ("equidist","equidistant $b_i$"),("smallcv","smallest early CV"),("largecv","largest early CV")]
MEAS=[("mean_tau","$\\tau'$"),("mean_nonseq","non-sequential detection"),("mean_seq","sequential detection")]
raw=collections.defaultdict(list)
for r in csv.DictReader(open(SRC)):
    f=int(round(float(r["fraction"])*100))
    for col,_ in MEAS:
        if r[col]!="": raw[(r["method"],f,r["ews"],col)].append(float(r[col]))
def val(method,f,ews,col):
    v=raw.get((method,f,ews,col),[]); return np.mean(v) if v else np.nan
YT=[0,0.2,0.4,0.6,0.8,1.0]; YTL=["0","0.2","0.4","0.6","0.8","1"]
def make_page(methods,out,startidx):
    nm=len(methods); fig,axs=plt.subplots(nm,3,figsize=(13,3.0*nm),squeeze=False)
    for ri,(mkey,mlab) in enumerate(methods):
        letter=chr(97+startidx+ri)
        for ci,(col,clab) in enumerate(MEAS):
            ax=axs[ri][ci]
            for e in EWS:
                c,ls=STYLE[e]; ys=[val(mkey,f,e,col) for f in FRACS]
                ax.plot(FRACS,ys,ls,color=c,lw=2.0,label=ELAB[e] if (ri==0 and ci==0) else None)
            ax.set_xlim(0,100); ax.set_ylim(0,1); ax.set_yticks(YT); ax.set_yticklabels(YTL,fontsize=12)
            ax.grid(alpha=.3); ax.tick_params(labelsize=12)
            ax.text(0.035,0.965,f"({letter}{ci+1})",transform=ax.transAxes,va="top",ha="left",
                    fontsize=15,bbox=dict(fc="white",ec="none",alpha=0.6,pad=1.0))
            if ri==0: ax.set_title(clab,fontsize=17)
            if ci==0: ax.set_ylabel(mlab,fontsize=15)
            if ri==nm-1: ax.set_xlabel("sentinel fraction (%)",fontsize=15)
    h,l=axs[0][0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=5,fontsize=16,frameon=False,bbox_to_anchor=(0.5,-0.012))
    fig.tight_layout(rect=[0,0.03,1,1]); fig.savefig(out,dpi=170,bbox_inches="tight",pad_inches=0); plt.close(fig)
    print("wrote",os.path.relpath(out,ROOT))
make_page(METHODS[:4],os.path.join(ROOT,"figures","fig_sentinel_p1.png"),0)
make_page(METHODS[4:],os.path.join(ROOT,"figures","fig_sentinel_p2.png"),4)

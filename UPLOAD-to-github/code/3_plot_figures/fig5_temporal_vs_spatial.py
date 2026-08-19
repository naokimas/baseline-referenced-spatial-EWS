"""Build the manuscript Fig 5 (DW,D,desc; 1x3) and the three 10-panel SI figures for the temporal-vs-spatial
comparison. Black curve = mean over the 400 (40 networks x 10 nodes) pairs; shaded band = mean +/- 1 std over
the same 400 pairs (from perpair_tau.csv / perpair_supf.csv). The dashed horizontal lines are the
corresponding spatial EWS values, which do not depend on the number of temporal samples L'."""
import os,csv,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
IMG=os.path.join(ROOT,"figures"); SID=os.path.join(ROOT,"Supplementary Material")
# The figure files themselves are not distributed with this repository; create their
# directories on demand so that a fresh clone can write into them.
os.makedirs(IMG,exist_ok=True); os.makedirs(SID,exist_ok=True)
CONDS=["DW $D$ asc","DW $D$ desc","DW $u$ asc","DW $u$ desc","MUT $D$ desc","MUT $u$ desc",
 "SIS $D$ asc","SIS $D$ desc","GR $D$ desc","GR $u$ desc"]
CLAB={"DW $D$ asc":"DW, $D$, asc","DW $D$ desc":"DW, $D$, desc","DW $u$ asc":"DW, $u$, asc","DW $u$ desc":"DW, $u$, desc",
 "MUT $D$ desc":"MUT, $D$, desc","MUT $u$ desc":"MUT, $u$, desc","SIS $D$ asc":"SIS, $D$, asc","SIS $D$ desc":"SIS, $D$, desc",
 "GR $D$ desc":"GR, $D$, desc","GR $u$ desc":"GR, $u$, desc"}
LET=["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)","(j)"]
COL={"Vdelta":"#1f77b4","Vrel":"#ff7f0e","CVrel":"#2ca02c"}
LAB={"Vdelta":"$V_\\Delta$","Vrel":"$V_{\\rm rel}$","CVrel":"CV$_{\\rm rel}$"}
TS=15.5; LS=14; TK=13   # title / label / tick font sizes (~1.5x the originals; titles ~Fig 4)
# ---- per-pair mean/std ----
def load(fn,keys):
    D={c:{} for c in CONDS}
    for r in csv.DictReader(open(os.path.join(ROOT,"data","results",fn))):
        D[r["cond"]][int(r["Lprime"])]=r
    return D
PT=load("perpair_tau.csv",None); PS=load("perpair_supf.csv",None)
def tau_curve(c):
    Ls=sorted(PT[c]); m=np.array([float(PT[c][L]["mean"]) for L in Ls]); sd=np.array([float(PT[c][L]["std"]) for L in Ls])
    return np.array(Ls,float),m,sd
def sup_curve(c,kind):
    p=("ns_" if kind=="nonseq" else "sq_"); Ls=sorted(PS[c])
    m=np.array([float(PS[c][L][p+"mean"]) if PS[c][L][p+"mean"]!="" else np.nan for L in Ls])
    sd=np.array([float(PS[c][L][p+"std"]) if PS[c][L][p+"std"]!="" else np.nan for L in Ls])
    return np.array(Ls,float),m,sd
# ---- spatial references ----
SPtau={}
for r in csv.DictReader(open(os.path.join(ROOT,"data","results","results_full.csv"))):
    key={("var","delta"):"Vdelta",("var","rel"):"Vrel",("CV","rel"):"CVrel"}.get((r["ews"],r["variant"]))
    if key: SPtau.setdefault(r["condition"],{})[key]=float(r["mean_tau"])
SPns={}; SPsq={}
for r in csv.DictReader(open(os.path.join(ROOT,"data","results","sentinel_results.csv"))):
    if abs(float(r["fraction"])-1.0)<1e-9 and r["ews"] in COL:
        SPns.setdefault(r["condition"],{}).setdefault(r["ews"],float(r["mean_nonseq"]))
        SPsq.setdefault(r["condition"],{}).setdefault(r["ews"],float(r["mean_seq"]))
def crossing(Ls,m,ref):
    idx=np.where(m>=ref-1e-9)[0]; return int(Ls[idx[0]]) if len(idx) else None
def draw(ax,Ls,m,sd,refs,title,ylab,circles):
    ax.fill_between(Ls,m-sd,m+sd,color="0.5",alpha=0.22,lw=0)
    ax.plot(Ls,m,color="black",lw=1.9,label="temporal variance")
    for e in ["Vdelta","Vrel","CVrel"]:
        ax.axhline(refs[e],color=COL[e],ls="--",lw=1.4,label=LAB[e],clip_on=False,zorder=4)
        if circles:
            xc=crossing(Ls,m,refs[e])
            if xc is not None: ax.plot([xc],[refs[e]],"o",color=COL[e],ms=5,zorder=5)
    ax.set_xscale("log"); ax.set_xlim(2,200); ax.set_ylim(0.0,1.02)
    ax.set_yticks([0,0.5,1]); ax.set_yticklabels(["0","0.5","1"])
    ax.set_title(title,loc="left",fontsize=TS,pad=4); ax.tick_params(labelsize=TK)
    if ylab: ax.set_ylabel(ylab,fontsize=LS)

# ============ MAIN: DW,D,desc 1x3 ============
c="DW $D$ desc"
fig,axs=plt.subplots(1,3,figsize=(13.6,4.3))
La,m,sd=tau_curve(c);        draw(axs[0],La,m,sd,SPtau[c],"(a) $\\tau'$","$\\tau'$",False)
Ln,m,sd=sup_curve(c,"nonseq");draw(axs[1],Ln,m,sd,SPns[c],"(b) Non-sequential test","sup-$F$ success fraction",False)
Ls,m,sd=sup_curve(c,"seq");   draw(axs[2],Ls,m,sd,SPsq[c],"(c) Sequential test","sup-$F$ success fraction",False)
for ax in axs: ax.set_xlabel("$L'$",fontsize=LS)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,fontsize=LS,frameon=False,bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(rect=[0,0.05,1,1]); fig.savefig(os.path.join(IMG,"fig_temporal_main.png"),dpi=200,bbox_inches="tight",pad_inches=0); plt.close(fig)
print("wrote figures/fig_temporal_main.png")

# ============ SI: three 10-panel figures ============
def si_fig(kind,SP,fname,ylab):
    fig,axes=plt.subplots(5,2,figsize=(9.6,13.4)); axes=axes.ravel()
    for ax,cc,let in zip(axes,CONDS,LET):
        if kind=="tau": Ls,m,sd=tau_curve(cc)
        else: Ls,m,sd=sup_curve(cc,kind)
        draw(ax,Ls,m,sd,SP[cc],f"{let} {CLAB[cc]}",None,False)
    for ax in axes[8:]: ax.set_xlabel("$L'$",fontsize=LS)
    for ax in axes[0::2]: ax.set_ylabel(ylab,fontsize=LS)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,fontsize=LS,frameon=False,bbox_to_anchor=(0.5,-0.008))
    fig.tight_layout(rect=[0,0.03,1,1]); fig.savefig(os.path.join(SID,fname),dpi=170,bbox_inches="tight",pad_inches=0); plt.close(fig)
    print("wrote",fname)
si_fig("tau",SPtau,"fig_si_temporal_tau.png","$\\tau'$")
si_fig("nonseq",SPns,"fig_si_temporal_nonseq.png","sup-$F$ success fraction")
si_fig("seq",SPsq,"fig_si_temporal_seq.png","sup-$F$ success fraction")
# crossings for SI text (mean over 400)
print("crossings (mean reaches spatial ref):")
for lab,SP,fn in [("tau",SPtau,tau_curve),("nonseq",SPns,None),("seq",SPsq,None)]:
    print(" ==",lab)
    for c in CONDS:
        Ls,m,_=(tau_curve(c) if lab=="tau" else sup_curve(c,lab))
        print("   ",CLAB[c],{e:(crossing(Ls,m,SP[c][e]) or ">200") for e in ["Vdelta","Vrel","CVrel"]})

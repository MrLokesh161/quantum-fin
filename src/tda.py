"""Classical TDA with Ripser. It supplies vectors compatible with a future VQC."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tda_features(point_clouds, figure_dir, stride=20):
    try: from ripser import ripser
    except ImportError: return pd.DataFrame(index=range(len(point_clouds))),"Skipped: ripser is not installed."
    rows=[]; example=None
    # Adjacent daily windows overlap heavily. Sampling reduces redundant PH work;
    # the resulting topological state is carried forward until the next sample.
    for cloud in point_clouds[::stride]:
        diagrams=ripser(cloud,maxdim=1)["dgms"]; row={}
        for dim,diagram in enumerate(diagrams):
            life=diagram[:,1]-diagram[:,0]; life=life[np.isfinite(life)]; p=life/life.sum() if life.sum()>0 else []
            prefix=f"tda_h{dim}"; row.update({f"{prefix}_betti":len(life),f"{prefix}_mean_lifetime":life.mean() if len(life) else 0.,f"{prefix}_median_lifetime":np.median(life) if len(life) else 0.,f"{prefix}_max_persistence":life.max() if len(life) else 0.,f"{prefix}_total_persistence":life.sum() if len(life) else 0.,f"{prefix}_persistence_variance":life.var() if len(life) else 0.,f"{prefix}_persistent_count":int((life>(np.median(life) if len(life) else 0)).sum()),f"{prefix}_long_lived_count":int((life>(life.mean()+life.std() if len(life) else 0)).sum()),f"{prefix}_persistence_entropy":float(-(p*np.log(p)).sum()) if len(p) else 0.})
        row["tda_euler_characteristic"]=row.get("tda_h0_betti",0)-row.get("tda_h1_betti",0)
        rows.append(row); example=example or diagrams
    if example:
        plt.figure(figsize=(6,5))
        for i,d in enumerate(example):
            d=d[np.isfinite(d).all(axis=1)]
            if len(d): plt.scatter(d[:,0],d[:,1],label=f"H{i}")
        plt.xlabel("Birth");plt.ylabel("Death");plt.legend();plt.title("Example persistence diagram");plt.tight_layout();plt.savefig(figure_dir/"persistence_diagram.png",dpi=180);plt.close()
        fig,ax=plt.subplots(figsize=(8,4));y=0
        for dim,d in enumerate(example):
            for birth,death in d[np.isfinite(d).all(axis=1)]: ax.plot([birth,death],[y,y],lw=2,label=f"H{dim}" if y==0 else None);y+=1
        ax.set_title("Example persistence barcode");ax.set_xlabel("Filtration");fig.tight_layout();fig.savefig(figure_dir/"persistence_barcode.png",dpi=180);plt.close(fig)
    sampled=pd.DataFrame(rows)
    full=sampled.reindex(range(len(point_clouds)//stride+1)).ffill().bfill()
    full.index=range(0,len(full)*stride,stride)
    return full.reindex(range(len(point_clouds))).ffill().bfill(),f"Completed with Ripser (every {stride} windows; carried forward between samples)."

"""Sliding-window persistent-homology summaries with Ripser."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tda_features(point_clouds, figure_dir, stride=20):
    if stride < 1:
        raise ValueError("TDA stride must be at least one.")
    try: from ripser import ripser
    except ImportError: return pd.DataFrame(index=range(len(point_clouds))),"Skipped: ripser is not installed."
    rows=[]; example=None
    sample_indexes=range(0,len(point_clouds),stride)
    # Each cloud is available at its end date. Normalize its state dimensions
    # locally so raw prices cannot dominate dimensionless indicators.
    for sample_index in sample_indexes:
        cloud=np.asarray(point_clouds[sample_index],dtype=float)
        if cloud.ndim!=2:
            raise ValueError(f"Point cloud {sample_index} must be a 2D array.")
        cloud=cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud)<2:
            diagrams=[np.empty((0,2)),np.empty((0,2))]
        else:
            scale=cloud.std(axis=0)
            scale[scale==0]=1.0
            cloud=(cloud-cloud.mean(axis=0))/scale
            try: diagrams=ripser(cloud,maxdim=1)["dgms"]
            except Exception as exc:
                raise RuntimeError(f"Persistent homology failed for point cloud {sample_index}: {exc}") from exc
        row={}
        for dim,diagram in enumerate(diagrams):
            diagram=np.asarray(diagram,dtype=float).reshape(-1,2)
            finite=diagram[np.isfinite(diagram).all(axis=1)]
            life=finite[:,1]-finite[:,0] if len(finite) else np.array([],dtype=float)
            life=life[life>0]
            p=life/life.sum() if life.sum()>0 else np.array([],dtype=float)
            epsilon=float(np.median(finite[:,1])) if len(finite) else 0.0
            betti=int(np.sum((diagram[:,0]<=epsilon)&(diagram[:,1]>epsilon))) if len(diagram) else 0
            prefix=f"tda_h{dim}"
            row.update({
                f"{prefix}_betti":betti,
                f"{prefix}_mean_lifetime":life.mean() if len(life) else 0.,
                f"{prefix}_median_lifetime":np.median(life) if len(life) else 0.,
                f"{prefix}_max_persistence":life.max() if len(life) else 0.,
                f"{prefix}_total_persistence":life.sum() if len(life) else 0.,
                f"{prefix}_persistence_variance":life.var() if len(life) else 0.,
                f"{prefix}_persistent_count":int((life>(np.median(life) if len(life) else 0)).sum()),
                f"{prefix}_long_lived_count":int((life>(life.mean()+life.std() if len(life) else 0)).sum()),
                f"{prefix}_persistence_entropy":float(-(p*np.log(p)).sum()) if len(p) else 0.,
            })
        row["tda_euler_characteristic"]=row.get("tda_h0_betti",0)-row.get("tda_h1_betti",0)
        rows.append(row)
        if example is None:
            example=diagrams
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
    sampled=pd.DataFrame(rows,index=list(sample_indexes))
    full=sampled.reindex(range(len(point_clouds))).ffill()
    return full,f"Completed with Ripser (every {stride} windows; carried forward from the most recent past sample)."

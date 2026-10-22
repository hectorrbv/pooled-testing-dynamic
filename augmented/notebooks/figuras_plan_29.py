"""Minimal Q2/Q1 maps for the October 20 discussion; no experiments here."""
import csv
from hashlib import sha256
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def plan_figures(root):
    path=root/"results/precio_q05_G/malla_2026-10-14_completa.csv"
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==32 and all(r["general_status"]=="exact" and r["policy_status"]=="evaluated" for r in rows)
    meta=json.loads(path.with_suffix(".meta.json").read_text())
    for source,digest in meta["source_sha256"].items():
        assert sha256((root/source).read_bytes()).hexdigest()==digest,source
    cmap=LinearSegmentedColormap.from_list("research_loss",["#fafafa","#b9cbd7","#526e80"])
    images={}
    def draw(name,panels,maximum):
        fig,axes=plt.subplots(1,2,figsize=(9.4,3.8),layout="constrained")
        for ax,(title,values) in zip(axes,panels):
            arr=np.array(values)
            im=ax.imshow(arr,cmap=cmap,vmin=0,vmax=maximum,aspect="auto")
            ax.set_xticks(range(4),[2,3,4,5]);ax.set_yticks(range(4),[1,2,3,4])
            ax.set(xlabel="G",ylabel="B",title=title)
            ax.tick_params(length=0)
            for spine in ax.spines.values():spine.set_visible(False)
            for (y,x),value in np.ndenumerate(arr):
                label="0" if value<1e-8 else f"{value:.1f}"
                ax.text(x,y,label,ha="center",va="center",fontsize=12,
                        color="white" if value>maximum*.72 else "#262626")
        fig.colorbar(im,ax=axes,shrink=.82,label="Pérdida (%)",ticks=[0,maximum/2,maximum])
        buffer=io.BytesIO();fig.savefig(buffer,format="png",dpi=160);plt.close(fig)
        images[name]=buffer.getvalue()
    def value(n,B,G,key):
        return float(next(r for r in rows if (int(r['n']),int(r['B']),int(r['G']))==(n,B,G))[key])
    q2=[(f"n = {n}",[[100*(1-value(n,B,G,"ratio")) for G in [2,3,4,5]] for B in [1,2,3,4]]) for n in [6,7]]
    q1=[(title,[[100*(1-min(value(n,B,G,key) for n in [6,7])) for G in [2,3,4,5]] for B in [1,2,3,4]])
        for title,key in [("Greedy","greedy_ratio"),("Rollout","rollout_ratio")]]
    draw("precio_laminar_G",q2,25)
    draw("greedy_rollout_G",q1,30)
    worst=min(rows,key=lambda r:float(r['ratio']))
    summary=dict(cases=32,grid_sha256=sha256(path.read_bytes()).hexdigest(),worst_q2=worst,
                 greedy_min=min(float(r['greedy_ratio']) for r in rows),
                 rollout_min=min(float(r['rollout_ratio']) for r in rows),
                 rollout_exact_cases=sum(abs(float(r['rollout_ratio'])-1)<1e-10 for r in rows),
                 information_checks=sum(r['information_check']=='pass' for r in rows),
                 q1_aggregation="minimum over n=6,7 separately for each B,G and policy",
                 q2_panels={title:values for title,values in q2},q1_panels={title:values for title,values in q1})
    out=root/'results/precio_q05_G/figuras';out.mkdir(exist_ok=True)
    aggregates=[]
    for B in [1,2,3,4]:
        for G in [2,3,4,5]:
            candidates=[r for r in rows if int(r['B'])==B and int(r['G'])==G]
            record=dict(B=B,G=G)
            for label,key in [('laminar_general','ratio'),('greedy','greedy_ratio'),
                              ('rollout','rollout_ratio'),('hierarchy','hierarchy_ratio')]:
                minimum=min(float(r[key]) for r in candidates)
                record[label+'_min_ratio']=minimum
                record[label+'_argmin_n']=';'.join(r['n'] for r in candidates if abs(float(r[key])-minimum)<1e-10)
            aggregates.append(record)
    with (out.parent/'resumen_por_G_B.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(aggregates[0]))
        writer.writeheader();writer.writerows(aggregates)
    for name,png in images.items():(out/(name+'.png')).write_bytes(png)
    (out/'datos_figuras.json').write_text(json.dumps(summary,indent=2)+'\n')
    return images,summary

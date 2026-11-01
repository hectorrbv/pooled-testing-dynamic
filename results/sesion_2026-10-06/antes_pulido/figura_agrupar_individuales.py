"""Render the grouping-versus-individuals experiment; no notebook is overwritten."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/diseno_mapa_2026-10-05'
r=json.loads((OUT/'agrupar_vs_individuales.json').read_text())
qs=sorted({x['q_decimal'] for x in r['rows']})
fig,axes=plt.subplots(1,2,figsize=(11.8,4.9),sharey=True)
fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.85,wspace=.15)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
cmap=LinearSegmentedColormap.from_list('group_gain',['#ffffff','#c6d5df','#34536b'])
cmap.set_bad('#dedede')
mx=max(x['gain_percent'] for x in r['rows'] if x['status']=='exact')
for ax,G in zip(axes,[2,3]):
 z=np.full((6,19),np.nan)
 for x in r['rows']:
  if x['G']==G and x['status']=='exact':z[x['B']-1,qs.index(x['q_decimal'])]=x['gain_percent']
 im=ax.imshow(z,aspect='auto',cmap=cmap,vmin=0,vmax=mx)
 ax.set_title(f'G = {G}',fontsize=13,pad=9)
 ax.set_xticks([1,5,9,13,17],labels=['0.1','0.3','0.5','0.7','0.9'])
 ax.set_yticks(range(6),labels=[str(b) for b in range(1,7)])
 ax.set_xlabel('q (probabilidad de estar sano)',fontsize=11)
 ax.tick_params(length=0)
 for spine in ax.spines.values():spine.set_visible(False)
 # A few exact values orient the eye; leave the complete grid uncluttered.
 for i in range(6):
  for j in [1,9,17]:
   if not np.isnan(z[i,j]):
    ax.text(j,i,f'{z[i,j]:.0f}%',ha='center',va='center',fontsize=9,
            color='white' if z[i,j]>.68*mx else '#262626')
axes[0].set_ylabel('B (pruebas disponibles)',fontsize=11)
fig.suptitle('Permitir grupos frente a usar solo individuales · n=6',fontsize=13,y=.98)
cax=fig.add_axes([.30,.10,.43,.035]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,50,100,150])
cb.set_label('Mejora de utilidad frente a individuales (%)',fontsize=11);cb.outline.set_visible(False)
fig.savefig(OUT/'agrupar_vs_individuales.png',dpi=160,bbox_inches='tight',facecolor='white')
plt.close(fig)
summary={'cases':len(r['rows']),'exact_cases':r['audit']['exact_cases'],
         'positive_gain':sum(x['status']=='exact' and x['gain_percent']>0 for x in r['rows']),
         'ties':sum(x['status']=='exact' and x['gain_percent']==0 for x in r['rows']),
         'largest_gain':max((x for x in r['rows'] if x['status']=='exact'),key=lambda x:x['gain_percent'])}
(OUT/'resumen.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

"""Render saved evidence standalone or inside notebook 28, without rerunning it."""
from pathlib import Path
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/diseno_mapa_2026-10-05'

def make_grouping_figure(r, *, field='gain_percent', label=None, normalization=None):
 qs=sorted({x['q_decimal'] for x in r['rows']})
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
 fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),sharey=True)
 fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.86,wspace=.15)
 cmap=LinearSegmentedColormap.from_list('group_gain',['#ffffff','#c6d5df','#34536b'])
 cmap.set_bad('#dedede')
 mx=max(x[field] for x in r['rows'] if x.get('status','exact')=='exact')
 def value_label(value):
  if field=='crossing_loss_percent':
   if value==0:return '0%'
   return '<0.1%' if value<.1 else f'{value:.1f}%'
  return f'{value:.0f}%'
 panels={}
 for ax,G in zip(axes,[2,3]):
  z=np.full((6,len(qs)),np.nan)
  for x in r['rows']:
   if x['G']==G and x.get('status','exact')=='exact':z[x['B']-1,qs.index(x['q_decimal'])]=x[field]
  panels[str(G)]=z.tolist()
  im=ax.imshow(z,aspect='auto',cmap=cmap,vmin=0,vmax=mx)
  ax.set_title(f'G = {G}',fontsize=13,pad=9)
  ax.set_xticks([1,5,9,13,17],labels=['0.1','0.3','0.5','0.7','0.9'])
  ax.set_yticks(range(6),labels=[str(b) for b in range(1,7)])
  ax.set_xlabel('q (probabilidad de estar sano)',fontsize=11)
  ax.tick_params(length=0)
  for spine in ax.spines.values():spine.set_visible(False)
  # Representative columns, plus the q=.3, B=2 example explained next.
  for i in range(6):
   for j in [1,9,17]:
    if not np.isnan(z[i,j]):
     ax.text(j,i,value_label(z[i,j]),ha='center',va='center',fontsize=9,
             color='white' if z[i,j]>.68*mx else '#262626')
  j=qs.index(.3)
  ax.text(j,1,value_label(z[1,j]),ha='center',va='center',fontsize=9,color='#262626')
 axes[0].set_ylabel('B (pruebas disponibles)',fontsize=11)
 ticks=None if field=='crossing_loss_percent' else [0,50,100,150]
 cax=fig.add_axes([.30,.10,.43,.035]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ticks)
 cb.set_label(label or 'Mejora frente a individuales (%) · blanco = empate',fontsize=11);cb.outline.set_visible(False)
 buf=io.BytesIO()
 fig.savefig(buf,format='png',dpi=160,bbox_inches='tight',facecolor='white')
 plt.close(fig)
 return buf.getvalue(),{'q':qs,'values_by_G':panels,'field':field,'n':6,'cases':len(r['rows']),
                       'normalization':normalization or '100*(optimal_general/individual_only - 1)',
                       'shared_scale':[0,mx]}


def main():
 r=json.loads((OUT/'agrupar_vs_individuales.json').read_text())
 png,_=make_grouping_figure(r)
 (OUT/'agrupar_vs_individuales.png').write_bytes(png)
 summary={'cases':len(r['rows']),'exact_cases':r['audit']['exact_cases'],
          'positive_gain':sum(x['status']=='exact' and x['gain_percent']>0 for x in r['rows']),
          'ties':sum(x['status']=='exact' and x['gain_percent']==0 for x in r['rows']),
          'largest_gain':max((x for x in r['rows'] if x['status']=='exact'),key=lambda x:x['gain_percent'])}
 (OUT/'resumen.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))


if __name__=='__main__':
 main()

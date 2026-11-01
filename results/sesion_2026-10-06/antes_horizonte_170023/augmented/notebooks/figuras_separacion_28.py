"""Readable overview, realized paths, and full trees for the exact 3 vs 2.5 case."""
from fractions import Fraction as F
from pathlib import Path
import io
import graphviz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, GRAY, BLUE, WARM = '#262626', '#777777', '#34536b', '#986347'


def people(ids):
    return ''.join('ABCDEF'[i] for i in ids) or 'ninguno'


def num(value):
    return f'{float(F(value)):g}'


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    return fig, ax


def text(ax, x, y, s, size=14, color=INK, **kw):
    return ax.text(x, y, s, ha='center', va='center', fontsize=size,
                   color=color, linespacing=1.45, **kw)


def box(ax, x, y, w, h, s, color=INK, size=14, fill='white', lw=1):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle='round,pad=0.006,rounding_size=0.015',
                               facecolor=fill, edgecolor=color, lw=lw, zorder=3))
    text(ax, x, y, s, size, color, zorder=4)


def line(ax, a, b, color=GRAY, lw=1):
    ax.plot([a[0],b[0]], [a[1],b[1]], color=color, lw=lw, zorder=1)


def png(fig):
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig);return buf.getvalue()


def make_separation_figures(evidence):
    images, metadata = {}, {}
    fig, ax = canvas(13.4, 3.8)
    ax.set_ylim(.225, 1)
    for name, cx, color, title in [('general',.245,WARM,'General · 3'),
                                    ('laminar',.755,BLUE,'Laminar · 2.5')]:
        tree=evidence['policies'][name]['tree']
        text(ax,cx,.945,title,19,color)
        box(ax,cx,.775,.12,.105,'AB',color,17)
        for j,branch in enumerate(tree['branches']):
            x=cx+(j-1)*.16
            line(ax,(cx,.715),(x,.47),color if j==1 else GRAY,1.7 if j==1 else .8)
            text(ax,x,.58,f"R = {branch['infected_count']}\nP = {branch['probability']}",12,
                 color if j==1 else GRAY,bbox={'facecolor':'white','edgecolor':'none','pad':2})
            child=branch['next']
            box(ax,x,.355,.139,.205,
                f"{people(child['pool'])} → …\n{num(child['value'])}",
                color if j==1 else GRAY,14,
                '#f6efea' if name=='general' and j==1 else '#edf2f5' if j==1 else 'white',
                1.7 if j==1 else .8)
    images['separacion_arbol_resumen']=png(fig)
    metadata['separacion_arbol_resumen']={'branches':evidence['root_branches'],
        'conditional_values_include_previous_credit':True,'subtrees_collapsed_after_second_action':True}

    fig, ax = canvas(11.8, 6.5)
    ax.set_ylim(.075, .93)
    box(ax,.5,.872,.11,.072,'AB',INK,17)
    line(ax,(.5,.83),(.5,.776),INK)
    text(ax,.56,.79,'R = 1',12,GRAY)
    for name,cx,color in [('general',.265,WARM),('laminar',.735,BLUE)]:
        witness=evidence['witness']['policies'][name]
        text(ax,cx,.795,'General' if name=='general' else 'Laminar',17,color)
        line(ax,(.5,.776),(cx,.713),color,1.5)
        steps=witness['steps'][1:]
        for j,step in enumerate(steps):
            y=.67-j*.16
            box(ax,cx,y,.15,.075,people(step['pool']),color,17,
                '#f6efea' if name=='general' and j==0 else '#edf2f5' if j==0 else 'white',
                1.8 if j==0 else 1)
            nxt=y-.16 if j<2 else .158
            line(ax,(cx,y-.044),(cx,nxt+.043),color,1.25)
            text(ax,cx+.073,(y+nxt)/2,f"R = {step['R']}",12,GRAY,
                 bbox={'facecolor':'white','edgecolor':'none','pad':2})
        ending=f"{', '.join('ABCDEF'[i] for i in witness['healthy'])}\n{witness['utility']}"
        box(ax,cx,.15,.24,.11,ending,color,14)
    images['separacion_ruta_comparada']=png(fig)
    metadata['separacion_ruta_comparada']=evidence['witness']
    return images, metadata


def export_full_trees(evidence, output):
    """Full policies retained as zoomable SVGs; selected realization highlighted."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    actual=evidence['witness']['infection_profile']
    for name,color in [('general',WARM),('laminar',BLUE)]:
        tree=evidence['policies'][name]['tree'];counter=0
        g=graphviz.Digraph(name,format='svg')
        g.attr(rankdir='TB',bgcolor='white',ordering='out',nodesep='.12',ranksep='.48',pad='.15')
        g.attr('node',fontname='Helvetica',fontsize='12',shape='box',style='rounded',margin='.1,.08')
        g.attr('edge',fontname='Helvetica',fontsize='10',arrowsize='.5')
        g.attr(label=f"{name.capitalize()} · n=6, B=4, G=3, q=0.5 · E[U]={num(tree['value'])}\nRuta resaltada: B y D infectados",labelloc='t',fontsize='18',fontname='Helvetica')
        def visit(node,on_path):
            nonlocal counter
            nid=f'n{counter}';counter+=1
            shade=color if on_path else '#a7a7a7'
            if node.get('stop'):
                label=f"Sanos: {people(node['healthy'])}\nU = {num(node['value'])}"
            else:
                label=f"{people(node['pool'])}\nb = {node['remaining_tests']} · E[U] = {num(node['value'])}"
            g.node(nid,label=label,color=shade,fontcolor=INK if on_path else GRAY,penwidth='1.6' if on_path else '.7')
            for branch in node.get('branches',[]):
                active=on_path and sum(actual[i] for i in node['pool'])==branch['infected_count']
                child=visit(branch['next'],active)
                g.edge(nid,child,label=f"R={branch['infected_count']} · P={branch['probability']}",
                       color=color if active else '#c6c6c6',fontcolor=color if active else GRAY,
                       penwidth='1.8' if active else '.6')
            return nid
        visit(tree,True)
        assert counter==evidence['policies'][name]['nodes_audited']
        (output/f'arbol_{name}_completo.svg').write_bytes(g.pipe(format='svg'))
        (output/f'arbol_{name}_completo.dot').write_text(g.source)

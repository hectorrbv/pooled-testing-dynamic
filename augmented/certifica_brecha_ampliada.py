"""Independent profile-polynomial certificate for a 10/9 limiting family."""
from datetime import datetime,timezone
from fractions import Fraction as F
from itertools import combinations,product
from pathlib import Path
from hashlib import sha256
import json
import sympy as sp
from augmented.experimento_sesion_29 import measure

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sesion_2026-10-12/brecha_ampliada'
s=sp.Symbol('s',real=True)


def minimum_quadratic(poly,left=F(19,20),right=F(1)):
    polynomial=sp.Poly(sp.expand(poly),s)
    assert polynomial.degree()<=2 or polynomial.is_zero
    coeff=[F(polynomial.nth(i)) for i in range(3)]
    points=[left,right]
    if coeff[2]>0:
        critical=-coeff[1]/(2*coeff[2])
        if left<critical<right:points.append(critical)
    vals=[sum(a*x**k for k,a in enumerate(coeff)) for x in points]
    return min(vals)


def symbolic_family():
    q=[s,sp.Rational(1,2),sp.Rational(1,2),s];u=[1,3,3,2]
    states=list(product((0,1),repeat=4))
    weights={z:sp.expand(sp.prod(q[i] if x==0 else 1-q[i] for i,x in enumerate(z))) for z in states}
    pools=[tuple(a) for k in [1,2] for a in combinations(range(4),k)]
    result={};inequalities=[]
    for mode in ['binary','laminar']:
        def obs(z,a):
            return int(any(z[i] for i in a)) if mode=='binary' else sum(z[i] for i in a)
        roots=[]
        for a in pools:
            root_value=0;branches=[]
            for r in sorted({obs(z,a) for z in states}):
                support=[z for z in states if obs(z,a)==r]
                cleared=set(a) if r==0 else set()
                choices=[]
                for b in [()]+pools:
                    if mode=='laminar' and set(a)&set(b) and not(set(a)<=set(b) or set(b)<=set(a)):continue
                    score=0
                    for t in sorted({obs(z,b) for z in support}):
                        leaf=[z for z in support if obs(z,b)==t]
                        healthy=cleared|(set(b) if t==0 else set()) if mode=='binary' else {i for i in range(4) if all(z[i]==0 for z in leaf)}
                        score+=sum(weights[z] for z in leaf)*sum(u[i] for i in healthy)
                    choices.append((b,sp.expand(score)))
                best_b,best=max(choices,key=lambda x:x[1].subs(s,sp.Rational(99,100)))
                for b,v in choices:
                    diff=sp.expand(best-v);minimum=minimum_quadratic(diff)
                    assert minimum>=0,(mode,a,r,b,diff,minimum)
                    inequalities.append(dict(mode=mode,first=a,result=r,best_second=best_b,alternative=b,
                        difference=str(diff),minimum_on_closed_interval=str(minimum)))
                root_value+=best
                branches.append(dict(result=r,second=best_b,unnormalized_value=str(best)))
            roots.append(dict(first=a,value=sp.expand(root_value),branches=branches))
        best=max(roots,key=lambda x:x['value'].subs(s,sp.Rational(99,100)))
        for row in roots:
            diff=sp.expand(best['value']-row['value']);minimum=minimum_quadratic(diff)
            assert minimum>=0,(mode,row,diff,minimum)
            inequalities.append(dict(mode=mode,best_first=best['first'],alternative_first=row['first'],
                                      difference=str(diff),minimum_on_closed_interval=str(minimum)))
        result[mode]=dict(value=str(best['value']),first=best['first'],
                         roots=[dict(row,value=str(row['value'])) for row in roots])
    assert sp.expand(sp.sympify(result['binary']['value'],locals={'s':s})-sp.Rational(5,2)*s*(1+s))==0
    assert sp.expand(sp.sympify(result['laminar']['value'],locals={'s':s})-3-sp.Rational(3,2)*s*s)==0
    result.update(n=4,B=2,G=2,q=['s','1/2','1/2','s'],u=[1,3,3,2],
                  interval='19/20 <= s < 1',limit_ratio='10/9',
                  endpoint_warning='At s=1, soft clearing knows A and D healthy initially. Formulas describe the left limit, not that endpoint instance.',
                  certification='All first tests and all second tests evaluated from raw profiles; every maximizing inequality checked exactly as a quadratic on [19/20,1].',
                  inequalities_checked=len(inequalities),inequalities=inequalities)
    return result


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    family=symbolic_family();examples=[]
    for health in ['.95','.99','.9999','.999999']:
        m=measure([health,'.5','.5',health],2,2,[1,3,3,2],heuristics=False)
        m['label']='family_s='+health
        b=F(m['policies']['binary']['value']);l=F(m['policies']['laminar']['value'])
        x=F(health);assert b==F(5,2)*x*(1+x) and l==3+F(3,2)*x*x
        m['ratio']=str(b/l);m['ratio_decimal']=float(b/l);examples.append(m)
    summaries=[]
    for p in sorted(OUT.glob('*/summary.json')):
        raw=json.loads(p.read_text())
        if 'calls' not in raw:continue
        summaries.append(dict(directory=p.parent.name,**raw))
    by_kind={}
    for r in summaries:
        b=r['best'];key=(b['mode'],b['B'])
        if key not in by_kind or b['ratio']>by_kind[key]['ratio']:by_kind[key]=b
    for (mode,B),row in by_kind.items():
        # All chosen search winners are small enough for the separate reference DP.
        if row['n']>6 or B>3:continue
        q=[F(max(1,min(999999,round(x*1000000))),1000000) for x in row['q']]
        u=[F(max(1,round(x*1000000)),1000000) for x in row['u']]
        m=measure(q,B,row['G'],u,heuristics=False)
        m.update(label=f'best_float_{mode}_B{B}_rationalized',float_candidate=row)
        b=F(m['policies']['binary']['value']);l=F(m['policies']['laminar']['value'])
        m['ratio']=str(b/l);m['ratio_decimal']=float(b/l);examples.append(m)
    sources=['augmented/certifica_brecha_ampliada.py','augmented/busqueda_brecha_ampliada.py',
             'augmented/busqueda_brecha_compilada.py','augmented/busca_brecha_presupuestos.py',
             'augmented/bellman_binario_exacto.py','augmented/bellman_perfiles.py','augmented/bm17_toy_solver.py']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),family=family,certified_examples=examples,
                completed_optimization_runs=len(summaries),floating_evaluations=sum(r['calls'] for r in summaries),
                optimization_runs=summaries,
                source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    (OUT/'certificados.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(symbolic_inequalities=family['inequalities_checked'],runs=len(summaries),
        floating_evaluations=result['floating_evaluations'],certified=[dict(label=x['label'],ratio=x['ratio_decimal']) for x in examples]),indent=2))


if __name__=='__main__':run()

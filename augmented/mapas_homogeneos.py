"""Generate the requested homogeneous action map with all root ties retained."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
import json
from time import perf_counter

from augmented.bellman_tipos import crear_solver
from augmented.provenance import write_canonical_csv

ROOT=Path(__file__).resolve().parent.parent


def one_grid(item):
    q,G=item
    start=perf_counter()
    v=crear_solver(1-q,G,backend='compiled')
    rows=[]
    for B in range(2,9):
        n=B*G
        opt=v(n,(),B)
        actions=v.action_values(n,(),B)
        sizes=[a[1] for a in v.optimal_actions(n,(),B)]
        single=next(w for a,w in actions if a[1]==1)
        group=max(w for a,w in actions if a[1]>1)
        rows.append(dict(q_sano=q,n=n,G=G,B=B,u=1,
            convencion='posterior_zero',clase='pathwise',
            optimo=round(opt,14),baseline_singletons=round(B*q,14),
            ratio_vs_singletons=round(opt/(B*q),12),
            tamanos_optimos=';'.join(map(str,sizes)),
            singleton_optimo=1 in sizes,grupo_optimo=any(k>1 for k in sizes),
            preferencia='empate_individual_grupo' if 1 in sizes and len(sizes)>1 else
                        'solo_individual' if sizes==[1] else 'solo_grupo',
            valor_singleton_primero=round(single,14),
            valor_mejor_grupo_primero=round(group,14),
            brecha_grupo_individual=round(group-single,14)))
    return rows,dict(q=q,G=G,seconds=perf_counter()-start,states=len(v.memo))


def map1(workers=2):
    work=[(round(i/20,2),G) for G in [2,3,4,5,8] for i in range(1,20)]
    rows,metrics=[],[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(one_grid,item):item for item in work}
        for i,future in enumerate(as_completed(tasks),1):
            chunk,metric=future.result();rows.extend(chunk);metrics.append(metric)
            if i%10==0 or i==len(work):
                print(f'Mapa 1: {i}/{len(work)} bloques completos, {len(rows)}/665 celdas',flush=True)
    rows.sort(key=lambda r:(r['G'],r['q_sano'],r['B']))
    metrics.sort(key=lambda r:(r['G'],r['q']))
    assert len(rows)==665
    hashes={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ['augmented/bellman_tipos.py','augmented/bellman_tipos_compilado.py','augmented/mapas_homogeneos.py']}
    out=ROOT/'results/mapa_homogeneo_1.csv'
    write_canonical_csv(out,rows,generator='augmented.mapas_homogeneos.map1',seed=None,
        params=dict(q=[round(i/20,2) for i in range(1,20)],G=[2,3,4,5,8],B=list(range(2,9)),
                    n='B*G: saturated population, also valid for any n>=B*G',
                    convention='posterior_zero',class_policy='pathwise_laminar',
                    backend='compiled exact-state float64',tie_tolerance='1e-10*max(1,abs(V))',
                    source_sha256=hashes))
    (ROOT/'results/mapa_homogeneo_1.runtime.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Wrote',out,flush=True)
    return rows


if __name__=='__main__':
    map1()

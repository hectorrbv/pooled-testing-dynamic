"""Hunt A: instances where optimal dynamic BINARY beats optimal LAMINAR augmented.
By V4 they must have Lambda>1, so sample q_i with some q_i>1/2."""
import sys, random, time, csv

from fractions import Fraction as F
from augmented.dp_perfiles_binario_conteo import DP, Lambda
from augmented.bellman_perfiles import BellmanPerfiles
seed=int(sys.argv[1]); budget=float(sys.argv[2]); nmax=int(sys.argv[3]); out=sys.argv[4]
rng=random.Random(seed); t0=time.time(); rows=[]; worst=None; nwins=0; tot=0
grid=[F(k,10) for k in range(1,10)]
while time.time()-t0<budget:
    n=rng.randint(3,nmax); G=rng.randint(2,min(3,n)); B=rng.randint(2,3)
    kind=rng.random()
    if kind<0.3: q=[rng.choice(grid[4:])]*n            # homogeneous q>=0.5
    elif kind<0.6: q=[rng.choice(grid) for _ in range(n)]
    else: q=[rng.choice([F(1,10),F(9,10),F(1,2),F(7,10),F(3,10)]) for _ in range(n)]
    u=[F(1)]*n if rng.random()<0.5 else [F(rng.randint(1,8),4) for _ in range(n)]
    lam=Lambda(q,G)
    if lam<=1: continue
    binv=DP(q,u,G,'binary').value(B)
    lamv=BellmanPerfiles([1-x for x in q],u,G,laminar=True).solve(B).value
    augv=DP(q,u,G,'count').value(B)
    tot+=1; r=lamv/binv
    if r<1: nwins+=1
    rows.append((n,B,G,[str(x) for x in q],[str(x) for x in u],str(lam),str(binv),str(lamv),str(augv),float(r)))
    if worst is None or r<worst[-1]: worst=rows[-1]
with open(out,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['n','B','G','q','u','Lambda','binary_dyn','laminar_aug','aug_unrestricted','ratio_lam_over_bin']); w.writerows(rows)
print('seed',seed,'instances',tot,'binary_beats_laminar',nwins,'min ratio',worst[-1] if worst else None)
print('worst',worst)

"""Hunt B: price of laminarity grid, laminar vs unrestricted augmented, q in {0.3,0.5,0.7}, u=1."""
import sys, time, csv

from fractions import Fraction as F
from augmented.dp_perfiles_binario_conteo import DP
from augmented.bellman_perfiles import BellmanPerfiles
out=sys.argv[1]; nmax=int(sys.argv[2]); rows=[]
for n in range(4,nmax+1):
  for q in (F(3,10),F(1,2),F(7,10)):
    for G in (2,3,4):
      for B in (2,3,4):
        if G>n or B>n: continue
        t=time.time()
        lam=BellmanPerfiles([1-q]*n,[F(1)]*n,G,laminar=True).solve(B).value
        aug=DP([q]*n,[F(1)]*n,G,'count').value(B)
        rows.append((n,float(q),G,B,str(lam),str(aug),float(lam/aug),round(time.time()-t,1)))
        print(rows[-1],flush=True)
with open(out,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['n','q','G','B','laminar','unrestricted','ratio','sec']); w.writerows(rows)

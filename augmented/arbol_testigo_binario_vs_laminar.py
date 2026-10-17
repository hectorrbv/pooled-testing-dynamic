import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-mac-Desktop-ASE-group-count-dynamic/ae158aa7-d20a-4b59-97b4-5e83d15eba80/scratchpad')
from fractions import Fraction as F
from augmented.dp_perfiles_binario_conteo import DP
from augmented.bellman_perfiles import BellmanPerfiles
names='abc'
def tree(dp, supp, b, cleared, depth=0):
    if b==0: return
    best=None
    for T in dp.pools:
        groups={}; z=supp
        while z:
            low=z&-z; idx=low.bit_length()-1; z^=low
            cnt=bin(idx&T).count('1'); key=(cnt>0) if dp.ch=='binary' else cnt
            groups[key]=groups.get(key,0)|low
        val=F(0); ptot=dp.P(supp); br=[]
        for key,sub in sorted(groups.items()):
            pr=dp.P(sub)/ptot
            if pr==0: continue
            h=dp._healthy_all(sub); new=h&~cleared
            r=sum(dp.u[i] for i in range(dp.n) if new>>i&1)
            val+=pr*(r+dp.V(sub,b-1,cleared|new)); br.append((key,sub,pr,new,r))
        if best is None or val>best[0]: best=(val,T,br)
    val,T,br=best
    print('  '*depth+f"test {{{''.join(names[i] for i in range(dp.n) if T>>i&1)}}}  value {float(val):.4f}")
    for key,sub,pr,new,r in br:
        print('  '*depth+f" -> result {key} (p={float(pr):.3f}) clears {{{''.join(names[i] for i in range(dp.n) if new>>i&1)}}} pays {float(r)}")
        tree(dp, sub, b-1, cleared|new, depth+1)
q=[F(9,10),F(7,10),F(7,10)]; u=[F(1)]*3; G=3; B=2
for ch in ('binary','count'):
    dp=DP(q,u,G,ch); print('=====',ch,'unrestricted, value',dp.value(B)); tree(dp,(1<<8)-1,B,0)
bp=BellmanPerfiles([1-x for x in q],u,G,laminar=True); sol=bp.solve(B); print('===== laminar augmented value',sol.value)
import json; t=bp.tree(B)
def show(node,d=0):
    if node is None or 'pool' not in node: return
    print('  '*d+f"pool {[names[i] for i in node['pool']]} value {node['value']}")
    for br in node.get('branches',[]):
        print('  '*d+f" -> count {br['infected_count']} p={br['probability']} newly {[names[i] for i in br['next'].get('newly_healthy',[])]}")
        show(br['next'],d+1)
show(t)

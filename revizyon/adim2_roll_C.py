import sys, os, json, numpy as np, tempfile
B="/home/user/manyetik_alan_olcumu"
sys.path[:0]=[B+"/revizyon",B,B+"/kmod_drivers",B+"/berry_data"]
import adim2_deflector as A, integrator_rev
from concurrent.futures import ProcessPoolExecutor
def task(t):
    os.chdir(tempfile.mkdtemp()); sys.modules["integrator"]=integrator_rev
    from fast_est import fast_measure
    amp,d=t; z=np.zeros(48); e=np.random.default_rng(30000).normal(0,1,48)*amp
    integrator_rev.DEFL_DE=z; integrator_rev.DEFL_ROLL=e
    f,_=fast_measure(z,z,direction=d,signed=True,n_turns=28,n_iter=4)
    return amp,d,f/1e-9
T=[(a,d) for a in (3e-7,3e-8,3e-9) for d in (+1,-1)]
with ProcessPoolExecutor(4) as p: R=list(p.map(task,T))
out={}
for a,d,f in R: out.setdefault(a,{})[d]=f
for a in sorted(out):
    fc,fcc=out[a][1],out[a][-1]
    print("roll rms %.0e: f_CW %.6g  f_CCW %.6g  C=%.4g  C/|f| %.2e"%(a,fc,fcc,abs(fc-fcc)/2,abs(fc-fcc)/2/abs(fc)),flush=True)
json.dump({str(k):v for k,v in out.items()},open(B+"/revizyon/sonuclar/adim2_roll_C.json","w"),indent=1)

import sys, os, numpy as np, tempfile
B="/home/user/manyetik_alan_olcumu"
sys.path[:0]=[B+"/revizyon",B,B+"/kmod_drivers",B+"/berry_data"]
import adim2_deflector as A, integrator_rev
from concurrent.futures import ProcessPoolExecutor
def task(t):
    os.chdir(tempfile.mkdtemp()); sys.modules["integrator"]=integrator_rev
    from fast_est import fast_measure
    kind,amp=t; z=np.zeros(48); rng=np.random.default_rng(30000)
    e=rng.normal(0,1,48)*amp
    integrator_rev.DEFL_DE = e if kind=="eps" else z
    integrator_rev.DEFL_ROLL = e if kind=="roll" else z
    f,_=fast_measure(z,z,direction=+1,signed=True)
    return kind,amp,f/1e-9
with ProcessPoolExecutor(4) as p:
    for r in p.map(task,[("eps",3e-7),("eps",3e-8),("roll",3e-7),("roll",3e-8)]):
        print("%s amp %.0e : f_CW = %.4g × hedef"%r, flush=True)

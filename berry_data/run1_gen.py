# Run 1: veri üretimi. Her config: f_true + tam kapalı yörünge (480 nokta).
# CO bir kez bulunur, sonra (a) spin track -> f, (b) 1-tur ince track -> orbit.
# Offline analiz (gürültü, BPM örnekleme, regresyon, fonksiyonel arama) ayrı.
import sys, os, json
sys.path.insert(0,"/tmp/spin_meas"); sys.path.insert(0,"/home/user/manyetik_alan_olcumu")
os.chdir("/home/user/manyetik_alan_olcumu")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def _init():
    sys.path.insert(0,"/tmp/spin_meas"); sys.path.insert(0,"/home/user/manyetik_alan_olcumu")
    os.chdir("/home/user/manyetik_alan_olcumu")
    os.dup2(os.open(os.devnull,os.O_WRONLY),1)

def gen(w,sigma,rng,NQ):
    nC=NQ//2; s=rng.normal(0,1,nC); d=rng.normal(0,1,nC)
    qf=np.sqrt(w)*s+np.sqrt(1-w)*d; qd=np.sqrt(w)*s-np.sqrt(1-w)*d
    a=np.empty(NQ); a[0::2]=qf; a[1::2]=qd; return a/np.std(a)*sigma

def work(task):
    w,seed=task
    from false_edm_4d import NQ, _T_rev, find_co_4d
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from integrator import integrate_particle
    CFG=json.load(open("params.json")); DT=float(CFG["dt"])
    rng=np.random.default_rng(11000+int(w*100)+seed)
    dx=gen(w,10e-6,rng,NQ); dy=gen(w,10e-6,rng,NQ)
    fields,y0,beta0,R0,p_mag,direction=setup_fields(CFG)
    T_rev=_T_rev(fields,beta0,R0)
    v_co,resid=find_co_4d(fields,p_mag,direction,dx,dy,np.zeros(NQ),T_rev)
    launch=_make_state(v_co,p_mag,direction,[0,0,direction])
    # (a) f: spin track
    fields.poincare_quad_index=0.0
    _,poin,pt=integrate_particle(launch,0.0,5e-4,DT,fields=fields,return_steps=5000,
                                 quad_dx=dx,quad_dy=dy)
    f=float(measure_dSy_dt_model(np.asarray(poin[:,7],float),np.asarray(pt,float)))
    # (b) orbit: 1-tur ince
    fields.poincare_quad_index=-1.0
    hist,_,_=integrate_particle(launch,0.0,T_rev,DT,fields=fields,return_steps=480,
                                quad_dx=dx,quad_dy=dy)
    return w,seed,f,resid,hist[:,0].copy(),hist[:,1].copy(),dx.copy(),dy.copy()

tasks=[(w,s) for w in (0.0,0.5,1.0) for s in range(8)]   # 24 config
W=[];F=[];RES=[];XO=[];YO=[];DX=[];DY=[]
prog="/tmp/spin_meas/run1_progress.txt"; open(prog,"w").write("start %d config\n"%len(tasks))
done=0
with ProcessPoolExecutor(4, initializer=_init) as pool:
    for w,seed,f,resid,xo,yo,dx,dy in pool.map(work, tasks):
        W.append(w);F.append(f);RES.append(resid);XO.append(xo);YO.append(yo);DX.append(dx);DY.append(dy)
        done+=1
        with open(prog,"a") as fh: fh.write("done %d/%d  w=%.1f f=%+.2e\n"%(done,len(tasks),w,f))
np.savez("/tmp/spin_meas/run1_data.npz", w=np.array(W), f=np.array(F), resid=np.array(RES),
         xo=np.array(XO), yo=np.array(YO), dx=np.array(DX), dy=np.array(DY))
open(prog,"a").write("DONE saved run1_data.npz\n")

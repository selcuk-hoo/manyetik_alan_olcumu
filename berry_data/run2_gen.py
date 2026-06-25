# Run 2: belirsizliği çöz. Simetrik-ağırlıklı (28 sim + 12 antisim), her config:
# f_true + TAM ince kapalı yörünge (480 nokta). Offline: zengin fonksiyonel f'i
# öngörebiliyor mu? Öngöremezse -> f sadece global yörünge fonksiyoneli değil
# (yerel spin içeriği var) -> kullanıcının orbit-spin uyumsuzluğu GERÇEK.
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
    rng=np.random.default_rng(22000+int(w*100)+seed)
    dx=gen(w,10e-6,rng,NQ); dy=gen(w,10e-6,rng,NQ)
    fields,y0,beta0,R0,p_mag,direction=setup_fields(CFG)
    T_rev=_T_rev(fields,beta0,R0)
    v_co,resid=find_co_4d(fields,p_mag,direction,dx,dy,np.zeros(NQ),T_rev)
    launch=_make_state(v_co,p_mag,direction,[0,0,direction])
    fields.poincare_quad_index=0.0
    _,poin,pt=integrate_particle(launch,0.0,5e-4,DT,fields=fields,return_steps=5000,quad_dx=dx,quad_dy=dy)
    f=float(measure_dSy_dt_model(np.asarray(poin[:,7],float),np.asarray(pt,float)))
    fields.poincare_quad_index=-1.0
    hist,_,_=integrate_particle(launch,0.0,T_rev,DT,fields=fields,return_steps=480,quad_dx=dx,quad_dy=dy)
    return w,seed,f,resid,hist[:,0].copy(),hist[:,1].copy(),dx.copy(),dy.copy()

tasks=[(1.0,s) for s in range(28)] + [(0.0,s) for s in range(12)]   # 28 sim + 12 antisim
W=[];F=[];RES=[];XO=[];YO=[];DX=[];DY=[]
prog="/tmp/spin_meas/run2_progress.txt"; open(prog,"w").write("start %d config\n"%len(tasks))
done=0
with ProcessPoolExecutor(4, initializer=_init) as pool:
    for w,seed,f,resid,xo,yo,dx,dy in pool.map(work, tasks):
        W.append(w);F.append(f);RES.append(resid);XO.append(xo);YO.append(yo);DX.append(dx);DY.append(dy)
        done+=1
        with open(prog,"a") as fh: fh.write("done %d/%d w=%.1f f=%+.2e\n"%(done,len(tasks),w,f))
np.savez("/tmp/spin_meas/run2_data.npz", w=np.array(W), f=np.array(F), resid=np.array(RES),
         xo=np.array(XO), yo=np.array(YO), dx=np.array(DX), dy=np.array(DY))
open(prog,"a").write("DONE\n")

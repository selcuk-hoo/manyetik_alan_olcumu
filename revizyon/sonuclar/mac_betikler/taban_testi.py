"""Geçiş 8 desenı: aa kanalı tabanı bilineer mi? f(Pa dx, Pa dy), f(Pa dx, 0), f(Pa dx, -Pa dy), f(0, Pa dy), f(dx, 0), f(dx,-dy)."""
import os, sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
BASE="/Users/selcuk/works/manyetik_alan_olcumu"; sys.path[:0]=[os.path.join(BASE,"revizyon"),BASE,os.path.join(BASE,"kmod_drivers")]
NQ=48
def proj():
    Ps=np.zeros((NQ,NQ))
    for k in range(NQ//2):
        a,b=2*k,2*k+1; Ps[a,a]=Ps[b,b]=Ps[a,b]=Ps[b,a]=0.5
    return Ps,np.eye(NQ)-Ps
def task(ad):
    from sahte_edm_donmus import measure_frozen
    h=json.load(open(os.path.join(BASE,"revizyon/sonuclar/bba_iter_cpp_donmus_state.json")))["hist"][-1]
    x,y=np.array(h["res_x"]),np.array(h["res_y"]); Ps,Pa=proj(); z=np.zeros(NQ)
    d={"aa":(Pa@x,Pa@y),"aa_dy0":(Pa@x,z),"aa_-dy":(Pa@x,-Pa@y),"aa_dx0":(z,Pa@y),"full_dy0":(x,z),"full_-dy":(x,-y)}[ad]
    r=measure_frozen(d[0],d[1],direction=-1,n_turns=28,n_iter=4)
    return ad, r["f"]/1e-9
if __name__=="__main__":
    out={}
    with ProcessPoolExecutor(6, initializer=__import__("build_response_matrix")._worker_init) as p:
        for ad,f in p.map(task,["aa","aa_dy0","aa_-dy","aa_dx0","full_dy0","full_-dy"]):
            out[ad]=f; print(f"{ad:9s} f* = {f:+.4f}", flush=True)
    json.dump(out,open(os.path.join(BASE,"revizyon/sonuclar/bba_gecis8_taban.json"),"w"),indent=1)

#!/usr/bin/env python3
"""Test 8: orgu-modeli hatasi (beta-beating) altinda drift modu saglamligi.
recon R_model^-1 (nominal); veri R_true (beta-beating'li). eps_beta tara."""
import json,sys,numpy as np
import os; _DIR=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_DIR)
import fodo_lattice as fl
cfg=json.load(open(os.path.join(_DIR,'..','params.json')))
OFF=50e-6; NOISE=1e-6; RAMP=10e-6; DQ0=100e-6; NEP=10; NSEED=15
def nominal(pl):
    Kxarc=fl.calibrate_K_x_arc(cfg) if pl=='x' else None
    beta,phi,Q=fl.compute_twiss_at_quads(cfg,pl,K_x_arc=Kxarc); KL=fl.signed_KL(cfg,pl)
    return beta,phi,Q,KL
def track_err(pl, eps_beta, seed):
    beta,phi,Q,KL=nominal(pl); N=len(beta)
    Rm=fl.build_response_matrix(beta,phi,Q,KL); Rm_inv=np.linalg.inv(Rm)
    rb=np.random.default_rng(50000+seed)
    bt=beta*(1+rb.normal(0,eps_beta,N)); pt=phi+rb.normal(0,eps_beta,N)   # beta-beating + faz hatasi
    Rt=fl.build_response_matrix(bt,pt,Q,KL)
    rng=np.random.default_rng(1000+seed)
    dq0=rng.normal(0,DQ0,N); b0=rng.normal(0,OFF,N); ramp=rng.normal(0,RAMP,N)
    y0=Rt@dq0+b0+rng.normal(0,NOISE,N)
    errs=[]
    for t in range(1,NEP+1):
        dqt=dq0+ramp*(t/NEP); yt=Rt@dqt+b0+rng.normal(0,NOISE,N)
        dqhat=Rm_inv@(yt-y0); errs.append(np.sqrt(np.mean((dqhat-(dqt-dq0))**2)))
    return np.mean(errs)
print(f"{'eps_beta':>9} {'y-track[um]':>12} {'x-track[um]':>12}")
for eps in (0.0,0.005,0.01,0.02,0.05,0.10):
    ey=np.median([track_err('y',eps,s) for s in range(NSEED)])
    ex=np.median([track_err('x',eps,s) for s in range(NSEED)])
    print(f"{eps:>9.3f} {ey*1e6:>12.2f} {ex*1e6:>12.2f}")
print("\n(LOCO sonrasi gercekci kalan beta-beating ~%1; hedef <10um)")

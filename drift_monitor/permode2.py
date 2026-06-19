#!/usr/bin/env python3
"""Singular-mod basina gurultu yukseltmesi (1/sigma) ve modun simetrik karakteri.
no-go ile uzlasma: en kotu modlar (yuksek-k simetrik) ne kadar yukseltiyor?"""
import json,sys,numpy as np
import os; _DIR=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_DIR)
import fodo_lattice as fl
cfg=json.load(open(os.path.join(_DIR,'..','params.json')))
def sym_frac(v):  # vektorun simetrik kismindaki guc orani
    N=len(v); s=v.copy()
    for c in range(N//2):
        m=0.5*(v[2*c]+v[2*c+1]); s[2*c]=m; s[2*c+1]=m
    return np.sum(s**2)/np.sum(v**2)
def nominal(pl):
    Kxarc=fl.calibrate_K_x_arc(cfg) if pl=='x' else None
    beta,phi,Q=fl.compute_twiss_at_quads(cfg,pl,K_x_arc=Kxarc); KL=fl.signed_KL(cfg,pl)
    return fl.build_response_matrix(beta,phi,Q,KL)
for pl in ('y',):
    R=nominal(pl); U,S,Vt=np.linalg.svd(R)
    print(f"duzlem {pl}: sigma_max={S[0]:.2f} sigma_min={S[-1]:.3f} kappa={S[0]/S[-1]:.0f}")
    print(f"{'mod':>4} {'sigma':>8} {'gurultu yukselt(1/sigma)':>22} {'simetrik guc%':>14}")
    for i in [0,1,2,5,10,20,40,44,45,46,47]:
        v=Vt[i]  # girdi-uzayi singular vektor
        print(f"{i:>4} {S[i]:>8.3f} {1.0/S[i]:>22.2f} {100*sym_frac(v):>13.1f}")
    # en kotu 8 modun ortalama simetrik gucu vs en iyi 8
    worst=np.mean([sym_frac(Vt[i]) for i in range(40,48)])
    best=np.mean([sym_frac(Vt[i]) for i in range(8)])
    print(f"\nen iyi 8 mod (buyuk sigma): ort simetrik guc %{100*best:.0f}")
    print(f"en kotu 8 mod (kucuk sigma): ort simetrik guc %{100*worst:.0f}")
    # false-EDM-kritik yuksek-k simetrik modun gurultu yukseltmesi
    print(f"\nen kotu modun (sigma_min) gurultu yukseltmesi: {1.0/S[-1]/(1.0/S[0]):.0f}x (en iyi moda gore)")

"""Enerji tanımının sahte EDM'ye etkisi: paper_runs 'sigma' desenleri (σ=10 μm, seed 0–2),
eski fırlatma (|p| sabit) vs toplam enerji sabit fırlatma. fast_measure_g ile aynı ayar."""
import sys, os, json, tempfile, numpy as np
B="/home/user/manyetik_alan_olcumu"
sys.path[:0]=[B+"/revizyon/tests",B,B+"/berry_data",B+"/kmod_drivers"]
from concurrent.futures import ProcessPoolExecutor
from ortak import Q_E, M_P, C_LIGHT, E0_MAGIC, R0

def make_state_E(v, p_mag, direction, spin):
    """Toplam enerji sabit fırlatma: kinetik(R0+x) = kinetik_ref − qΦ(R0+x), Φ = −E0 R0 ln(R/R0)."""
    x = v[0]
    W_ref = np.sqrt((p_mag*C_LIGHT)**2 + (M_P*C_LIGHT**2)**2)
    Wk = W_ref - Q_E*(-E0_MAGIC*R0*np.log((R0+x)/R0))
    p = np.sqrt(Wk**2 - (M_P*C_LIGHT**2)**2)/C_LIGHT
    return [v[0], v[1], 0.0, p*direction*v[2], p*direction*v[3], p*direction, spin[0], spin[1], spin[2]]

def run(task):
    os.chdir(tempfile.mkdtemp())
    corrected, seed = task
    import false_edm_4d, false_edm_mode_scan
    if corrected:
        false_edm_4d._make_state = make_state_E; false_edm_mode_scan._make_state = make_state_E
    from paper_runs import fast_measure_g
    import paper_runs
    if corrected: paper_runs._make_state = make_state_E
    rng = np.random.default_rng(3000 + seed); dx = rng.normal(0,10e-6,48); dy = rng.normal(0,10e-6,48)
    f, _ = fast_measure_g(dx, dy)
    return corrected, seed, f

with ProcessPoolExecutor(4) as p:
    R = list(p.map(run, [(c,s) for s in (0,1,2) for c in (False,True)]))
for s in (0,1,2):
    fo = [f for c,ss,f in R if ss==s and not c][0]; fn=[f for c,ss,f in R if ss==s and c][0]
    print("seed %d: eski %.4e  enerji-sabit %.4e  oran %.3f"%(s,fo,fn,fn/fo))

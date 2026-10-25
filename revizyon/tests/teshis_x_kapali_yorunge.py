"""Hipotez testi: kapalı yörünge bulucu fırlatmada toplam enerjiyi sabit tutarsa yatay R farkı kaybolur mu?"""
import sys, os, json, tempfile, numpy as np
B="/home/user/manyetik_alan_olcumu"
sys.path[:0]=[B+"/revizyon/tests",B,B+"/berry_data",B+"/kmod_drivers"]
import test_k2_yorunge as T
from ortak import Q_E, M_P, C_LIGHT, E0_MAGIC, R0
from concurrent.futures import ProcessPoolExecutor

def make_state_E(v, p_mag, direction, spin):
    """Toplam enerji sabit: kinetik(R0+x) = kinetik_ref − qΦ(R0+x), Φ = −E0 R0 ln(R/R0)."""
    x = v[0]
    W_ref = np.sqrt((p_mag*C_LIGHT)**2 + (M_P*C_LIGHT**2)**2)
    phi = -E0_MAGIC*R0*np.log((R0+x)/R0)
    Wk = W_ref - Q_E*phi
    p = np.sqrt(Wk**2 - (M_P*C_LIGHT**2)**2)/C_LIGHT
    return [v[0], v[1], 0.0, p*direction*v[2], p*direction*v[3], p*direction, spin[0], spin[1], spin[2]]

def run(task):
    os.chdir(tempfile.mkdtemp())
    corrected, amp = task
    import false_edm_4d, false_edm_mode_scan
    if corrected:
        false_edm_4d._make_state = make_state_E; false_edm_mode_scan._make_state = make_state_E
    import build_response_matrix as brm
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    cfg = json.load(open(B+"/params.json")); cfg["direction"]=-1.0; cfg["t2"]=1e-4
    fields, y0, beta0, R0_, p_mag, d = false_edm_mode_scan.setup_fields(cfg)
    dx=np.zeros(48); dx[T.QI]=amp; z=np.zeros(48)
    v_co,_=find_co_4d(fields,p_mag,d,dx,z,z,_T_rev(fields,beta0,R0_),n_turns=28,n_iter=4)
    launch=(make_state_E if corrected else false_edm_mode_scan._make_state)(v_co,p_mag,d,[0,0,d])
    fields.poincare_quad_index=999.0
    for f in ("cod_data.txt","rf.txt"):
        if os.path.exists(f): os.remove(f)
    integrate_particle(launch,0.0,1e-4,float(cfg["dt"]),fields=fields,return_steps=10,quad_dy=z,quad_dx=dx)
    return brm.read_cod_quads(24)[0]

with ProcessPoolExecutor(4) as p:
    r = list(p.map(run, [(False,T.DQ),(False,0.0),(True,T.DQ),(True,0.0)]))
axis = list(p for p in [None])
with ProcessPoolExecutor(2) as p:
    a = list(p.map(T._orbit, [("axis","x",T.DQ,1e-3,-1,[]),("axis","x",0.0,1e-3,-1,[])]))
ax = (np.array(a[0][0])-np.array(a[1][0]))/T.DQ
co_old = (r[0]-r[1])/T.DQ; co_new=(r[2]-r[3])/T.DQ
print("eski CO (|p| sabit)  vs eksen-ortalaması: %.3e"%(np.linalg.norm(co_old-ax)/np.linalg.norm(ax)))
print("yeni CO (enerji sabit) vs eksen-ortalaması: %.3e"%(np.linalg.norm(co_new-ax)/np.linalg.norm(ax)))

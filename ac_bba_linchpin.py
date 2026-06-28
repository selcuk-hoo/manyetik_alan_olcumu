#!/usr/bin/env python3
"""
ac_bba_linchpin.py — LINCHPIN testi: all-quad AC-BBA → kalan sahte-EDM.

Soru (kmod_bba_plani.md §6.2): 48-quad K-modülasyon → per-quad demet-quad ofset
ölçümü → düzeltme → KALAN geometrik-faz sahte-EDM, gerçekçi koşullarda
(kapasitif-BPM gürültüsü, sonlu modülasyon derinliği, optik-model β-beating)
hedefin (1 nrad/s) altına iner mi?

Yapı:
  [1] İstatistiksel hassasiyet σ_stat: tur-tur BPM gürültüsü demodüle edilince
      per-quad ofset hatası. Gerçekçi (f_rev, entegrasyon süresi, σ_BPM).
  [2] Sistematik (β-beating) süpürmesi: gerçek optik T_true ≠ model T_model →
      düzeltme artığı e = o − ô. Bu artık ofset desenini DOĞRULANMIŞ estimator'a
      (false_edm_4d.measure_false_edm) misalignment olarak verip kalan sahte-EDM.
  [3] σ²-ölçekleme doğrulaması (p≈2) — over-claim koruması.

Tüm tepkiler analytic_kmod FODO Twiss'inden (lineer model); kalan sahte-EDM
C++ spin estimator'ından. Konservatif: perfect-steering CO sönümünü saymaz
(estimator'a ofset=misalignment verince CO amplifikasyonu EKLENİR → üst sınır).

Kullanım:
  python3 ac_bba_linchpin.py --stat            # sadece istatistiksel hassasiyet (hızlı)
  python3 ac_bba_linchpin.py --calib --workers 7   # estimator σ²-kalibrasyonu
  python3 ac_bba_linchpin.py --sweep --workers 7   # β-beating make-or-break
"""
import os, sys, json, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import analytic_kmod as ak
from ac_bba_observability import build_T, recon_acbba

with open(os.path.join(BASE, "params.json")) as _f:
    CFG = json.load(_f)
NFODO = int(CFG['nFODO']); NQ = 2 * NFODO

# ── revolüsyon frekansı (T_rev, false_edm_4d ile aynı) ────────────────────
M2 = 0.938272046; AMU = 1.792847356; C = 299792458.0
_pmag = M2 / np.sqrt(AMU)
BETA0 = _pmag / np.sqrt(_pmag**2 + M2**2)
CIRC = (2*np.pi*CFG['R0'] + 4*NFODO*CFG['driftLen'] + 2*NFODO*CFG['quadLen'])
T_REV = CIRC / (BETA0 * C)
F_REV = 1.0 / T_REV


# ─────────────────────────────────────────────────────────────────────────
# Lineer kapalı-yörünge modeli (demet-quad ofseti o_j üretmek için)
# ─────────────────────────────────────────────────────────────────────────

def co_kernel(config, plane):
    """G[i,j] = √(βiβj)/(2sinπQ) cos(|φi−φj|−πQ): kick→orbit cevabı."""
    g = float(config.get('g1', 0.21))
    beta, phi, Q = ak.compute_twiss_at_quads(config, g, plane)
    n = len(beta); denom = 2.0*np.sin(np.pi*Q); sb = np.sqrt(beta)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = sb[i]*sb[j]/denom*np.cos(abs(phi[i]-phi[j])-np.pi*Q)
    return G, beta, phi, Q


def beam_quad_offset(config, dm, plane):
    """Misalignment dm → demet-quad ofseti o_j = x_co,j − dm_j.

    Kaçık quad feed-down kick'i θ_j = (KL işaretli)·dm_j (BPM=quad konumu).
    x_co = G @ θ ; o = x_co − dm.
    """
    nFODO = int(config['nFODO']); L_q = float(config['quadLen'])
    Brho = ak.compute_Brho(config); K = abs(config['g1'])/Brho
    KL = ak.signed_KL(nFODO, K, L_q, plane)     # işaretli K·L
    G, *_ = co_kernel(config, plane)
    theta = KL * dm                              # her quad feed-down kick
    x_co = G @ theta
    return x_co - dm


def perturbed_T(config, plane, depth, beat_eps, rng):
    """β-beating'li 'gerçek' optik tepkisi T_true.

    Gradyanlara quad-bazlı küçük rastgele hata vermek yerine, üretilen β,φ'ye
    doğrudan ε mertebeli bozulma ekleriz (LOCO-artığı β-beating temsili):
    β_i → β_i(1+ε·ξ_i), φ_i → φ_i + ε·ζ_i.  T'nin yapısı analytic_kmod ile aynı.
    """
    nFODO = int(config['nFODO']); L_q = float(config['quadLen'])
    Brho = ak.compute_Brho(config); g = float(config['g1'])
    K = abs(g)/Brho; dK = depth*K
    beta, phi, Q = ak.compute_twiss_at_quads(config, g, plane)
    if beat_eps > 0:
        beta = beta * (1.0 + beat_eps*rng.normal(0, 1, len(beta)))
        phi = phi + beat_eps*rng.normal(0, 1, len(phi))
    KL = ak.signed_KL(nFODO, dK, L_q, plane)
    return ak.build_R_analytic(beta, phi, Q, KL)


# ─────────────────────────────────────────────────────────────────────────
# [1] İstatistiksel hassasiyet
# ─────────────────────────────────────────────────────────────────────────

def stat_precision(depth=0.02, sigma_bpm=1e-6, t_int=1.0):
    """Per-quad ofset ölçüm hatası σ_stat,j = σ_BPM·√(2/N_tur)/√(Σ_i T_ij²)."""
    out = {}
    for plane in ('y', 'x'):
        T, beta, phi, Q = build_T(CFG, plane, depth)
        colpow = np.einsum('ij,ij->j', T, T)
        N_turns = F_REV * t_int
        sigma = sigma_bpm*np.sqrt(2.0/N_turns)/np.sqrt(colpow)
        out[plane] = sigma
    return out, F_REV, T_REV


# ─────────────────────────────────────────────────────────────────────────
# [2] β-beating-sınırlı düzeltme artığı
# ─────────────────────────────────────────────────────────────────────────

def bba_residual_offset(config, dm_x, dm_y, depth, beat_eps, sigma_bpm,
                        t_int, rng):
    """AC-BBA ölç-düzelt → kalan demet-quad ofset deseni (x,y).

    o = gerçek demet-quad ofseti; T_true (β-beating'li) ile genlik A=T_true·o;
    model T_model (nominal) ile projeksiyon ô; düzeltme ô'yu sıfırlar → artık
    e = o − ô. İstatistiksel gürültü genliğe √(2/N) ölçeğiyle eklenir.
    """
    N_turns = F_REV * t_int
    sig_amp = sigma_bpm*np.sqrt(2.0/N_turns)     # demodüle genlik gürültüsü
    res = {}
    for plane, dm in (('y', dm_y), ('x', dm_x)):
        o = beam_quad_offset(config, dm, plane)
        T_model, *_ = build_T(config, plane, depth)
        T_true = perturbed_T(config, plane, depth, beat_eps, rng)
        A = T_true * o[None, :] + rng.normal(0, sig_amp, (NQ, NQ))
        o_hat = recon_acbba(T_model, A)
        res[plane] = o - o_hat
    return res['x'], res['y']


# ─────────────────────────────────────────────────────────────────────────
# Estimator köprüsü (false_edm_4d)
# ─────────────────────────────────────────────────────────────────────────

def _est(task):
    """worker: misalignment (dx,dy) → |sahte-EDM| [rad/s]."""
    dx, dy = task
    sys.path.insert(0, os.path.join(BASE, "berry_data"))
    from false_edm_4d import measure_false_edm
    f, resid = measure_false_edm(np.asarray(dx), np.asarray(dy), np.zeros(NQ))
    return abs(f), resid


def run_estimator(patterns, workers):
    import build_response_matrix as brm
    from concurrent.futures import ProcessPoolExecutor
    out = []
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=brm._worker_init) as pool:
            for r in pool.map(_est, patterns):
                out.append(r)
    else:
        import tempfile; os.chdir(tempfile.mkdtemp())
        for t in patterns:
            out.append(_est(t))
    return out


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", action="store_true")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--workers", "-w", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--tint", type=float, default=1.0)
    ap.add_argument("--depth", type=float, default=0.02)
    ap.add_argument("--sigbpm", type=float, default=1e-6)
    args = ap.parse_args()
    os.chdir(BASE)

    if args.stat:
        sig, frev, trev = stat_precision(args.depth, args.sigbpm, args.tint)
        print("="*68)
        print("[1] İSTATİSTİKSEL PER-QUAD OFSET HASSASİYETİ")
        print("="*68)
        print(f"  T_rev={trev*1e6:.3f} μs  f_rev={frev/1e3:.1f} kHz")
        print(f"  Entegrasyon t_int={args.tint:.1f}s → N_turns={frev*args.tint:.3e}")
        print(f"  depth(ΔK/K)={args.depth*100:.1f}%  σ_BPM={args.sigbpm*1e6:.1f}μm")
        for plane in ('y', 'x'):
            s = sig[plane]
            print(f"  {plane}: σ_stat per-quad = {s.min()*1e9:.2f}–{s.max()*1e9:.2f} nm "
                  f"(ort {s.mean()*1e9:.2f} nm)")
        print("  → istatistiksel taban hedefin ÇOK altında; gerçek sınır SİSTEMATİK.")
        return

    if args.calib:
        # Estimator mutlak ölçek: f(σ)=A·σ²; beyaz misalignment, çok seed.
        sigmas = [10e-6, 5e-6, 2.5e-6]
        pats = []
        meta = []
        for sg in sigmas:
            for sd in range(args.seeds):
                rng = np.random.default_rng(3000+sd)
                dx = rng.normal(0, sg, NQ); dy = rng.normal(0, sg, NQ)
                pats.append((dx, dy)); meta.append(sg)
        print(f"Kalibrasyon: {len(pats)} estimator koşumu...")
        res = run_estimator(pats, args.workers)
        meta = np.array(meta); fs = np.array([r[0] for r in res])
        # log-log eğim
        for sg in sigmas:
            m = meta == sg
            print(f"  σ={sg*1e6:4.1f}μm: |f|={fs[m].mean():.3e} ± {fs[m].std():.1e} rad/s")
        A = np.mean(fs/meta**2)
        p = np.polyfit(np.log(meta), np.log(fs), 1)[0]
        print(f"  Ölçek A = ⟨f/σ²⟩ = {A:.3e} rad/s/m²   (f@10μm={A*(10e-6)**2:.2e})")
        print(f"  log-log eğim p = {p:.3f}  (geometrik faz beklenen p=2.00)")
        np.save(os.path.join(BASE,"acbba_calib.npy"), np.column_stack([meta,fs]))
        return

    if args.sweep:
        # β-beating make-or-break: her ε için kalan ofset deseni → estimator.
        epss = [0.0, 0.001, 0.005, 0.01, 0.05]
        sigma_mis = 10e-6
        pats = []; meta = []
        for eps in epss:
            for sd in range(args.seeds):
                rng = np.random.default_rng(5000+sd)
                dx = rng.normal(0, sigma_mis, NQ); dy = rng.normal(0, sigma_mis, NQ)
                ex, ey = bba_residual_offset(CFG, dx, dy, args.depth, eps,
                                             args.sigbpm, args.tint, rng)
                pats.append((ex, ey)); meta.append(eps)
        print(f"β-beating süpürmesi: {len(pats)} estimator koşumu...")
        res = run_estimator(pats, args.workers)
        meta = np.array(meta); fs = np.array([r[0] for r in res])
        print("="*68)
        print("[2] LINCHPIN — β-beating sınırlı KALAN sahte-EDM")
        print("="*68)
        print(f"  Giriş misalignment σ={sigma_mis*1e6:.0f}μm, depth={args.depth*100:.0f}%, "
              f"t_int={args.tint:.0f}s, σ_BPM={args.sigbpm*1e6:.0f}μm")
        print(f"  Hedef: 1 nrad/s = 1e-9; gerçek EDM = 9.81e-10 rad/s")
        print(f"  {'β-beating':>10} {'kalan ofset RMS':>16} {'|sahte-EDM|':>16} {'hedef?':>8}")
        for eps in epss:
            m = meta == eps
            fm = fs[m].mean(); fsd = fs[m].std()
            ok = "EVET" if fm < 1e-9 else "hayır"
            print(f"  {eps*100:8.1f}% {'':6} {'':6} {fm:.3e} ± {fsd:.1e}   {ok}")
        np.save(os.path.join(BASE,"acbba_sweep.npy"), np.column_stack([meta,fs]))
        return

    ap.print_help()


if __name__ == "__main__":
    main()

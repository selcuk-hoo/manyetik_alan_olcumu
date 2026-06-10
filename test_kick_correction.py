#!/usr/bin/env python3
"""test_kick_correction.py — Kick sayısı vs False EDM testi.

Amaç:
  k=2 yörünge hizalama hatası için N korrektör kicki false EDM'yi ne kadar
  bastırır? Analitik Courant-Snyder yeşil fonksiyonu ile optimal kick vektörü
  hesaplanır, ardından tam simülasyon ile dSy/dt ölçülür.

Test A1 — k=2 sabit:
  Δy_j = A₂ × F_{k=2,cos}[j], N_corr ∈ {0, 2, 4, 8, 12, 24, 48}
  Teorik beklenti: equidistant QF korrektörler için k=2 cos+sin bileşenlerini
  tam kapatmak en az N=8 gerektirir.

Test A2 — rassal hizalama:
  RMS(Δy) = A_RAND, M realizasyon, N_corr ∈ {0, 4, 8, 24, 48}
  Amaç: sabit korrektör konumları farklı hizalama desenlerinde ne kadar iyi?

Korrektör modeli:
  Gerçek makinede bağımsız dipol korrektör ≡ ince-lens yaklaşımında kaydırılmış
  kuadrupol: θ = K₁L × δy. Korrektör kick'i, korrektör konumundaki quad'a ek
  Δy olarak modellenir.

Çıktı:
  test_kick_correction.png — iki alt-grafik
  Terminal tablo
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Fiziksel sabitler ────────────────────────────────────────────────────────
M2  = 0.938272046   # proton kütlesi [GeV/c²]
AMU = 1.792847356   # anormal manyetik moment

# ── Test parametreleri ───────────────────────────────────────────────────────
Q_Y          = 2.68    # dikey betatron tune (params.json'da tanımsız → hardcode)
A_K2         = 1e-5    # k=2 sabit test genliği [m] = 10 μm
T2           = 5e-4    # simülasyon süresi [s]
CO_TURNS     = 12      # kapalı yörünge bulma tur sayısı
CO_ITER      = 1       # Newton yinelemesi sayısı
RETURN_STEPS = 3000    # Poincaré kayıt kapasitesi
N_CORR_K2    = [0, 2, 4, 8, 12, 24, 48]
N_CORR_RAND  = [0, 4, 8, 24, 48]
M_RAND       = 5       # rassal realizasyon sayısı
A_RAND       = 1e-5    # rassal hizalama RMS [m]


# ── FODO analitik parametreler ───────────────────────────────────────────────

def _fodo_lattice(config):
    """FODO ızgara parametrelerini analitik olarak hesaplar.

    Dönüş: (n_q, KL, beta, psi)
      n_q  : toplam kuadrupol sayısı = 2×nFODO
      KL   : ince-lens odak kuvveti [m⁻¹]  (K₁ × L)
      beta : her quad pozisyonunda beta fonksiyonu [m], boy n_q
      psi  : her quad pozisyonunda betatron faz ilerlemesi [rad], boy n_q
    """
    nFODO   = int(config["nFODO"])
    n_q     = 2 * nFODO
    R0      = float(config["R0"])
    quadLen = float(config["quadLen"])
    dLen    = float(config["driftLen"])
    g1      = float(config["g1"])

    # Bρ = p[GeV/c] / 0.3  [T·m]
    p_magic = M2 / np.sqrt(AMU)
    Brho    = p_magic / 0.3
    KL      = g1 * quadLen / Brho

    # Çevre → hücre uzunluğu → beta fonksiyonları
    circ   = 2*np.pi*R0 + 4*nFODO*dLen + 2*nFODO*quadLen
    L_half = circ / (2 * nFODO)           # QF↔QD mesafesi [m]
    mu     = 2*np.pi*Q_Y / nFODO          # hücre başına faz ilerlemesi [rad]
    beta_F = L_half * (1 + np.sin(mu/2)) / np.sin(mu)
    beta_D = L_half * (1 - np.sin(mu/2)) / np.sin(mu)

    # Çift indeks → QF (beta_F), tek indeks → QD (beta_D)
    beta = np.where(np.arange(n_q) % 2 == 0, beta_F, beta_D)
    psi  = np.arange(n_q) * (2*np.pi*Q_Y / n_q)
    return n_q, KL, beta, psi


def build_green_matrix(beta, psi):
    """Courant-Snyder Green fonksiyonu matrisi.

    G[i,j] = √(β_i β_j) × cos(|ψ_i − ψ_j| − πQ) / (2 sin(πQ))

    Kapalı yörünge yanıtı: y_CO = KL × G @ Δy
    """
    i_idx, j_idx = np.meshgrid(np.arange(len(beta)),
                                np.arange(len(beta)), indexing='ij')
    dpsi = np.abs(psi[i_idx] - psi[j_idx])
    return (np.sqrt(beta[i_idx] * beta[j_idx])
            * np.cos(dpsi - np.pi * Q_Y)
            / (2 * np.sin(np.pi * Q_Y)))


# ── Kick düzeltmesi ──────────────────────────────────────────────────────────

def apply_kick_correction(quad_dy, G, n_corr):
    """N eşit aralıklı QF korrektörle en küçük kareler yörünge düzeltmesi.

    Parametreler:
      quad_dy : orijinal quad dikey hataları [m], boy n_q
      G       : Green matrisi, (n_q × n_q)
      n_corr  : korrektör sayısı

    Dönüş:
      dy_new     : düzeltilmiş quad dizisi [m]
      corr_idx   : korrektör indeksleri
      orbit_norm : ||G @ dy_new|| / √n_q  (artık orbit etkin değeri)
    """
    n_q   = len(quad_dy)
    orbit = G @ quad_dy
    norm_fn = lambda v: float(np.linalg.norm(G @ v) / np.sqrt(n_q))

    if n_corr == 0:
        return quad_dy.copy(), np.array([], dtype=int), norm_fn(quad_dy)

    qf_idx = np.arange(0, n_q, 2)    # [0, 2, 4, …, n_q-2]: QF pozisyonları
    n_qf   = len(qf_idx)

    if n_corr >= n_q:
        corr_idx = np.arange(n_q)
    elif n_corr >= n_qf:
        corr_idx = qf_idx
    else:
        step     = n_qf // n_corr
        corr_idx = qf_idx[::step][:n_corr]

    # G[:, corr_idx] @ Δy_c = −(G @ dy_orig)  →  en küçük kareler çözümü
    delta, _, _, _ = np.linalg.lstsq(G[:, corr_idx], -orbit, rcond=None)

    dy_new = quad_dy.copy()
    dy_new[corr_idx] += delta
    return dy_new, corr_idx, norm_fn(dy_new)


# ── Yardımcılar ─────────────────────────────────────────────────────────────

def _suppress_stdout():
    """C++ verbose çıktısını /dev/null'a yönlendirir."""
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


# ── Paralel worker ───────────────────────────────────────────────────────────

def _worker(task):
    """Tek simülasyon: kapalı yörünge bul → spin takip → dSy/dt ölç.

    Görev demeti: (label, quad_dy_list, t2, co_turns, co_iter, return_steps)
    Dönüş: (label, slope [rad/s], co_off [μm])
    """
    label, quad_dy_list, t2, co_turns, co_iter, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state, C
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)

    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    dt    = float(config.get("dt", 1e-11))
    circ  = (2*np.pi*R0
             + 4*fields.nFODO*fields.driftLen
             + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)
    dy    = np.asarray(quad_dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        v_co, _ = find_closed_orbit(fields, p_mag, direction, dy, dt, T_rev,
                                     n_turns=co_turns, n_iter=co_iter)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y_launch, 0.0, t2, dt,
            fields=fields,
            return_steps=return_steps,
            quad_dy=dy,
        )
    finally:
        _restore_stdout(saved)

    sy    = np.asarray(poin[:, 7], float)
    ts    = np.asarray(poin_t,     float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    co_off = float(np.hypot(v_co[0], v_co[1]) * 1e6)   # [μm]
    return label, slope, co_off


# ── Ana rutin ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    with open("params.json") as fh:
        config = json.load(fh)

    n_q, KL, beta, psi = _fodo_lattice(config)
    G = build_green_matrix(beta, psi)

    from fourier_reconstruct import fodo_basis
    antisym = config.get("smooth_antisym_fodo", True)
    F2, _   = fodo_basis(n_q, [2], antisym)
    dy_k2   = A_K2 * F2[:, 0]     # k=2 cos bileşeni, genlik A_K2

    nFODO = int(config["nFODO"])
    print(f"FODO parametreleri: nFODO={nFODO}, n_q={n_q}, KL={KL:.4f} m⁻¹")
    print(f"  β_F={float(beta[0]):.1f} m,  β_D={float(beta[1]):.1f} m,  Q_y={Q_Y}")

    # ── Test A1: k=2 sabit ────────────────────────────────────────────────
    tasks_k2    = []
    orms_k2     = {}
    for nc in N_CORR_K2:
        dy_corr, _, orms = apply_kick_correction(dy_k2, G, nc)
        orms_k2[nc] = orms * KL * 1e6    # artık orbit RMS [μm]
        tasks_k2.append((
            f"k2_N{nc}",
            dy_corr.tolist(),
            T2, CO_TURNS, CO_ITER, RETURN_STEPS,
        ))

    # ── Test A2: rassal hizalama ─────────────────────────────────────────
    rng = np.random.default_rng(42)
    rand_patterns = [rng.standard_normal(n_q) * A_RAND for _ in range(M_RAND)]
    tasks_rand = []
    orms_rand  = {}
    for m_idx, dy_rand in enumerate(rand_patterns):
        for nc in N_CORR_RAND:
            dy_corr, _, orms = apply_kick_correction(dy_rand, G, nc)
            orms_rand[(m_idx, nc)] = orms * KL * 1e6
            tasks_rand.append((
                f"rand_m{m_idx}_N{nc}",
                dy_corr.tolist(),
                T2, CO_TURNS, CO_ITER, RETURN_STEPS,
            ))

    all_tasks = tasks_k2 + tasks_rand
    n_total   = len(all_tasks)
    n_workers = min(mp.cpu_count(), n_total)
    print(f"\nToplam {n_total} simülasyon, {n_workers} işçi ile başlatılıyor...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker, all_tasks)

    elapsed = time.time() - t0

    # Sonuç sözlüğü
    res_map = {lbl: (sl, co) for lbl, sl, co in results}

    # ── Tablo A1 ─────────────────────────────────────────────────────────
    slopes_k2 = [res_map[f"k2_N{nc}"][0] for nc in N_CORR_K2]
    ref_k2    = slopes_k2[0]

    print(f"\n{'─'*72}")
    print("Test A1: k=2 cos, A=10 μm — korrektör sayısı taraması")
    print(f"{'─'*72}")
    hdr = f"{'N_corr':>8}  {'dSy/dt [rad/s]':>18}  {'|bastırma|':>12}  {'CO artık RMS [μm]':>20}"
    print(hdr)
    print('─' * len(hdr))
    for nc, sl in zip(N_CORR_K2, slopes_k2):
        ratio = abs(sl / ref_k2) if ref_k2 != 0 else float('nan')
        print(f"{nc:>8d}  {sl:>18.4e}  {ratio:>12.4f}  {orms_k2[nc]:>20.4f}")

    # ── Tablo A2 ─────────────────────────────────────────────────────────
    rand_by_nc = {nc: [] for nc in N_CORR_RAND}
    for m_idx in range(M_RAND):
        for nc in N_CORR_RAND:
            rand_by_nc[nc].append(res_map[f"rand_m{m_idx}_N{nc}"][0])

    print(f"\n{'─'*72}")
    print(f"Test A2: Rassal hizalama, RMS=10 μm, M={M_RAND} realizasyon")
    print(f"{'─'*72}")
    hdr2 = f"{'N_corr':>8}  {'medyan |dSy/dt|':>18}  {'std |dSy/dt|':>16}  {'bast. oranı':>12}"
    print(hdr2)
    print('─' * len(hdr2))
    ref_rand = np.median(np.abs(rand_by_nc[0])) or 1.0
    for nc in N_CORR_RAND:
        arr = np.abs(rand_by_nc[nc])
        print(f"{nc:>8d}  {np.median(arr):>18.4e}  {np.std(arr):>16.4e}  "
              f"{np.median(arr)/ref_rand:>12.4f}")

    print(f"\nToplam süre: {elapsed:.1f} s")

    # ── Figür ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Kick Düzeltmesi: Korrektör Sayısı vs False EDM", fontsize=13)

    # Sol panel — k=2 sabit
    ax = axes[0]
    ax.semilogy(N_CORR_K2, np.abs(slopes_k2), 'bo-', markersize=8, linewidth=1.5,
                label='|dSy/dt| (simülasyon)')
    ax.axhline(np.abs(ref_k2), color='gray', ls='--', alpha=0.6, label='N=0 (düzeltmesiz)')
    ax.set_xlabel("Korrektör sayısı N")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Test A1: k=2, A=10 μm (sabit desen)")
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(N_CORR_K2)

    # Sağ panel — rassal desen kutu-bıyık grafiği
    ax = axes[1]
    data_boxes = [np.abs(rand_by_nc[nc]) for nc in N_CORR_RAND]
    bp = ax.boxplot(data_boxes,
                    positions=range(len(N_CORR_RAND)),
                    patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_xticks(range(len(N_CORR_RAND)))
    ax.set_xticklabels([str(nc) for nc in N_CORR_RAND])
    ax.set_xlabel("Korrektör sayısı N")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title(f"Test A2: Rassal hizalama, RMS=10 μm, M={M_RAND}")
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_kick_correction.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()

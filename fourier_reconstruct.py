#!/usr/bin/env python3
"""fourier_reconstruct.py — Fourier harmonik geri çatım kalitesi raporu

Kullanım:
  python3 fourier_reconstruct.py

Veri dosyalarına göre otomatik mod:
  R_dy_1_c0.npy varsa → çok-konfig (tüm cN dosyaları yığılır)
  R_dy_1.npy varsa    → tek-konfig

params.json'daki harmoniklerle baz kurulur ve geri çatım kalitesi
(genlik, faz, hata yüzdesi) raporlanır. LASSO, Greedy yok.
"""
import json, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


# ── Fourier baz matrisi ──────────────────────────────────────────────────────

def fodo_basis(n_q, k_list, antisym=True):
    """FODO antisimetrik Fourier baz matrisi.
    F[j,:] = (-1)^j × {1, cos(2πk·(j//2)/N), sin(...)} burada N = n_q//2.
    """
    N = n_q // 2
    j = np.arange(n_q)
    s = (-1.0)**j if antisym else np.ones(n_q)
    n = j // 2
    cols, meta = [], []
    for k in k_list:
        if k == 0:
            cols.append(s.copy())
            meta.append((k, 'dc'))
        else:
            cols.append(s * np.cos(2*np.pi*k*n/N))
            cols.append(s * np.sin(2*np.pi*k*n/N))
            meta.append((k, 'cos'))
            meta.append((k, 'sin'))
    return np.column_stack(cols), meta


# ── yardımcılar ─────────────────────────────────────────────────────────────

def amp_phase_from_coeffs(a, meta):
    """LSQ katsayı vektörü → {k: (A, φ)} sözlüğü."""
    by_k = {}
    for coef, (k, kind) in zip(a, meta):
        by_k.setdefault(k, {})[kind] = coef
    out = {}
    for k, d in by_k.items():
        if 'dc' in d:
            out[k] = (abs(d['dc']), 0.0)
        else:
            ac, as_ = d.get('cos', 0.0), d.get('sin', 0.0)
            out[k] = (np.sqrt(ac**2 + as_**2), np.arctan2(as_, ac))
    return out


def truth_from_cfg(harmonics_cfg):
    """params.json harmonik listesi → {k: (A, φ)} sözlüğü."""
    out = {}
    for h in harmonics_cfg:
        k = h['k']
        ac = h.get('amp_cos', 0.0)
        as_ = h.get('amp_sin', 0.0) if k != 0 else 0.0
        A = np.sqrt(ac**2 + as_**2)
        phi = np.arctan2(as_, ac) if k != 0 else 0.0
        out[k] = (A, phi)
    return out


def phase_diff(phi_fit, phi_true):
    """İki faz arasındaki en küçük açı (0..π)."""
    d = phi_fit - phi_true
    d = (d + np.pi) % (2*np.pi) - np.pi  # [-π, π]'e taşı
    return abs(d)


# ── ana fit + rapor ──────────────────────────────────────────────────────────

def fit_report(label, dR_list, delta_list, dy_true, k_list, truth_cfg, antisym):
    """Yığılmış sistemi fit et, sonuçları tablola."""
    n_q  = dR_list[0].shape[0]
    F, meta = fodo_basis(n_q, k_list, antisym)
    baz  = F.shape[1]

    M = np.vstack([dR @ F for dR in dR_list])
    b = np.concatenate(delta_list)

    sv    = np.linalg.svd(M, compute_uv=False)
    kappa = sv[0] / sv[-1] if sv[-1] > 1e-20 else np.inf
    rank  = int(np.sum(sv > sv[0] * 1e-3))

    a, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    dy_hat = F @ a

    fit   = amp_phase_from_coeffs(a, meta)
    truth = truth_from_cfg(truth_cfg)
    rms   = np.std(dy_hat - dy_true) * 1e6
    cor   = np.corrcoef(dy_true, dy_hat)[0, 1] if np.std(dy_hat) > 1e-20 else float('nan')

    # ── başlık ──
    n_cfg = len(dR_list)
    cfg_str = f"{n_cfg} konfig" if n_cfg > 1 else "1 konfig"
    det_str = "TAM BELİRLİ" if rank >= baz else f"underdetermined  rank {rank}/{baz}"
    print(f"\n{'━'*65}")
    print(f"  {label}   {cfg_str}   {det_str}   κ = {kappa:.2e}")
    print(f"{'━'*65}")
    print(f"  {'k':>2}   {'── tahmin ──':^20}   {'── gerçek ──':^20}   ΔA/A      Δφ")
    print(f"  {'-'*2}   {'-'*20}   {'-'*20}   {'-'*6}    {'-'*6}")

    for k in sorted(set(fit) | set(truth)):
        A_fit, phi_fit   = fit.get(k, (0.0, 0.0))
        A_true, phi_true = truth.get(k, (0.0, 0.0))

        fit_str  = f"{A_fit*1e6:8.1f} μm  ∠{phi_fit:+5.2f}"
        if A_true > 0:
            true_str = f"{A_true*1e6:8.1f} μm  ∠{phi_true:+5.2f}"
            err_A    = abs(A_fit - A_true) / A_true * 100
            err_phi  = phase_diff(phi_fit, phi_true)
            err_str  = f"{err_A:6.1f}%"
            phi_str  = f"{err_phi:.2f} rad"
        else:
            true_str = "        —           "
            err_str  = "     —"
            phi_str  = "    —"

        print(f"  {k:>2}   {fit_str}   {true_str}   {err_str}   {phi_str}")

    print(f"  {'-'*63}")
    print(f"  Profil: RMS hata = {rms:.1f} μm   korelasyon = {cor:.3f}")
    print(f"{'━'*65}")

    return dy_hat


# ── veri yükleme ─────────────────────────────────────────────────────────────

def load_data():
    """Mevcut dosyalara göre veri yükle. (çok-konfig öncelikli)"""
    dRy, dRx, dy_list, dx_list = [], [], [], []
    dy_true = dx_true = None

    # çok-konfig
    for n in range(10):
        if not os.path.exists(f"R_dy_1_c{n}.npy"):
            break
        dRy.append(np.load(f"R_dy_2_c{n}.npy") - np.load(f"R_dy_1_c{n}.npy"))
        dRx.append(np.load(f"R_dx_2_c{n}.npy") - np.load(f"R_dx_1_c{n}.npy"))
        d = np.load(f"kmod_test_c{n}.npz")
        dy_list.append(d["delta_y"])
        dx_list.append(d["delta_x"])
        if dy_true is None:
            dy_true = d["dy_gercek"]
            dx_true = d["dx_gercek"]

    if dRy:
        print(f"  Çok-konfig: {len(dRy)} konfigürasyon yüklendi (c0..c{len(dRy)-1})")
        return dRy, dRx, dy_list, dx_list, dy_true, dx_true

    # tek-konfig
    if os.path.exists("R_dy_1.npy"):
        dRy = [np.load("R_dy_2.npy") - np.load("R_dy_1.npy")]
        dRx = [np.load("R_dx_2.npy") - np.load("R_dx_1.npy")]
        d = np.load("kmod_reconstruction_test.npz")
        dy_list = [d["delta_y"]]
        dx_list = [d["delta_x"]]
        dy_true = d["dy_gercek"]
        dx_true = d["dx_gercek"]
        print("  Tek-konfig: R_dy_1.npy yüklendi")
        return dRy, dRx, dy_list, dx_list, dy_true, dx_true

    return None, None, None, None, None, None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(BASE)

    with open("params.json") as f:
        cfg = json.load(f)

    antisym = cfg.get("smooth_antisym_fodo", True)
    dy_cfg  = cfg.get("dy_harmonics", [])
    dx_cfg  = cfg.get("dx_harmonics", [])
    ky      = sorted({h['k'] for h in dy_cfg})
    kx      = sorted({h['k'] for h in dx_cfg})
    recon_ky = cfg.get("recon_k_list_dy", None)  # opsiyonel sızıntı testi bazı

    print("Fourier Rekonstrüksiyon Kalite Raporu")
    print(f"  dy harmonikleri (gerçek): k = {ky}")
    print(f"  dx harmonikleri (gerçek): k = {kx}")

    dRy, dRx, dy_list, dx_list, dy_true, dx_true = load_data()
    if dRy is None:
        print("\nHata: hiçbir veri dosyası bulunamadı.")
        print("Önce build_response_matrix.py ve test_kmod_reconstruction.py çalıştırın.")
        return

    # ── dikey (dy) ──
    if ky:
        fit_report("DİKEY (dy)  [baz = gerçek]",
                   dRy, dy_list, dy_true, ky, dy_cfg, antisym)

        # Opsiyonel sızıntı testi: baz ≠ gerçek
        if recon_ky is not None and sorted(recon_ky) != ky:
            print(f"\n  [Sızıntı testi: baz = k={sorted(recon_ky)}, gerçek = k={ky}]")
            fit_report("DİKEY (dy)  [baz ≠ gerçek — sızıntı]",
                       dRy, dy_list, dy_true, sorted(recon_ky), dy_cfg, antisym)

    # ── yatay (dx) ──
    if kx:
        fit_report("YATAY (dx)  [baz = gerçek]",
                   dRx, dx_list, dx_true, kx, dx_cfg, antisym)


if __name__ == "__main__":
    main()

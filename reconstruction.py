#!/usr/bin/env python3
"""
reconstruction.py — Adaptif Fourier harmonik geri çatımı (k-modülasyon verisi)

Girdi:
  R_dy_1.npy, R_dy_2.npy       (build_response_matrix.py ürünü)
  R_dx_1.npy, R_dx_2.npy
  kmod_reconstruction_test.npz (test_kmod_reconstruction.py ürünü, delta_y/delta_x)
  params.json                  (k_search_max, threshold, antisim, gerçek harmonikler)

Algoritma (greedy matching pursuit):
  Aktif harmonik seti A = {} başlat.
  Aday harmonikler: k = 0, 1, ..., k_search_max (FODO seviyesi).
  Her adımda:
    Her k ∉ A için, F_test = FODO-antisim Fourier bazı (A ∪ {k} için)
    M_test = ΔR · F_test;  â = lstsq(M_test, Δy);  rezidüel hesapla
    En düşük rezidüeli veren k* → A'ya ekle
  Durdur: rezidüel düşüşü < threshold × önceki_rezidüel olduğunda
  Son A ile rekonstrüksiyon, her k için (genlik, faz) raporla
"""
import json
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def fodo_fourier_basis(n_q, k_list, antisym=True):
    """FODO seviyesinde Fourier baz matrisi.
    F[j, ·] = sign(j) × {1, cos(2πk·n/N), sin(2πk·n/N)} burada n = j//2.
    sign(j) = (-1)^j antisym=True ise, aksi halde 1.
    k=0 için yalnız bir sütun (sabit terim, sign ile çarpılır).
    """
    n_fodo = n_q // 2
    j = np.arange(n_q)
    sign = np.where(j % 2 == 0, 1.0, -1.0 if antisym else 1.0)
    fodo_idx = j // 2

    cols = []
    col_meta = []   # her sütunun (k, 'cos'/'sin'/'dc') etiketi
    for k in k_list:
        if k == 0:
            cols.append(sign * np.ones(n_q))
            col_meta.append((0, 'dc'))
        else:
            cols.append(sign * np.cos(2*np.pi*k*fodo_idx/n_fodo))
            cols.append(sign * np.sin(2*np.pi*k*fodo_idx/n_fodo))
            col_meta.append((k, 'cos'))
            col_meta.append((k, 'sin'))
    F = np.column_stack(cols)
    return F, col_meta


def fit_basis(dR, delta, F):
    """ΔR·F üzerinden lstsq çöz, katsayılar + rezidüel normu döndür."""
    M = dR @ F
    a, _, _, _ = np.linalg.lstsq(M, delta, rcond=None)
    residual = delta - M @ a
    return a, np.linalg.norm(residual)


def greedy_search(dR, delta, k_max, threshold, antisym):
    """Adaptif greedy harmonik seçimi.
    Döndürür: (seçilen_k_listesi, son_F, son_a, son_meta, rezidüel_geçmişi)"""
    n_q = dR.shape[0]
    candidates = list(range(k_max + 1))   # k = 0, 1, ..., k_max
    selected = []
    history = []

    # Başlangıç rezidüeli: hiç fit yok, |Δy| kendisi
    prev_res = np.linalg.norm(delta)
    history.append((None, prev_res))

    while True:
        best_k = None
        best_res = np.inf
        for k in candidates:
            if k in selected:
                continue
            trial = selected + [k]
            F_trial, _ = fodo_fourier_basis(n_q, trial, antisym=antisym)
            _, res = fit_basis(dR, delta, F_trial)
            if res < best_res:
                best_res = res
                best_k = k

        if best_k is None:
            break

        # Rezidüel düşüşü yeterli mi?
        drop = (prev_res - best_res) / prev_res
        if drop < threshold:
            break

        selected.append(best_k)
        history.append((best_k, best_res))
        prev_res = best_res

    F_final, meta_final = fodo_fourier_basis(n_q, selected, antisym=antisym)
    a_final, res_final = fit_basis(dR, delta, F_final)
    return selected, F_final, a_final, meta_final, history


def harmonics_to_amp_phase(a, meta):
    """LSQ katsayılarını (k, amplitüd, faz) listesine dönüştür.
    amplitüd = √(a_cos² + a_sin²), faz = atan2(a_sin, a_cos)."""
    by_k = {}
    for coef, (k, kind) in zip(a, meta):
        d = by_k.setdefault(k, {})
        d[kind] = coef
    out = []
    for k in sorted(by_k):
        d = by_k[k]
        if 'dc' in d:
            out.append((k, abs(d['dc']), 0.0, d['dc'], 0.0))
        else:
            ac = d.get('cos', 0.0)
            as_= d.get('sin', 0.0)
            amp = np.sqrt(ac*ac + as_*as_)
            phase = np.arctan2(as_, ac)
            out.append((k, amp, phase, ac, as_))
    return out


def truth_harmonics(harmonics_cfg):
    """params.json'daki harmonik listesini (k, amp, phase, ac, as) formuna çevir."""
    out = []
    for h in harmonics_cfg:
        k = h["k"]
        ac = h.get("amp_cos", 0.0)
        as_= h.get("amp_sin", 0.0)
        amp = np.sqrt(ac*ac + as_*as_)
        phase = np.arctan2(as_, ac) if k > 0 else 0.0
        out.append((k, amp, phase, ac, as_))
    return out


def print_report(label, dR, delta, gercek_full, selected, a, meta, history,
                 truth_cfg):
    n_q = len(gercek_full)
    F_final, _ = fodo_fourier_basis(n_q, selected, antisym=True)
    geri = F_final @ a

    print(f"\n{'=' * 72}")
    print(f"  {label}  —  ΔR boyut {dR.shape}, Δ RMS = {np.std(delta)*1e6:.2f} μm")
    print(f"{'=' * 72}")

    print(f"\nGreedy arama geçmişi:")
    print(f"  {'adım':>4}  {'seçilen k':>10}  {'rezidüel norm':>14}  {'düşüş %':>9}")
    prev = history[0][1]
    print(f"  {0:>4}  {'(başlangıç)':>10}  {prev:14.4e}  {'—':>9}")
    for i, (k, res) in enumerate(history[1:], 1):
        drop = (prev - res) / prev * 100
        print(f"  {i:>4}  {k:>10d}  {res:14.4e}  {drop:8.2f}%")
        prev = res
    print(f"  Seçilen harmonikler: {selected}")

    found = harmonics_to_amp_phase(a, meta)
    truth = truth_harmonics(truth_cfg)
    truth_map = {t[0]: t for t in truth}

    print(f"\nTespit edilen harmonikler vs gerçek:")
    print(f"  {'k':>3}  {'A_tahmin':>11}  {'φ_tahmin':>10}  {'A_gercek':>11}  {'φ_gercek':>10}  {'|ΔA|/A %':>10}")
    print(f"  {'-'*3}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*10}")
    all_ks = sorted(set([f[0] for f in found] + [t[0] for t in truth]))
    for k in all_ks:
        f = next((x for x in found if x[0] == k), None)
        t = truth_map.get(k)
        a_tah = f[1] if f else 0.0
        p_tah = f[2] if f else 0.0
        a_gr  = t[1] if t else 0.0
        p_gr  = t[2] if t else 0.0
        err   = abs(a_tah - a_gr) / a_gr * 100 if a_gr > 1e-15 else float('nan')
        flag  = "  ← spurious" if (f and not t) else ("  ← kaçırıldı" if (t and not f) else "")
        a_tah_um = a_tah * 1e6
        a_gr_um  = a_gr  * 1e6
        if np.isnan(err):
            print(f"  {k:>3d}  {a_tah_um:8.2f} μm  {p_tah:8.3f}  {a_gr_um:8.2f} μm  {p_gr:8.3f}     —    {flag}")
        else:
            print(f"  {k:>3d}  {a_tah_um:8.2f} μm  {p_tah:8.3f}  {a_gr_um:8.2f} μm  {p_gr:8.3f}  {err:7.2f}%{flag}")

    err_rms = np.std(geri - gercek_full) * 1e6
    cor     = np.corrcoef(gercek_full, geri)[0, 1]
    print(f"\nTam rekonstrüksiyon hata (smooth+gurultu tabanı dahil):")
    print(f"  hata RMS = {err_rms:.3f} μm   korelasyon = {cor:.6f}")
    return geri


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    k_max     = config.get("k_search_max", 12)
    threshold = config.get("greedy_residual_threshold", 0.02)
    antisym   = config.get("smooth_antisym_fodo", True)

    # ΔR'leri hesapla
    R_dy_1 = np.load("R_dy_1.npy")
    R_dy_2 = np.load("R_dy_2.npy")
    R_dx_1 = np.load("R_dx_1.npy")
    R_dx_2 = np.load("R_dx_2.npy")
    dR_dy  = R_dy_2 - R_dy_1
    dR_dx  = R_dx_2 - R_dx_1

    # Ölçüm sonuçlarını yükle
    data = np.load("kmod_reconstruction_test.npz")
    delta_y    = data["delta_y"]
    delta_x    = data["delta_x"]
    dy_gercek  = data["dy_gercek"]
    dx_gercek  = data["dx_gercek"]

    print(f"Adaptif Fourier rekonstrüksiyonu")
    print(f"  k_search_max = {k_max}")
    print(f"  greedy_residual_threshold = {threshold} (= %{threshold*100:.0f} düşüş)")
    print(f"  FODO antisimetrik baz: {antisym}")

    # dy
    selected_y, F_y, a_y, meta_y, hist_y = greedy_search(
        dR_dy, delta_y, k_max, threshold, antisym)
    geri_y = print_report("DİKEY (dy)", dR_dy, delta_y, dy_gercek,
                          selected_y, a_y, meta_y, hist_y,
                          config.get("dy_harmonics", []))

    # dx
    selected_x, F_x, a_x, meta_x, hist_x = greedy_search(
        dR_dx, delta_x, k_max, threshold, antisym)
    geri_x = print_report("YATAY (dx)", dR_dx, delta_x, dx_gercek,
                          selected_x, a_x, meta_x, hist_x,
                          config.get("dx_harmonics", []))

    np.savez("reconstruction_result.npz",
             dy_gercek=dy_gercek, dy_geri=geri_y,
             dx_gercek=dx_gercek, dx_geri=geri_x,
             selected_k_y=np.array(selected_y),
             selected_k_x=np.array(selected_x),
             a_y=a_y, a_x=a_x)
    print("\nSonuçlar 'reconstruction_result.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""test_cross_correlation.py — Üç açık soruya sayısal yanıt.

S1: CO=True (kapalı yörünge üzerinde fırlatma) gerçek hızlandırıcıda
    kullanılabilir mi? Tek noktada kick yeterli mi, yoksa dağıtık
    düzelticiler mi gerekir?

S2: dSy/dt = Σ_ij c_ij A_i A_j ikinci-dereceden formun katsayı matrisi.
    k=2,3,4 cos bileşenlerinden 3×3 c_ij ölçülür. Hangi mod çiftleri
    birbirini bastırır (negatif c_ij)?

S3: c_ij özdeğer ayrışımından false EDM'yi minimize eden mod kombinasyonu.
    En küçük özdeğerin özvektörü doğrulama simülasyonuyla test edilir.

Yöntem:
  9 CO=True simülasyonu (k2, k3, k4 köşegen + 3 çift pos + 3 çift neg)
  + 1 CO=False karşılaştırma = toplam 10 simülasyon (paralel).

  c_ij hesaplama formülü:
    c_ii = F_i / A²                                      (köşegen)
    c_ij = [F(+A,+A) - F(+A,-A)] / (4A²)               (köşegen dışı)

  Bu formül, F'nin tam karesel form olmasını gerektirmez — sign bilgisi
  doğrudan +/- kombinasyonlardan çıkarılır.
"""

import json, os, sys, time, numpy as np
import multiprocessing as mp

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Worker fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def _suppress_stdout():
    """C++ stdout'u /dev/null'a yönlendirir. (fd 1 düzeyinde)"""
    import os
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    import os
    os.dup2(fd, 1)
    os.close(fd)


def _worker_co_true(task):
    """CO=True stroboskopik eğim ölçümü. Task: (label, [(k,ac,as),...], t2, co_turns)"""
    label, modes_amps, t2, co_turns = task
    import os, sys, json, time, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state, C
    from fourier_reconstruct import fodo_basis
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    n_q = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    dt = float(config.get("dt", 1e-11))

    dy = np.zeros(n_q)
    for k, ac, asin in modes_amps:
        Fk, _ = fodo_basis(n_q, [k], antisym)
        dy += Fk[:, 0] * float(ac) + Fk[:, 1] * float(asin)

    circ = 2*np.pi*R0 + 4*fields.nFODO*fields.driftLen + 2*fields.nFODO*fields.quadLen
    T_rev = circ / (beta0 * C)

    t0 = time.time()
    # C++ verbose çıktısını bastır
    saved_fd = _suppress_stdout()
    try:
        v_co, resid = find_closed_orbit(fields, p_mag, direction, dy, dt, T_rev,
                                        n_turns=co_turns, n_iter=1)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])

        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(y_launch, 0.0, t2, dt, fields=fields,
                                              return_steps=3000, quad_dy=dy)
    finally:
        _restore_stdout(saved_fd)

    sy = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    co_off_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)
    return label, slope, co_off_mm, time.time() - t0


def _worker_co_false(task):
    """CO=False stroboskopik eğim ölçümü (y=0 fırlatma). Task: (label, [(k,ac,as),...], t2)"""
    label, modes_amps, t2 = task
    import os, sys, json, time, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields, C
    from fourier_reconstruct import fodo_basis
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, y0, _, _, _, _ = setup_fields(config)
    n_q = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    dt = float(config.get("dt", 1e-11))

    dy = np.zeros(n_q)
    for k, ac, asin in modes_amps:
        Fk, _ = fodo_basis(n_q, [k], antisym)
        dy += Fk[:, 0] * float(ac) + Fk[:, 1] * float(asin)

    fields.poincare_quad_index = 0.0
    t0 = time.time()
    saved_fd = _suppress_stdout()
    try:
        _, poin, poin_t = integrate_particle(y0, 0.0, t2, dt, fields=fields,
                                              return_steps=3000, quad_dy=dy)
    finally:
        _restore_stdout(saved_fd)
    sy = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    return label, slope, 0.0, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Ana program
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    A  = 1e-5    # 10 μm mod katsayısı
    t2 = 5e-4    # ~109 tur
    co_turns = 24   # n_iter=1 ile tek geçiş yeterli
    ctx = mp.get_context("spawn")

    MODES = [2, 3, 4]
    n = len(MODES)

    # ── Tüm görevleri tek listede topla ─────────────────────────────────────
    all_tasks_true  = []
    all_tasks_false = []

    # S1: CO=False, k=2 karşılaştırma
    all_tasks_false.append(("co_false_k2", [(2, A, 0.0)], t2))

    # S2 köşegenler: tek mod
    for k in MODES:
        all_tasks_true.append((f"diag_k{k}", [(k, A, 0.0)], t2, co_turns))

    # S2 çift +A+A
    for i, ki in enumerate(MODES):
        for j, kj in enumerate(MODES):
            if j > i:
                all_tasks_true.append(
                    (f"pp_{ki}_{kj}", [(ki, A, 0.0), (kj,  A, 0.0)], t2, co_turns))

    # S2 çift +A-A
    for i, ki in enumerate(MODES):
        for j, kj in enumerate(MODES):
            if j > i:
                all_tasks_true.append(
                    (f"pm_{ki}_{kj}", [(ki, A, 0.0), (kj, -A, 0.0)], t2, co_turns))

    total = len(all_tasks_true) + len(all_tasks_false)
    print(f"\n{'='*65}")
    print(f"  ÇAPRAZ KORELASYON ANALİZİ — Üç Açık Soru")
    print(f"  A = {A*1e6:.0f} μm,  t2 = {t2*1e3:.1f} ms,  co_turns = {co_turns}")
    print(f"  Toplam simülasyon: {total}  (paralel)")
    print(f"{'='*65}")

    t_wall = time.time()
    with ctx.Pool(min(total, 6)) as pool:
        results_true_raw  = pool.map(_worker_co_true,  all_tasks_true)
        results_false_raw = pool.map(_worker_co_false, all_tasks_false)

    wall = time.time() - t_wall
    print(f"  Bitti — duvar saati: {wall:.0f}s")

    Fco  = {r[0]: (r[1], r[2]) for r in results_true_raw}    # label → (slope, co_mm)
    Fno  = {r[0]: (r[1], r[2]) for r in results_false_raw}

    # ── SORU 1 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  S1: CO=True vs CO=False — k=2, A={A*1e6:.0f}μm")
    print(f"{'='*65}")

    s_true  = Fco.get("diag_k2",  (float("nan"), 0))[0]
    s_false = Fno.get("co_false_k2", (float("nan"), 0))[0]
    ratio   = abs(s_false) / abs(s_true) if abs(s_true) > 1e-30 else float("inf")
    co_off  = Fco.get("diag_k2", (0, 0))[1]

    print(f"\n  CO=True  (kapalı yörünge fırlatması):")
    print(f"    dSy/dt = {s_true:+.3e} rad/s   |CO ofseti| = {co_off:.3f} mm")
    print(f"\n  CO=False (y=0 ideal başlangıç):")
    print(f"    dSy/dt = {s_false:+.3e} rad/s")
    print(f"\n  |CO=False| / |CO=True| = {ratio:.1e}×")
    print(f"""
  YORUM:
  CO=True ≡ parçacık, misalignment'lı kafes kapalı yörüngesi üzerinde.
  Gerçek bir hızlandırıcıda bu, ring boyunca dağıtık BPM ölçümü +
  harmonik analiz + korrektör uygulamasıyla sağlanır. Yörünge düzeltmesi
  sonrası parçacık zaten bu "kapalı yörüngeye" oturtulmuş sayılır.

  TEK KİCK: Ring boyunca dalganın tek noktada bükülmesi kapalı yörüngeyi
  yalnızca kısmen değiştirir. Tek kick ile betatron amplitüdü k=2 bazında
  azaltılabilir ancak diğer modlar etkilenmez → CO=True rejimi elde edilmez.
  Tam bastırım için en az 2N korrektör (N = düzeltilecek mod sayısı) gerekir.
""")

    # ── SORU 2 ──────────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  S2: Çapraz Korelasyon Matrisi c_ij  [k=2,3,4 cos; A={A*1e6:.0f}μm]")
    print(f"{'='*65}")

    c = np.zeros((n, n))
    for i, k in enumerate(MODES):
        fi = Fco.get(f"diag_k{k}", (0.0, 0))[0]
        c[i, i] = fi / A**2

    for i, ki in enumerate(MODES):
        for j, kj in enumerate(MODES):
            if j > i:
                fpp = Fco.get(f"pp_{ki}_{kj}", (0.0, 0))[0]
                fpm = Fco.get(f"pm_{ki}_{kj}", (0.0, 0))[0]
                c_ij = (fpp - fpm) / (4 * A**2)
                c[i, j] = c_ij
                c[j, i] = c_ij

    # Matris yazdır
    header = "         " + "  ".join(f"{'k='+str(k):>12}" for k in MODES)
    print(f"\n  {header}")
    for i, ki in enumerate(MODES):
        row = f"  k={ki}    " + "  ".join(f"{c[i,j]:>12.4e}" for j in range(n))
        print(row)

    # Bireysel ölçüm özeti
    print(f"\n  Bireysel mod dSy/dt değerleri (A={A*1e6:.0f}μm):")
    for i, k in enumerate(MODES):
        fi = Fco.get(f"diag_k{k}", (0.0, 0))[0]
        print(f"    k={k}: {fi:+.4e} rad/s   (c_{k}{k} = {c[i,i]:+.4e})")

    print(f"\n  Çift mod dSy/dt (A her ikisine, cos-cos):")
    for i, ki in enumerate(MODES):
        for j, kj in enumerate(MODES):
            if j > i:
                fpp = Fco.get(f"pp_{ki}_{kj}", (0.0, 0))[0]
                fpm = Fco.get(f"pm_{ki}_{kj}", (0.0, 0))[0]
                expected_no_cross = (c[i,i] + c[j,j]) * A**2
                cross_contrib = fpp - expected_no_cross
                print(f"    k={ki}+k={kj}(+): {fpp:+.4e}  k={ki}+k={kj}(-): {fpm:+.4e}"
                      f"  c_{ki}{kj}={c[i,j]:+.4e}")
                pct = abs(cross_contrib/expected_no_cross)*100 if abs(expected_no_cross)>1e-30 else 0
                print(f"       çapraz katkı: {cross_contrib:+.4e} rad/s  "
                      f"({pct:.0f}% köşegen toplamından)")

    # Özdeğer analizi
    try:
        eigvals, eigvecs = np.linalg.eigh(c)
        idx = np.argsort(eigvals)
        eigvals  = eigvals[idx]
        eigvecs  = eigvecs[:, idx]
        ok_eigen = True
    except Exception as e:
        print(f"  [özdeğer hatası: {e}]")
        ok_eigen = False

    if ok_eigen:
        print(f"\n  Özdeğer ayrışımı:")
        for lam, v in zip(eigvals, eigvecs.T):
            v_str = ", ".join(f"{x:+.3f}" for x in v)
            sign_tag = " ← negatif!" if lam < 0 else ""
            print(f"    λ = {lam:+.4e}   v = [{v_str}]{sign_tag}")

        if np.any(eigvals < 0):
            print(f"\n  *** Negatif özdeğer var — bu mod kombinasyonu ters")
            print(f"      yönde false EDM üretir. Aralarında geçişte |dSy/dt|=0 noktası var.")
        else:
            min_to_max = eigvals[0] / eigvals[-1]
            print(f"\n  Tüm özdeğerler pozitif → c_ij pozitif yarı-tanımlı")
            print(f"  λ_min/λ_max = {min_to_max:.3f}")

    # ── SORU 3 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  S3: Minimum False EDM Konfigürasyonu")
    print(f"{'='*65}")

    if ok_eigen:
        v_min   = eigvecs[:, 0]
        lam_min = eigvals[0]
        v_max   = eigvecs[:, -1]
        lam_max = eigvals[-1]

        pred_min = lam_min * A**2
        pred_max = lam_max * A**2

        modes_opt = [(MODES[i], float(v_min[i]) * A, 0.0) for i in range(n)]
        modes_wrs = [(MODES[i], float(v_max[i]) * A, 0.0) for i in range(n)]

        print(f"\n  En küçük özdeğer λ_min = {lam_min:.4e} rad/s·m⁻²")
        print(f"  Özvektör (optimal kombinasyon): {v_min}")
        print(f"  Optimal mod katsayıları (A_k·v_k):")
        for i, (k, ac, _) in enumerate(modes_opt):
            print(f"    k={k}: A_cos = {ac*1e6:+.3f} μm  ({v_min[i]:+.4f} × {A*1e6:.0f}μm)")
        print(f"  Teorik dSy/dt = λ_min × A² = {pred_min:+.4e} rad/s")

        print(f"\n  En büyük özdeğer λ_max = {lam_max:.4e} rad/s·m⁻²")
        print(f"  Teorik dSy/dt_max = λ_max × A² = {pred_max:+.4e} rad/s")

        print(f"\n  Doğrulama simülasyonları (optimal + en kötü)...")
        verify_tasks = [
            ("opt_min", modes_opt, t2, co_turns),
            ("opt_max", modes_wrs, t2, co_turns),
        ]
        t0 = time.time()
        with ctx.Pool(2) as pool:
            r_verify = pool.map(_worker_co_true, verify_tasks)
        print(f"  Doğrulama süresi: {time.time()-t0:.0f}s")

        Fv = {r[0]: r[1] for r in r_verify}
        slope_min = Fv.get("opt_min", float("nan"))
        slope_max = Fv.get("opt_max", float("nan"))

        f_k2 = Fco.get("diag_k2", (0.0, 0))[0]

        print(f"\n  Optimal (λ_min): ölçülen = {slope_min:+.4e}  tahmin = {pred_min:+.4e}")
        print(f"  En kötü (λ_max): ölçülen = {slope_max:+.4e}  tahmin = {pred_max:+.4e}")

        ratio_opt = abs(f_k2) / abs(slope_min) if abs(slope_min) > 1e-30 else float("inf")
        ratio_wrs = abs(slope_max) / abs(f_k2)  if abs(f_k2) > 1e-30 else float("inf")

        print(f"\n  Karşılaştırma (A = {A*1e6:.0f}μm):")
        print(f"    k=2 tek mod (temel):   {f_k2:+.4e} rad/s")
        print(f"    Optimal kombinasyon:   {slope_min:+.4e} rad/s  "
              f"({ratio_opt:.1f}× bastırım vs k=2)")
        print(f"    En kötü kombinasyon:   {slope_max:+.4e} rad/s  "
              f"({ratio_wrs:.1f}× büyütme vs k=2)")

        print(f"""
  YORUM — OPTİMİZASYON:
  c_ij matrisi, belirli bir toplam mod genliği (||A||=sabit) için
  hangi karışımın minimum/maksimum false EDM verdiğini söyler.

  Pratik uygulama:
  - Mevcut hata desenini Fourier analiz et (orbit ölçümünden)
  - c_ij'yi kullanarak her modun yönünü (işaretini) tahmin et
  - Orbit korrektörleriyle kötü moda katkıyı azalt
  - "Spin ölçümü ile optimize etmek" = dSy/dt'yi A(t)'ye göre geri
    besleme yaparak minimize etmek → en az 2k+1 spin ölçümü gerekir
    (quadratik fit için)
""")

    # ── NİHAİ ÖZET ──────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  NİHAİ ÖZET")
    print(f"{'='*65}")
    print(f"\n  S1: CO=True/False oranı = {ratio:.2e}×")
    print(f"      → CO=True gerçek hızlandırıcıda uygulanabilir (dağıtık orbit düzeltimi)")
    print(f"      → Tek kick: sadece kısmi bastırım, CO=True rejimi elde edilemez")
    if ok_eigen:
        print(f"\n  S2: c_ij matris özellikleri:")
        print(f"      Özdeğerler = {[f'{lam:.3e}' for lam in eigvals]}")
        neg = "NEGATİF özdeğer var" if np.any(eigvals < 0) else "tümü pozitif"
        print(f"      → {neg}")
        print(f"      → Modlar arası {'güçlü' if abs(c[0,1])/(c[0,0]+1e-30)>0.1 else 'zayıf'}"
              f" çapraz korelasyon")
        print(f"\n  S3: Optimal kombinasyon (λ_min özvektörü):")
        print(f"      Ölçülen: {slope_min:+.3e} rad/s  (teorik: {pred_min:+.3e})")
        if ok_eigen and abs(slope_min) > 1e-30:
            print(f"      k=2 tek moda göre bastırım: {abs(f_k2)/abs(slope_min):.1f}×")

#!/usr/bin/env python3
"""teshis_p_azimut.py — Ek rapor madde 9: fırlatma noktasına bağımlılık nereden geliyor?

Hipotez: sert kenarlı saptırıcı modelinde (alan saptırıcı sınırında aniden başlar, potansiyel basamağı
yok) elektrik alanı saptırıcı içindeki radyal harekete göre iş yapar; bu yüzden AYNI kapalı yörüngenin
farklı hücrelerdeki alansız bölgelerinde |p| farklıdır. Eski prosedür (|p| = p_sihirli, fırlatma
hücre k'nın ARC1 çıkışında) böylece her k için farklı bir yörünge enerjisi seçer.

Test: σ = 10 μm, seed 0 deseni (rng 3000), CCW. Eski prosedürle (|p| = p_sihirli) hücre 0'dan kapalı
yörünge üzerinde fırlat; hücre k = 0, 6, 12, 18'in ARC2 girişinde |p|'yi ölç (ARC1 çıkışından ARC2
girişine kadar yalnız sürüklenme + kuadrupol → |p| sabit, yani bu, hücre k'dan fırlatma noktasındaki
|p|'dir). δ_k = |p|/p_sihirli − 1.
Öngörü (teshis_enerji_ABC sonuçlarından, f doğrusal δ'da, eğim −2.906e9 × hedef / birim δ):
   f_k(eski) = 1088, 2729, −456, 95 ; f* = 82 → δ(fırlatma k) = (f_k − 82)/(−2.906e9·1e-9)
   δ_k(hücre 0 yörüngesinde) = δ(fırlatma 0) − δ(fırlatma k)  →  0, +5.65e-7, −5.31e-7, −3.41e-7
Kontrol: kaçıklık sıfırken δ_k ≡ 0 olmalı.
"""
import os, sys, json, tempfile
import numpy as np

B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon/tests", B + "/revizyon", B, B + "/berry_data", B + "/kmod_drivers"]
OUT = B + "/revizyon/sonuclar/teshis_p_azimut.json"


def pmag_state(v, p_mag, direction, spin):
    return [v[0], v[1], 0.0, p_mag * direction * v[2], p_mag * direction * v[3], p_mag * direction,
            spin[0], spin[1], spin[2]]


def run(sig):
    os.chdir(tempfile.mkdtemp())
    import false_edm_4d, false_edm_mode_scan
    false_edm_4d._make_state = pmag_state; false_edm_mode_scan._make_state = pmag_state
    from false_edm_mode_scan import setup_fields
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    cfg = json.load(open(B + "/params.json"))
    rng = np.random.default_rng(3000)
    dx = rng.normal(0, sig, 48); dy = rng.normal(0, sig, 48)
    fields, y0, beta0, R0_, p_mag, d = setup_fields(cfg)
    z = np.zeros(48)
    Trev = _T_rev(fields, beta0, R0_)
    v_co, resid = find_co_4d(fields, p_mag, d, dx, dy, z, Trev, n_turns=28, n_iter=4)
    launch = pmag_state(v_co, p_mag, d, [0.0, 0.0, d])
    res = dict(sigma_um=sig * 1e6, co_resid=float(resid), cells={})
    for k in (0, 6, 12, 18):
        fields.poincare_quad_index = float(2 * k + 1)          # tek → hücre k'nın ARC2 girişi
        _, poin, _ = integrate_particle(launch, 0.0, 6 * Trev, float(cfg["dt"]), fields=fields,
                                        return_steps=10, quad_dx=dx, quad_dy=dy, quad_tilt=z)
        p = np.sqrt(poin[:, 3] ** 2 + poin[:, 4] ** 2 + poin[:, 5] ** 2)
        res["cells"][k] = dict(delta_mean=float(p.mean() / p_mag - 1), delta_std=float(p.std() / p_mag),
                               n=int(len(p)))
    return res


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2) as pool:
        R = list(pool.map(run, [10e-6, 0.0]))
    json.dump(R, open(OUT, "w"), indent=1)
    pred = {0: 0.0, 6: 5.65e-7, 12: -5.31e-7, 18: -3.41e-7}
    for r in R:
        print(f"σ = {r['sigma_um']:.0f} μm (CO artığı {r['co_resid']:.1e}):")
        d0 = r["cells"][0]["delta_mean"]
        for k, c in r["cells"].items():
            print(f"  hücre {k:2d}: δ = {c['delta_mean']:+.3e} (tur-tur std {c['delta_std']:.1e}, n={c['n']}); "
                  f"δ − δ_0 = {c['delta_mean'] - d0:+.3e}   öngörü {pred[k]:+.2e}")

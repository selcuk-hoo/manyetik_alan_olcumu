#!/usr/bin/env python3
"""teshis_enerji_ABC.py — sahte EDM ölçüm prosedüründe fırlatma enerjisinin fiziksel tanımı.

Desen: paper_runs 'sigma', σ = 10 μm, seed 0 (rng 3000). Yön CCW. Tam ölçüm ayarı (28/4, t2 = 3e-4 s).
Fırlatma modları:
  pmag   : (şu anki) kapalı yörünge üzerinde, momentum büyüklüğü sihirli değer
  energy : kapalı yörünge üzerinde, toplam enerji = tasarım (eksenden sihirli momentumla giren parçacık)
  δ      : energy modunun momentumuna ek bağıl sapma (kapalı yörünge aramasında da uygulanır)
Fırlatma azimutu: desen 2k kuadrupol kaydırılır (halka simetrik → fizik aynı, başlangıç noktası değişir).
Her koşumda: f = dS_y/dt (model-fit) ve yatay presesyon hızı ω_h = d/dt atan2(S_x, S_z).

A: energy, δ ∈ {−1e-6, −1e-7, 0, +1e-7, +1e-6}, k = 0     → f(δ), ω_h(δ)
B: pmag ve energy, k ∈ {0, 6, 12, 18}                       → azimut bağımlılığı
C: ω_h(δ*) = 0 noktası (A'dan doğrusal) → f(δ*)             → donmuş spin noktası
"""
import os, sys, json, tempfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor

B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon/tests", B, B + "/berry_data", B + "/kmod_drivers"]
from ortak import Q_E, M_P, C_LIGHT, E0_MAGIC, R0

OUT = B + "/revizyon/sonuclar/teshis_enerji_ABC.json"


def make_launcher(mode, delta):
    def make_state(v, p_mag, direction, spin):
        x = v[0]
        if mode == "pmag":
            p = p_mag
        else:
            W_ref = np.sqrt((p_mag * C_LIGHT) ** 2 + (M_P * C_LIGHT ** 2) ** 2)
            Wk = W_ref - Q_E * (-E0_MAGIC * R0 * np.log((R0 + x) / R0))
            p = np.sqrt(Wk ** 2 - (M_P * C_LIGHT ** 2) ** 2) / C_LIGHT
        p *= (1.0 + delta)
        return [v[0], v[1], 0.0, p * direction * v[2], p * direction * v[3], p * direction,
                spin[0], spin[1], spin[2]]
    return make_state


def run(task):
    os.chdir(tempfile.mkdtemp())
    mode, delta, k = task
    import false_edm_4d, false_edm_mode_scan
    ms = make_launcher(mode, delta)
    false_edm_4d._make_state = ms; false_edm_mode_scan._make_state = ms
    from false_edm_mode_scan import setup_fields, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    cfg = json.load(open(B + "/params.json"))
    rng = np.random.default_rng(3000)
    dx = np.roll(rng.normal(0, 10e-6, 48), -2 * k); dy = np.roll(rng.normal(0, 10e-6, 48), -2 * k)
    fields, y0, beta0, R0_, p_mag, d = setup_fields(cfg)
    z = np.zeros(48)
    v_co, resid = find_co_4d(fields, p_mag, d, dx, dy, z, _T_rev(fields, beta0, R0_), n_turns=28, n_iter=4)
    launch = ms(v_co, p_mag, d, [0.0, 0.0, d])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrate_particle(launch, 0.0, 3e-4, float(cfg["dt"]), fields=fields,
                                     return_steps=4000, quad_dx=dx, quad_dy=dy, quad_tilt=z)
    t = np.asarray(pt, float)
    f = float(measure_dSy_dt_model(np.asarray(poin[:, 7], float), t))
    ang = np.unwrap(np.arctan2(np.asarray(poin[:, 6], float), np.asarray(poin[:, 8], float) * d))
    w_h = float(np.polyfit(t, ang, 1)[0])
    return dict(mode=mode, delta=delta, k=k, f=f, f_target=f / 1e-9, w_h=w_h,
                co_x_um=float(v_co[0] * 1e6), co_resid=float(resid))


def main():
    A = [("energy", d, 0) for d in (-1e-6, -1e-7, 0.0, 1e-7, 1e-6)]
    Bt = [(m, 0.0, k) for m in ("pmag", "energy") for k in (0, 6, 12, 18) if not (m == "energy" and k == 0)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["mode"], r["delta"], r["k"]) for r in res}
    todo = [t for t in A + Bt if t not in done]
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(run, todo):
            res.append(r); json.dump(res, open(OUT, "w"), indent=1)
            print(f"{r['mode']:6s} δ={r['delta']:+.0e} k={r['k']:2d}: f = {r['f_target']:9.2f} × hedef, "
                  f"ω_h = {r['w_h']:+.4e} rad/s, x_co(fırlatma) = {r['co_x_um']:+.2f} μm", flush=True)
    # C: ω_h(δ) = 0 noktası
    a = sorted([r for r in res if r["mode"] == "energy" and r["k"] == 0], key=lambda r: r["delta"])
    dl = np.array([r["delta"] for r in a]); wh = np.array([r["w_h"] for r in a])
    slope, icpt = np.polyfit(dl, wh, 1)
    dstar = float(-icpt / slope)
    if not any(r["mode"] == "energy" and abs(r["delta"] - dstar) < 1e-12 and r["k"] == 0 for r in res):
        r = run(("energy", dstar, 0)); r["C_test"] = True
        res.append(r); json.dump(res, open(OUT, "w"), indent=1)
        print(f"C: δ* = {dstar:+.3e} → f = {r['f_target']:.2f} × hedef, ω_h = {r['w_h']:+.3e}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
false_edm_4d.py — Sahte-EDM'in (dS_y/dt) hizalama deseniyle ilişkisi, 4D kapalı
yörüngede tam spin takibiyle.

İki çalışma:
  (1) sahte EDM vs Ŝ   : Ŝ = (simetrik güç − antisimetrik güç)/(toplam) ∈[-1,1]
      Ŝ kontrollü desenlerle (antisim→+sim) taranır; dx,dy aynı simetri yapısı.
  (2) sahte EDM vs tilt : dx,dy sabit rastgele; quad tilt RMS taranır.
      Tilt İŞARETLİ: her quad için N(0,θ) (hem + hem −).

Ölçüm reçetesi (doğrulanmış): 4D kapalı yörüngede (x,x',y,y') tek ideal parçacık
fırlat (betatron öl) → EDMSwitch=0 → her tur S_y örnekle → model-fit seküler eğim.
4D CO, betatron pozisyon varyansını (var x + var y) 4-boyutlu Newton ile minimize
ederek bulunur (lattis lineer → kuadratik çukur, 1-2 adım yeter).

Kullanım:
  python3 false_edm_4d.py --test                  # hızlı akıl sağlığı
  python3 false_edm_4d.py --scan-s --workers 7    # Ŝ kampanyası
  python3 false_edm_4d.py --scan-tilt --workers 7 # tilt kampanyası
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = "/home/user/manyetik_alan_olcumu"
sys.path.insert(0, _BASE)
import build_response_matrix as brm
from false_edm_mode_scan import (setup_fields, _make_state,
                                  measure_dSy_dt_model, C)

with open(os.path.join(_BASE, "params.json")) as _f:
    CFG = json.load(_f)
NQ = 2 * int(CFG["nFODO"])
DT = float(CFG["dt"])


def _T_rev(fields, beta0, R0):
    circ = (2 * np.pi * R0 + 4 * fields.nFODO * fields.driftLen
            + 2 * fields.nFODO * fields.quadLen)
    return circ / (beta0 * C)


def find_co_4d(fields, p_mag, direction, dx, dy, tilt, T_rev,
               n_turns=28, n_iter=2, dG=None):
    """4D kapalı yörünge (x,x',y,y'): betatron pozisyon varyansını minimize et."""
    from integrator import integrate_particle
    spin = [0.0, 0.0, direction]
    fields.poincare_quad_index = 0.0
    t_probe = n_turns * T_rev

    def var4(v):
        st = _make_state(v, p_mag, direction, spin)
        _, poin, _ = integrate_particle(st, 0.0, t_probe, DT, fields=fields,
                                        return_steps=10, quad_dx=dx, quad_dy=dy,
                                        quad_tilt=tilt, quad_dG=dG)
        if poin is None or len(poin) < 5:
            return 1e30
        return float(np.var(poin[:, 0]) + np.var(poin[:, 1]))

    v = np.zeros(4)
    h = np.array([2e-4, 2e-5, 2e-4, 2e-5])
    for _ in range(n_iter):
        f0 = var4(v)
        g = np.zeros(4); H = np.zeros((4, 4)); fp = np.zeros(4)
        for i in range(4):
            ei = np.zeros(4); ei[i] = h[i]
            fpi = var4(v + ei); fmi = var4(v - ei); fp[i] = fpi
            g[i] = (fpi - fmi) / (2 * h[i])
            H[i, i] = (fpi - 2 * f0 + fmi) / (h[i] ** 2)
        for i in range(4):
            for j in range(i + 1, 4):
                ei = np.zeros(4); ei[i] = h[i]; ej = np.zeros(4); ej[j] = h[j]
                fpp = var4(v + ei + ej)
                H[i, j] = H[j, i] = (fpp - fp[i] - fp[j] + f0) / (h[i] * h[j])
        try:
            dv = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dv)):
            break
        v = v + dv
        h *= 0.3
    resid = np.sqrt(max(var4(v), 0.0))
    fields.poincare_quad_index = -1.0
    return v, resid


def measure_false_edm(dx, dy, tilt, t2=5e-4, return_steps=5000):
    """4D CO'da tek ideal parçacıkla seküler dS_y/dt [rad/s]."""
    from integrator import integrate_particle
    fields, y0, beta0, R0, p_mag, direction = setup_fields(CFG)
    T_rev = _T_rev(fields, beta0, R0)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, tilt, T_rev)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrate_particle(launch, 0.0, t2, DT, fields=fields,
                                     return_steps=return_steps,
                                     quad_dx=dx, quad_dy=dy, quad_tilt=tilt)
    sv = np.asarray(poin[:, 7], float)
    slope = float(measure_dSy_dt_model(sv, np.asarray(pt, float)))
    return slope, resid


def s_hat(arr):
    """Ŝ = (||sym||² − ||antisym||²)/(toplam); hücre içi QF/QD."""
    qf = arr[0::2]; qd = arr[1::2]
    s = 0.5 * (qf + qd); d = 0.5 * (qf - qd)
    num = np.sum(s ** 2) - np.sum(d ** 2)
    den = np.sum(s ** 2) + np.sum(d ** 2)
    return num / den if den > 0 else 0.0


def gen_pattern(w, sigma, rng):
    """w∈[0,1] simetrik ağırlık (w=0 antisim, w=1 sim); Ŝ≈2w−1."""
    nC = NQ // 2
    s = rng.normal(0, 1, nC); d = rng.normal(0, 1, nC)
    qf = np.sqrt(w) * s + np.sqrt(1 - w) * d
    qd = np.sqrt(w) * s - np.sqrt(1 - w) * d
    arr = np.empty(NQ); arr[0::2] = qf; arr[1::2] = qd
    arr *= sigma / np.std(arr)
    return arr


# ── kampanya worker'ları ──────────────────────────────────────────────────
def _w_s(task):
    w, seed, sigma = task
    rng = np.random.default_rng(10000 + seed)
    dx = gen_pattern(w, sigma, rng); dy = gen_pattern(w, sigma, rng)
    f, resid = measure_false_edm(dx, dy, np.zeros(NQ))
    sh = 0.5 * (s_hat(dx) + s_hat(dy))
    return sh, abs(f), resid


def _w_tilt(task):
    theta, seed, sigma = task
    rng = np.random.default_rng(20000 + seed)
    dx = gen_pattern(0.5, sigma, rng); dy = gen_pattern(0.5, sigma, rng)
    tilt = rng.normal(0, theta, NQ) if theta > 0 else np.zeros(NQ)  # İŞARETLİ
    f, resid = measure_false_edm(dx, dy, tilt)
    return theta, abs(f), resid


def _run(tasks, fn, workers):
    out = []
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=brm._worker_init) as pool:
            for r in pool.map(fn, tasks):
                out.append(r)
    else:
        import tempfile; os.chdir(tempfile.mkdtemp())
        for t in tasks:
            out.append(fn(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--scan-s", action="store_true")
    ap.add_argument("--scan-tilt", action="store_true")
    ap.add_argument("--workers", "-w", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=10e-6)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    os.chdir(_BASE)

    if args.test:
        import tempfile; os.chdir(tempfile.mkdtemp())
        print("=== AKIL SAĞLIĞI: σ² ölçeklemesi (antisim ve sim desen) ===")
        for w, lab in ((0.0, "antisim"), (1.0, "sim")):
            for sigma in (10e-6, 20e-6):
                rng = np.random.default_rng(42)
                dx = gen_pattern(w, sigma, rng); dy = gen_pattern(w, sigma, rng)
                t = time.time()
                f, resid = measure_false_edm(dx, dy, np.zeros(NQ))
                print(f"  {lab} σ={sigma*1e6:.0f}μm: |f|={abs(f):.3e} rad/s, "
                      f"Ŝ={0.5*(s_hat(dx)+s_hat(dy)):+.2f}, CO artık={resid*1e6:.2f}μm "
                      f"({time.time()-t:.0f}s)")
        return

    if args.scan_s:
        ws = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
        tasks = [(w, sd, args.sigma) for w in ws for sd in range(args.seeds)]
        print(f"Ŝ kampanyası: {len(tasks)} koşum (σ={args.sigma*1e6:.0f}μm)...")
        res = _run(tasks, _w_s, args.workers)
        arr = np.array([(sh, f) for sh, f, _ in res])
        np.save(os.path.join(_DIR, "fedm_vs_shat.npy"), arr)
        _plot_s(arr)

    if args.scan_tilt:
        thetas = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
        tasks = [(th, sd, args.sigma) for th in thetas for sd in range(args.seeds)]
        print(f"tilt kampanyası: {len(tasks)} koşum...")
        res = _run(tasks, _w_tilt, args.workers)
        arr = np.array([(th, f) for th, f, _ in res])
        np.save(os.path.join(_DIR, "fedm_vs_tilt.npy"), arr)
        _plot_tilt(arr)


def _setup_plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "savefig.dpi": 600,
                         "savefig.bbox": "tight"})
    return plt


def _plot_s(arr):
    plt = _setup_plot()
    fig, ax = plt.subplots(figsize=(3.375, 2.6))
    ax.scatter(arr[:, 0], arr[:, 1] * 1e9, s=10, color="C0", alpha=0.7)
    ax.set_xlabel(r"$\hat S$  (−1: antisimetrik,  +1: simetrik)")
    ax.set_ylabel(r"$|dS_y/dt|$  [nrad/s]")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(_DIR, "fig8_fedm_vs_shat.png"))
    print("Kaydedildi: drift_monitor/fig8_fedm_vs_shat.png")


def _plot_tilt(arr):
    plt = _setup_plot()
    fig, ax = plt.subplots(figsize=(3.375, 2.6))
    ax.scatter(arr[:, 0] * 1e3, arr[:, 1] * 1e9, s=10, color="C1", alpha=0.7)
    ax.set_xlabel(r"quad tilt RMS  $\theta$  [mrad]")
    ax.set_ylabel(r"$|dS_y/dt|$  [nrad/s]")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(_DIR, "fig9_fedm_vs_tilt.png"))
    print("Kaydedildi: drift_monitor/fig9_fedm_vs_tilt.png")


if __name__ == "__main__":
    main()

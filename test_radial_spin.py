#!/usr/bin/env python3
"""test_radial_spin.py — Başlangıç polarizasyonu radyal olursa ne olur?

SORU: Spin-sürülü trimde başlangıç polarizasyonu boylamsal yerine radyal
olsa sistemin hassasiyeti artar mı?

TEORİK BEKLENTİ: Hem gerçek EDM dönüşü (radyal E alanı → Ω radyal) hem
de sahte EDM dönüşü (dikey yörünge sapması × quad gradyanı → B_x radyal
→ Ω radyal) RADYAL eksen etrafındadır. Thomas-BMT: dS/dt = Ω × S.
Ω = Ω_x x̂ için dS_y/dt = −Ω_x·S_z. Boylamsal spin (S_z=1) → tam sinyal;
radyal spin (S_x=1, S_z=0) → BİRİNCİ MERTEBEDE SIFIR. Yani radyal
polarizasyon hassasiyeti artırmaz; her iki sinyale birden kör olur.
Radyal demet ancak boylamsal-B sistematiği (Ω_z) için bir "boş kanal"
(null channel) olabilir.

TEST: 4 koşum (CO=False, t2=1ms):
  1. boylamsal spin + 100μm rastgele kaçıklık (seed=321)  → referans f
  2. radyal spin    + aynı kaçıklık                       → beklenti ~0
  3. boylamsal spin + kaçıklık yok + EDMSwitch=1          → saf EDM sinyali
  4. radyal spin    + kaçıklık yok + EDMSwitch=1          → beklenti ~0

Çıktı: test_radial_spin.json, test_radial_spin.png
"""

import json
import os
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

PATTERN_RMS  = 1e-4
PATTERN_SEED = 321
T2           = 1e-3
RETURN_STEPS = 6000


def _worker(task):
    """Tek spin koşumu: (etiket, dy, spin_kipi, edm_switch) → dSy/dt ve Sy(t)."""
    label, dy_list, spin_mode, edm_switch = task

    import os, sys, json, tempfile, shutil
    import numpy as np
    sys.path.insert(0, BASE)
    tmp = tempfile.mkdtemp(prefix=f"rspin_{os.getpid()}_")
    os.chdir(tmp)

    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open(os.path.join(BASE, "params.json")) as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    fields.poincare_quad_index = 0.0
    fields.EDMSwitch = float(edm_switch)
    dt = float(config.get("dt", 1e-11))

    # Spin başlangıcı: yerel (sx, sy, sz)
    if spin_mode == "radial":
        y0[6:9] = [1.0, 0.0, 0.0]
    # "longitudinal": setup_fields varsayılanı [0, 0, direction]

    dy = np.asarray(dy_list, dtype=float)

    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null)
    try:
        _, poin, poin_t = integrate_particle(
            y0, 0.0, T2, dt, fields=fields,
            return_steps=RETURN_STEPS, quad_dy=dy)
        t_arr = np.asarray(poin_t, float)
        sy = np.asarray(poin[:, 7], float)
        slope = float(np.polyfit(t_arr, sy, 1)[0])
        # Sy(t) örneklemesi figür için (60 nokta)
        idx = np.linspace(0, len(sy) - 1, 60).astype(int)
        trace = {"t": t_arr[idx].tolist(), "sy": sy[idx].tolist()}
    finally:
        os.dup2(fd, 1); os.close(fd)
        os.chdir(BASE)
        shutil.rmtree(tmp, ignore_errors=True)

    return label, {"dSy_dt": slope, "trace": trace}


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    n_q = 2 * int(config.get("nFODO", 24))

    P = np.random.default_rng(PATTERN_SEED).standard_normal(n_q) * PATTERN_RMS
    Z = np.zeros(n_q)

    tasks = [
        ("long_pat", P.tolist(), "longitudinal", 0.0),
        ("rad_pat",  P.tolist(), "radial",       0.0),
        ("long_edm", Z.tolist(), "longitudinal", 1.0),
        ("rad_edm",  Z.tolist(), "radial",       1.0),
    ]
    ctx = mp.get_context("spawn")
    print(f"{len(tasks)} simülasyon ({mp.cpu_count()} işçi)...")
    with ctx.Pool(processes=min(mp.cpu_count(), len(tasks))) as pool:
        res = dict(pool.map(_worker, tasks))

    print(f"\n{'─'*64}")
    print(f"{'koşum':>10} {'spin':>13} {'kaynak':>10} {'dSy/dt [rad/s]':>16}")
    print(f"{'─'*64}")
    rows = [("long_pat", "boylamsal", "kaçıklık"),
            ("rad_pat",  "radyal",    "kaçıklık"),
            ("long_edm", "boylamsal", "EDM"),
            ("rad_edm",  "radyal",    "EDM")]
    for key, spin, src in rows:
        print(f"{key:>10} {spin:>13} {src:>10} {res[key]['dSy_dt']:>+16.4e}")

    r_pat = abs(res["rad_pat"]["dSy_dt"] / res["long_pat"]["dSy_dt"])
    r_edm = (abs(res["rad_edm"]["dSy_dt"] / res["long_edm"]["dSy_dt"])
             if res["long_edm"]["dSy_dt"] != 0 else float("nan"))
    print(f"\nRadyal/boylamsal oran — kaçıklık: {r_pat:.2e}, EDM: {r_edm:.2e}")

    out = {"_aciklama": "Radyal vs boylamsal başlangıç polarizasyonu "
                        "(CO=False, t2=1ms, desen seed=321 100μm RMS)",
           "sonuclar": {k: {"dSy_dt": v["dSy_dt"]} for k, v in res.items()},
           "oran_kaciklik": r_pat, "oran_edm": r_edm}
    with open("test_radial_spin.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("Kaydedildi: test_radial_spin.json")

    # Figür: Sy(t) izleri
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (k1, k2, title) in zip(axes, [
            ("long_pat", "rad_pat", "Kaçıklık (100μm RMS, seed=321)"),
            ("long_edm", "rad_edm", "Saf EDM (η=1.88×10⁻¹⁵)")]):
        for key, lbl, c in ((k1, "boylamsal", "C0"), (k2, "radyal", "C3")):
            tr = res[key]["trace"]
            ax.plot(np.array(tr["t"]) * 1e3, tr["sy"], c, label=lbl)
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("S_y")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_radial_spin.png", dpi=150)
    print("Figür: test_radial_spin.png")
    print(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()

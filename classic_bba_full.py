#!/usr/bin/env python3
"""
classic_bba_full.py — T5 UÇTAN-UCA C++ SONUCU (separation_bba_testleri.md §5.1)

Tam zincir, tamamı gerçek dinamikle (C++):
  1) 47 quad × 2 düzlem klasik BBA: quad_dG modülasyonu + komşu-düzeltici
     bump + 2-noktalı tarama → null → merkez kestirimi (CO-oturtmalı okuma).
     (cell-0 QF quad_dG okumaz [tuzak #8] → sim'de quad 0 kaçıklığı 0 alınır
     ve ölçülmez; gerçek makinede o quad başka yolla modüle edilir.)
  2) Düzeltme: kestirilen merkezler düşülür (düzeltici ≡ merkez kayması).
  3) Kalan sahte-EDM, spin izleyicisiyle DOĞRUDAN ölçülür
     (kmod_drivers.fast_est.fast_measure; 4D-CO + model-fit) — formül
     kestirimi DEĞİL.

Gürültüsüz koşum → sistematik taban; istatistik (BPM gürültüsü) analitik
çalışmadan ayrıca rapor edilir (yol gösterici). Çıktı: JSONL artımlı +
kmod_drivers/paper_runs_results.json [bba_full].

Kullanım: python3 classic_bba_full.py -w 4 [--scan 150e-6]
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import R_perquad, sym_anti_projectors, NQ, G_NOM

EPS = 0.02
JSONL = "/tmp/kmod_recover/bba_full.jsonl"
os.makedirs("/tmp/kmod_recover", exist_ok=True)


def _orbit_xy(task):
    """CO-oturtmalı tek koşum: (dx, dy, dG) → (x_COD, y_COD) 48-BPM [m]."""
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.path.insert(0, os.path.join(BASE, "berry_data"))
    from false_edm_mode_scan import setup_fields, _make_state
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    from build_response_matrix import read_cod_quads
    dx, dy, dG, tag = task
    dx = np.array(dx); dy = np.array(dy); dG = np.array(dG)
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    zeros = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, zeros,
                             T_rev, n_turns=14, n_iter=2)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 999.0
    for fname in ("cod_data.txt", "rf.txt"):
        if os.path.exists(fname):
            os.remove(fname)
    integrate_particle(launch, 0.0, float(cfg.get("t2", 1e-3)),
                       float(cfg["dt"]), fields=fields, return_steps=10,
                       quad_dy=dy, quad_dx=dx, quad_tilt=zeros, quad_dG=dG)
    x, y = read_cod_quads(NQ // 2)
    rec = {"tag": tag, "x": [float(v) for v in x], "y": [float(v) for v in y],
           "resid": float(resid), "t": time.time()}
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--scan", type=float, default=150e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    m_dx = rng.normal(0.0, 10e-6, NQ); m_dx[0] = 0.0   # quad 0: tuzak #8
    m_dy = rng.normal(0.0, 10e-6, NQ); m_dy[0] = 0.0

    # Bump ölçekleme/knob seçimi için DÜZLEM-ÖZEL analitik tepki (yalnız kılavuz;
    # sonuç değil). x-düzlemi için dikey R'yi kullanmak yanlış düzeltici + yanlış
    # genlik verir (dx hatası ~4× şişer) — bu yüzden her düzlem kendi R'siyle.
    from analytic_kmod import (compute_twiss_at_quads, signed_KL,
                               build_R_analytic, compute_Brho)
    with open(os.path.join(BASE, "params.json")) as _f:
        _cfg = json.load(_f)
    _nF = int(_cfg["nFODO"]); _Lq = float(_cfg["quadLen"])
    _Brho = compute_Brho(_cfg)

    def _build_R0(plane):
        beta, phi, Q = compute_twiss_at_quads(_cfg, G_NOM, plane)
        KL = signed_KL(_nF, abs(G_NOM) / _Brho, _Lq, plane)
        return build_R_analytic(beta, phi, Q, KL)

    R0_model = {"y": _build_R0("y"), "x": _build_R0("x")}

    quads = list(range(1, NQ))
    # görevler: her quad × düzlem × {−δ, +δ} × {mod aç, kapa}
    tasks = []
    meta = []
    for i in quads:
        cands = [(i - 1) % NQ, (i + 1) % NQ, (i - 2) % NQ, (i + 2) % NQ]
        for plane in ("y", "x"):
            Rp = R0_model[plane]
            j = max(cands, key=lambda jj: abs(Rp[i, jj]))   # düzlem-özel düzeltici
            for sgn in (-1.0, +1.0):
                dxx, dyy = m_dx.copy(), m_dy.copy()
                shift = sgn * args.scan / Rp[i, j]          # düzlem-özel genlik
                if plane == "y":
                    dyy[j] += shift
                else:
                    dxx[j] += shift
                dG_on = np.zeros(NQ); dG_on[i] = EPS
                for on in (1, 0):
                    tasks.append((list(dxx), list(dyy),
                                  list(dG_on if on else np.zeros(NQ)),
                                  f"q{i}_{plane}_{'p' if sgn>0 else 'm'}_{on}"))
            meta.append((i, plane, j))

    print(f"BBA taraması: {len(tasks)} CO-oturtmalı koşum ({args.workers} işçi)")
    import build_response_matrix as brm
    with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
        recs = list(pool.map(_orbit_xy, tasks))
    by = {r["tag"]: r for r in recs}

    est_x = np.zeros(NQ); est_y = np.zeros(NQ)
    for (i, plane, j) in meta:
        pts = []
        for sgn_lbl in ("m", "p"):
            on = np.array(by[f"q{i}_{plane}_{sgn_lbl}_1"][plane])
            off = np.array(by[f"q{i}_{plane}_{sgn_lbl}_0"][plane])
            A = on - off
            pts.append((off[i], A))
        (x1, A1), (x2, A2) = pts
        s = (A2 - A1) / (x2 - x1)                    # eğim vektörü (48 BPM)
        a = A1 - s * x1
        w = s ** 2
        x_star = -np.sum(w * (a / np.where(s == 0, 1, s))) / np.sum(w)
        if plane == "y":
            est_y[i] = x_star
        else:
            est_x[i] = x_star

    err_x = est_x - m_dx; err_y = est_y - m_dy
    err_x[0] = err_y[0] = 0.0                        # quad 0 ölçülmedi (m=0)
    P_sym, P_anti = sym_anti_projectors()
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    print("\n=== BBA MERKEZ-KESTİRİM HATALARI (C++, gürültüsüz) ===")
    for lbl, e in (("dx", err_x), ("dy", err_y)):
        print(f"  {lbl}: RMS={rms(e)*1e6:.3f} μm  sym={rms(P_sym@e)*1e6:.3f}  "
              f"anti={rms(P_anti@e)*1e6:.3f} μm")

    # ── kalan sahte-EDM: SPİN İZLEYİCİSİYLE doğrudan (formül değil) ──
    sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
    from fast_est import fast_measure
    print("\nSpin ölçümleri (ham + düzeltilmiş)...")
    f_raw, r1 = fast_measure(m_dx, m_dy)
    f_cor, r2 = fast_measure(m_dx - est_x, m_dy - est_y)
    print("\n=== UÇTAN-UCA SONUÇ (C++ spin izleyici) ===")
    print(f"  ham sahte-EDM        |f| = {f_raw:.3e} rad/s ({f_raw/1e-9:.0f}× hedef)")
    print(f"  BBA-düzeltme sonrası |f| = {f_cor:.3e} rad/s ({f_cor/1e-9:.2f}× hedef)")
    print(f"  bastırma = {f_raw/max(f_cor,1e-30):.0f}×   (CO artıkları {r1*1e6:.2f}/{r2*1e6:.2f} μm)")

    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data["bba_full"] = {
        "err_x_rms": rms(err_x), "err_y_rms": rms(err_y),
        "err_x_sym": rms(P_sym @ err_x), "err_y_sym": rms(P_sym @ err_y),
        "err_x_anti": rms(P_anti @ err_x), "err_y_anti": rms(P_anti @ err_y),
        "f_raw": f_raw, "f_corrected": f_cor, "seed": args.seed,
        "est_x": [float(v) for v in est_x], "est_y": [float(v) for v in est_y],
        "m_dx": [float(v) for v in m_dx], "m_dy": [float(v) for v in m_dy]}
    json.dump(data, open(path, "w"), indent=1)
    print(f"  kaydedildi → [bba_full]   [toplam {time.time()-t0:.0f} s]")


if __name__ == "__main__":
    main()

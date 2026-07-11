#!/usr/bin/env python3
"""diag_bbeat.py — β-beat altında BBA null'u neden kayıyor? (tanı, C++)

Birkaç quad için, β-beat GÖMÜLÜ, dikey düzlemde İNCE tarama (9 nokta) yap;
tepki A[i] vs demet konumu x'i incele: (a) lineer mi, (b) null gerçek merkeze
mi oturuyor, (c) 2-nokta kestirimi ince-taramadan farklı mı (→ kabalık artefaktı).
Aynı deseni β-beat=0 ile de koş (kontrol). Sonuç fizik/artefakt ayrımı.

Kullanım: python3 diag_bbeat.py -w 4
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import NQ, G_NOM
from analytic_kmod import compute_twiss_at_quads, signed_KL, build_R_analytic, compute_Brho

EPS = 0.02


def _orbit(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.path.insert(0, os.path.join(BASE, "berry_data"))
    from false_edm_mode_scan import setup_fields, _make_state
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    from build_response_matrix import read_cod_quads
    dx, dy, dG, tag = task
    dx = np.array(dx); dy = np.array(dy); dG = np.array(dG); zeros = np.zeros(NQ)
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, zeros,
                             T_rev, n_turns=14, n_iter=2)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 999.0
    for fn in ("cod_data.txt", "rf.txt"):
        if os.path.exists(fn):
            os.remove(fn)
    integrate_particle(launch, 0.0, float(cfg.get("t2", 1e-3)), float(cfg["dt"]),
                       fields=fields, return_steps=10, quad_dy=dy,
                       quad_dx=dx, quad_tilt=zeros, quad_dG=dG)
    x, y = read_cod_quads(NQ // 2)
    return {"tag": tag, "x": [float(v) for v in x], "y": [float(v) for v in y]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--quads", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--npts", type=int, default=9)
    ap.add_argument("--half", type=float, default=200e-6)
    ap.add_argument("--orbit-scale", type=float, default=1.0,
                    help="ölçülen quad HARİÇ diğer kaçıklıkları ölçekle (yörünge→nefes proxy)")
    ap.add_argument("--plane", choices=["x", "y"], default="y")
    args = ap.parse_args()
    t0 = time.time()
    PL = args.plane

    rng = np.random.default_rng(0)
    # ölçülen düzlemin kaçıklık deseni; iki düzlem de aynı seed-akışıyla üretilir
    m_dx = rng.normal(0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0, 10e-6, NQ); m_dy[0] = 0.0
    m_meas = m_dx if PL == "x" else m_dy
    bbeat = rng.normal(0, 0.01, NQ); bbeat[0] = 0.0

    cfg = json.load(open("params.json"))
    beta, phi, Q = compute_twiss_at_quads(cfg, G_NOM, PL)
    Brho = compute_Brho(cfg)
    KL = signed_KL(NQ // 2, abs(G_NOM) / Brho, 0.4, PL)
    R0 = build_R_analytic(beta, phi, Q, KL)

    deltas = np.linspace(-args.half, args.half, args.npts)
    tasks = []
    for i in args.quads:
        j = max([(i-1) % NQ, (i+1) % NQ, (i-2) % NQ, (i+2) % NQ],
                key=lambda jj: abs(R0[i, jj]))
        # ölçülen quad i'nin merkezi SABİT; diğerleri ×orbit_scale (nefes kaynağı)
        base_x = (m_dx * args.orbit_scale).copy()
        base_y = (m_dy * args.orbit_scale).copy()
        base_x[i] = m_dx[i]; base_y[i] = m_dy[i]
        for bb_lbl, bb in (("bb0", np.zeros(NQ)), ("bb1", bbeat)):
            for kd, d in enumerate(deltas):
                dx = base_x.copy(); dy = base_y.copy()
                if PL == "x":
                    dx[j] += d / R0[i, j]
                else:
                    dy[j] += d / R0[i, j]
                for on in (1, 0):
                    dG = bb.copy()
                    if on:
                        dG[i] += EPS
                    tasks.append((list(dx), list(dy), list(dG),
                                  f"q{i}_{bb_lbl}_{kd}_{on}"))

    print(f"{len(tasks)} koşum ({args.workers} işçi), düzlem={PL}...")
    import build_response_matrix as brm
    with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
        recs = list(pool.map(_orbit, tasks))
    by = {r["tag"]: r for r in recs}

    for i in args.quads:
        print(f"\n=== quad {i}  (düzlem {PL}; gerçek merkez m = {m_meas[i]*1e6:+.3f} μm) ===")
        for bb_lbl in ("bb0", "bb1"):
            xs, As = [], []
            for kd in range(args.npts):
                on = np.array(by[f"q{i}_{bb_lbl}_{kd}_1"][PL])
                off = np.array(by[f"q{i}_{bb_lbl}_{kd}_0"][PL])
                xs.append(off[i]); As.append(on - off)
            xs = np.array(xs); As = np.array(As)
            # en güçlü BPM'de lineerlik + null
            b = int(np.argmax(np.abs(As[args.npts//2])))
            Ab = As[:, b]
            p1 = np.polyfit(xs, Ab, 1)                     # lineer fit
            p2 = np.polyfit(xs, Ab, 2)                     # kuadratik (lineerlik testi)
            null_lin = -p1[1] / p1[0]
            resid_lin = np.std(Ab - np.polyval(p1, xs))
            # tüm-BPM ağırlıklı null (full koddaki gibi), ince + 2-nokta
            def wnull(idx):
                s = (As[idx[1]] - As[idx[0]]) / (xs[idx[1]] - xs[idx[0]])
                a = As[idx[0]] - s * xs[idx[0]]
                return -np.sum(s * a) / np.sum(s ** 2)
            null_2pt = wnull((0, -1))
            # ince: ardışık çiftlerin ortalaması yerine tam LSQ
            S = np.array([np.polyfit(xs, As[:, bb2], 1)[0] for bb2 in range(NQ)])
            Aint = np.array([np.polyfit(xs, As[:, bb2], 1)[1] for bb2 in range(NQ)])
            null_fine = -np.sum(S * Aint) / np.sum(S ** 2)
            print(f"  [{bb_lbl}] en güçlü BPM: null={null_lin*1e6:+.3f} μm  "
                  f"lin-artık={resid_lin/abs(p1[0])/args.half*100:.2f}% "
                  f"kuad/lin={abs(p2[0]*args.half/p1[0])*100:.1f}%")
            print(f"         ağırlıklı null: 2-nokta={null_2pt*1e6:+.3f} μm  "
                  f"ince(9pt-LSQ)={null_fine*1e6:+.3f} μm  "
                  f"→ hata 2pt={((null_2pt-m_meas[i]))*1e6:+.3f}, "
                  f"ince={((null_fine-m_meas[i]))*1e6:+.3f} μm")
    print(f"\n[toplam {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

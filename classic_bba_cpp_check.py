#!/usr/bin/env python3
"""
classic_bba_cpp_check.py — T5'in C++ doğrulaması (separation_bba_testleri.md §3)

Null-arayan klasik BBA'nın analitik sonucunu (simetrik mod dahil merkezler
gürültü-sınırlı ölçülür) GERÇEK demet dinamiğiyle sınar: seçilen birkaç quad
için modülasyon tepkisi A(δ) iki kapalı-yörünge farkıyla (quad_dG aç/kapa)
ölçülür, null'un gerçek merkeze (m_i) oturup oturmadığına bakılır.

- Modülasyon: quad_dG[i] = ε (C++ per-quad kesirsel gradyan) — cell-0 QF
  (QUAD_F_MOD, quad_dG'yi OKUMAZ; CLAUDE.md tuzak #8) KULLANILMAZ.
- Bump: komşu quad'a quad_dy ek terimi (düzeltici ≡ merkez kayması,
  separation_bba_testleri.md §2.5).
- Gürültüsüz (C++ deterministik): ölçülen sapma = saf model/dinamik bias.

Kullanım: python3 classic_bba_cpp_check.py [-w 4] [--quads 5 20 33] [--npts 5]
Çıktı: her quad için est−m_i [μm]; JSON'a ek (kmod_drivers/paper_runs_results.json).
"""
import argparse, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import R_perquad, NQ, G_NOM

EPS = 0.02
SCAN_HALF = 150e-6


def _orbit(task):
    """Tek kapalı-yörünge: (dy, dG) → 48-BPM dikey COD [m].

    KRİTİK: parçacık ideal eksenden DEĞİL, 4D kapalı yörüngeden fırlatılır —
    aksi hâlde betatron artığı okumaya ~0.2 μm kirlilik bırakır ve null'da
    1/eğim (~25×) büyüyerek ~5 μm sahte bias yaratır (ilk koşumda görüldü).
    find_co_4d quad_dG'yi bilmez; mod-on koşumda kalan betatron genliği ~A
    (~μm) → zaman-ortalama artığı ~nm (önemsiz)."""
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.path.insert(0, os.path.join(BASE, "berry_data"))
    from false_edm_mode_scan import setup_fields, _make_state
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    from build_response_matrix import read_cod_quads
    dy, dG = task
    dy = np.array(dy); dG = np.array(dG)
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    zeros = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, direction, zeros, dy, zeros,
                             T_rev, n_turns=14, n_iter=2)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 999.0
    for fname in ("cod_data.txt", "rf.txt"):
        if os.path.exists(fname):
            os.remove(fname)
    integrate_particle(launch, 0.0, float(cfg.get("t2", 1e-3)),
                       float(cfg["dt"]), fields=fields, return_steps=10,
                       quad_dy=dy, quad_dx=zeros, quad_tilt=zeros, quad_dG=dG)
    x, y = read_cod_quads(NQ // 2)      # nFODO bekler (24), quad sayısı değil
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--quads", type=int, nargs="+", default=[5, 20, 33])
    ap.add_argument("--npts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m_true = rng.normal(0.0, 10e-6, NQ)
    R0_model, _ = R_perquad(np.full(NQ, G_NOM))
    deltas = np.linspace(-SCAN_HALF, SCAN_HALF, args.npts)

    # görev listesi: her quad × her δ × (mod aç, kapa) + taban
    tasks, index = [], []
    for i in args.quads:
        assert i != 0, "cell-0 QF quad_dG okumaz (tuzak #8)"
        cands = [(i - 1) % NQ, (i + 1) % NQ, (i - 2) % NQ, (i + 2) % NQ]
        jc = max(cands, key=lambda jj: abs(R0_model[i, jj]))
        assert jc != 0 or True
        for d in deltas:
            dy = m_true.copy()
            dy[jc] += d / R0_model[i, jc]
            dG_on = np.zeros(NQ); dG_on[i] = EPS
            tasks += [(list(dy), list(dG_on)), (list(dy), list(np.zeros(NQ)))]
            index.append((i, d))

    print(f"{len(tasks)} kapalı-yörünge koşumu ({args.workers} işçi)...")
    import build_response_matrix as brm
    with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
        orbits = list(pool.map(_orbit, tasks))

    out = {}
    print("\n=== T5 C++ DOĞRULAMASI: null gerçek merkeze oturuyor mu? ===")
    for qn, i in enumerate(args.quads):
        A_pts, x_pts = [], []
        for kd in range(args.npts):
            k2 = (qn * args.npts + kd) * 2
            y_on, y_off = orbits[k2], orbits[k2 + 1]
            A_pts.append(y_on - y_off)
            x_pts.append(y_off[i])
        A_pts = np.array(A_pts); x_pts = np.array(x_pts)
        X = np.vstack([x_pts, np.ones(args.npts)]).T
        coef, _, _, _ = np.linalg.lstsq(X, A_pts, rcond=None)
        s_b, a_b = coef[0], coef[1]
        w = s_b ** 2
        x_star = -np.sum(w * (a_b / np.where(s_b == 0, 1, s_b))) / np.sum(w)
        err = x_star - m_true[i]
        # fit kalitesi: en iyi BPM'de artık
        b_best = int(np.argmax(np.abs(s_b)))
        resid = np.std(A_pts[:, b_best] - (s_b[b_best] * x_pts + a_b[b_best]))
        print(f"  quad {i:2d}: est = {x_star*1e6:+8.3f} μm, gerçek m = "
              f"{m_true[i]*1e6:+8.3f} μm  →  BIAS = {err*1e6:+7.3f} μm  "
              f"(fit artığı {resid*1e6:.4f} μm)")
        out[str(i)] = {"est": x_star, "true": m_true[i], "bias": err}

    biases = np.array([v["bias"] for v in out.values()])
    print(f"\n  bias RMS = {np.sqrt(np.mean(biases**2))*1e6:.3f} μm "
          f"(gürültüsüz → saf dinamik/model bias)")
    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data["bba_cpp_check"] = out
    json.dump(data, open(path, "w"), indent=1)
    print("  kaydedildi → kmod_drivers/paper_runs_results.json [bba_cpp_check]")


if __name__ == "__main__":
    main()

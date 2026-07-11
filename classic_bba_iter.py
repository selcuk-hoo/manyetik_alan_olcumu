#!/usr/bin/env python3
"""
classic_bba_iter.py — İTERATİF BBA: β-beat/nefes çöküşünün çözümü testi
(separation_bba_testleri.md §5.2)

Teşhis: β-beat %1 altında BBA çöküyordu çünkü modülasyon, DİĞER kaçık quad'ların
kurduğu büyük (0.37 mm) yörüngeyi "nefes"le yeniden taşır ve β-beat bu nefesi
feed-down'dan saptırıp null'u ~2 μm kaydırır. İzole test kesin gösterdi: yörünge
sıfırlanınca bias 0.02 μm'e düşer. → Çözüm: yörüngeyi küçült.

İterasyon (standart BBA pratiği "önce orbit düzelt"in doğal hâli):
  geçiş k: kalan-kaçıklık makinesinde (offsets = residual) BBA → merkez kestir
           → residual −= est  (merkezlere "taşı" → yörünge küçülür)
  → her geçişte yörünge ~geometrik oranla küçülür → nefes-bias küçülür.
Her geçiş sonunda kalan sahte-EDM SPİN İZLEYİCİYLE ölçülür (merkez-RMS değil).

Gürültüsüz + ofsetsiz (ikisi ayrı doğrulandı: ofset iptal, gürültü 1/√N) —
burada tek soru β-beat/nefes'in iterasyonla çözülüp çözülmediği.

Kullanım: python3 classic_bba_iter.py -w 4 --passes 3 --bbeat 0.01
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import R_perquad, sym_anti_projectors, NQ, G_NOM
from classic_bba_full import _orbit_xy, estimate, EPS
from analytic_kmod import compute_twiss_at_quads, signed_KL, build_R_analytic, compute_Brho


def build_meta(R0_model, scan):
    """Her (quad,düzlem) için düzlem-özel düzeltici j ve bump shift."""
    meta = []
    for i in range(1, NQ):
        cands = [(i-1) % NQ, (i+1) % NQ, (i-2) % NQ, (i+2) % NQ]
        for plane in ("y", "x"):
            Rp = R0_model[plane]
            j = max(cands, key=lambda jj: abs(Rp[i, jj]))
            shift = scan / Rp[i, j]
            meta.append((i, plane, j, shift))
    return meta


def run_pass(base_x, base_y, bbeat, meta, pool):
    """Kalan-kaçıklık makinesinde bir BBA geçişi → est_x, est_y, KALİTE (C++).

    Kalite = BPM'ler arası null tutarlılığı. Kötü quad'da (dikiş/dispersiyon)
    farklı BPM'ler farklı sıfır-geçişi görür; o quad'ın düzeltmesi kısılır.
    q = 1/(1 + (BPM-null yayılımı / scan)²) ∈ (0,1].
    """
    tasks = []
    for (i, plane, j, shift) in meta:
        for sgn in (-1.0, +1.0):
            dxx, dyy = base_x.copy(), base_y.copy()
            if plane == "y":
                dyy[j] += sgn * shift
            else:
                dxx[j] += sgn * shift
            dG_on = bbeat.copy(); dG_on[i] += EPS
            dG_off = bbeat.copy()
            for on in (1, 0):
                tasks.append((list(dxx), list(dyy),
                              list(dG_on if on else dG_off),
                              f"q{i}_{plane}_{'p' if sgn>0 else 'm'}_{on}"))
    recs = list(pool.map(_orbit_xy, tasks))
    by = {r["tag"]: r for r in recs}

    est_x = np.zeros(NQ); est_y = np.zeros(NQ)
    qual_x = np.ones(NQ); qual_y = np.ones(NQ)
    for (i, plane, j, shift) in meta:
        r = {}
        for sgn_lbl in ("m", "p"):
            for on in (1, 0):
                r[(sgn_lbl, on)] = np.array(by[f"q{i}_{plane}_{sgn_lbl}_{on}"][plane])
        x1 = r[("m", 0)][i]; x2 = r[("p", 0)][i]
        A1 = r[("m", 1)] - r[("m", 0)]; A2 = r[("p", 1)] - r[("p", 0)]
        s = (A2 - A1) / (x2 - x1); a = A1 - s * x1
        x_star = -np.sum(s * a) / np.sum(s ** 2)          # ağırlıklı ortak null
        # per-BPM null'lar (yeterince güçlü eğimli BPM'ler) → yayılım → kalite
        strong = np.abs(s) > 0.1 * np.max(np.abs(s))
        nulls = -a[strong] / s[strong]
        w = (s[strong] ** 2)
        spread = np.sqrt(np.sum(w * (nulls - x_star) ** 2) / np.sum(w))
        span = abs(x2 - x1)
        q = 1.0 / (1.0 + (spread / (0.15 * span)) ** 2)   # yayılım ~%15 scan'de q=0.5
        if plane == "y":
            est_y[i] = x_star; qual_y[i] = q
        else:
            est_x[i] = x_star; qual_x[i] = q
    return est_x, est_y, qual_x, qual_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--scan", type=float, default=150e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bbeat", type=float, default=0.01)
    ap.add_argument("--relax", type=float, default=0.5,
                    help="under-relaxation kazancı: res -= relax·kalite·est")
    ap.add_argument("--resume", action="store_true",
                    help="kayıtlı residual'dan devam et (restart-güvenli)")
    args = ap.parse_args()
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    m_dx = rng.normal(0.0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0.0, 10e-6, NQ); m_dy[0] = 0.0
    bbeat = rng.normal(0.0, args.bbeat, NQ) if args.bbeat > 0 else np.zeros(NQ)
    bbeat[0] = 0.0

    cfg = json.load(open(os.path.join(BASE, "params.json")))
    nF = int(cfg["nFODO"]); Lq = float(cfg["quadLen"]); Brho = compute_Brho(cfg)
    def R0(plane):
        beta, phi, Q = compute_twiss_at_quads(cfg, G_NOM, plane)
        return build_R_analytic(beta, phi, Q, signed_KL(nF, abs(G_NOM)/Brho, Lq, plane))
    R0_model = {"y": R0("y"), "x": R0("x")}
    meta = build_meta(R0_model, args.scan)
    P_sym, _ = sym_anti_projectors()
    rms = lambda a: float(np.sqrt(np.mean(a**2)))

    # yörünge büyüklüğü göstergesi (analitik, kılavuz): ||R·residual||
    Ry = R0_model["y"]
    def orbit_rms(rx, ry):
        return rms(Ry @ ry) * 1e6  # μm, dikey (kaba gösterge)

    sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
    from fast_est import fast_measure

    print(f"=== İTERATİF BBA (β-beat {args.bbeat*100:.1f}%, {args.passes} geçiş) ===")
    f_raw, _ = fast_measure(m_dx, m_dy)
    print(f"  ham sahte-EDM = {f_raw:.3e} ({f_raw/1e-9:.0f}× hedef); "
          f"başlangıç yörünge ~{orbit_rms(m_dx,m_dy):.0f} μm (dikey)")

    # restart-güvenli durum dosyası (residual arrayleri) → --resume ile devam
    STATE = "/tmp/kmod_recover/bba_iter_state.json"
    os.makedirs("/tmp/kmod_recover", exist_ok=True)
    hist = []
    res_x, res_y = m_dx.copy(), m_dy.copy()
    p0 = 1
    if args.resume and os.path.exists(STATE):
        st = json.load(open(STATE))
        res_x = np.array(st["res_x"]); res_y = np.array(st["res_y"])
        hist = st["hist"]; p0 = st["next_pass"]
        print(f"  [resume] geçiş {p0}'dan devam (kayıtlı {len(hist)} geçiş)")

    with ProcessPoolExecutor(args.workers, initializer=__import__('build_response_matrix')._worker_init) as pool:
        for p in range(p0, args.passes + 1):
            est_x, est_y, qx, qy = run_pass(res_x, res_y, bbeat, meta, pool)
            # under-relaxation × kalite: kötü/tutarsız quad'ın düzeltmesi kısılır
            res_x = res_x - args.relax * qx * est_x
            res_y = res_y - args.relax * qy * est_y
            res_x[0] = res_y[0] = 0.0
            f_p, resid_co = fast_measure(res_x, res_y)
            print(f"  geçiş {p}: kalan sym dx {rms(P_sym@res_x)*1e6:.3f} / "
                  f"dy {rms(P_sym@res_y)*1e6:.3f} μm  yörünge ~{orbit_rms(res_x,res_y):.1f} μm"
                  f"  min-kalite x/y {qx.min():.2f}/{qy.min():.2f}"
                  f"  →  sahte-EDM {f_p:.3e} ({f_p/1e-9:.2f}× hedef)")
            hist.append({"pass": p, "f": f_p, "sym_dx": rms(P_sym@res_x),
                         "sym_dy": rms(P_sym@res_y), "orbit_um": orbit_rms(res_x, res_y),
                         "minqual_x": float(qx.min()), "minqual_y": float(qy.min())})
            # artımlı kayıt: sonuç + restart durumu
            pth = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
            dd = json.load(open(pth)) if os.path.exists(pth) else {}
            dd["bba_iter"] = {"bbeat": args.bbeat, "seed": args.seed, "relax": args.relax,
                              "f_raw": f_raw, "passes": hist}
            json.dump(dd, open(pth, "w"), indent=1)
            json.dump({"res_x": list(res_x), "res_y": list(res_y), "hist": hist,
                       "next_pass": p + 1}, open(STATE, "w"))

    print(f"\n  ham {f_raw/1e-9:.0f}× → {args.passes} geçiş sonrası "
          f"{hist[-1]['f']/1e-9:.2f}× hedef  (toplam bastırma {f_raw/max(hist[-1]['f'],1e-30):.0f}×)")
    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data["bba_iter"] = {"bbeat": args.bbeat, "seed": args.seed, "f_raw": f_raw,
                        "passes": hist}
    json.dump(data, open(path, "w"), indent=1)
    print(f"  kaydedildi → [bba_iter]   [toplam {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

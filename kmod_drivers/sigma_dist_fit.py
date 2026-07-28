#!/usr/bin/env python3
"""sigma_dist_fit.py — SAHTE-EDM DAĞILIMI ve y=kx² FİT'İNİN KUYRUK-BAĞIMLILIĞI.

Hipotez (kullanıcı gözlemi, Omarov Fig. 9a): sahte-EDM dağılımı σ-başına AĞIR
KUYRUKLU (f ∝ x_CO·y_CO = yörünge², yörünge de rezonansa yakın ağır-kuyruklu).
Omarov'un `y=kx²` en-küçük-kareler fiti büyük-y noktalarına aşırı ağırlık verir →
birkaç EKSTREM seed fitin k'sını yukarı çeker; MEDYAN çok daha aşağıda kalır.
Sonuç: bizim medyan/RMS'imiz (~1.6e-6) ile Omarov fiti (1.5e-5) arasındaki ~10×,
FİZİK farkı değil, ÖZET-İSTATİSTİK farkı (kuyruk-fiti vs medyan) olabilir.

Bu betik Omarov'un analizini TAKLİT eder:
  • σ ∈ {2.5, 5, 10} μm, her σ'da N bağımsız seed (varsayılan 40).
  • Her σ için: medyan, ortalama, RMS, p10/p50/p90, max, min.
  • y=kx² LS fiti (i) TÜM bulut (Omarov-tarzı), (ii) yalnız medyanlar,
    (iii) yalnız max'lar üzerinden — üç k karşılaştırılır.
  • Her k'nın σ=10μm'de öngördüğü f ile Omarov 1.5e-5 kıyaslanır.

BEKLENEN SONUÇ: k_all (bulut) ≫ k_medyan; k_all·10² ~ Omarov 1.5e-5 iken
k_medyan·10² ~ bizim 1e-6. Bu, ~10×'in kuyruk-fiti artefaktı olduğunu gösterir.

Kullanım (lokal — ağır spin koşumları; container SIGTERM verir):
    python3 kmod_drivers/sigma_dist_fit.py -n 40 -w 12
    python3 kmod_drivers/sigma_dist_fit.py -n 80 -w 12 --sigmas 2.5,5,10,20
UYARI: N×len(σ) tam sahte-EDM ölçümü. N=40, 3σ = 120 koşum (~çekirdek başına
birkaç dk). Paralelliği çıktı zaman-damgalarından doğrula.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
os.chdir(BASE)

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sigma_dist_fit_results.json")
OMAROV_10UM = 1.5e-5   # rad/s, Fig. 9a fit değeri @ σ=10 μm


def measure(args):
    """Bağımsız seed: (σ_index, σ, seed) → |f|. C++ printf (fd 1) susturulur."""
    sidx, sg, seed = args
    from paper_runs import fast_measure_g, NQ
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass
    # σ-başına BAĞIMSIZ desen (Omarov "random rms σ" gibi): seed'i σ ile ayrıştır
    rng = np.random.default_rng(7000 + sidx * 1000 + seed)
    dx = rng.normal(0, sg, NQ)
    dy = rng.normal(0, sg, NQ)
    devnull = os.open(os.devnull, os.O_WRONLY); saved = os.dup(1); os.dup2(devnull, 1)
    try:
        f, _ = fast_measure_g(dx, dy, g_scale=1.0)
    finally:
        os.dup2(saved, 1); os.close(saved); os.close(devnull)
    return sg, float(f)


def k_fit(sigmas_um, fvals):
    """y = k·x² en-küçük-kareler (LS): k = Σ(f·σ²)/Σ(σ⁴)."""
    x2 = np.asarray(sigmas_um, float) ** 2
    y = np.asarray(fvals, float)
    return float(np.sum(y * x2) / np.sum(x2 ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--seeds", type=int, default=40)
    ap.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--sigmas", type=str, default="2.5,5,10")
    a = ap.parse_args()
    sig_um = [float(x) for x in a.sigmas.split(",")]
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass
    print(f"{len(sig_um)}×{a.seeds} = {len(sig_um)*a.seeds} koşum, {a.workers} çekirdek. "
          f"Omarov Fig 9a: {OMAROV_10UM:.1e} @ 10μm.\n", flush=True)

    tasks = [(i, s * 1e-6, sd) for i, s in enumerate(sig_um) for sd in range(a.seeds)]
    vals = {s: [] for s in sig_um}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(measure, t) for t in tasks]
        for fut in as_completed(futs):
            sg, f = fut.result()
            vals[sg * 1e6].append(f)
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"  [{done:3d}/{len(tasks)}] t={time.time()-t0:.0f}s", flush=True)

    # ── per-σ istatistik ──
    print("\n=== σ-BAŞINA DAĞILIM ===")
    stat = {}
    for s in sig_um:
        a_ = np.array(vals[s])
        st = dict(median=float(np.median(a_)), mean=float(a_.mean()),
                  rms=float(np.sqrt((a_**2).mean())), p90=float(np.percentile(a_, 90)),
                  mx=float(a_.max()), mn=float(a_.min()), n=len(a_))
        stat[s] = st
        print(f"  σ={s:4.1f}μm (n={st['n']}): medyan={st['median']:.2e} "
              f"ort={st['mean']:.2e} RMS={st['rms']:.2e} p90={st['p90']:.2e} "
              f"max={st['mx']:.2e}  (max/medyan={st['mx']/st['median']:.1f}×)")

    # ── y=kx² fit: bulut vs medyan vs max ──
    cloud_x = [s for s in sig_um for _ in vals[s]]
    cloud_y = [f for s in sig_um for f in vals[s]]
    k_all = k_fit(cloud_x, cloud_y)
    k_med = k_fit(sig_um, [stat[s]["median"] for s in sig_um])
    k_max = k_fit(sig_um, [stat[s]["mx"] for s in sig_um])
    print("\n=== y=kx² FİT (Omarov-tarzı) ===")
    for name, k in (("TÜM bulut (Omarov gibi)", k_all),
                    ("yalnız MEDYAN", k_med), ("yalnız MAX", k_max)):
        f10 = k * 100.0
        print(f"  {name:26s}: k={k:.3e}  → f(10μm)={f10:.2e}  "
              f"(Omarov/bu = {OMAROV_10UM/f10:.1f}×)")
    print("\n  YORUM: k_all ≫ k_medyan ise → Omarov fiti kuyruk-domine; ~10× fark")
    print("  fizik değil özet-istatistik. k_all·10² ~ 1.5e-5 iken medyan ~1e-6 beklenir.")

    out = {"seeds": a.seeds, "sigmas_um": sig_um, "per_sigma": stat,
           "k_all": k_all, "k_median": k_med, "k_max": k_max,
           "omarov_10um": OMAROV_10UM, "f_all": {str(s): vals[s] for s in sig_um}}
    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[kaydedildi: {RESULT}]  [toplam {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

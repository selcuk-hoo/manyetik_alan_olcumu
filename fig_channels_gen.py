#!/usr/bin/env python3
"""fig_channels_gen.py — Fig-3 (bilineer kanal ayrışımı) VERİ KAMPANYASI.

Sahte-EDM, yatay ve dikey kaçıklık desenlerinin bilineer fonksiyonelidir:
    f = f_ss + f_sa + f_as + f_aa
Her desen simetrik (s) + antisimetrik (a) parçaya ayrılır; dört kanal, dört
AYRI izleyici koşumuyla (her düzlemde bir projekte parça, başka kusur yok)
DOĞRUDAN ölçülür — ağırlıklar W_ij hiç bilinmez, izleyici onları içerir.

  f_ss = f(P_sym v_x, P_sym v_y)   f_sa = f(P_sym v_x, P_anti v_y)
  f_as = f(P_anti v_x, P_sym v_y)  f_aa = f(P_anti v_x, P_anti v_y)

Estimatör: berry_data/false_edm_4d.measure_false_edm (4D-CO + model-fit; makale
§II reçetesi). Bilineerlik VARSAYILMAZ, ÖLÇÜLÜR: dört imzalı kanalın toplamı
tam-desen sahte-EDM'ine %0.1 içinde eşit olmalı.

Çıktı: kmod_drivers/paper_runs_results.json ["channels"] (fig_crsep gibi;
make_orbit_figures.fig_channels() bunu okuyup çizer).

Kullanım:
  python3 fig_channels_gen.py -w 7                 # (a) 3-seed @10μm + (b) σ-tarama
  python3 fig_channels_gen.py -w 7 --seeds 3
"""
import os, sys, json, time, argparse
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "berry_data"))
import build_response_matrix as brm
from make_orbit_figures import sym_anti_projectors, NQ
from false_edm_4d import measure_false_edm

TARGET = 1e-9  # 1 nrad/s ↔ 1e-29 e·cm hedef
RESULTS = os.path.join(_DIR, "kmod_drivers", "paper_runs_results.json")

# kanal → (yatay parça, dikey parça); 'f' = tam (projeksiyonsuz)
CMAP = {"ss": ("s", "s"), "sa": ("s", "a"),
        "as": ("a", "s"), "aa": ("a", "a"), "full": ("f", "f")}


def _pattern(seed, sigma):
    """Temsili desen (classic_bba_iter ile AYNI konvansiyon: seed 0 → f_raw=356×):
    default_rng(seed) → dx sonra dy; index 0 (cell-0 QF) sıfır (tuzak #8)."""
    rng = np.random.default_rng(seed)
    dx = rng.normal(0.0, sigma, NQ); dx[0] = 0.0
    dy = rng.normal(0.0, sigma, NQ); dy[0] = 0.0
    return dx, dy


def _w_channel(task):
    """Tek kanal ölçümü (paralel worker)."""
    seed, sigma, channel = task
    Ps, Pa = sym_anti_projectors()
    dx, dy = _pattern(seed, sigma)
    parts_x = {"s": Ps @ dx, "a": Pa @ dx, "f": dx}
    parts_y = {"s": Ps @ dy, "a": Pa @ dy, "f": dy}
    ci, cj = CMAP[channel]
    f, _ = measure_false_edm(parts_x[ci], parts_y[cj], np.zeros(NQ))
    return seed, float(sigma), channel, float(f / TARGET)   # × hedef (imzalı)


def _run(tasks, workers):
    out = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=brm._worker_init) as pool:
            for r in pool.map(_w_channel, tasks):
                out.append(r)
    else:
        import tempfile; os.chdir(tempfile.mkdtemp())
        for t in tasks:
            out.append(_w_channel(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=3, help="(a) paneli seed sayısı")
    ap.add_argument("--sigma-a", type=float, default=10e-6, help="(a) paneli σ")
    ap.add_argument("--sigmas-b", type=str, default="2.5e-6,5e-6,10e-6",
                    help="(b) paneli σ taraması (seed 0)")
    args = ap.parse_args()
    os.chdir(_DIR)

    chans = ["ss", "sa", "as", "aa"]
    seeds = list(range(args.seeds))
    sig_a = args.sigma_a
    sigmas_b = [float(s) for s in args.sigmas_b.split(",")]

    # (a): tüm seed × (4 kanal + full) @ σ_a
    tasks_a = [(sd, sig_a, ch) for sd in seeds for ch in chans + ["full"]]
    # (b): seed 0 × 4 kanal × σ taraması (σ_a zaten (a)'da seed 0 için var)
    tasks_b = [(0, sg, ch) for sg in sigmas_b if abs(sg - sig_a) > 1e-12
               for ch in chans]

    print(f"=== Fig-3 kanal kampanyası: {len(tasks_a)+len(tasks_b)} koşum "
          f"(workers={args.workers}) ===", flush=True)
    t0 = time.time()
    res = _run(tasks_a + tasks_b, args.workers)
    print(f"  bitti ({time.time()-t0:.0f}s)", flush=True)

    # topla
    A = {ch: [None]*len(seeds) for ch in chans + ["full"]}
    B = {ch: {} for ch in chans}
    for sd, sg, ch, fv in res:
        if abs(sg - sig_a) < 1e-12 and ch in A:
            A[ch][sd] = fv
        if sd == 0 and ch in B:
            B[ch][f"{sg:.3e}"] = fv
    # (b): σ_a seed-0 değerini (a)'dan al
    for ch in chans:
        B[ch][f"{sig_a:.3e}"] = A[ch][0]

    # bilineerlik kontrolü: full / Σ(4 kanal)
    sums = [A["ss"][i] + A["sa"][i] + A["as"][i] + A["aa"][i] for i in seeds]
    ratios = [A["full"][i] / s if abs(s) > 1e-30 else float("nan")
              for i, s in zip(seeds, sums)]
    print("  bilineerlik  full/Σ:", [f"{r:.4f}" for r in ratios])
    for ch in chans:
        print(f"    {ch}: {[f'{A[ch][i]:+.1f}×' for i in seeds]}  "
              f"(ort |{np.mean([abs(A[ch][i]) for i in seeds]):.1f}×|)")

    data = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    data["channels"] = {
        "sigma_a": sig_a, "seeds": seeds, "channels": chans,
        "a": {ch: A[ch] for ch in chans + ["full"]},
        "b_sigmas": sorted(set(sigmas_b) | {sig_a}),
        "b": {ch: [B[ch][f"{sg:.3e}"] for sg in sorted(set(sigmas_b) | {sig_a})]
              for ch in chans},
        "bilinearity_full_over_sum": ratios,
    }
    json.dump(data, open(RESULTS, "w"), indent=1)
    print(f"\nKaydedildi → {RESULTS} [\"channels\"]", flush=True)
    print("Figürü üret:  python3 -c \"from make_orbit_figures import "
          "fig_channels; fig_channels()\"", flush=True)


if __name__ == "__main__":
    main()

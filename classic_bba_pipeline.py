#!/usr/bin/env python3
"""classic_bba_pipeline.py — ÇOK-SEED TAM BORU HATTI (pahalı, C++).

Her seed için: tam BBA iterasyonu (ölçülen C++ R ile yönlendirme, 3 geçiş) →
son bir SVD yörünge düzeltmesi (antisimetrik/yörünge-görünür artığı temizler) →
nihai sahte-EDM (C++ spin). Amaç: boru hattı tabanının SEED DAĞILIMI
(tek-seed bilineer saçılımı yerine sağlam bir sayı).

Boru hattı fiziği (makale §II.D, eq:channels):
  BBA → simetrik kanal f_ss'i azaltır (matris-tersleme yapamaz)
  yörünge düzeltme → antisimetrik kanalları (f_aa, f_as, f_sa) siler
  BİRLİKTE → hedefe; tek başına hiçbiri.

RESTART-GÜVENLİ: per-seed durum + geçiş-içi orbit ara-kayıt (classic_bba_iter'den)
+ tamamlanan seed results JSON'da → atlanır.  OTOMATİK devam: aynı komutu
tekrar çalıştır, biten seed'ler atlanır (--resume bayrağı YOK).

Kullanım: python3 classic_bba_pipeline.py -w 4 --seeds 0 1 2 3 4
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import sym_anti_projectors, NQ, G_NOM
from classic_bba_iter import build_meta, run_pass, _pass_ckpt
import build_response_matrix as brm

RESULTS = os.path.join(BASE, "kmod_drivers", "pipeline_multiseed.json")
Ps, Pa = sym_anti_projectors()
rms = lambda a: float(np.sqrt(np.mean(a**2)))


def orbit_correct(res, R, rcond):
    """Kesik-SVD yörünge düzeltmesi: yörünge-görünür (antisim) modları kaldır."""
    return res - np.linalg.pinv(R, rcond=rcond) @ (R @ res)


def load_results():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"seeds": {}}


def save_results(d):
    json.dump(d, open(RESULTS, "w"), indent=1)


def run_bba_seed(seed, R0_model, meta, bbeat_amp, passes, relax, scan, workers):
    """Tek seed için tam BBA iterasyonu → (res_x, res_y). Restart-güvenli."""
    rkey = f"pipe_s{seed}"
    STATE = f"/tmp/kmod_recover/{rkey}_state.json"
    os.makedirs("/tmp/kmod_recover", exist_ok=True)

    # seed 0: mevcut cpp koşumunun artığını yeniden kullan (10 saat tasarruf)
    seed0_state = "/tmp/kmod_recover/bba_iter_cpp_state.json"
    if seed == 0 and not os.path.exists(STATE) and os.path.exists(seed0_state):
        st = json.load(open(seed0_state))
        if st.get("next_pass", 0) > passes:
            print(f"  [seed 0] mevcut cpp artığı yeniden kullanılıyor", flush=True)
            return np.array(st["res_x"]), np.array(st["res_y"])

    rng = np.random.default_rng(seed)
    m_dx = rng.normal(0.0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0.0, 10e-6, NQ); m_dy[0] = 0.0
    bbeat = rng.normal(0.0, bbeat_amp, NQ); bbeat[0] = 0.0

    res_x, res_y = m_dx.copy(), m_dy.copy(); p0 = 1
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        res_x = np.array(st["res_x"]); res_y = np.array(st["res_y"]); p0 = st["next_pass"]
        print(f"  [seed {seed}] geçiş {p0}'dan devam", flush=True)
    else:
        import glob as _glob
        for f in _glob.glob(f"/tmp/kmod_recover/{rkey}_orbits_p*.jsonl"):
            os.remove(f)

    with ProcessPoolExecutor(workers, initializer=brm._worker_init) as pool:
        for p in range(p0, passes + 1):
            est_x, est_y, qx, qy = run_pass(res_x, res_y, bbeat, meta, pool, p, rkey)
            res_x = res_x - relax * qx * est_x
            res_y = res_y - relax * qy * est_y
            res_x[0] = res_y[0] = 0.0
            json.dump({"res_x": list(res_x), "res_y": list(res_y), "next_pass": p + 1},
                      open(STATE, "w"))
            if os.path.exists(_pass_ckpt(p, rkey)):
                os.remove(_pass_ckpt(p, rkey))
            print(f"  [seed {seed}] geçiş {p} bitti "
                  f"(sym dx {rms(Ps@res_x)*1e6:.2f}/dy {rms(Ps@res_y)*1e6:.2f} μm)", flush=True)
    return res_x, res_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--bbeat", type=float, default=0.01)
    ap.add_argument("--relax", type=float, default=0.5)
    ap.add_argument("--scan", type=float, default=150e-6)
    args = ap.parse_args()
    t0 = time.time()

    Rx = np.load(os.path.join(BASE, "R_dx_1.npy"))
    Ry = np.load(os.path.join(BASE, "R_dy_1.npy"))
    R0_model = {"y": Ry, "x": Rx}
    meta = build_meta(R0_model, args.scan)

    sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
    from fast_est import fast_measure

    data = load_results()
    print(f"=== ÇOK-SEED BORU HATTI (seeds {args.seeds}, ölçülen R) ===", flush=True)

    for seed in args.seeds:
        if str(seed) in data["seeds"] and data["seeds"][str(seed)].get("done"):
            r = data["seeds"][str(seed)]
            print(f"[seed {seed}] zaten tamam: BBA {r['f_bba']/1e-9:.1f}× → "
                  f"OC {r['f_oc_best']/1e-9:.2f}× (atlanıyor)", flush=True)
            continue

        print(f"\n[seed {seed}] BBA iterasyonu başlıyor...", flush=True)
        res_x, res_y = run_bba_seed(seed, R0_model, meta, args.bbeat,
                                    args.passes, args.relax, args.scan, args.workers)

        # BBA sonrası (yörünge düzeltmesi öncesi)
        f_bba, _ = fast_measure(res_x, res_y)
        # yörünge düzeltmesi (rcond taraması) + nihai f
        oc = []
        for rc in (0.05, 0.02, 0.01):
            cx = orbit_correct(res_x, Rx, rc); cy = orbit_correct(res_y, Ry, rc)
            f, _ = fast_measure(cx, cy)
            oc.append({"rcond": rc, "f": f,
                       "sym_dx": rms(Ps@cx), "sym_dy": rms(Ps@cy),
                       "anti_dx": rms(Pa@cx), "anti_dy": rms(Pa@cy)})
            print(f"  [seed {seed}] OC rcond={rc}: f={f/1e-9:.2f}×", flush=True)
        f_oc_best = min(o["f"] for o in oc)

        data["seeds"][str(seed)] = {
            "done": True, "seed": seed,
            "sym_dx_res": rms(Ps@res_x), "sym_dy_res": rms(Ps@res_y),
            "anti_dx_res": rms(Pa@res_x), "anti_dy_res": rms(Pa@res_y),
            "f_bba": f_bba, "oc": oc, "f_oc_best": f_oc_best}
        save_results(data)
        print(f"[seed {seed}] BİTTİ: BBA {f_bba/1e-9:.1f}× → OC-en-iyi "
              f"{f_oc_best/1e-9:.2f}×  [{time.time()-t0:.0f}s]", flush=True)

    # özet dağılım
    done = [data["seeds"][s] for s in data["seeds"] if data["seeds"][s].get("done")]
    if done:
        fb = np.array([d["f_bba"] for d in done]) / 1e-9
        fo = np.array([d["f_oc_best"] for d in done]) / 1e-9
        print(f"\n=== DAĞILIM ({len(done)} seed) ===", flush=True)
        print(f"  BBA sonrası:   {fb}  → medyan {np.median(fb):.1f}×", flush=True)
        print(f"  BBA+OC:        {fo}  → medyan {np.median(fo):.2f}×  "
              f"(min {fo.min():.2f}, max {fo.max():.2f})", flush=True)
    print(f"kaydedildi → {RESULTS}  [toplam {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""cwccw_telafi.py — CW/CCW beam-reversal-tek sahte-EDM artığı, ENSEMBLE.

Gerçek EDM beam-reversal-TEK (odd); CW/CCW farkında hayatta kalır:
    C = (f_CW - f_CCW)/2                       [Omarov Eq. C1]
Geometrik-faz sahte-EDM'in de beam-reversal-TEK bir ARTIĞI vardır (CW/CCW
demetler biraz farklı kapalı yörünge sürer) — bu C'de kalır. İŞARETLİ slope
şart (|f_CW|−|f_CCW| YANLIŞ; fast_measure(signed=True)).

Hesaplananlar (N seed, temsili desen default_rng(seed), 10μm):
  - C_raw : ham kaçıklığın CW/CCW artığı (antisim domine → yüzlerce×)
  - C_oc  : yörünge düzeltmesi (kesik-SVD, rcond) SONRASI artık (§tab:chain 2.satır)
  - pure-sim / pure-anti : σ=10μm saf desenlerin artığı (§polarity 14×/491×)

Makaledeki tek-seed "62×" (orbit-corr+CW/CCW) ve "1.6×" (BBA+OC+CW/CCW; --bba)
sayılarını ENSEMBLE'a çevirir.

Kullanım:
  python3 kmod_drivers/cwccw_telafi.py -w 5 --nseed 5
  python3 kmod_drivers/cwccw_telafi.py -w 5 --nseed 5 --bba   # + BBA yolu (pahalı)
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

JSONL = "/tmp/kmod_recover/cwccw_telafi.jsonl"    # artımlı (restart-güvenli)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
os.chdir(BASE)
import build_response_matrix as brm
from make_orbit_figures import sym_anti_projectors, NQ
from classic_bba_pipeline import orbit_correct, build_meta, run_bba_seed
from fast_est import fast_measure

TARGET = 1e-9
RCOND = 0.01
Rx = np.load(os.path.join(BASE, "R_dx_1.npy"))
Ry = np.load(os.path.join(BASE, "R_dy_1.npy"))


def _pattern(seed, sigma=10e-6):
    rng = np.random.default_rng(seed)
    dx = rng.normal(0, sigma, NQ); dx[0] = 0.0
    dy = rng.normal(0, sigma, NQ); dy[0] = 0.0
    return dx, dy


def _C(dx, dy):
    """Beam-reversal-tek artık (f_CW - f_CCW)/2, İŞARETLİ."""
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True)
    return 0.5 * (fcw - fccw)


def _task(t):
    import tempfile; os.chdir(tempfile.mkdtemp())
    kind, seed = t
    if kind == "raw":
        dx, dy = _pattern(seed);  return kind, seed, _C(dx, dy)
    if kind == "oc":
        dx, dy = _pattern(seed)
        cx = orbit_correct(dx, Rx, RCOND); cy = orbit_correct(dy, Ry, RCOND)
        return kind, seed, _C(cx, cy)
    if kind in ("sym", "anti"):
        Ps, Pa = sym_anti_projectors(); P = Ps if kind == "sym" else Pa
        dx, dy = _pattern(seed)
        return kind, seed, _C(P @ dx, P @ dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=5)
    ap.add_argument("--nseed", type=int, default=5)
    ap.add_argument("--bba", action="store_true",
                    help="§VII: her seed BBA+OC → CW/CCW artığı (PAHALI, ~saatler)")
    ap.add_argument("--passes", type=int, default=3, help="--bba modu geçiş sayısı")
    args = ap.parse_args()

    seeds = list(range(args.nseed))

    # ── §VII: BBA + son OC + CW/CCW artığı (ensemble) ──────────────────
    if args.bba:
        R0_model = {"y": Ry, "x": Rx}
        meta = build_meta(R0_model, 150e-6)
        OUTB = os.path.join(BASE, "kmod_drivers", "cwccw_bba_out.json")
        out = json.load(open(OUTB)) if os.path.exists(OUTB) else {}
        for s in seeds:
            if str(s) in out:
                print(f"  seed {s}: atlandı ({out[str(s)]:.2f}×)", flush=True); continue
            rx, ry = run_bba_seed(s, R0_model, meta, 0.01, args.passes, 0.5,
                                  150e-6, args.workers)
            cx = orbit_correct(rx, Rx, RCOND); cy = orbit_correct(ry, Ry, RCOND)
            C = _C(cx, cy)
            out[str(s)] = abs(C) / TARGET
            json.dump(out, open(OUTB, "w"), indent=1)
            print(f"  seed {s}: BBA+OC+CW/CCW |C| = {abs(C)/TARGET:.2f}×", flush=True)
        a = np.array([out[str(s)] for s in seeds])
        print(f"\n=== §VII BBA+OC+CW/CCW ({args.nseed} seed) ===")
        print(f"  medyan {np.median(a):.2f}×  [{a.min():.2f}, {a.max():.2f}]")
        print(f"Kaydedildi: {OUTB}")
        return
    tasks = [("raw", s) for s in seeds] + [("oc", s) for s in seeds] \
        + [("sym", s) for s in seeds] + [("anti", s) for s in seeds]
    # resume: tamamlanan (kind,seed) atla
    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    res = {"raw": {}, "oc": {}, "sym": {}, "anti": {}}
    if os.path.exists(JSONL):
        for ln in open(JSONL):
            try:
                r = json.loads(ln); res[r["kind"]][r["seed"]] = r["C"]
            except Exception:
                pass
    todo = [t for t in tasks if t[1] not in res[t[0]]]
    print(f"=== CW/CCW artık ENSEMBLE: {len(todo)}/{len(tasks)} koşum "
          f"(İŞARETLİ; {len(tasks)-len(todo)} atlandı) ===", flush=True)
    t0 = time.time(); done = 0
    if todo:
        with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
            for kind, seed, C in pool.map(_task, todo):
                res[kind][seed] = C
                with open(JSONL, "a") as fh:
                    fh.write(json.dumps({"kind": kind, "seed": seed, "C": C}) + "\n")
                done += 1
                print(f"  [{done}/{len(todo)}] {kind} seed {seed}: "
                      f"|C|={abs(C)/TARGET:.1f}×  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  bitti ({time.time()-t0:.0f}s)", flush=True)

    def stats(d, lbl):
        a = np.abs(np.array([d[s] for s in seeds])) / TARGET
        g = np.exp(np.mean(np.log(a + 1e-30)))
        print(f"  {lbl:26s} medyan {np.median(a):7.1f}×  geomean {g:7.1f}×  "
              f"[{a.min():.1f}, {a.max():.1f}]")
        return dict(median=float(np.median(a)), geomean=float(g),
                    lo=float(a.min()), hi=float(a.max()),
                    per_seed=[float(x) for x in a])
    print(f"\n=== SONUÇ (|C|/hedef, {args.nseed} seed) ===")
    out = {}
    out["raw"]  = stats(res["raw"],  "ham (CW/CCW)")
    out["oc"]   = stats(res["oc"],   "orbit-corr + CW/CCW")
    out["sym"]  = stats(res["sym"],  "pure-simetrik")
    out["anti"] = stats(res["anti"], "pure-antisimetrik")
    out["rcond"] = RCOND; out["nseed"] = args.nseed
    json.dump(out, open(os.path.join(BASE, "kmod_drivers", "cwccw_telafi_out.json"), "w"),
              indent=1)
    print("\nKaydedildi: kmod_drivers/cwccw_telafi_out.json")
    print("  → makale: tab:chain 'orbit corr + CW/CCW' satırı = 'orbit-corr+CW/CCW' medyanı")


if __name__ == "__main__":
    main()

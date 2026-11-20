#!/usr/bin/env python3
"""ofis_v2_kosulari.py — Ofis bilgisayarı için ikinci tur koşular (makale revizyonu, Kasım 2026).

İş A 'hafta' : Haftalık senaryoda TEK DEMET (W1_E2, rijit BPM + doğru model) ile İKİ DEMET (W1_E4)
               karşılaştırmasını 5 desenden 25 desene çıkarır (seed 5–24). Gerekçe: makalenin "iki demet
               tek demete göre bir mertebe kazandırır" iddiası 15 değere (medyan 0.65 vs 0.05) dayanıyor.
               Aynı ölçüm: donmus_adim6.task (donmuş spin, kısa ayar 14 tur / 2 iter), saat 24/96/168.
               Çıktı: sonuclar/donmus_adim6_kisa.jsonl (mevcut dosyaya EKLER; seed 0–4 korunur).
İş B 'polar' : Kuadrupol polaritesini çevirmenin saptırıcı ε×φ sahte EDM'sine etkisi, 4 desenden 12
               desene (seed 4–11). saptirici_testleri.py'nin polar_defl (yalnız saptırıcı hataları) ve
               polar_tam (r = 1e-6/√h, 168 h trim-öncesi gerçek durum) testleri; burada normal polarite de
               aynı betikle ölçülür. Omarov birleşimi ½(odd₊ + odd₋).
               Çıktı: sonuclar/saptirici_testleri.jsonl (EKLER).
Kullanım:  python3 ofis_v2_kosulari.py hafta -w 8
           python3 ofis_v2_kosulari.py polar -w 8
           python3 ofis_v2_kosulari.py ozet
Her iki iş de kesilirse aynı komutla kaldığı yerden devam eder.
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)

HAFTA_SEEDS = list(range(5, 25))
HAFTA_SCEN = [("W1", "E2"), ("W1", "E4")]
POLAR_SEEDS = list(range(4, 12))


def hafta(workers):
    from concurrent.futures import ProcessPoolExecutor
    import donmus_adim6 as D6
    out = os.path.join(REV, "sonuclar", "donmus_adim6_kisa.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["world"], r["method"], r["hour"], r["direction"]) for r in map(json.loads, open(out))}
    T = [(s, w, m, h, d, 14, 2) for s in HAFTA_SEEDS for (w, m) in HAFTA_SCEN for h in (24, 96, 168)
         for d in (-1, 1) if (s, w, m, h, d) not in done]
    print(f"hafta: {len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(D6.task, T):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"seed {r['seed']} {r['world']}-{r['method']} {r['hour']} sa yön {r['direction']:+d}: "
                  f"f*={r['f_target']:.4g}", flush=True)


def polar(workers):
    from concurrent.futures import ProcessPoolExecutor
    import dinamik_saptirici as DS
    import saptirici_testleri as ST
    # 1) yörünge döngüsü anlık görüntüleri (deterministik; seed 0–3 aynen yeniden üretilir)
    DS.SEEDS = range(max(POLAR_SEEDS) + 1)
    DS.RATES = [1e-7, 1e-6, 1e-5]
    DS.lin()
    # 2) spin
    done = set()
    if os.path.exists(ST.OUT):
        done = {ST.key({k: v for k, v in r.items() if k not in ("f", "f_target", "w1", "delta_star")})
                for r in map(json.loads, open(ST.OUT))}
    T = []
    for s in POLAR_SEEDS:
        for d in (-1, 1):
            for test, flip in (("polar_defl", False), ("polar_defl", True), ("polar_tam", False), ("polar_tam", True)):
                t = dict(test=test, flip=flip, seed=s, direction=d)
                if ST.key(t) not in done:
                    T.append(t)
    print(f"polar: {len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(ST._task, T):
            open(ST.OUT, "a").write(json.dumps(r) + "\n")
            print(r["test"], "flip", r["flip"], "seed", r["seed"], "yön", r["direction"], f"f*={r['f_target']:.4g}",
                  flush=True)


def ozet():
    # A: hafta
    rows = [json.loads(l) for l in open(os.path.join(REV, "sonuclar", "donmus_adim6_kisa.jsonl"))]
    d = {}
    for r in rows:
        if r["hour"] > 0:
            d.setdefault((r["world"], r["method"], r["hour"], r["seed"]), {})[r["direction"]] = r["f"] / 1e-9
    for (w, m) in HAFTA_SCEN:
        c = np.array([abs(v[1] - v[-1]) / 2 for k, v in d.items() if k[:2] == (w, m) and len(v) == 2])
        if len(c):
            print(f"hafta {w}-{m}: n={len(c)} C medyan {np.median(c):.3g}  %84 {np.percentile(c, 84):.3g}  "
                  f"maks {c.max():.3g}  hedef-üstü {(c > 1).sum()}/{len(c)}")
    # B: polarite (normal polarite aynı betikten ya da dinamik_saptirici'den)
    import saptirici_testleri as ST
    import dinamik_saptirici as DS
    g = {}
    for r in map(json.loads, open(ST.OUT)):
        if r["test"].startswith("polar"):
            g.setdefault((r["test"], r["flip"], r["seed"]), {})[r["direction"]] = r["f"] / 1e-9
    if os.path.exists(DS.OUT_EDM):
        for r in map(json.loads, open(DS.OUT_EDM)):
            if r["rate"] == 1e-6 and r["hour"] == 168:
                g.setdefault(("polar_tam", False, r["seed"]), {})[r["direction"]] = r["f"] / 1e-9
    odd = {k: (v[1] - v[-1]) / 2 for k, v in g.items() if len(v) == 2}
    for test in ("polar_defl", "polar_tam"):
        ss = sorted({k[2] for k in odd if k[0] == test and (test, False, k[2]) in odd and (test, True, k[2]) in odd})
        if not ss:
            continue
        p = np.array([odd[(test, False, s)] for s in ss]); m = np.array([odd[(test, True, s)] for s in ss])
        comb = np.abs(p + m) / 2
        print(f"{test}: n={len(ss)}  |odd+| medyan {np.median(np.abs(p)):.3g}  |odd-| medyan {np.median(np.abs(m)):.3g}"
              f"  Omarov ½|odd++odd-| medyan {np.median(comb):.3g} maks {comb.max():.3g}"
              f"  korelasyon(odd+,odd-) {np.corrcoef(p, m)[0, 1]:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("is_", choices=["hafta", "polar", "ozet"])
    ap.add_argument("-w", type=int, default=8); a = ap.parse_args()
    {"hafta": lambda: hafta(a.w), "polar": lambda: polar(a.w), "ozet": ozet}[a.is_]()

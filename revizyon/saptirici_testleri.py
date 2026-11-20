#!/usr/bin/env python3
"""saptirici_testleri.py — Saptırıcı kararlılığı gereksinimi için dört ek test (dinamik_saptirici.py'nin devamı).

T1 'dakika' : trim aralığı 1 dk (saatlik yerine), r = 1e-6/√h. Beklenti (tahmin): C ∝ r²T → ~60× küçülür.
T2 'eps'/'roll': yalnız alan (ε) ya da yalnız roll (φ) sürüklenir, r = 1e-6/√h, saatlik trim. Darboğaz hangisi?
T3 'ortak'  : r = 1e-7/√h saatlik koşunun 168 h trim-öncesi durumuna büyük bir ORTAK ε (1e-5, 1e-4) eklenir.
              Donmuş spin enerjisine yeniden kilitlenince C değişmiyorsa ortak mod (ortak HV beslemesi)
              gerçekten zararsızdır; yalnız bağımsız kısım önemlidir.
T4 'polar'  : kuadrupol polaritesi çevrilir (config'te g0, g1, g2 işareti; CLAUDE.md tuzak 8). (a) yalnız
              saptırıcı hataları (kuadrupoller kusursuz), (b) r = 1e-6 168 h trim-öncesi gerçek durum.
              Omarov birleşimi: tek kanal ortalaması ½(odd₊ + odd₋). ε×φ terimi polariteyle işaret
              değiştiriyorsa söner.
Yörünge döngüsü dinamik_saptirici.run_one ile aynı (doğrusal, %0.2 gradyan hatalı gerçek makine, 14 nm,
TSVD 1e-4, ortak trim + enerji geri beslemesi = ortalamalar çıkarılır). Spin: C++ donmuş spin (14 tur, 2 iter).
Çıktı: sonuclar/saptirici_testleri.jsonl
Kullanım: python3 saptirici_testleri.py lin ; python3 saptirici_testleri.py edm -w 4 ; ... summ
"""
import os, sys, json, argparse, tempfile
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
sys.path.insert(0, REV)
import dinamik_saptirici as DS
NQ = 48
OUT_LIN = os.path.join(REV, "sonuclar", "saptirici_testleri_lin.json")
OUT = os.path.join(REV, "sonuclar", "saptirici_testleri.jsonl")
SEEDS = range(4)


def run_mode(S, seed, rate, trim_h, which="both", hours=168):
    """dinamik_saptirici.run_one'ın hızlandırılmış kopyası (sabit sözde-ters) + yalnız-ε / yalnız-φ seçeneği."""
    import adim2_deflector as A2
    rng = np.random.default_rng(70_000 + seed)
    P = {}
    for pl in ("x", "y"):
        A = S[pl]["S"]; nrm = np.linalg.norm(A, axis=0)
        P[pl] = (np.linalg.pinv(A / nrm, rcond=DS.RC) / nrm[:, None])
    dm = {pl: rng.normal(0, DS.DM0, NQ) for pl in ("x", "y")}
    de = {pl: np.zeros(NQ) for pl in ("x", "y")}
    on = {"x": which in ("both", "eps"), "y": which in ("both", "roll")}
    n = int(round(hours / trim_h)); pre = None
    for k in range(1, n + 1):
        for pl in ("x", "y"):
            dm[pl] += rng.normal(0, DS.Q_RATE * np.sqrt(trim_h), NQ)
            step = rng.normal(0, rate * np.sqrt(trim_h), NQ)
            if on[pl]:
                de[pl] += step
            de[pl] -= de[pl].mean()
        if k == n:
            pre = {pl: [dm[pl].tolist(), de[pl].tolist()] for pl in ("x", "y")}
        for pl in ("x", "y"):
            s = S[pl]
            y = s["Tq"] @ dm[pl] + s["Td"] @ de[pl] + rng.normal(0, DS.SIG_N, 2 * NQ)
            est = P[pl] @ y
            dm[pl] -= est[:NQ]; de[pl] -= est[NQ:]
    return pre


def lin():
    S = DS.mats(); out = {}
    for s in SEEDS:
        out[f"dakika_{s}"] = dict(test="dakika", seed=s, snap=run_mode(S, s, 1e-6, 1 / 60))
        out[f"eps_{s}"] = dict(test="eps", seed=s, snap=run_mode(S, s, 1e-6, 1.0, "eps"))
        out[f"roll_{s}"] = dict(test="roll", seed=s, snap=run_mode(S, s, 1e-6, 1.0, "roll"))
        print("seed", s, "tamam", flush=True)
    json.dump(out, open(OUT_LIN, "w"))


def _one_cfg(dx, dy, direction, delta, dG, flip, n_turns=14, n_iter=2, t2=3e-4):
    """sahte_edm_donmus._one ile aynı; flip=True → kuadrupol polaritesi çevrilir."""
    import sahte_edm_donmus as SD
    import integrator_rev
    import false_edm_4d, false_edm_mode_scan
    ms = SD.launcher(delta)
    false_edm_4d._make_state = ms; false_edm_mode_scan._make_state = ms
    from false_edm_mode_scan import setup_fields, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    cfg = json.load(open(os.path.join(BASE, "params.json"))); cfg["direction"] = float(direction)
    if flip:
        for k in ("g0", "g1", "g2"):
            cfg[k] = -float(cfg[k])
    fields, y0, beta0, R0_, p_mag, d = setup_fields(cfg)
    z = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, d, dx, dy, z, _T_rev(fields, beta0, R0_),
                             n_turns=n_turns, n_iter=n_iter, dG=dG)
    launch = ms(v_co, p_mag, d, [0.0, 0.0, d])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrator_rev.integrate_particle(launch, 0.0, t2, float(cfg["dt"]), fields=fields,
                                                    return_steps=4000, quad_dx=dx, quad_dy=dy,
                                                    quad_tilt=z, quad_dG=dG)
    t = np.asarray(pt, float)
    f = float(measure_dSy_dt_model(np.asarray(poin[:, 7], float), t))
    ang = np.unwrap(np.arctan2(np.asarray(poin[:, 6], float), np.asarray(poin[:, 8], float) * d))
    return f, float(np.polyfit(t, ang, 1)[0])


def measure(dx, dy, direction, dG, flip):
    """measure_frozen'ın polarite seçenekli kopyası (aynı enterpolasyon + üçüncü nokta kuralı)."""
    import sahte_edm_donmus as SD
    cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())
    try:
        f0, w0 = _one_cfg(dx, dy, direction, 0.0, dG, flip)
        d1 = -w0 / SD.S_W_DELTA
        f1, w1 = _one_cfg(dx, dy, direction, d1, dG, flip)
        fs = f0 + (f1 - f0) * (0.0 - w0) / (w1 - w0)
        ds = d1 * (0.0 - w0) / (w1 - w0)
        if abs(w1) > SD.W1_ESIK:
            fs, w2 = _one_cfg(dx, dy, direction, ds, dG, flip)
        return dict(f=float(fs), w0=w0, w1=w1, delta_star=float(ds))
    finally:
        os.chdir(cwd)


def tasks():
    lin_ = json.load(open(OUT_LIN))
    dyn = json.load(open(DS.OUT_LIN))["runs"]
    T = []
    for s in SEEDS:
        for d in (-1, 1):
            T.append(dict(test="dakika", seed=s, direction=d))
            T.append(dict(test="eps", seed=s, direction=d))
            T.append(dict(test="roll", seed=s, direction=d))
            for amp in (1e-5, 1e-4):
                T.append(dict(test="ortak", amp=amp, seed=s, direction=d))
            T.append(dict(test="polar_defl", flip=False, seed=s, direction=d))
            T.append(dict(test="polar_defl", flip=True, seed=s, direction=d))
            T.append(dict(test="polar_tam", flip=True, seed=s, direction=d))
    return T


def key(t):
    return json.dumps({k: t[k] for k in sorted(t)})


def _task(t):
    import integrator_rev
    from build_rtrue import dG_pattern
    dG = dG_pattern(0.002)
    s = t["seed"]; flip = t.get("flip", False)
    if t["test"] in ("dakika", "eps", "roll"):
        sn = json.load(open(OUT_LIN))[f"{t['test']}_{s}"]["snap"]
    elif t["test"] == "ortak":
        sn = json.load(open(DS.OUT_LIN))["runs"][f"1e-07_1_{s}"]["snaps"]["168"]
    else:
        sn = json.load(open(DS.OUT_LIN))["runs"][f"1e-06_1_{s}"]["snaps"]["168"]
    dx, dy = np.asarray(sn["x"][0]), np.asarray(sn["y"][0])
    eps, roll = np.asarray(sn["x"][1], float), np.asarray(sn["y"][1], float)
    if t["test"] == "ortak":
        eps = eps + t["amp"]
    if t["test"] == "polar_defl":
        dx = np.zeros(NQ); dy = np.zeros(NQ)
    integrator_rev.DEFL_DE = eps; integrator_rev.DEFL_ROLL = roll
    r = measure(dx, dy, t["direction"], dG, flip)
    return dict(t, f=r["f"], f_target=r["f"] / 1e-9, w1=r["w1"], delta_star=r["delta_star"])


def edm(workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(OUT):
        done = {key({k: v for k, v in r.items() if k not in ("f", "f_target", "w1", "delta_star")})
                for r in map(json.loads, open(OUT))}
    T = [t for t in tasks() if key(t) not in done]
    print(f"{len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_task, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(r["test"], r.get("amp", ""), r.get("flip", ""), "seed", r["seed"], "yön", r["direction"],
                  f"f*={r['f_target']:.4g}", flush=True)
    summ()


def summ():
    rows = [json.loads(l) for l in open(OUT)]
    g = {}
    for r in rows:
        k = (r["test"], r.get("amp"), r.get("flip"))
        g.setdefault(k, {}).setdefault(r["seed"], {})[r["direction"]] = r["f"] / 1e-9
    odd = {}
    for k, v in g.items():
        odd[k] = {s: (d[1] - d[-1]) / 2 for s, d in v.items() if len(d) == 2}
    for k in sorted(odd, key=str):
        c = np.abs(list(odd[k].values()))
        if len(c):
            print(f"{str(k):32s} n={len(c)} C medyan {np.median(c):.3g} maks {c.max():.3g}")
    # Omarov birleşimi: ½(odd₊ + odd₋)
    for base, plus in (("polar_defl", ("polar_defl", None, False)), ("polar_tam", None)):
        minus = odd.get((base, None, True), {})
        if base == "polar_defl":
            p = odd.get(plus, {})
        else:   # tam durum, normal polarite: dinamik_saptirici 168 h trim-öncesi (r=1e-6)
            p = {}
            for l in open(DS.OUT_EDM):
                r = json.loads(l)
                if r["rate"] == 1e-6 and r["hour"] == 168:
                    p.setdefault(r["seed"], {})[r["direction"]] = r["f"] / 1e-9
            p = {s: (d[1] - d[-1]) / 2 for s, d in p.items() if len(d) == 2}
        ss = sorted(set(p) & set(minus))
        if ss:
            comb = np.abs([(p[s] + minus[s]) / 2 for s in ss])
            print(f"{base}: odd+ {np.round([p[s] for s in ss], 3)}  odd- {np.round([minus[s] for s in ss], 3)}"
                  f"  Omarov ½(odd++odd-) medyan {np.median(comb):.3g} maks {comb.max():.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["lin", "edm", "summ"])
    ap.add_argument("-w", type=int, default=4); a = ap.parse_args()
    {"lin": lin, "edm": lambda: edm(a.w), "summ": summ}[a.mode]()

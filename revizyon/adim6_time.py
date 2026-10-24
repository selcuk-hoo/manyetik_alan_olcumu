#!/usr/bin/env python3
"""adim6_time.py — PRAB revizyonu Adım 6: zaman senaryosu (Hakem B).

Senaryo (1 hafta, adım Δt = 1 saat):
  t=0     : BBA kalibrasyonu → kuadrupol artığı m0 ~ N(0, σ_BBA) (iid), BPM–kuad farkı δ = 0.
            Statik BPM ofsetleri referans (golden) okumada sadeleşir.
  her adım: ATL yer hareketi g(s,t): uzayda periyodik rastgele yürüyüş, adım varyansı A·Δt·Δs
            (Shiltsev; A = 1e-5 μm²/(s·m) kötümser). Kuadrupoller zemini izler.
  Dünyalar:
   W1 (önerilen): BPM kuadrupole rijit bağlı; δ(t) her BPM'de bağımsız rastgele yürüyüş,
                  hız r_δ [μm/√gün]. Okuma r = R m_eff − (q + δ) ; q = kuadrupol fiziksel konumu,
                  m_eff = q − c (c: kuadrupol içi düzeltici, merkez kayması eşdeğeri).
   W0 (makale)  : BPM'ler ayrı ayaklarda ama AYNI zeminde (ATL'yi izler) + bağımsız ayak kayması
                  (hız r_δ). Okuma r = R m_eff − (g + δ).
  Kestirim (her adım, referans = son BBA): y = r − r_ref (+ düzeltici bilgisi), TSVD:
   W1-E4: iki demet, R − I modeli (BPM'in kuadrupolle hareket ettiğini bilir)
   W1-E2: tek demet, R − I modeli
   W0-E3: iki demet, makale modeli (BPM sabit sanır)
   none : izleme yok (yalnız periyodik BBA da yok)
  BBA tetikleme: monitör δ'yı göremez; beklenen σ_δ(t) = r_δ √(t − t_BBA) eşik σ_δ* = 0.3 μm'ye
   ulaşınca BBA yapılır (m_eff → yeni artık N(0,σ_BBA), δ → 0, referans yenilenir).
  Gürültü: σ_n = sqrt( σ_tur² / (f_dev·T_ort) + σ_taban² ), demet başına, her BPM/düzlem bağımsız.

Çıktılar:
  series : doğrusal zaman serisi (simetrik/antisimetrik artık rms), birçok seed
  edm    : seçili zamanlarda C++ sahte EDM (azaltılmış estimatör ayarı, fast_est varsayılanı)
  detect : en küçük tespit edilebilir simetrik sürüklenme (3σ) vs σ_tur, T_ort, σ_taban
  plot
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(REV)
for p in (REV, BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
import adim4_correlated as A4

OUTDIR = A4.OUTDIR
NQ = 48
I = np.eye(NQ)
TARGET = 1e-9
HOUR = 3600.0
F_REV = 1.0 / 3.7e-6                     # tur frekansı [Hz] (T_rev ≈ 3.7 μs)
P = dict(A=1e-5 * 1e-12,                 # ATL katsayısı [m²/(s·m)]  (1e-5 μm²/(s·m))
         sig_bba=0.3e-6,                 # BBA artığı [m] (tahmin parametresi)
         r_delta=0.2e-6,                 # δ kayma hızı [m/√gün]
         sig_delta_star=0.3e-6,          # BBA tetikleme eşiği [m]
         sig_turn=10e-6, T_avg=60.0, sig_floor=10e-9,
         hours=168, rc=3e-3)


def sig_n(sig_turn, T_avg, sig_floor):
    return float(np.sqrt(sig_turn ** 2 / (F_REV * T_avg) + sig_floor ** 2))


R = A4.load_R()


def pinv(M, rc):
    n = np.linalg.norm(M, axis=0)
    return np.linalg.pinv(M / n, rcond=rc) / n[:, None]


EST = {}
for pl in "xy":
    Rc, Rw = R[pl]["ccw"], R[pl]["cw"]
    EST[pl] = {"E4": pinv(np.vstack([Rc - I, Rw - I]), P["rc"]),
               "E2": pinv(Rc - I, P["rc"]),
               "E3": pinv(np.vstack([Rc, Rw]), P["rc"])}


def atl_step(rng, pl, dt):
    """Bir zaman adımında zemin artımı: periyodik rastgele yürüyüş (düzlem çıkarılmış), adım
    varyansı A·dt·Δs."""
    w = A4.gen("rw", rng)                          # adım varyansı Δs (birim a=1)
    w = A4.remove_plane(w, pl)
    return w * np.sqrt(P["A"] * dt)


def simulate(seed, world, method, record_hours=None):
    """Zaman serisi. Durum (düzlem başına): q kuadrupol fiziksel konumu, c kuadrupol içi düzeltici
    (merkez kayması eşdeğeri), g zemin, d = δ. Etkin kaçıklık m_eff = q − c.
    Referans: son BBA anındaki okuma ve düzeltici değeri."""
    rng = np.random.default_rng(500_000 + seed)    # çevresel dizi (zemin, δ, BBA) — senaryodan bağımsız
    rng_n = np.random.default_rng(900_000 + seed)  # BPM gürültüsü
    sn = sig_n(P["sig_turn"], P["T_avg"], P["sig_floor"])
    st = {pl: dict(q=rng.normal(0, P["sig_bba"], NQ), c=np.zeros(NQ), g=np.zeros(NQ), d=np.zeros(NQ))
          for pl in "xy"}

    def reading(pl):
        s = st[pl]
        meff = s["q"] - s["c"]
        bpm = (s["q"] + s["d"]) if world == "W1" else (s["g"] + s["d"])
        return (R[pl]["ccw"] @ meff - bpm + rng_n.normal(0, sn, NQ),
                R[pl]["cw"] @ meff - bpm + rng_n.normal(0, sn, NQ))

    ref = {pl: (reading(pl), st[pl]["c"].copy()) for pl in "xy"}
    t_bba = 0
    out = {"hours": [], "sym": [], "anti": [], "bba": [], "m_eff": {}}
    for h in range(P["hours"] + 1):
        if h > 0:
            for pl in "xy":
                dg = atl_step(rng, pl, HOUR)
                st[pl]["g"] += dg; st[pl]["q"] += dg
                st[pl]["d"] += rng.normal(0, P["r_delta"] * np.sqrt(1 / 24), NQ)
            bba_due = P["r_delta"] * np.sqrt((h - t_bba) / 24) >= P["sig_delta_star"]
            if method != "none":
                for pl in ("xy" if method != "bba_only" else ""):   # bba_only: izleme yok, BBA var
                    s = st[pl]
                    r1, r2 = reading(pl)
                    (q1, q2), c_ref = ref[pl]
                    y1, y2 = r1 - q1, r2 - q2
                    if world == "W1":
                        # r = (R−I) m_eff − c − δ + n → bilinen düzeltici değişimini geri ekle
                        dc = s["c"] - c_ref
                        y1, y2 = y1 + dc, y2 + dc
                        est = (EST[pl]["E4"] @ np.concatenate([y1, y2]) if method == "E4"
                               else EST[pl]["E2"] @ y1)
                        # est ≈ m_eff − m_eff_ref ; m_eff'i referans değerine geri çek
                        s["c"] = s["c"] + est
                    else:
                        # makale modeli r = R m_eff (BPM sabit sanılır). Düzelticinin bilinen etkisi
                        # R·(−Δc)'yi geri ekle → y ≈ R Δq (model gözüyle) ; c'yi referansa göre MUTLAK ayarla
                        dc = s["c"] - c_ref
                        y1, y2 = y1 + R[pl]["ccw"] @ dc, y2 + R[pl]["cw"] @ dc
                        est = EST[pl]["E3"] @ np.concatenate([y1, y2])
                        s["c"] = c_ref + est
                if bba_due:
                    for pl in "xy":
                        s = st[pl]
                        s["c"] = s["q"] - rng.normal(0, P["sig_bba"], NQ)   # yeni BBA artığı
                        s["d"] = np.zeros(NQ)                                # BPM ofseti yeniden kalibre
                        ref[pl] = (reading(pl), s["c"].copy())
                    t_bba = h; out["bba"].append(h)
        meff = {pl: st[pl]["q"] - st[pl]["c"] for pl in "xy"}
        out["hours"].append(h)
        out["sym"].append(float(np.mean([np.std(A4.PS @ meff[pl]) for pl in "xy"]) * 1e6))
        out["anti"].append(float(np.mean([np.std(A4.PA @ meff[pl]) for pl in "xy"]) * 1e6))
        if record_hours and h in record_hours:
            out["m_eff"][h] = (meff["x"].tolist(), meff["y"].tolist())
    return out


SCEN = [("W1", "E4"), ("W1", "E2"), ("W0", "E3"), ("W1", "bba_only"), ("W1", "none")]


def run_series(nseed=20):
    out = {"params": {k: v for k, v in P.items()},
           "sig_n_m": sig_n(P["sig_turn"], P["T_avg"], P["sig_floor"]), "runs": {}}
    for w, m in SCEN:
        S = [simulate(s, w, m) for s in range(nseed)]
        sym = np.array([r["sym"] for r in S])
        out["runs"][f"{w}_{m}"] = {"hours": S[0]["hours"], "sym_med": np.median(sym, 0).tolist(),
                                   "sym_p16": np.percentile(sym, 16, 0).tolist(),
                                   "sym_p84": np.percentile(sym, 84, 0).tolist(),
                                   "anti_med": np.median([r["anti"] for r in S], 0).tolist(),
                                   "bba_hours": S[0]["bba"]}
        r = out["runs"][f"{w}_{m}"]
        print(f"{w}-{m}: simetrik artık medyan (μm) t=0 {r['sym_med'][0]:.3f}, 24h {r['sym_med'][24]:.3f}, "
              f"96h {r['sym_med'][96]:.3f}, 168h {r['sym_med'][168]:.3f}; BBA saatleri {r['bba_hours'][:6]}",
              flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "adim6_series.json"), "w"), indent=1)


EDM_HOURS = [0, 6, 24, 48, 96, 168]
JSONL = os.path.join(OUTDIR, "adim6_edm.jsonl")


def _task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    from fast_est import fast_measure
    seed, w, m, h = task
    o = simulate(seed, w, m, record_hours=[h])
    dx, dy = (np.array(a) for a in o["m_eff"][h])
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True)
    return dict(seed=seed, world=w, method=m, hour=h, C=abs(fcw - fccw) / 2 / TARGET,
                f_cw=fcw / TARGET, f_ccw=fccw / TARGET,
                sym=float(np.mean([np.std(A4.PS @ dx), np.std(A4.PS @ dy)]) * 1e6))


def run_edm(seeds, workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(JSONL):
        done = {(d["seed"], d["world"], d["method"], d["hour"]) for d in map(json.loads, open(JSONL))}
    tasks = []
    for s in seeds:
        for w, m in SCEN:
            for h in EDM_HOURS:
                if h == 0 and (w, m) != ("W1", "none"):
                    continue                         # t=0 hepsinde aynı (BBA artığı)
                if (s, w, m, h) not in done:
                    tasks.append((s, w, m, h))
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_task, tasks):
            open(JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} {r['world']}-{r['method']} t={r['hour']}h: C={r['C']:.4f}× "
                  f"(sim {r['sym']:.3f} μm)", flush=True)


def run_detect():
    """En küçük tespit edilebilir simetrik sürüklenme: kestirimin simetrik gürültüsünün 3 katı.
    Doğrusal: σ_sym,est = k·σ_n ; k = medyan std(P_s · P · n)/σ_n, 400 gürültü örneği."""
    out = {"k": {}, "grid": []}
    rng = np.random.default_rng(1)
    for name in ("E4", "E2", "E3"):
        ks = []
        for pl in "xy":
            Pm = EST[pl][name]
            nrow = Pm.shape[1]
            ks += [np.std(A4.PS @ (Pm @ rng.normal(0, 1, nrow))) for _ in range(400)]
        out["k"][name] = float(np.median(ks))
    for st in (1e-6, 10e-6, 100e-6):
        for fl in (0.0, 10e-9, 100e-9):
            for T in (1e-3, 1e-2, 0.1, 1, 10, 100, 1000):
                sn = sig_n(st, T, fl)
                out["grid"].append({"sig_turn": st, "floor": fl, "T_avg": T, "sig_n": sn,
                                    **{f"min_{k}": 3 * v * sn for k, v in out["k"].items()}})
    json.dump(out, open(os.path.join(OUTDIR, "adim6_detect.json"), "w"), indent=1)
    print("k (σ_sym,est / σ_n):", out["k"])
    for g in out["grid"]:
        if g["T_avg"] in (0.01, 1, 100):
            print(f"σ_tur {g['sig_turn']*1e6:5.0f} μm, taban {g['floor']*1e9:4.0f} nm, T {g['T_avg']:6.2f} s: "
                  f"σ_n {g['sig_n']*1e9:7.2f} nm → min sürüklenme E4 {g['min_E4']*1e6:.4f} μm, "
                  f"E2 {g['min_E2']*1e6:.4f} μm")


def run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    col = {"W1_E4": "#D55E00", "W1_E2": "#E69F00", "W0_E3": "#0072B2", "W1_bba_only": "#009E73",
           "W1_none": "#595959"}
    lab = {"W1_E4": "rigid BPM, two beams", "W1_E2": "rigid BPM, one beam",
           "W0_E3": "BPM on own stand, paper model", "W1_bba_only": "periodic BBA only",
           "W1_none": "no monitoring, no BBA"}
    S = json.load(open(os.path.join(OUTDIR, "adim6_series.json")))
    for k, r in S["runs"].items():
        h = np.array(r["hours"])
        axs[0].semilogy(h, r["sym_med"], color=col[k], label=lab[k])
        axs[0].fill_between(h, r["sym_p16"], r["sym_p84"], color=col[k], alpha=0.12)
    axs[0].set_xlabel("time [h]"); axs[0].set_ylabel(r"symmetric residual [$\mu$m]")
    axs[0].set_title("(a) residual misalignment (linear, 20 seeds)", fontsize=8.5)
    axs[0].legend(fontsize=6.5)
    if os.path.exists(JSONL):
        E = [json.loads(l) for l in open(JSONL)]
        t0 = [e for e in E if e["hour"] == 0]
        for k in col:
            w, m = k.split("_", 1)
            pts = [e for e in E if e["world"] == w and e["method"] == m] + \
                  ([] if (w, m) == ("W1", "none") else [dict(e, world=w, method=m) for e in t0])
            hs = sorted(set(e["hour"] for e in pts))
            med = [np.median([e["C"] for e in pts if e["hour"] == hh]) for hh in hs]
            axs[1].semilogy(hs, med, "o-", color=col[k], ms=3, label=lab[k])
            for e in pts:
                axs[1].scatter(e["hour"], max(e["C"], 1e-5), color=col[k], s=6, alpha=0.4)
        axs[1].axhline(1, color="#D55E00", ls="--", lw=1)
        axs[1].set_xlabel("time [h]"); axs[1].set_ylabel(r"false EDM $C$ [target]")
        axs[1].set_title("(b) false EDM (C++ tracking, 3 seeds)", fontsize=8.5)
    D = json.load(open(os.path.join(OUTDIR, "adim6_detect.json")))
    for st, ls in ((1e-6, ":"), (10e-6, "-"), (100e-6, "--")):
        for fl, c in ((0.0, "#595959"), (10e-9, "#0072B2"), (100e-9, "#D55E00")):
            g = [x for x in D["grid"] if x["sig_turn"] == st and x["floor"] == fl]
            axs[2].loglog([x["T_avg"] for x in g], [x["min_E4"] * 1e6 for x in g], ls, color=c, lw=1,
                          label=f"{st*1e6:.0f} $\\mu$m/turn, floor {fl*1e9:.0f} nm")
    axs[2].set_xlabel(r"averaging time $T_{\rm avg}$ [s]")
    axs[2].set_ylabel(r"min. detectable sym. drift (3$\sigma$) [$\mu$m]")
    axs[2].set_title("(c) two-beam detection limit", fontsize=8.5)
    axs[2].legend(fontsize=5.5, ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"adim6.{ext}"), dpi=200, bbox_inches="tight")
    print("adim6.pdf/png yazıldı")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["series", "edm", "detect", "plot"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"series": run_series, "edm": lambda: run_edm(a.seeds, a.workers),
     "detect": run_detect, "plot": run_plot}[a.mode]()

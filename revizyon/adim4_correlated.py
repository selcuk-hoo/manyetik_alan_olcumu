#!/usr/bin/env python3
"""adim4_correlated.py — PRAB revizyonu Adım 4: korelasyonlu hizalama hataları (Hakem A, madde 2).

Modeller (her düzlem bağımsız; kuadrupoller s boyunca eşit aralıklı, Δs = C/48 = 16.66 m):
  iid : korelasyonsuz Gauss (makalenin modeli)
  rw  : rastgele yürüyüş — ardışık kuadrupoller arası artım N(0, a²Δs), yani
        ⟨(y_i − y_j)²⟩ = a²|s_i − s_j|  (göreli hata ∝ √mesafe). Halka kapalı olduğundan
        periyodik Brownian köprüsü (ölçüm ağının kapanması) kullanılır.
  atl : Shiltsev ATL yasası ⟨ΔY²⟩ = A·T·L. Sabit T için uzaysal yapı rw ile AYNIDIR
        (a² = A·T); farkı yalnız genliğin zamanla √T büyümesidir (Adım 6'da kullanılır).
        Burada ayrı örnek üretilmez, rw ile eşdeğer olduğu raporlanır.
  mix : rw + iid; komşu-fark varyansının yarısı rw'den, yarısı yerel (iid) hatadan.

Düzlem çıkarma: rijit öteleme/eğim demetçe izlenir ve zararsızdır; ölçüm bir referans
düzleme göre yapılır. Dikeyde m=0,1 (ofset + eğim), yatayda m=1 (öteleme) harmonikleri
çıkarılır. Normalizasyon: düzlem çıkarıldıktan sonra rms = 10 μm (Şekil 4 ile karşılaştırma).

Modlar: stats | rcond | edm | plot
"""
import os, sys, json, argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = os.path.join(BASE, "revizyon")
for p in (REV, BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
OUTDIR = os.path.join(REV, "sonuclar")
NQ = 48
TARGET = 1e-9
CFG = json.load(open(os.path.join(BASE, "params.json")))
L_DEF = 2 * np.pi * CFG["R0"] / NQ
CIRC = NQ * L_DEF + 2 * NQ * CFG["driftLen"] + NQ * CFG["quadLen"]
DS = CIRC / NQ                                    # ardışık kuadrupoller arası [m]
THETA = 2 * np.pi * np.arange(NQ) / NQ            # kuadrupol azimutları (eşit aralık)


def projectors():
    Ps = np.zeros((NQ, NQ)); Pa = np.zeros((NQ, NQ))
    for k in range(NQ // 2):
        a, b = 2 * k, 2 * k + 1
        Ps[a, a] = Ps[b, b] = Ps[a, b] = Ps[b, a] = 0.5
        Pa[a, a] = Pa[b, b] = 0.5
        Pa[a, b] = Pa[b, a] = -0.5
    return Ps, Pa


PS, PA = projectors()


def remove_plane(v, plane):
    """Dikey: m=0,1; yatay: m=1 harmoniklerini en küçük karelerle çıkar."""
    cols = [np.cos(THETA), np.sin(THETA)] + ([np.ones(NQ)] if plane == "y" else [])
    A = np.array(cols).T
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def gen(model, rng):
    """Normalizasyonsuz desen (birim adım)."""
    if model == "iid":
        return rng.normal(0, 1, NQ)
    steps = rng.normal(0, np.sqrt(DS), NQ)        # a = 1 m^{1/2} birimi
    w = np.cumsum(steps)
    w = w - np.arange(1, NQ + 1) / NQ * w[-1]     # periyodik köprü (kapanış)
    if model == "rw":
        return w
    if model == "mix":
        # komşu-fark varyansı: rw → a²Δs ; iid σ² → 2σ²; eşit pay için σ² = Δs/2
        return w + rng.normal(0, np.sqrt(DS / 2), NQ)
    raise ValueError(model)


def _seed_rng(model, seed, plane):
    """Sabit, tekrar üretilebilir tohum (model, seed, düzlem)."""
    code = {"iid": 1, "rw": 2, "mix": 3}[model]
    return np.random.default_rng(code * 100_000 + seed * 2 + (plane == "y"))


def pattern(model, seed, plane, rms=10e-6, plane_removed=True):
    v = gen(model, _seed_rng(model, seed, plane))
    if plane_removed:
        v = remove_plane(v, plane)
    return v * rms / np.std(v)


def harmonic_power(v):
    """Hücre harmoniği m=0..12 güç kesirleri (simetrik ve antisimetrik ayrı).
    Simetrik bileşenin hücre başına değeri s_k=(v_2k+v_2k+1)/2, antisimetrik a_k=(v_2k−v_2k+1)/2;
    her ikisinin 24-noktalı DFT'si (m ve 24−m birleştirilir)."""
    s = (v[0::2] + v[1::2]) / 2; a = (v[0::2] - v[1::2]) / 2
    out = {}
    for name, u in (("sym", s), ("anti", a)):
        F = np.abs(np.fft.fft(u)) ** 2
        p = np.array([F[0]] + [F[m] + F[24 - m] for m in range(1, 12)] + [F[12]])
        out[name] = p
    tot = out["sym"].sum() + out["anti"].sum()
    return {k: (v_ / tot).tolist() for k, v_ in out.items()}


def run_stats(nseed=2000):
    out = {"DS_m": DS, "circ_m": CIRC, "models": {}}
    for model in ("iid", "rw", "mix"):
        r = {}
        for pr in (False, True):
            vs = [pattern(model, s, pl, plane_removed=pr) for s in range(nseed) for pl in "xy"]
            sym = np.array([np.std(PS @ v) for v in vs]); anti = np.array([np.std(PA @ v) for v in vs])
            nn = np.array([np.std(np.diff(np.r_[v, v[0]])) for v in vs])
            far = np.array([np.std(v[:24] - v[24:]) for v in vs])      # karşıt kuadrupoller
            vf = sym ** 2 / (sym ** 2 + anti ** 2)
            hp = [harmonic_power(v) for v in vs]
            r["plane_removed" if pr else "raw"] = {
                "sym_rms_um": float(np.median(sym) * 1e6), "anti_rms_um": float(np.median(anti) * 1e6),
                "ratio_sym_over_anti_median": float(np.median(sym / anti)),
                "ratio_p16": float(np.percentile(sym / anti, 16)),
                "ratio_p84": float(np.percentile(sym / anti, 84)),
                "sym_variance_fraction": float(np.median(vf)),
                "neighbor_diff_rms_um": float(np.median(nn) * 1e6),
                "opposite_diff_rms_um": float(np.median(far) * 1e6),
                "harm_sym": np.mean([h["sym"] for h in hp], axis=0).tolist(),
                "harm_anti": np.mean([h["anti"] for h in hp], axis=0).tolist()}
            d = r["plane_removed" if pr else "raw"]
            print(f"{model:4s} {'düzlem çıkarılmış' if pr else 'ham            '}: sim {d['sym_rms_um']:.2f} μm, "
                  f"antisim {d['anti_rms_um']:.2f} μm, oran {d['ratio_sym_over_anti_median']:.2f} "
                  f"[{d['ratio_p16']:.2f}, {d['ratio_p84']:.2f}], sim varyans kesri "
                  f"{d['sym_variance_fraction']:.3f}; komşu fark {d['neighbor_diff_rms_um']:.2f} μm, "
                  f"karşıt-kuad fark {d['opposite_diff_rms_um']:.2f} μm  (toplam rms 10 μm'ye normalize)")
        out["models"][model] = r
    json.dump(out, open(os.path.join(OUTDIR, "adim4_stats.json"), "w"), indent=1)


# ─────────────────────── düzeltme + sahte EDM ───────────────────────
RCONDS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def load_R():
    return {pl: {t: np.load(os.path.join(BASE, f"R_{k}_{'1' if t=='ccw' else 'cw'}.npy"))
                 for t in ("ccw", "cw")} for pl, k in (("x", "dx"), ("y", "dy"))}


def residual(R, dm, method, rc, rng, sig_n):
    """Statik (mutlak) düzeltme, Mq düzelticiler, lab-sabit BPM, BPM ofseti yok (Şekil 4 elması
    ile aynı koşul). E1: tek demet (CCW), E3: iki demet."""
    n1 = rng.normal(0, sig_n, NQ); n2 = rng.normal(0, sig_n, NQ)
    if method == "raw":
        return dm
    if method == "E1":
        A, y = R["ccw"], R["ccw"] @ dm + n1
    else:
        A = np.vstack([R["ccw"], R["cw"]]); y = np.concatenate([R["ccw"] @ dm + n1, R["cw"] @ dm + n2])
    nrm = np.linalg.norm(A, axis=0)
    return dm - (np.linalg.pinv(A / nrm, rcond=rc) @ y) / nrm


def best_rconds(sig_n=14e-9, nseed=200):
    R = load_R(); out = {}
    for model in ("iid", "rw", "mix"):
        for m in ("E1", "E3"):
            for pl in ("x", "y"):
                def cost(rc):
                    return np.median([np.std(residual(R[pl], pattern(model, 1000 + s, pl), m, rc,
                                                      np.random.default_rng(s), sig_n))
                                      for s in range(nseed)])
                best = min(RCONDS, key=cost)
                out[f"{model}_{m}_{pl}"] = best
    json.dump(out, open(os.path.join(OUTDIR, "adim4_rcond.json"), "w"), indent=1)
    return out


EDM_JSONL = os.path.join(OUTDIR, "adim4_edm.jsonl")


def make(seed, model, method, sig_n=14e-9):
    R = load_R(); rc = json.load(open(os.path.join(OUTDIR, "adim4_rcond.json")))
    rng = np.random.default_rng(77_000 + seed)
    res = {}
    for pl in ("x", "y"):
        dm = pattern(model, seed, pl)
        res[pl] = residual(R[pl], dm, method, rc.get(f"{model}_{method}_{pl}", 1e-2), rng, sig_n)
    return res["x"], res["y"]


def _edm_task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    from fast_est import fast_measure
    seed, model, method = task
    dx, dy = make(seed, model, method)
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    return dict(seed=seed, model=model, method=method, f_cw=fcw / TARGET, f_ccw=fccw / TARGET,
                C=abs(fcw - fccw) / 2 / TARGET,
                sym_x=float(np.std(PS @ dx) * 1e6), sym_y=float(np.std(PS @ dy) * 1e6),
                anti_x=float(np.std(PA @ dx) * 1e6), anti_y=float(np.std(PA @ dy) * 1e6))


def run_edm(seeds, workers, models, methods):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(EDM_JSONL):
        done = {(d["seed"], d["model"], d["method"]) for d in map(json.loads, open(EDM_JSONL))}
    tasks = [(s, mo, me) for s in seeds for mo in models for me in methods if (s, mo, me) not in done]
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_edm_task, tasks):
            open(EDM_JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} {r['model']} {r['method']}: C={r['C']:.4f}× "
                  f"(sim {r['sym_x']:.2f}/{r['sym_y']:.2f}, antisim {r['anti_x']:.2f}/{r['anti_y']:.2f} μm)",
                  flush=True)


def run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    st = json.load(open(os.path.join(OUTDIR, "adim4_stats.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    m = np.arange(13)
    col = {"iid": "#595959", "rw": "#0072B2", "mix": "#E69F00"}
    lab = {"iid": "uncorrelated", "rw": r"random walk ($\propto\sqrt{L}$)", "mix": "random walk + local"}
    for mo in ("iid", "rw", "mix"):
        d = st["models"][mo]["plane_removed"]
        axs[0].semilogy(m, d["harm_sym"], "-o", color=col[mo], ms=3, label=lab[mo] + ", sym")
        axs[0].semilogy(m, d["harm_anti"], "--s", color=col[mo], ms=3, label=lab[mo] + ", anti")
    axs[0].set_xlabel("cell harmonic $m$"); axs[0].set_ylabel("fraction of variance")
    axs[0].set_title("(a) harmonic content (plane removed)", fontsize=9)
    axs[0].legend(fontsize=6)
    if os.path.exists(EDM_JSONL):
        rows = [json.loads(l) for l in open(EDM_JSONL)]
        xs = {"raw": 0, "E1": 1, "E3": 2}; off = {"iid": -0.15, "rw": 0.0, "mix": 0.15}
        for mo in ("iid", "rw", "mix"):
            for me in ("raw", "E1", "E3"):
                v = [r["C"] for r in rows if r["model"] == mo and r["method"] == me]
                axs[1].scatter(np.full(len(v), xs[me] + off[mo]), np.maximum(v, 1e-5), color=col[mo],
                               s=14, label=lab[mo] if me == "raw" else None)
        axs[1].axhline(1, color="#D55E00", ls="--", lw=1)
        axs[1].set_yscale("log"); axs[1].set_xticks([0, 1, 2])
        axs[1].set_xticklabels(["no correction", "one-beam OC", "two-beam OC"])
        axs[1].set_ylabel(r"false EDM $C$ [units of target]")
        axs[1].set_title(r"(b) 10 $\mu$m rms (plane removed), 14 nm, magnetic-in-quad", fontsize=8.5)
        axs[1].legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"adim4.{ext}"), dpi=200, bbox_inches="tight")
    print("adim4.pdf/png yazıldı")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats", "rcond", "edm", "plot"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["iid", "rw", "mix"])
    ap.add_argument("--methods", nargs="+", default=["raw", "E1", "E3"])
    a = ap.parse_args()
    {"stats": run_stats, "rcond": lambda: print(best_rconds()),
     "edm": lambda: run_edm(a.seeds, a.workers, a.models, a.methods), "plot": run_plot}[a.mode]()

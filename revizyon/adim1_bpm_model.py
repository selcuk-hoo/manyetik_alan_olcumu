#!/usr/bin/env python3
"""adim1_bpm_model.py — PRAB revizyonu Adım 1: BPM modeli kontrolü.

Soru: BPM'ler kuadrupolle birlikte hareket ediyorsa (makale Sec. V.B), simetrik
sürüklenme tek demetli yörüngede de görünür mü? Görünürse iki-demet toplamına
gerek kalır mı? Hangi ρ aralığında iki-demet yöntemi üstündür?

Model (her düzlem ayrı, BBA kalibrasyonu SONRASI fark okuması):
  Δm   : kuadrupol manyetik merkezlerinin sürüklenmesi (gerçek, bilinmez)
  u    : BPM elektriksel merkezinin hareketi, iki aile:
           "k0"  : u = ρ·Δm                     (bağımsız BPM hareketi yok)
                   ρ=0 → laboratuvara sabit (a);  ρ=1 → kuadrupole rijit bağlı (b)
           "eq"  : u = ρ·Δm + √(1−ρ²)·ξ         (ξ: Δm'den bağımsız, aynı rms'li
                   BPM hareketi; corr(u,Δm)=ρ, |u|≈|Δm|) — kötümser aile
  Δr_b : demet b'nin (CW/CCW) BPM okumasındaki değişim
           Δr_b = R_b·Δm − u + n_b       (n_b: BPM gürültüsü, demetler arası bağımsız)
  Statik BPM ofseti ve BBA artığı referans okumada sadeleşir (fark ölçümü).

Düzeltme: makaledeki gibi, kuadrupol içindeki manyetik düzeltici ≡ merkez kayması
(separation_bba_testleri.md §2.5); etkin artık m' = Δm − ĉ. Sahte EDM yalnız m''ye
bağlıdır (alan g·(x−m')).

Kestiriciler (hepsi TSVD, rcond=0.01 — makaledeki twobeam_oc.py ile aynı):
  E1 tek-demet standart OC : ĉ = pinv(R_ccw) Δr_ccw                (BPM'i sabit sanır)
  E2 tek-demet model-bilen : ĉ = pinv(R_ccw − ρI) Δr_ccw           (ρ'yu bilir)
  E3 iki-demet (makale)    : ĉ = pinv([R_ccw; R_cw]) [Δr_ccw; Δr_cw]
  E4 iki-demet model-bilen : ĉ = pinv([R_ccw−ρI; R_cw−ρI]) [...]

Modlar:
  recon : doğrusal cebir, çok seed (C++ yok) → simetrik/antisimetrik artık rms
  edm   : seçili durumlar için C++ spin izleyiciyle kalan sahte EDM
          C = ½|f_CW − f_CCW| (twobeam_oc.py ile aynı metrik), JSONL'e artımlı yazar
  plot  : figürler (PDF + PNG)

Kullanım:
  python3 revizyon/adim1_bpm_model.py recon
  python3 revizyon/adim1_bpm_model.py edm -w 4 --seeds 0 1 2
  python3 revizyon/adim1_bpm_model.py plot
"""
import os, sys, json, argparse, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
OUTDIR = os.path.join(BASE, "revizyon", "sonuclar")
os.makedirs(OUTDIR, exist_ok=True)

NQ = 48
RCOND = 0.01
TARGET = 1e-9
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
METHODS = ["E0", "E1", "E2", "E3", "E4"]
MLABEL = {"E0": "düzeltme yok", "E1": "tek demet, standart OC",
          "E2": "tek demet, ρ-bilen", "E3": "iki demet (makale)",
          "E4": "iki demet, ρ-bilen"}


def projectors():
    """Hücre-içi simetrik/antisimetrik projektörler (makale Eq. 4)."""
    Ps = np.zeros((NQ, NQ)); Pa = np.zeros((NQ, NQ))
    for k in range(NQ // 2):
        a, b = 2 * k, 2 * k + 1
        Ps[a, a] = Ps[b, b] = Ps[a, b] = Ps[b, a] = 0.5
        Pa[a, a] = Pa[b, b] = 0.5
        Pa[a, b] = Pa[b, a] = -0.5
    return Ps, Pa


PS, PA = projectors()
I = np.eye(NQ)


def load_R():
    """C++ ile ölçülmüş tepki matrisleri (BPM = kuadrupol girişi, lab çerçevesi).
    R_d?_1.npy: CCW (direction=−1), R_d?_cw.npy: CW (direction=+1)."""
    R = {}
    for pl, key in (("x", "dx"), ("y", "dy")):
        R[pl] = {"ccw": np.load(os.path.join(BASE, f"R_{key}_1.npy")),
                 "cw": np.load(os.path.join(BASE, f"R_{key}_cw.npy"))}
    return R


def drift_pattern(rng, kind, rms):
    """Sürüklenme deseni: 'sym' (yalnız simetrik) veya 'gen' (korelasyonsuz)."""
    v = rng.normal(0.0, 1.0, NQ)
    if kind == "sym":
        v = PS @ v
    return v * rms / np.std(v)


RCONDS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]


def estimators(Rc, Rw, rho, rcond=RCOND):
    """Dört kestiricinin pseudo-inverse'leri (TSVD; tekil değer < rcond·s_max atılır)."""
    pinv = lambda M: np.linalg.pinv(M, rcond=rcond)
    return {"E1": ("one", pinv(Rc)),
            "E2": ("one", pinv(Rc - rho * I)),
            "E3": ("two", pinv(np.vstack([Rc, Rw]))),
            "E4": ("two", pinv(np.vstack([Rc - rho * I, Rw - rho * I])))}


def residuals(Rp, dm, rho, xi, n_ccw, n_cw, est, family="k0"):
    """Bir düzlem için her kestiricinin etkin artığı m' = Δm − ĉ."""
    kap = np.sqrt(max(0.0, 1.0 - rho ** 2)) if family == "eq" else 0.0
    u = rho * dm + kap * xi
    r_ccw = Rp["ccw"] @ dm - u + n_ccw
    r_cw = Rp["cw"] @ dm - u + n_cw
    out = {"E0": dm.copy()}
    for name, (kind, P) in est.items():
        y = r_ccw if kind == "one" else np.concatenate([r_ccw, r_cw])
        out[name] = dm - P @ y
    return out


def make_case(seed, kind, rms, rho, sig_n, family="k0", rcond=RCOND):
    """Tek seed için (x, y) düzlemleri: gerçek sürüklenme + tüm kestirici artıkları.
    Rastgele sayılar ρ'dan bağımsız üretilir → aynı seed farklı ρ'da aynı Δm, ξ, n."""
    rng = np.random.default_rng(10_000 + seed)
    res = {}
    for pl in ("x", "y"):
        dm = drift_pattern(rng, kind, rms)
        xi = drift_pattern(rng, "gen", rms)        # bağımsız BPM hareketi (aynı rms)
        n1 = rng.normal(0.0, sig_n, NQ); n2 = rng.normal(0.0, sig_n, NQ)
        res[pl] = residuals(R[pl], dm, rho, xi, n1, n2, est_for(pl, rho, rcond), family)
    return res


R = load_R()
_EST = {}


def est_for(pl, rho, rcond):
    key = (pl, rho, rcond)
    if key not in _EST:
        _EST[key] = estimators(R[pl]["ccw"], R[pl]["cw"], rho, rcond)
    return _EST[key]


# ─────────────────────────── recon modu ───────────────────────────
def run_recon(nseed=200):
    out = {"params": {"nseed": nseed, "rcond_fixed": RCOND, "rconds": RCONDS, "rhos": RHOS,
                      "drift_rms_um": 5.0, "noise_levels_m": [14e-9, 1e-6]},
           "rows": []}
    for fam, kind, sig_n, rho in [(f, k, n, r) for f in ("k0", "eq") for k in ("sym", "gen")
                                  for n in (14e-9, 1e-6) for r in RHOS]:
        # her rcond için tüm seed'lerde artıklar: acc[rc][m][comp] = liste (x ve y düzlemi)
        acc = {}
        for rc in RCONDS:
            acc[rc] = {m: {"sym": [], "anti": [], "tot": []} for m in METHODS}
            for s_ in range(nseed):
                res = make_case(s_, kind, 5e-6, rho, sig_n, fam, rc)
                for m in METHODS:
                    for pl in ("x", "y"):
                        acc[rc][m]["sym"].append(np.std(PS @ res[pl][m]))
                        acc[rc][m]["anti"].append(np.std(PA @ res[pl][m]))
                        acc[rc][m]["tot"].append(np.std(res[pl][m]))
        row = {"family": fam, "kind": kind, "sig_n": sig_n, "rho": rho}
        for m in METHODS:
            # sabit rcond (makale) ve en iyi rcond (toplam artık medyanını en aza indiren)
            best = min(RCONDS, key=lambda rc: np.median(acc[rc][m]["tot"]))
            row[f"{m}_best_rcond"] = best
            for tag, rc in (("fix", RCOND), ("best", best)):
                for comp in ("sym", "anti", "tot"):
                    a = np.array(acc[rc][m][comp]) * 1e6
                    row[f"{m}_{tag}_{comp}_med"] = float(np.median(a))
                    row[f"{m}_{tag}_{comp}_p16"] = float(np.percentile(a, 16))
                    row[f"{m}_{tag}_{comp}_p84"] = float(np.percentile(a, 84))
        out["rows"].append(row)
        print(f"{fam} {kind} σn={sig_n*1e9:5.0f}nm ρ={rho:4.2f} | sim artık μm fix/best: "
              + "  ".join(f"{m} {row[m+'_fix_sym_med']:.3f}/{row[m+'_best_sym_med']:.3f}"
                          for m in METHODS), flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "adim1_recon.json"), "w"), indent=1)
    # koşullanma bilgisi
    cond = {}
    for pl in ("x", "y"):
        Rc, Rw = R[pl]["ccw"], R[pl]["cw"]
        for rho in (0.0, 0.5, 0.9, 1.0):
            cond[f"{pl}_rho{rho}"] = {
                "one": float(np.linalg.cond(Rc - rho * I)),
                "two": float(np.linalg.cond(np.vstack([Rc - rho * I, Rw - rho * I])))}
    json.dump(cond, open(os.path.join(OUTDIR, "adim1_cond.json"), "w"), indent=1)
    print(json.dumps(cond, indent=1))


# ─────────────────────────── edm modu ───────────────────────────
EDM_RHOS = [0.0, 0.5, 1.0]
EDM_JSONL = os.path.join(OUTDIR, "adim1_edm.jsonl")


def best_rcond(fam, kind, sig_n, rho, method):
    """recon modunun seçtiği en iyi TSVD eşiği (recon önce koşulmalı)."""
    d = json.load(open(os.path.join(OUTDIR, "adim1_recon.json")))
    for r in d["rows"]:
        if (r["family"], r["kind"], r["sig_n"], r["rho"]) == (fam, kind, sig_n, rho):
            return r.get(f"{method}_best_rcond", RCOND)
    raise KeyError((fam, kind, sig_n, rho))


def _edm_task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    from fast_est import fast_measure
    seed, rho, method, sig_n = task
    rc = best_rcond("k0", "sym", sig_n, rho, method)
    res = make_case(seed, "sym", 5e-6, rho, sig_n, "k0", rc)
    dx, dy = res["x"][method], res["y"][method]
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    return dict(seed=seed, rho=rho, method=method, sig_n=sig_n, rcond=rc,
                f_cw=fcw / TARGET, f_ccw=fccw / TARGET, C=abs(fcw - fccw) / 2 / TARGET,
                sym_x=float(np.std(PS @ dx) * 1e6), sym_y=float(np.std(PS @ dy) * 1e6),
                anti_x=float(np.std(PA @ dx) * 1e6), anti_y=float(np.std(PA @ dy) * 1e6))


def run_edm(seeds, workers, sig_n):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(EDM_JSONL):
        for line in open(EDM_JSONL):
            d = json.loads(line)
            done.add((d["seed"], d["rho"], d["method"], d["sig_n"]))
    tasks = []
    for s in seeds:
        tasks.append((s, 0.0, "E0", sig_n))                    # E0 ρ'dan bağımsız
        for rho in EDM_RHOS:
            ms = ["E1", "E3"] if rho == 0.0 else ["E1", "E2", "E3", "E4"]
            tasks += [(s, rho, m, sig_n) for m in ms]
    tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks)} görev (her biri CW+CCW, ~2×9 dk tek çekirdek)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_edm_task, tasks):
            with open(EDM_JSONL, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} ρ={r['rho']} {r['method']}: C={r['C']:.4f}× "
                  f"(sim {r['sym_x']:.3f}/{r['sym_y']:.3f} μm) [{time.time()-t0:.0f}s]",
                  flush=True)


# ─────────────────────────── plot modu ───────────────────────────
def run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 9, "savefig.bbox": "tight"})
    d = json.load(open(os.path.join(OUTDIR, "adim1_recon.json")))
    col = {"E0": "#595959", "E1": "#009E73", "E2": "#E69F00", "E3": "#0072B2", "E4": "#D55E00"}
    mk = {"E0": "x", "E1": "s", "E2": "^", "E3": "o", "E4": "D"}
    lab = {"E0": "no correction", "E1": "one beam, standard OC",
           "E2": r"one beam, $\rho$-aware", "E3": "two beams (paper)",
           "E4": r"two beams, $\rho$-aware"}
    for fam, kind in [(f, k) for f in ("k0", "eq") for k in ("sym", "gen")]:
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
        for j, sig_n in enumerate((14e-9, 1e-6)):
            rows = [r for r in d["rows"] if r["family"] == fam and r["kind"] == kind and r["sig_n"] == sig_n]
            rho = np.array([r["rho"] for r in rows])
            for i, comp in enumerate(("sym", "anti")):
                ax = axs[i, j]
                for m in METHODS:
                    med = np.array([r[f"{m}_best_{comp}_med"] for r in rows])
                    lo = np.array([r[f"{m}_best_{comp}_p16"] for r in rows])
                    hi = np.array([r[f"{m}_best_{comp}_p84"] for r in rows])
                    fx = np.array([r[f"{m}_fix_{comp}_med"] for r in rows])
                    ax.plot(rho, med, marker=mk[m], color=col[m], label=lab[m], ms=4)
                    ax.plot(rho, fx, ls=":", color=col[m], lw=1)
                    ax.fill_between(rho, lo, hi, color=col[m], alpha=0.12)
                ax.set_yscale("log")
                ax.set_title(f"{'symmetric' if comp=='sym' else 'antisymmetric'} residual, "
                             f"BPM noise {sig_n*1e9:.0f} nm" if sig_n < 1e-7 else
                             f"{'symmetric' if comp=='sym' else 'antisymmetric'} residual, "
                             f"BPM noise 1 $\\mu$m", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(r"rms residual [$\mu$m]")
                if i == 1:
                    ax.set_xlabel(r"BPM–quadrupole correlation $\rho$")
        axs[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{'Symmetric' if kind=='sym' else 'Uncorrelated'} drift, 5 $\\mu$m rms "
                     f"per plane; BPM motion {'$u=\\rho\\,\\Delta m$' if fam=='k0' else '$u=\\rho\\,\\Delta m+\\sqrt{1-\\rho^2}\\,\\xi$'}; median, 16–84% band, {d['params']['nseed']} seeds\nsolid: best TSVD cut per method; dotted: rcond = 0.01 (paper)",
                     fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUTDIR, f"adim1_recon_{fam}_{kind}.{ext}"), dpi=200)
        plt.close(fig)
    # sahte EDM figürü: (a) C = ½|f_CW − f_CCW|, (b) tek yön max(|f_CW|,|f_CCW|)
    if os.path.exists(EDM_JSONL):
        rows = [json.loads(l) for l in open(EDM_JSONL)]
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
        off = {"E0": -0.06, "E1": -0.03, "E2": 0.0, "E3": 0.03, "E4": 0.06}
        for ax, key in zip(axs, ("C", "fmax")):
            for m in METHODS:
                pts = [r for r in rows if r["method"] == m]
                if not pts:
                    continue
                rhos = EDM_RHOS if m == "E0" else sorted(set(r["rho"] for r in pts))
                for k, rho in enumerate(rhos):
                    sel = pts if m == "E0" else [r for r in pts if r["rho"] == rho]
                    vals = [r["C"] if key == "C" else max(abs(r["f_cw"]), abs(r["f_ccw"]))
                            for r in sel]
                    ax.scatter(np.full(len(vals), rho + off[m]), np.maximum(vals, 1e-5),
                               color=col[m], marker=mk[m], s=18,
                               label=lab[m] if k == 0 else None)
            ax.axhline(1.0, color="#D55E00", ls="--", lw=1)
            ax.text(0.3, 1.3, "target", color="#D55E00", fontsize=8)
            ax.set_yscale("log")
            ax.set_xlabel(r"BPM–quadrupole correlation $\rho$")
        axs[0].set_ylabel("false EDM [units of target]")
        axs[0].set_title(r"(a) $C=\frac{1}{2}|f_{\rm CW}-f_{\rm CCW}|$", fontsize=9)
        axs[1].set_title(r"(b) single beam, $\max(|f_{\rm CW}|,|f_{\rm CCW}|)$", fontsize=9)
        fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(r"Symmetric drift 5 $\mu$m rms, BPM noise 14 nm, $u=\rho\,\Delta m$, 3 seeds",
                     fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUTDIR, f"adim1_edm.{ext}"), dpi=200)
        plt.close(fig)
    print("figürler:", OUTDIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["recon", "edm", "plot"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--nseed", type=int, default=200)
    ap.add_argument("--sig-n", type=float, default=14e-9)
    a = ap.parse_args()
    if a.mode == "recon":
        run_recon(a.nseed)
    elif a.mode == "edm":
        run_edm(a.seeds, a.workers, a.sig_n)
    else:
        run_plot()

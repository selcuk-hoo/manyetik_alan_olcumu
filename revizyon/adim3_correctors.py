#!/usr/bin/env python3
"""adim3_correctors.py — PRAB revizyonu Adım 3: manyetik ve elektrostatik düzelticiler (Hakem A).

Düzeltici aileleri (her düzlem için 48'er):
  Mq  : kuadrupol İÇİNDE manyetik dipol ≡ kuadrupol merkez kayması (makalenin örtük modeli,
        separation_bba_testleri.md §2.5). Tekme CCW'de işaret çevirir.
  Md  : saptırıcı konumunda manyetik dipol (yatay: düşey B_v, düşey: radyal B_r) [T].
        Tekme CCW'de işaret çevirir → FARK kanalı.
  Ed  : saptırıcı konumunda elektrostatik düzeltici (yatay: alan trimi ε, düşey: roll φ).
        Tekme iki demette AYNI → TOPLAM kanalı.
  MEd : Md + Ed birlikte (96 düzeltici / 96 denklem).

Hedefler:
  zero   : iki demetin BPM okumasını sıfırla (klasik yörünge düzeltmesi)
  center : demeti kestirilen kuadrupol merkezlerinden geçir (okuma = m̂, lab-sabit BPM)

Senaryo (Adım 1 ile aynı): BBA sonrası saf simetrik kuadrupol sürüklenmesi 5 μm rms/düzlem,
lab-sabit BPM (ρ=0), 14 nm gürültü, nominal model. m̂: iki-demet fit (E3, TSVD).

Modlar: matrix | check | edm | plot
"""
import os, sys, json, argparse, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = os.path.join(BASE, "revizyon")
for p in (REV, BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
import integrator_rev
sys.modules["integrator"] = integrator_rev
OUTDIR = os.path.join(REV, "sonuclar")
NQ = 48
TARGET = 1e-9
DELTA_B = 1e-6        # manyetik düzeltici matris pertürbasyonu [T]
FAM = ("eps", "roll", "Bv", "Br")


def set_defl(eps=None, roll=None, Bv=None, Br=None):
    z = np.zeros(NQ)
    integrator_rev.DEFL_DE = z if eps is None else np.asarray(eps)
    integrator_rev.DEFL_ROLL = z if roll is None else np.asarray(roll)
    integrator_rev.DEFL_BV = z if Bv is None else np.asarray(Bv)
    integrator_rev.DEFL_BR = z if Br is None else np.asarray(Br)


def cod_run(task):
    """Kapalı yörünge (eksenden fırlatma + tur ortalaması). task = (dir, dx, dy, corr dict, t2)."""
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.modules["integrator"] = integrator_rev
    import build_response_matrix as brm
    brm.integrate_particle = integrator_rev.integrate_particle
    d, dx, dy, corr, t2 = task
    set_defl(**corr)
    cfg = json.load(open(os.path.join(BASE, "params.json")))
    cfg["direction"] = float(d); cfg["t2"] = t2
    alanlar, st = brm.setup_fields(cfg)
    return brm.run_sim(alanlar, st, cfg, np.asarray(dy), np.asarray(dx))


def MPATH(fam, d):
    if fam in ("eps", "roll"):
        return os.path.join(OUTDIR, f"D_{fam}_{d}.npy")          # Adım 2'de üretildi
    return os.path.join(OUTDIR, f"M_{fam}_{d}.npy")


def run_matrix(workers, t2=3e-4):
    """Manyetik düzeltici tepki matrisleri M_Bv (B_v → x), M_Br (B_r → y) [m/T], CW + CCW."""
    from concurrent.futures import ProcessPoolExecutor
    z = np.zeros(NQ)
    for d, tag in ((-1, "ccw"), (+1, "cw")):
        if os.path.exists(MPATH("Br", tag)):
            print(tag, "mevcut"); continue
        tasks = [(d, z, z, {}, t2)]
        for k in range(NQ):
            b = z.copy(); b[k] = DELTA_B; tasks.append((d, z, z, {"Bv": b}, t2))
        for k in range(NQ):
            b = z.copy(); b[k] = DELTA_B; tasks.append((d, z, z, {"Br": b}, t2))
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool:
            res = list(pool.map(cod_run, tasks))
        x0, y0 = res[0]
        Mv = np.array([(res[1 + k][0] - x0) / DELTA_B for k in range(NQ)]).T
        Mr = np.array([(res[1 + NQ + k][1] - y0) / DELTA_B for k in range(NQ)]).T
        np.save(MPATH("Bv", tag), Mv); np.save(MPATH("Br", tag), Mr)
        print(f"{tag}: M_Bv, M_Br yazıldı ({time.time()-t0:.0f}s)", flush=True)


def load():
    """Kuadrupol R ve dört düzeltici ailesinin matrisleri, düzlem başına."""
    M = {}
    for pl, rk, e, m in (("x", "dx", "eps", "Bv"), ("y", "dy", "roll", "Br")):
        M[pl] = {"R": {t: np.load(os.path.join(BASE, f"R_{rk}_{'1' if t=='ccw' else 'cw'}.npy"))
                       for t in ("ccw", "cw")},
                 "E": {t: np.load(MPATH(e, t)) for t in ("ccw", "cw")},
                 "M": {t: np.load(MPATH(m, t)) for t in ("ccw", "cw")}}
    return M


def run_check():
    """İşaret kontrolü: her ailenin CW/CCW korelasyonu ve toplam/fark kanalı oranı."""
    M = load(); out = {}
    for pl in ("x", "y"):
        for fam in ("R", "E", "M"):
            a, b = M[pl][fam]["ccw"], M[pl][fam]["cw"]
            c = np.mean([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(NQ)])
            r = np.linalg.norm(a + b) / np.linalg.norm(a - b)
            out[f"{pl}_{fam}"] = {"corr_cw_ccw": float(c), "sum_over_diff": float(r)}
            print(f"[{pl}] {fam}: ortalama sütun corr(CW,CCW) = {c:+.3f}, |toplam|/|fark| = {r:.2f}")
    json.dump(out, open(os.path.join(OUTDIR, "adim3_check.json"), "w"), indent=1)


# ───────────────────────── senaryo + düzeltme ─────────────────────────
def projectors():
    Ps = np.zeros((NQ, NQ))
    for k in range(NQ // 2):
        a, b = 2 * k, 2 * k + 1
        Ps[a, a] = Ps[b, b] = Ps[a, b] = Ps[b, a] = 0.5
    return Ps


def tsvd(A, y, rc):
    n = np.linalg.norm(A, axis=0)
    return (np.linalg.pinv(A / n, rcond=rc) @ y) / n


CASES = ["none", "Mq", "Md_zero", "Ed_zero", "MEd_zero", "Md_center", "Ed_center", "MEd_center"]


def make_case(seed, case, sig_n=14e-9, drift=5e-6, rc_fit=3e-3, rc_corr=1e-3):
    """Kuadrupol artıkları (dx, dy) + saptırıcı düzeltici ayarları (eps, roll, Bv, Br) +
    düzeltme sonrası öngörülen okuma rms'leri."""
    M = load(); Ps = projectors()
    rng = np.random.default_rng(10_000 + seed)      # Adım 1 ile aynı desen/gürültü üreteci
    q = {}; corr = {}; pred = {}
    for pl in ("x", "y"):
        R, E, Mg = M[pl]["R"], M[pl]["E"], M[pl]["M"]
        dm = Ps @ rng.normal(0, 1, NQ); dm *= drift / np.std(dm)
        rng.normal(0, 1, NQ)                         # Adım 1'deki ξ çekimiyle hizalı kal
        n1 = rng.normal(0, sig_n, NQ); n2 = rng.normal(0, sig_n, NQ)
        y = np.concatenate([R["ccw"] @ dm + n1, R["cw"] @ dm + n2])
        Sq = np.vstack([R["ccw"], R["cw"]])
        mhat = tsvd(Sq, y, rc_fit)
        ce = np.zeros(NQ); cm = np.zeros(NQ); res = dm.copy()
        if case == "Mq":
            res = dm - mhat
        elif case != "none":
            fam, goal = case.split("_")
            target = np.zeros(2 * NQ) if goal == "zero" else np.concatenate([mhat, mhat])
            A = {"Md": np.vstack([Mg["ccw"], Mg["cw"]]),
                 "Ed": np.vstack([E["ccw"], E["cw"]]),
                 "MEd": np.hstack([np.vstack([Mg["ccw"], Mg["cw"]]),
                                   np.vstack([E["ccw"], E["cw"]])])}[fam]
            c = tsvd(A, target - y, rc_corr)
            if fam == "Md":
                cm = c
            elif fam == "Ed":
                ce = c
            else:
                cm, ce = c[:NQ], c[NQ:]
        q[pl] = res
        e_key, m_key = ("eps", "Bv") if pl == "x" else ("roll", "Br")
        corr[e_key] = ce; corr[m_key] = cm
        # öngörülen (doğrusal) okuma ve "demet − kuadrupol merkezi" rms'i (gerçek dm ile)
        orb = {t: R[t] @ res + E[t] @ ce + Mg[t] @ cm for t in ("ccw", "cw")}
        # etkin kuadrupol merkezi: Mq'da kaydırılmış merkez = res; diğerlerinde kuadrupol yerinde (dm)
        center = res if case in ("none", "Mq") else dm
        pred[pl] = {"orbit_rms_um": {t: float(np.std(orb[t]) * 1e6) for t in orb},
                    "beam_minus_center_rms_um": {t: float(np.std(orb[t] - center) * 1e6)
                                                 for t in orb},
                    "corr_E_rms": float(np.std(ce)), "corr_M_rms_T": float(np.std(cm)),
                    "sym_res_um": float(np.std(Ps @ res) * 1e6)}
    return q["x"], q["y"], corr, pred


EDM_JSONL = os.path.join(OUTDIR, "adim3_edm.jsonl")


def _edm_task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.modules["integrator"] = integrator_rev
    from fast_est import fast_measure
    seed, case = task
    dx, dy, corr, pred = make_case(seed, case)
    set_defl(**corr)
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    return dict(seed=seed, case=case, f_cw=fcw / TARGET, f_ccw=fccw / TARGET,
                C=abs(fcw - fccw) / 2 / TARGET, even=(fcw + fccw) / 2 / TARGET, pred=pred)


def run_edm(seeds, workers, cases):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(EDM_JSONL):
        done = {(json.loads(l)["seed"], json.loads(l)["case"]) for l in open(EDM_JSONL)}
    tasks = [(s, c) for s in seeds for c in cases if (s, c) not in done]
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_edm_task, tasks):
            open(EDM_JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} {r['case']}: C={r['C']:.4f}× çift={r['even']:.4g}× "
                  f"(f_CW {r['f_cw']:.4g}, f_CCW {r['f_ccw']:.4g})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["matrix", "check", "pred", "edm", "plot"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cases", nargs="+", default=CASES)
    a = ap.parse_args()
    if a.mode == "matrix":
        run_matrix(a.workers)
    elif a.mode == "check":
        run_check()
    elif a.mode == "pred":
        for c in CASES:
            _, _, corr, pred = make_case(0, c)
            print(c, json.dumps({pl: {k: (v if not isinstance(v, float) else round(v, 4))
                                      for k, v in pred[pl].items()} for pl in pred}))
    elif a.mode == "plot":
        pass                                     # dosya sonunda tanımlı; aşağıda çağrılır
    else:
        run_edm(a.seeds, a.workers, a.cases)


def run_plot():
    """C ve çift kısım (f_CW+f_CCW)/2, senaryo başına, 3 seed (log)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    rows = [json.loads(l) for l in open(EDM_JSONL)]
    lab = {"none": "none", "Mq": "Mq\n(in quad)", "Md_zero": "Md\nzero", "Ed_zero": "Ed\nzero",
           "MEd_zero": "M+Ed\nzero", "Md_center": "Md\ncenter", "Ed_center": "Ed\ncenter",
           "MEd_center": "M+Ed\ncenter"}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, key, t in zip(axs, ("C", "even"), (r"(a) $C=\frac{1}{2}|f_{\rm CW}-f_{\rm CCW}|$",
                                              r"(b) $|f_{\rm CW}+f_{\rm CCW}|/2$")):
        for i, c in enumerate(CASES):
            v = [abs(r[key]) for r in rows if r["case"] == c]
            ax.scatter(np.full(len(v), i), np.maximum(v, 1e-5), color="#0072B2", s=16, zorder=3)
        ax.axhline(1, color="#D55E00", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_xticks(range(len(CASES)))
        ax.set_xticklabels([lab[c] for c in CASES], fontsize=7)
        ax.set_title(t, fontsize=9); ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("false EDM [units of target]")
    fig.suptitle(r"Symmetric drift 5 $\mu$m, lab-fixed BPM, 14 nm noise, 3 seeds; "
                 r"Mq: magnetic in quad, Md: magnetic at deflector, Ed: electrostatic at deflector",
                 fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"adim3_edm.{ext}"), dpi=200, bbox_inches="tight")
    print("adim3_edm.pdf/png yazıldı")


if __name__ == "__main__" and sys.argv[1:2] == ["plot"]:
    run_plot()

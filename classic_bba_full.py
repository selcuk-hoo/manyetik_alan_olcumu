#!/usr/bin/env python3
"""
classic_bba_full.py — T5 UÇTAN-UCA C++ SONUCU (separation_bba_testleri.md §5.1)

Tam zincir, tamamı gerçek dinamikle (C++):
  1) 47 quad × 2 düzlem klasik BBA: quad_dG modülasyonu + komşu-düzeltici
     bump + 2-noktalı tarama → null → merkez kestirimi (CO-oturtmalı okuma).
     (cell-0 QF quad_dG okumaz [tuzak #8] → sim'de quad 0 kaçıklığı 0 alınır
     ve ölçülmez; gerçek makinede o quad başka yolla modüle edilir.)
  2) Düzeltme: kestirilen merkezler düşülür (düzeltici ≡ merkez kayması).
  3) Kalan sahte-EDM, spin izleyicisiyle DOĞRUDAN ölçülür
     (kmod_drivers.fast_est.fast_measure; 4D-CO + model-fit) — formül
     kestirimi DEĞİL.

Gürültüsüz koşum → sistematik taban; istatistik (BPM gürültüsü) analitik
çalışmadan ayrıca rapor edilir (yol gösterici). Çıktı: JSONL artımlı +
kmod_drivers/paper_runs_results.json [bba_full].

Kullanım: python3 classic_bba_full.py -w 4 [--scan 150e-6]
"""
import argparse, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_orbit_figures import R_perquad, sym_anti_projectors, NQ, G_NOM

EPS = 0.02
JSONL = "/tmp/kmod_recover/bba_full.jsonl"
os.makedirs("/tmp/kmod_recover", exist_ok=True)


def _orbit_xy(task):
    """CO-oturtmalı tek koşum: (dx, dy, dG) → (x_COD, y_COD) 48-BPM [m]."""
    import tempfile
    os.chdir(tempfile.mkdtemp())
    sys.path.insert(0, os.path.join(BASE, "berry_data"))
    from false_edm_mode_scan import setup_fields, _make_state
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    from build_response_matrix import read_cod_quads
    dx, dy, dG, tag = task
    dx = np.array(dx); dy = np.array(dy); dG = np.array(dG)
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    T_rev = _T_rev(fields, beta0, R0)
    zeros = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, zeros,
                             T_rev, n_turns=14, n_iter=2)
    launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    fields.poincare_quad_index = 999.0
    for fname in ("cod_data.txt", "rf.txt"):
        if os.path.exists(fname):
            os.remove(fname)
    integrate_particle(launch, 0.0, float(cfg.get("t2", 1e-3)),
                       float(cfg["dt"]), fields=fields, return_steps=10,
                       quad_dy=dy, quad_dx=dx, quad_tilt=zeros, quad_dG=dG)
    x, y = read_cod_quads(NQ // 2)
    rec = {"tag": tag, "x": [float(v) for v in x], "y": [float(v) for v in y],
           "resid": float(resid), "t": time.time()}
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def estimate(by, meta, noise_sigma=0.0, offset=None, rng=None):
    """BBA merkez kestirimi. noise_sigma: averaj-sonrası okuma gürültüsü (her 4
    okumaya bağımsız); offset: statik per-BPM ofset (A'da iptal, tarama ekseninde
    est'i +offset[i] kaydırır → golden-orbit sürüşü geri alır)."""
    if offset is None:
        offset = np.zeros(NQ)
    est_x = np.zeros(NQ); est_y = np.zeros(NQ)
    for (i, plane, j) in meta:
        reads = {}
        for sgn_lbl in ("m", "p"):
            for on in (1, 0):
                v = np.array(by[f"q{i}_{plane}_{sgn_lbl}_{on}"][plane]) + offset
                if noise_sigma > 0:
                    v = v + rng.normal(0.0, noise_sigma, NQ)
                reads[(sgn_lbl, on)] = v
        pts = []
        for sgn_lbl in ("m", "p"):
            A = reads[(sgn_lbl, 1)] - reads[(sgn_lbl, 0)]
            pts.append((reads[(sgn_lbl, 0)][i], A))
        (x1, A1), (x2, A2) = pts
        s = (A2 - A1) / (x2 - x1)
        a = A1 - s * x1
        x_star = -np.sum(s * a) / np.sum(s ** 2)      # ağırlıklı null (w=s²)
        if plane == "y":
            est_y[i] = x_star
        else:
            est_x[i] = x_star
    return est_x, est_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--scan", type=float, default=150e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bbeat", type=float, default=0.01,
                    help="per-quad statik gradyan hatası (β-beat), C++ dinamiğine gömülür")
    ap.add_argument("--bpm-offset", type=float, default=100e-6,
                    help="statik per-BPM ofset [m] (golden-orbit'te iptal)")
    ap.add_argument("--bpm-noise", type=float, default=1e-6,
                    help="tek-atış BPM gürültüsü [m]")
    ap.add_argument("--navg", type=str, default="100,10000",
                    help="N_avg listesi (nokta başına atış ortalaması)")
    args = ap.parse_args()
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    m_dx = rng.normal(0.0, 10e-6, NQ); m_dx[0] = 0.0   # quad 0: tuzak #8
    m_dy = rng.normal(0.0, 10e-6, NQ); m_dy[0] = 0.0

    # β-beat: per-quad statik gradyan hatası (gerçek makine optiği ≠ model);
    # C++ dinamiğine gömülür (dG'ye eklenir), quad 0 hariç (cell-0 QF quad_dG
    # okumaz, tuzak #8). BPM ofseti/gürültüsü okuma-katmanı → post-işlemde.
    bbeat = rng.normal(0.0, args.bbeat, NQ) if args.bbeat > 0 else np.zeros(NQ)
    bbeat[0] = 0.0
    bpm_off = rng.normal(0.0, args.bpm_offset, NQ) if args.bpm_offset > 0 else np.zeros(NQ)

    # Bump ölçekleme/knob seçimi için DÜZLEM-ÖZEL analitik tepki (yalnız kılavuz;
    # sonuç değil). x-düzlemi için dikey R'yi kullanmak yanlış düzeltici + yanlış
    # genlik verir (dx hatası ~4× şişer) — bu yüzden her düzlem kendi R'siyle.
    from analytic_kmod import (compute_twiss_at_quads, signed_KL,
                               build_R_analytic, compute_Brho)
    with open(os.path.join(BASE, "params.json")) as _f:
        _cfg = json.load(_f)
    _nF = int(_cfg["nFODO"]); _Lq = float(_cfg["quadLen"])
    _Brho = compute_Brho(_cfg)

    def _build_R0(plane):
        beta, phi, Q = compute_twiss_at_quads(_cfg, G_NOM, plane)
        KL = signed_KL(_nF, abs(G_NOM) / _Brho, _Lq, plane)
        return build_R_analytic(beta, phi, Q, KL)

    R0_model = {"y": _build_R0("y"), "x": _build_R0("x")}

    quads = list(range(1, NQ))
    # görevler: her quad × düzlem × {−δ, +δ} × {mod aç, kapa}
    tasks = []
    meta = []
    for i in quads:
        cands = [(i - 1) % NQ, (i + 1) % NQ, (i - 2) % NQ, (i + 2) % NQ]
        for plane in ("y", "x"):
            Rp = R0_model[plane]
            j = max(cands, key=lambda jj: abs(Rp[i, jj]))   # düzlem-özel düzeltici
            for sgn in (-1.0, +1.0):
                dxx, dyy = m_dx.copy(), m_dy.copy()
                shift = sgn * args.scan / Rp[i, j]          # düzlem-özel genlik
                if plane == "y":
                    dyy[j] += shift
                else:
                    dxx[j] += shift
                dG_on = bbeat.copy(); dG_on[i] += EPS     # modülasyon + β-beat
                dG_off = bbeat.copy()                      # yalnız β-beat
                for on in (1, 0):
                    tasks.append((list(dxx), list(dyy),
                                  list(dG_on if on else dG_off),
                                  f"q{i}_{plane}_{'p' if sgn>0 else 'm'}_{on}"))
            meta.append((i, plane, j))

    print(f"BBA taraması: {len(tasks)} CO-oturtmalı koşum ({args.workers} işçi)")
    import build_response_matrix as brm
    with ProcessPoolExecutor(args.workers, initializer=brm._worker_init) as pool:
        recs = list(pool.map(_orbit_xy, tasks))
    by = {r["tag"]: r for r in recs}

    P_sym, P_anti = sym_anti_projectors()
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    sys.path.insert(0, os.path.join(BASE, "kmod_drivers"))
    from fast_est import fast_measure

    def report_err(tag, ex, ey):
        ex = ex.copy(); ey = ey.copy(); ex[0] = ey[0] = 0.0
        print(f"  [{tag}] dx: RMS {rms(ex)*1e6:.3f} μm (sym {rms(P_sym@ex)*1e6:.3f}"
              f", anti {rms(P_anti@ex)*1e6:.3f})  |  dy: RMS {rms(ey)*1e6:.3f} μm "
              f"(sym {rms(P_sym@ey)*1e6:.3f}, anti {rms(P_anti@ey)*1e6:.3f})")
        return rms(P_sym @ ex), rms(P_sym @ ey)

    # ── (A) β-beat gömülü, gürültüsüz kestirim (asıl transparanlık testi) ──
    est_x, est_y = estimate(by, meta)
    print(f"\n=== BBA MERKEZ-KESTİRİM HATALARI (C++; β-beat={args.bbeat*100:.1f}%) ===")
    report_err("β-beat, gürültüsüz", est_x - m_dx, est_y - m_dy)

    # ── (B) BPM ofseti: golden-orbit sürüşünde iptal (nicel gösterim) ──
    est_x_off, est_y_off = estimate(by, meta, offset=bpm_off)
    res_x_off = m_dx - (est_x_off - bpm_off)      # golden-orbit: ofset geri alınır
    res_y_off = m_dy - (est_y_off - bpm_off)
    res_x_clean = m_dx - est_x
    off_cancel = rms(res_x_off - res_x_clean) + rms(res_y_off - (m_dy - est_y))
    print(f"\n=== BPM OFSETİ ({args.bpm_offset*1e6:.0f} μm) ===")
    print(f"  golden-orbit sürüşü sonrası artık ile ofsetsiz artık farkı: "
          f"{off_cancel*1e9:.2f} nm → ofset İPTAL oluyor (Huang ile tutarlı)")

    # ── kalan sahte-EDM: SPİN İZLEYİCİSİYLE (formül değil) ──
    print("\nSpin ölçümleri...")
    f_raw, r1 = fast_measure(m_dx, m_dy)
    f_bb, r2 = fast_measure(m_dx - est_x, m_dy - est_y)
    print(f"  ham |f| = {f_raw:.3e} ({f_raw/1e-9:.0f}× hedef)")
    print(f"  β-beat sonrası |f| = {f_bb:.3e} ({f_bb/1e-9:.2f}× hedef), bastırma "
          f"{f_raw/max(f_bb,1e-30):.0f}×")

    # ── (C) BPM gürültüsü: N_avg taraması (merkez-hata RMS + C++ f çapası) ──
    navgs = [int(x) for x in args.navg.split(",")]
    K = 60
    rng2 = np.random.default_rng(9999)
    noise_out = {}
    print(f"\n=== BPM GÜRÜLTÜSÜ ({args.bpm_noise*1e6:.1f} μm/atış) — N_avg taraması ===")
    for na in navgs:
        sig = args.bpm_noise / np.sqrt(na)         # averaj-sonrası okuma gürültüsü
        sx_list, sy_list = [], []
        ex_last = ey_last = None
        for _ in range(K):
            ex, ey = estimate(by, meta, noise_sigma=sig, offset=bpm_off, rng=rng2)
            # golden-orbit: ofset geri alınır → artık = (est−off) − m
            ex = (ex - bpm_off) - m_dx; ey = (ey - bpm_off) - m_dy
            ex[0] = ey[0] = 0.0
            sx_list.append(rms(P_sym @ ex)); sy_list.append(rms(P_sym @ ey))
            ex_last, ey_last = ex, ey
        # C++ çapa: son gürültülü artıkla doğrudan spin ölçümü
        f_na, _ = fast_measure(-ex_last, -ey_last)
        print(f"  N_avg={na:6d} (σ_read={sig*1e9:.0f} nm): sym-hata dx "
              f"{np.mean(sx_list)*1e6:.3f} / dy {np.mean(sy_list)*1e6:.3f} μm  →  "
              f"C++ |f| = {f_na:.3e} ({f_na/1e-9:.2f}× hedef)")
        noise_out[str(na)] = {"sig": sig, "sym_dx": np.mean(sx_list),
                              "sym_dy": np.mean(sy_list), "f_cpp": f_na}

    print("\n=== ÖZET ===")
    print(f"  β-beat {args.bbeat*100:.0f}% + BPM ofset {args.bpm_offset*1e6:.0f} μm "
          f"(iptal) + gürültü {args.bpm_noise*1e6:.1f} μm:")
    noise_summary = "  ".join(f"[N_avg={n}: {noise_out[str(n)]['f_cpp']/1e-9:.1f}× hedef]"
                              for n in navgs)
    print(f"  gürültüsüz taban {f_bb/1e-9:.2f}× hedef;  gürültü: {noise_summary}")

    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data["bba_full_syst"] = {
        "bbeat": args.bbeat, "bpm_offset": args.bpm_offset,
        "bpm_noise": args.bpm_noise, "seed": args.seed,
        "f_raw": f_raw, "f_bbeat_noiseless": f_bb,
        "sym_dx": rms(P_sym @ (est_x - m_dx)), "sym_dy": rms(P_sym @ (est_y - m_dy)),
        "offset_cancel_nm": off_cancel * 1e9, "noise": noise_out}
    json.dump(data, open(path, "w"), indent=1)
    print(f"  kaydedildi → [bba_full_syst]   [toplam {time.time()-t0:.0f} s]")


if __name__ == "__main__":
    main()

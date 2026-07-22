#!/usr/bin/env python3
"""
make_orbit_figures.py — makale figürleri (İngilizce etiketli)

"Proton EDM Deneyinde Sahte EDM Sinyalinin Yörünge Düzeltmesiyle Bastırılması
ve Sınırları" makalesi (makale_orbit_bastirma.md) için analitik figürler.
C++ GEREKTİRMEZ; per-quad Twiss + kapalı-yörünge makinesi analytic_kmod.py
yapı taşları üzerine kuruludur (C++ ile %1 içinde doğrulanmış,
squid_bpm_test.md §5.5).

Üretilen figürler (İngilizce etiket; PNG'ler .gitignore'da → git add -f):
  fig_orbit_suppression.png : ANA figür — ulaşılabilir sahte-EDM vs σ
                              (belgelenmiş zincir: ham/CW-CCW/orbit-düzeltme;
                              kaynak: omarov.md §10, /tmp scriptleri C++)
  fig_orbit_modes.png       : R'nin SV spektrumu (simetrik içerik renkli)
                              + G_k = C/|Q²-k²| kazanç yasası
  fig_orbit_breathing.png   : dağıtık-frekans per-quad K-mod'un optik-nefesle
                              çöküşü (squid_bpm_test.md §5-6 reprodüksiyonu)
  fig_orbit_lockin.png      : tek-frekans ΔR + lock-in: corr tuzağı ve
                              simetrik-bileşen hatası vs β-beat (§9.5 repro)

Kullanım: python3 make_orbit_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_kmod import (compute_Brho, quad_matrix, drift_matrix,
                           propagate_twiss, phase_step, build_R_analytic)

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

with open("params.json") as f:
    CFG = json.load(f)

nFODO = int(CFG["nFODO"])
NQ    = 2 * nFODO
L_q   = float(CFG["quadLen"])
L_d   = float(CFG["driftLen"])
R0    = float(CFG["R0"])
G_NOM = float(CFG.get("g1", 0.21))
BRHO  = compute_Brho(CFG)
L_def = np.pi * R0 / nFODO
L_mid = 2.0 * L_d + L_def
L_wrap = 2.0 * L_d + L_def


# ─────────────────────────────────────────────────────
# Per-quad Twiss + tepki matrisi (dikey düzlem)
# ─────────────────────────────────────────────────────

def signed_K_vertical(g_arr):
    """Dikey düzlemde işaretli K dizisi: QF defokalize (−), QD fokalize (+)."""
    K = np.zeros(NQ)
    for k in range(nFODO):
        K[2*k]   = -abs(g_arr[2*k])   / BRHO
        K[2*k+1] = +abs(g_arr[2*k+1]) / BRHO
    return K


def twiss_perquad(K):
    """Her quad'ın kendi K'sıyla Twiss (β, φ, Q) — nefes bu yolla kendiliğinden var."""
    mats = []
    for k in range(nFODO):
        mats += [quad_matrix(K[2*k], L_q), drift_matrix(L_mid),
                 quad_matrix(K[2*k+1], L_q), drift_matrix(L_wrap)]
    M = np.eye(2)
    for m in mats:
        M = m @ M
    cos_mu = (M[0, 0] + M[1, 1]) / 2.0
    if abs(cos_mu) >= 1.0:
        raise ValueError("kararsız latis")
    sin_mu = np.sign(M[0, 1]) * np.sqrt(1.0 - cos_mu**2)
    beta0  = M[0, 1] / sin_mu
    alpha0 = (M[0, 0] - M[1, 1]) / (2.0 * sin_mu)

    beta, alpha, phi = beta0, alpha0, 0.0
    beta_arr, phi_arr = np.zeros(NQ), np.zeros(NQ)
    i_el = 0
    for k in range(nFODO):
        for jq in (2*k, 2*k+1):
            beta_arr[jq], phi_arr[jq] = beta, phi          # quad girişi = BPM
            for m in (mats[i_el], mats[i_el+1]):           # quad + takip eden drift
                dphi = phase_step(m, beta, alpha)
                beta, alpha = propagate_twiss(m, beta, alpha)
                phi += dphi
            i_el += 2
    Q = phi / (2.0 * np.pi)
    return beta_arr, phi_arr, Q


def R_perquad(g_arr):
    """Per-quad gradyanlı tepki matrisi R[BPM_i, quad_j] (feed-down kick tabanlı)."""
    K = signed_K_vertical(g_arr)
    beta, phi, Q = twiss_perquad(K)
    KL = K * L_q
    return build_R_analytic(beta, phi, Q, KL), Q


def sym_anti_projectors():
    """Hücre-içi simetrik (QF,QD aynı-işaret) / antisimetrik (zıt) projektörler."""
    P_sym = np.zeros((NQ, NQ))
    P_anti = np.zeros((NQ, NQ))
    for k in range(nFODO):
        a, b = 2*k, 2*k+1
        P_sym[a, a] = P_sym[b, b] = P_sym[a, b] = P_sym[b, a] = 0.5
        P_anti[a, a] = P_anti[b, b] = 0.5
        P_anti[a, b] = P_anti[b, a] = -0.5
    return P_sym, P_anti


# Okabe–Ito renk-körü-güvenli kategorik palet (bilimsel standart)
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#595959"}

plt.rcParams.update({
    # LaTeX-standart görünüm: Computer Modern serif (mathtext "cm" ile birebir);
    # usetex gerektirmez, makale gövde fontuyla aynı aileden görünür.
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    # Tek, tutarlı boyut şeması (figür başına override YOK): eksen-ismi ≥ tik-sayısı
    # hiyerarşisi korunur; hepsi aynı puntoda → PDF'te aynı görünür.
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.7, "lines.linewidth": 1.6, "grid.alpha": 0.22,
    "axes.grid": True, "legend.frameon": False,
})
COL1 = (3.4, 2.75)   # tek-kolon baskı boyutu (fontlar gerçek boyutta render olur)
COL2 = (7.0, 3.0)    # çift-kolon (figure*) baskı boyutu


def panel_label(ax, s):
    """(a)/(b) panel etiketi — her zaman kutunun DIŞINDA, sol-üstte (tutarlı).
    mathtext \\mathbf ile gerçek kalın (cmr10'un bold varyantı yok)."""
    ax.text(0.0, 1.02, rf"$\mathbf{{{s}}}$", transform=ax.transAxes,
            va="bottom", ha="left")


# ═════════════════════════════════════════════════════
# FIG 1 (ANA): ulaşılabilir sahte-EDM vs σ
#   Sayılar: omarov.md §10 (C++, /tmp/cwccw_telafi, orbit_duzeltme):
#   σ=10 μm'de ham≈1000×, +CW/CCW→474×, +orbit-düzeltme→62× hedef; f ∝ σ².
# ═════════════════════════════════════════════════════

# ── C++ izleyiciyle ÖLÇÜLEN ensemble noktaları (× hedef) ──
# RAW: düzeltme yok, 3 seed; kaynak paper_runs_results.json ["sigma"] (kmod_drivers).
RAW_F = {2.5: [67.7, 3.6, 99.1],
         5.0: [271.4, 14.5, 396.6],
         10.0: [1089.0, 58.2, 1587.2]}
# ORBIT-CORR: derin kesik-SVD (rcond=0.01), 3 seed (10,11,12); her σ'da fast_est C++.
# kaynak: /tmp/sigma5_corrected.py, /tmp/sigma20_corrected.py.
MEAS_F = {5.0:  [0.52, 0.62, 1.42],
          10.0: [1.79, 2.72, 5.73],
          20.0: [7.36, 11.09, 23.13]}
# Yörünge düzeltmesi + CW/CCW ENSEMBLE tabanı (σ=10, 5 seed, işaretli);
# medyan 47× [0.6-72] (cwccw_telafi_out.json; tab:chain 2. satır).
ENS_FLOOR_SIGMA, ENS_FLOOR_F = 10.0, 47.0
# BBA + son yörünge düzeltmesi — 5-seed ensemble @σ=10μm (sabit rcond=0.01);
# medyan/min/max × hedef. Kaynak: kmod_drivers/pipeline_multiseed.json.
ENS_BBAOC_SIGMA = 10.0
ENS_BBAOC = [0.068, 0.026, 0.217]       # medyan, min, max (5 seed, hepsi hedef-altı)
# İki-demet (CW/CCW) yörünge düzeltme — kalibre referans, %1 β-beat, 14nm, 5 seed;
# odd artık C=½|f_CW−f_CCW|/target. Kaynak: kmod_drivers/twobeam_oc_14nm_g20.json.
ENS_TWOBEAM_SIGMA = 10.0
ENS_TWOBEAM = [0.043, 0.000, 0.620]     # medyan, min, max (15 seed, hepsi hedef-altı)


def _sig_fit(data):
    """log f = p log σ + c uydur; medyan/min/max döndür."""
    sig = np.array(sorted(data))
    xs = np.array([s for s in sig for _ in data[s]])
    ys = np.array([f for s in sig for f in data[s]])
    p, c = np.polyfit(np.log(xs), np.log(ys), 1)
    med = np.array([np.median(data[s]) for s in sig])
    lo  = np.array([np.min(data[s]) for s in sig])
    hi  = np.array([np.max(data[s]) for s in sig])
    return sig, med, lo, hi, p, c


def fig_suppression():
    r_sig, r_med, r_lo, r_hi, r_p, r_c = _sig_fit(RAW_F)
    m_sig, m_med, m_lo, m_hi, m_p, m_c = _sig_fit(MEAS_F)

    fig, ax = plt.subplots(figsize=COL1)
    ax.set_xscale("log"); ax.set_yscale("log")

    # RAW — ölçülen tek-demet (düzeltme yok); noktalar bağ çizgisiyle (fit değil)
    ax.errorbar(r_sig, r_med, yerr=[r_med - r_lo, r_hi - r_med],
                fmt="o-", color=OI["grey"], ms=6, lw=1.3, capsize=3, zorder=6,
                ecolor=OI["grey"], label="raw")

    # ORBIT-CORR — ölçülen tek-demet
    ax.errorbar(m_sig, m_med, yerr=[m_med - m_lo, m_hi - m_med],
                fmt="s--", color=OI["green"], ms=6, lw=1.3, capsize=3, zorder=6,
                ecolor=OI["green"], label="orbit corr.")

    # BBA + son yörünge düzeltmesi — 5-seed ensemble @σ=10
    em, elo, ehi = ENS_BBAOC
    ax.errorbar([ENS_BBAOC_SIGMA - 0.6], [em], yerr=[[em - elo], [ehi - em]],
                fmt="*", color=OI["purple"], ms=15, capsize=4, zorder=7,
                ecolor=OI["purple"], mec="k", mew=0.4, label="BBA $+$ orbit corr.")

    # İki-demet (CW/CCW) yörünge düzeltmesi
    tm, tlo, thi = ENS_TWOBEAM
    ax.errorbar([ENS_TWOBEAM_SIGMA + 0.9], [tm], yerr=[[tm - tlo], [thi - tm]],
                fmt="D", color=OI["blue"], ms=8, capsize=4, zorder=7,
                ecolor=OI["blue"], mec="k", mew=0.4, label="two-beam orbit corr.")

    ax.axhline(1.0, color=OI["verm"], ls="--", lw=1.1)
    ax.text(1.15, 1.35, "target", color=OI["verm"])

    ax.set_xlabel("rms quad misalignment $\\sigma$  [$\\mu$m]")
    ax.set_ylabel("false EDM  [units of target]")
    sec = ax.secondary_yaxis("right",
                             functions=(lambda v: v * 1e-29, lambda v: v / 1e-29))
    sec.set_ylabel("equivalent EDM  [$e\\!\\cdot\\!$cm]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              fontsize=7.5, columnspacing=1.2, handletextpad=0.4)
    ax.set_xlim(1, 100); ax.set_ylim(1e-2, 3e3)
    fig.savefig("fig_orbit_suppression.png")
    plt.close(fig)
    print(f"fig_orbit_suppression.png yazıldı "
          f"(raw p={r_p:.3f}, orbit-corr p={m_p:.3f})")


# ═════════════════════════════════════════════════════
# FIG 2: mod yapısı — SV spektrumu (simetrik içerik) + G_k yasası
# ═════════════════════════════════════════════════════

def fig_modes():
    g0 = np.full(NQ, G_NOM)
    R, Q = R_perquad(g0)
    U, s, Vt = np.linalg.svd(R)
    P_sym, _ = sym_anti_projectors()
    fsym = np.array([np.linalg.norm(P_sym @ Vt[i]) ** 2 for i in range(NQ)])

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=COL2)  # ax2=(a)gain, ax1=(b)SVD

    # ── (a) kazanç yasası G_k ──
    k = np.arange(1, nFODO + 1)
    C_law, Q2 = 24.8, 5.03
    Gk = C_law / np.abs(Q2 - k.astype(float) ** 2)
    ax2.semilogy(k, Gk, "o-", color=OI["blue"], ms=4)
    ax2.axvline(np.sqrt(Q2), color="k", ls=":", lw=1)
    ax2.text(np.sqrt(Q2) * 1.15, 0.05, "$Q_y$")
    ax2.text(3.5, 9, "antisymmetric\n(low $k$)", color=OI["verm"], fontsize=8.5)
    ax2.text(17, 0.2, "symmetric\n($k\\!\\approx\\!24$)", color=OI["blue"],
             ha="left", fontsize=8.5)
    ax2.set_xlabel("azimuthal harmonic $k$")
    ax2.set_ylabel("orbit gain  $G_k = C/|Q^2-k^2|$")
    panel_label(ax2, "(a)")

    # ── (b) SV spektrumu, simetrik içerik renkli ──
    sc = ax1.scatter(np.arange(NQ), s / s[0], c=fsym, cmap="coolwarm",
                     vmin=0, vmax=1, s=30, edgecolors="k", linewidths=0.3)
    ax1.set_yscale("log")
    ax1.set_xlabel("singular-value index of $R$")
    ax1.set_ylabel("$\\sigma_i / \\sigma_0$")
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("$\\|P_{\\rm sym} v_i\\|^2$")
    ax1.text(6, 0.6, "antisym.\n(visible)", fontsize=8)
    ax1.text(30, 0.01, "symmetric\n(blind)", fontsize=8, ha="left")
    panel_label(ax1, "(b)")

    fig.tight_layout()
    fig.savefig("fig_orbit_modes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"fig_orbit_modes.png yazıldı (cond={s[0]/s[-1]:.0f})")


# ═════════════════════════════════════════════════════
# FIG 3: optik-nefes — dağıtık-frekans per-quad K-mod'un çöküşü
#   squid_bpm_test.md §5-6 reprodüksiyonu (analitik; C++ ile %1 doğrulanmıştı)
# ═════════════════════════════════════════════════════

def fig_breathing(eps=0.02, seed=0):
    rng = np.random.default_rng(seed)
    dy = rng.uniform(-100e-6, 100e-6, NQ)           # ±100 μm kaçıklık
    g0 = np.full(NQ, G_NOM)
    R0m, _ = R_perquad(g0)
    y0 = R0m @ dy                                    # nominal kapalı yörünge

    # Tam (nefes DAHİL) per-quad genlik: quad i'nin g'si %2 artınca TÜM optik değişir
    A_full = np.zeros((NQ, NQ))                      # [BPM, modüle edilen quad]
    C_full = np.zeros((NQ, NQ))                      # kalibrasyon: yalnız dy_i=1
    for i in range(NQ):
        gi = g0.copy(); gi[i] *= (1.0 + eps)
        Ri, _ = R_perquad(gi)
        A_full[:, i] = (Ri - R0m) @ dy
        e = np.zeros(NQ); e[i] = 1.0
        C_full[:, i] = (Ri - R0m) @ e

    # Nefessiz (yalnız feed-down) idealizasyon: optik sabit, yalnız kendi kick'i %2
    A_fd = eps * R0m * dy[None, :]                   # A_fd[b,i] = ε·R0[b,i]·dy_i
    C_fd = eps * R0m

    def recon(A, C):
        num = np.sum(C * A, axis=0)
        den = np.sum(C * C, axis=0)
        return num / den                             # per-quad projeksiyon (48 BPM)

    dy_full_1 = A_full[0] / C_full[0]                # tek BPM
    dy_fd_1   = A_fd[0]   / C_fd[0]
    dy_full_48 = recon(A_full, C_full)
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    c1, cf, c48 = corr(dy_full_1, dy), corr(dy_fd_1, dy), corr(dy_full_48, dy)

    # Kaldıraç ayrıştırması: en duyarlı quad'da tam-dy vs yalnız-kendi-dy tepkisi
    i_big = int(np.argmax(np.abs(A_full[0])))
    gi = g0.copy(); gi[i_big] *= (1.0 + eps)
    Ri, _ = R_perquad(gi)
    e = np.zeros(NQ); e[i_big] = dy[i_big]
    A_own = ((Ri - R0m) @ e)[0]
    A_all = A_full[0, i_big]

    ratio = abs(A_all / A_own)
    fig, ax1 = plt.subplots(figsize=COL1)

    um = 1e6
    ax1.scatter(dy * um, dy_fd_1 * um, s=22, color=OI["blue"], alpha=0.9,
                label="no breathing")
    ax1.scatter(dy * um, dy_full_1 * um, s=26, marker="x", color=OI["verm"],
                label="with breathing")
    lim = 110
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-4 * lim, 450)
    ax1.set_xlabel("true quad offset $dy_j$  [$\\mu$m]")
    ax1.set_ylabel("reconstructed offset  [$\\mu$m]")
    ax1.legend(loc="upper left")

    fig.savefig("fig_orbit_breathing.png", bbox_inches="tight")
    plt.close(fig)
    print(f"fig_orbit_breathing.png yazıldı (corr: fd={cf:+.3f}, 1BPM={c1:+.2f}, "
          f"48BPM={c48:+.2f}, kaldıraç={ratio:.0f}×)")


# ═════════════════════════════════════════════════════
# FIG 4: tek-frekans ΔR + lock-in — corr tuzağı ve β-beat felaketi
#   squid_bpm_test.md §9.5 reprodüksiyonu (30 seed)
# ═════════════════════════════════════════════════════

def fig_lockin(nseed=40, noise_floor=10e-9):
    g1 = np.full(NQ, G_NOM)
    dR_model = R_perquad(g1 * 1.02)[0] - R_perquad(g1)[0]
    P_sym, P_anti = sym_anti_projectors()

    # σ_g = per-quad fraksiyonel GRADYAN hatası; gerçek β-beat ≈ 5.1×σ_g
    # (tam-tur Twiss'ten). Gerçekçi LOCO ~%1-2 β-beat = %0.2-0.4 gradyan hatası.
    sigma_g_list = [0.0, 0.002, 0.004, 0.01]        # → β-beat ≈ 0, 1%, 2%, 5%
    BB_FACTOR = 5.1
    res = {sg: {"corr": [], "sym": [], "anti": []} for sg in sigma_g_list}
    rng = np.random.default_rng(1)
    for sg in sigma_g_list:
        for _ in range(nseed):
            dy = rng.uniform(-100e-6, 100e-6, NQ)
            # kararsız gerçekleme çıkarsa yeniden örnekle (büyük σ_g kuyruğu)
            for _try in range(50):
                delta_g = rng.normal(0.0, sg, NQ) if sg > 0 else np.zeros(NQ)
                gt = g1 * (1.0 + delta_g)                   # gerçek makine (β-beat)
                try:
                    dR_true = R_perquad(gt * 1.02)[0] - R_perquad(gt)[0]
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"σ_g={sg}: 50 denemede kararlı gerçekleme yok")
            meas = dR_true @ dy + rng.normal(0, noise_floor, NQ)
            dy_hat = np.linalg.solve(dR_model, meas)
            err = dy_hat - dy
            res[sg]["corr"].append(np.corrcoef(dy, dy_hat)[0, 1])
            res[sg]["sym"].append(np.sqrt(np.mean((P_sym @ err) ** 2)))
            res[sg]["anti"].append(np.sqrt(np.mean((P_anti @ err) ** 2)))

    sig_sym = np.sqrt(np.mean((P_sym @ np.random.default_rng(2)
                               .uniform(-100e-6, 100e-6, (200, NQ)).T) ** 2))

    bb = np.array([sg * BB_FACTOR * 100 for sg in sigma_g_list])   # β-beat %
    sym_e = [np.mean(res[sg]["sym"]) * 1e6 for sg in sigma_g_list]
    anti_e = [np.mean(res[sg]["anti"]) * 1e6 for sg in sigma_g_list]

    fig, ax = plt.subplots(figsize=COL1)
    ax.set_yscale("log")
    # gerçekçi LOCO bandı (~%1-2 β-beat)
    ax.axvspan(1.0, 2.0, color=OI["green"], alpha=0.10)
    ax.text(1.5, max(sym_e) * 0.75, "LOCO", ha="center", color=OI["green"],
            fontsize=8)
    ax.plot(bb, sym_e, "o-", color=OI["verm"], ms=6, label="symmetric error")
    ax.plot(bb, anti_e, "s-", color=OI["blue"], ms=6, label="antisymmetric error")
    ax.axhline(sig_sym * 1e6, color="k", ls="--", lw=1.1)
    ax.text(bb[-1], sig_sym * 1e6 * 1.18, "symmetric signal", ha="right",
            fontsize=8)
    ax.set_xlabel("$\\beta$-beat  rms $\\Delta\\beta/\\beta$  [%]")
    ax.set_ylabel("reconstruction error  [$\\mu$m]")
    ax.legend(loc="lower right")
    fig.savefig("fig_orbit_lockin.png")
    plt.close(fig)
    print("fig_orbit_lockin.png yazıldı:",
          {f"{sg:g}": (round(np.mean(res[sg]['sym'])*1e6), round(np.mean(res[sg]['corr']), 2))
           for sg in sigma_g_list})


# ═════════════════════════════════════════════════════
# FIG 5 (C++): σ² doğrulaması — kmod_drivers/paper_runs.py sigma çıktısından
# ═════════════════════════════════════════════════════

def fig_sigma():
    import json as _j
    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    with open(path) as fh:
        rows = _j.load(fh)["sigma"]["rows"]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sig = np.array([r["sigma_um"] for r in rows])
    fm = np.array([r["f_mean"] for r in rows])
    for r in rows:
        ax.loglog([r["sigma_um"]] * len(r["f_all"]), r["f_all"], "o",
                  color="tab:blue", alpha=0.45, ms=6)
    ax.loglog(sig, fm, "s-", color="tab:blue", ms=9, lw=2,
              label="ensemble mean (3 seeds)")

    # log-log fit → üs p
    p, b = np.polyfit(np.log(sig), np.log(fm), 1)
    xs = np.linspace(2, 12, 50)
    ax.loglog(xs, np.exp(b) * xs ** p, "--", color="tab:red", lw=1.5,
              label=f"power-law fit:  $f \\propto \\sigma^{{{p:.2f}}}$")

    ax.set_xlabel("rms quadrupole misalignment $\\sigma$  [$\\mu$m]")
    ax.set_ylabel("|false EDM|  $|dS_y/dt|$  [rad/s]")
    ax.set_title("False EDM scales as $\\sigma^2$ — geometric-phase signature\n"
                 "(C++ spin tracking, 4D closed orbit + model-fit estimator)",
                 fontsize=11)
    ax.text(0.03, 0.72, "quadratic scaling = product of two\nmisalignment-driven "
            "spin rotations\n(no linear leakage)", transform=ax.transAxes,
            fontsize=9, style="italic", color="tab:gray")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_orbit_sigma.png")
    plt.close(fig)
    print(f"fig_orbit_sigma.png yazıldı (p={p:.3f})")


# ═════════════════════════════════════════════════════
# FIG (C++): bilineer kanal ayrışımı f_ss/f_sa/f_as/f_aa
#   Kaynak: kmod_drivers/paper_runs_results.json ["channels"]
#   (üretici: fig_channels_gen.py). (a) 3-seed @10μm bar; (b) σ² ölçekleme.
# ═════════════════════════════════════════════════════
def fig_channels():
    import json as _j
    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    with open(path) as fh:
        d = _j.load(fh)["channels"]
    chans = d["channels"]                       # ["ss","sa","as","aa"]
    labels = {"ss": r"$f_{ss}$", "sa": r"$f_{sa}$",
              "as": r"$f_{as}$", "aa": r"$f_{aa}$"}
    # Okabe–Ito renk + kanal başına marker şekli (renk-tek-kodlamadan kaçın)
    CH = {"ss": (OI["blue"], "o"), "sa": (OI["orange"], "s"),
          "as": (OI["verm"], "^"), "aa": (OI["purple"], "D")}
    seeds = d["seeds"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=COL2)

    # ── (a) σ=σ_a: kanal büyüklükleri (seed dağılımı + ort. yatay çizgi) ──
    x = np.arange(len(chans))
    for k, ch in enumerate(chans):
        col, mk = CH[ch]
        pts = [abs(d["a"][ch][i]) for i in range(len(seeds))]
        axa.scatter(np.full(len(pts), x[k]), pts, s=26, color=col, marker=mk,
                    edgecolor="k", linewidth=0.3, zorder=5)
        m = np.mean(pts)
        axa.plot([x[k] - 0.28, x[k] + 0.28], [m, m], "-", color=col, lw=2,
                 zorder=4)
    axa.set_yscale("log")
    axa.set_xticks(x); axa.set_xticklabels([labels[c] for c in chans])
    axa.set_xlim(-0.5, len(chans) - 0.5)
    axa.set_xlabel("channel")
    axa.set_ylabel(r"$|f|$  [units of target]")
    panel_label(axa, "(a)")
    rr = d.get("bilinearity_full_over_sum", [])

    # ── (b) σ² ölçekleme: her kanal ──
    sig = np.array(d["b_sigmas"]) * 1e6         # μm
    for ch in chans:
        col, mk = CH[ch]
        y = np.array([abs(v) for v in d["b"][ch]])
        axb.plot(sig, y, "-", color=col, marker=mk, ms=5, label=labels[ch])
    ybig = np.array([abs(v) for v in d["b"][max(chans, key=lambda c: abs(d["b"][c][-1]))]])
    axb.plot(sig, ybig[-1] * (sig / sig[-1])**2, "k--", lw=1, alpha=0.6,
             label=r"$\propto\sigma^2$")
    axb.set_xscale("log"); axb.set_yscale("log")
    # yatay eksen: bilimsel notasyon yerine düz sayı etiketleri (veri σ'larında)
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    axb.xaxis.set_major_locator(FixedLocator(sig))
    axb.xaxis.set_minor_locator(NullLocator())
    axb.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axb.set_xlabel(r"misalignment rms  $\sigma$  [$\mu$m]")
    axb.set_ylabel(r"$|f|$  [units of target]")
    axb.legend(ncol=2, loc="upper left")
    panel_label(axb, "(b)")

    fig.tight_layout()
    fig.savefig("fig_orbit_channels.png")
    plt.close(fig)
    print(f"fig_orbit_channels.png yazıldı (bilineerlik full/Σ={rr})")


# ═════════════════════════════════════════════════════
# FIG 7 (C++): BBA yakınsaması — simetriği indirir, antisimetrikte tıkanır
#   Kaynak: kmod_drivers/paper_runs_results.json ["bba_iter_cpp"]
#   (ölçülen-matris BBA, %1 β-beat, 3 geçiş; f = C++ spin izleyici).
#   Per-geçiş anti artık kaydedilmediğinden gömülü sayılarla çizilir.
# ═════════════════════════════════════════════════════

# bba_iter_cpp (ölçülen matris, %1 gradyan hatası ≈ %5 β-beat, 5 geçiş);
# f × hedef, artıklar μm. Kaynak: paper_runs_results.json["bba_iter_cpp"].
BBA_PASS   = [1, 2, 3, 4, 5, 6, 7, 8]
BBA_F      = [72.8, 16.7, 28.4, 20.9, 10.5, 4.06, 1.05, 0.094]
BBA_SYM_DX = [4.27, 2.90, 2.29, 1.92, 1.51, 1.25, 1.09, 0.90]
BBA_SYM_DY = [3.27, 1.70, 0.91, 0.50, 0.28, 0.16, 0.10, 0.06]
BBA_F_RAW  = 356.0
# geçiş-5 artığının kanal ayrışımı (× hedef): antisim domine, sim ihmal edilebilir
# (channel_split_out.json: f_anti=7.86×, f_sym=9e-5×)
BBA_P5_FANTI, BBA_P5_FSYM = 7.9, 0.0
# BBA + son yörünge düzeltmesi sonrası — 5-seed ensemble (sabit rcond=0.01)
BBA_OC_LO, BBA_OC_HI = 0.03, 0.22


def fig_bba_convergence():
    fig, ax = plt.subplots(figsize=COL1)
    ax.set_yscale("log")

    # sahte-EDM f (× hedef): ham (pass 0) + geçiş başına
    ax.plot([0], [BBA_F_RAW], "D", color=OI["verm"], ms=8, zorder=6)
    ax.text(0, BBA_F_RAW * 1.7, "raw", color=OI["verm"], va="bottom",
            ha="center", fontsize=8)
    ax.plot(BBA_PASS, BBA_F, "D-", color=OI["verm"], ms=6, lw=1.6,
            label="BBA only")

    # son yörünge düzeltmesi — hızlandırıcı (yıldız)
    ax.plot([3.4], [0.12], "*", color=OI["green"], ms=16, zorder=7,
            mec="k", mew=0.4, label="BBA $+$ final orbit corr.")

    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.text(8.4, 1.3, "target", ha="right", fontsize=8)

    ax.set_xlabel("BBA pass")
    ax.set_ylabel("false EDM  $|f|$  [units of target]")
    ax.set_xticks(BBA_PASS)
    ax.set_xlim(-0.5, 8.7)
    ax.set_ylim(1e-2, 1e3)
    ax.legend(loc="upper right")
    fig.savefig("fig_orbit_bba_convergence.png")
    plt.close(fig)
    print("fig_orbit_bba_convergence.png yazıldı")


if __name__ == "__main__":
    fig_bba_convergence()
    fig_suppression()
    fig_modes()
    fig_breathing()
    fig_lockin()
    for fn in (fig_channels,):
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print(f"{fn.__name__} atlandı ({e})")
    print("\nTüm analitik figürler üretildi. C++ gerektiren figürler için bkz. "
          "makale_orbit_bastirma.md §5 (kmod_drivers/fast_est.py vb.).")

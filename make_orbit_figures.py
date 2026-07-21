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


plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})


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
    sigma = np.logspace(0, 2, 200)          # 1–100 μm

    r_sig, r_med, r_lo, r_hi, r_p, r_c = _sig_fit(RAW_F)
    m_sig, m_med, m_lo, m_hi, m_p, m_c = _sig_fit(MEAS_F)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # RAW — ölçülen tek-demet (düzeltme yok)
    ax.loglog(sigma, np.exp(r_c) * sigma ** r_p, "-", color="#888", lw=1.6,
              zorder=3)
    ax.errorbar(r_sig, r_med, yerr=[r_med - r_lo, r_hi - r_med],
                fmt="o", color="#555", ms=8, capsize=4, zorder=6,
                ecolor="#999",
                label=f"raw (no correction): $p={r_p:.2f}$")

    # ORBIT-CORR — ölçülen tek-demet
    ax.loglog(sigma, np.exp(m_c) * sigma ** m_p, "--", color="tab:green",
              lw=1.6, zorder=3)
    ax.errorbar(m_sig, m_med, yerr=[m_med - m_lo, m_hi - m_med],
                fmt="s", color="tab:green", ms=8, capsize=4, zorder=6,
                ecolor="tab:green",
                label=f"orbit correction only: $p={m_p:.2f}$")

    # BBA + son yörünge düzeltmesi — 5-seed ensemble @σ=10 (tam şema tabanı)
    em, elo, ehi = ENS_BBAOC
    ax.errorbar([ENS_BBAOC_SIGMA - 0.6], [em], yerr=[[em - elo], [ehi - em]],
                fmt="*", color="tab:purple", ms=17, capsize=5, zorder=7,
                ecolor="tab:purple", mec="k", mew=0.5,
                label="BBA $+$ orbit corr. (5 seeds): all sub-target")

    # İki-demet (CW/CCW) yörünge düzeltmesi — kalibre ref, %1 β-beat, 5-seed
    tm, tlo, thi = ENS_TWOBEAM
    ax.errorbar([ENS_TWOBEAM_SIGMA + 0.9], [tm], yerr=[[tm - tlo], [thi - tm]],
                fmt="D", color="tab:blue", ms=11, capsize=5, zorder=7,
                ecolor="tab:blue", mec="k", mew=0.5,
                label="two-beam orbit corr. (calib.\\ ref, 15 seeds): all sub-target")

    ax.axhline(1.0, color="tab:red", ls="--", lw=1.3)
    ax.text(1.1, 1.3, "target ($10^{-29}\\,e\\!\\cdot\\!$cm)",
            color="tab:red", fontsize=9.5)

    ax.set_xlabel("rms quadrupole misalignment $\\sigma$  [μm]")
    ax.set_ylabel("single-beam false EDM  [units of target]")
    sec = ax.secondary_yaxis("right",
                             functions=(lambda v: v * 1e-29, lambda v: v / 1e-29))
    sec.set_ylabel("equivalent EDM  [$e\\!\\cdot\\!$cm]")
    ax.set_title("The false EDM scales as $\\sigma^2$; single-beam correction alone\n"
                 "leaves a symmetric floor, reached by BBA or by two-beam correction",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, title="tracking (median, full spread)")
    ax.set_xlim(1, 100); ax.set_ylim(1e-2, 3e3)
    fig.tight_layout()
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

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.6))  # ax2=sol(gain), ax1=sağ(SVD)

    sc = ax1.scatter(np.arange(NQ), s / s[0], c=fsym, cmap="coolwarm",
                     vmin=0, vmax=1, s=42, edgecolors="k", linewidths=0.4)
    ax1.set_yscale("log")
    ax1.set_xlabel("singular-value index of $R$ (orbit response)")
    ax1.set_ylabel("$\\sigma_i / \\sigma_0$")
    ax1.set_title(f"Orbit response modes  (cond $R$ = {s[0]/s[-1]:.0f})")
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("symmetric content of mode  $\\|P_{sym} v_i\\|^2$")
    ax1.annotate("orbit-VISIBLE:\nantisymmetric (QF/QD opposite)",
                 xy=(4, 0.7), xytext=(10, 0.55), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax1.annotate("orbit-BLIND:\nsymmetric (QF/QD together)",
                 xy=(44, s[-3]/s[0]), xytext=(18, 3.5e-3), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))

    k = np.arange(1, nFODO + 1)
    C_law, Q2 = 24.8, 5.03
    Gk = C_law / np.abs(Q2 - k.astype(float) ** 2)
    ax2.semilogy(k, Gk, "o-", color="tab:green", ms=5)
    ax2.axvline(np.sqrt(Q2), color="k", ls=":", lw=1)
    ax2.text(np.sqrt(Q2) * 1.1, 0.055, "betatron tune\n$Q \\approx 2.3$", fontsize=9)
    ax2.annotate("antisymmetric patterns:\nlow $k$, near resonance → LARGE orbit",
                 xy=(2, Gk[1]), xytext=(6, 12), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax2.annotate("symmetric patterns:\n$k \\approx 24 \\gg Q$ → orbit suppressed",
                 xy=(24, Gk[-1]), xytext=(11, 0.28), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax2.set_xlabel("azimuthal harmonic $k$ of the kick pattern")
    ax2.set_ylabel("orbit gain  $G_k = C/|Q^2 - k^2|$")
    ax2.set_title("Why the symmetric pattern is orbit-blind")

    fig.suptitle("Mode structure of the orbit response (analytic lattice, "
                 f"$Q_y$ = {Q:.2f})", y=1.02, fontsize=12)
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
    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))

    um = 1e6
    ax1.scatter(dy * um, dy_fd_1 * um, s=30, color="tab:green", alpha=0.85,
                label=f"no breathing (feed-down only): corr = {cf:+.3f}")
    ax1.scatter(dy * um, dy_full_48 * um, s=30, marker="s", color="tab:purple",
                alpha=0.7, label=f"with breathing, 48 BPMs: corr = {c48:+.2f}")
    ax1.scatter(dy * um, dy_full_1 * um, s=34, marker="x", color="tab:red",
                label=f"with breathing, 1 BPM: corr = {c1:+.2f}")
    lim = 110
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-4 * lim, 4 * lim)
    ax1.set_xlabel("true quad offset $dy_j$  [μm]")
    ax1.set_ylabel("reconstructed offset  [μm]")
    ax1.set_title("Per-quad AC-modulation readout fails: optics breathing\n"
                  "swamps the feed-down signal ($%.0f\\times$ at the worst quad)"
                  % ratio)
    ax1.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout()
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

    x = np.arange(len(sigma_g_list))
    sym_e = [np.mean(res[sg]["sym"]) * 1e6 for sg in sigma_g_list]
    anti_e = [np.mean(res[sg]["anti"]) * 1e6 for sg in sigma_g_list]
    corrs = [np.mean(res[sg]["corr"]) for sg in sigma_g_list]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    w = 0.36
    ax.bar(x - w/2, sym_e, w, color="tab:red", label="symmetric-component error")
    ax.bar(x + w/2, anti_e, w, color="tab:blue", label="antisymmetric-component error")
    ax.axhline(sig_sym * 1e6, color="k", ls="--", lw=1.2)
    ax.text(-0.42, sig_sym * 1e6 * 1.15,
            f"symmetric signal itself ({sig_sym*1e6:.0f} μm rms)", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sg*BB_FACTOR*100:.0f}%\n($\\sigma_g${sg*100:g}%)"
                        for sg in sigma_g_list])
    ax.set_xlabel("β-beat  (rms $\\Delta\\beta/\\beta$;  gradient error $\\sigma_g$ below)")
    # gerçekçi LOCO bandı (~%1-2 β-beat)
    ax.axvspan(0.5, 2.5, color="tab:green", alpha=0.07)
    ax.text(1.5, ax.get_ylim()[1]*0.5, "realistic\n(LOCO)", ha="center",
            fontsize=8, color="tab:green")
    ax.set_ylabel("reconstruction error  [μm]")
    for xi, c in zip(x, corrs):
        ax.text(xi, ax.get_ylim()[0] * 1.6, f"corr = {c:.2f}", ha="center",
                fontsize=9, color="tab:gray")
    ax.set_title("Single-frequency $\\Delta R$ inversion at the lock-in noise "
                 f"floor ({noise_floor*1e9:.0f} nm):\neven at a realistic 1% "
                 "β-beat the symmetric error exceeds the signal", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.98, 0.80, "correlation stays 'good' while the\nsymmetric component "
            "fails —\ncorr is a misleading metric here",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            style="italic", color="tab:gray")
    fig.tight_layout()
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

    ax.set_xlabel("rms quadrupole misalignment $\\sigma$  [μm]")
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
# FIG 6 (C++): CR-ayrım körlüğü — paper_runs.py crsep çıktısından
# ═════════════════════════════════════════════════════

def fig_crsep():
    import json as _j
    path = os.path.join(BASE, "kmod_drivers", "paper_runs_results.json")
    with open(path) as fh:
        d = _j.load(fh)["crsep"]

    labels = ["single-beam closed orbit\n(ordinary BPM reading)",
              "counter-rotating beam separation\n(CW $-$ CCW, Omarov's observable)"]
    sym_v = [np.mean(d["sym"]["cod_rms"]) * 1e6, np.mean(d["sym"]["cr_rms"]) * 1e6]
    anti_v = [np.mean(d["anti"]["cod_rms"]) * 1e6, np.mean(d["anti"]["cr_rms"]) * 1e6]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w/2, anti_v, w, color="tab:blue",
           label="antisymmetric pattern (orbit-visible)")
    ax.bar(x + w/2, sym_v, w, color="tab:red",
           label="symmetric pattern (drives the false EDM)")
    ax.set_ylim(0, max(anti_v) * 1.3)
    for xi, (a, s) in enumerate(zip(anti_v, sym_v)):
        ax.text(xi, max(a, s) * 1.04, f"suppression {a/s:.1f}×",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("orbit signature, rms  [μm]  (10 μm rms pattern)")
    ax.set_title("The counter-rotating-beam separation shares the orbit's "
                 "blindness\nto the symmetric misalignment pattern "
                 "(C++ tracker, 3 seeds)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.30, 0.60, "similar suppression factors →\nmeasuring and zeroing "
            "the CR separation\ncannot see (or fix) the symmetric\ncomponent "
            "either", transform=ax.transAxes, ha="center",
            fontsize=9, style="italic", color="tab:gray")
    fig.tight_layout()
    fig.savefig("fig_orbit_crsep.png")
    plt.close(fig)
    print("fig_orbit_crsep.png yazıldı")


# ═════════════════════════════════════════════════════
# FIG 3 (C++): bilineer kanal ayrışımı f_ss/f_sa/f_as/f_aa
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
    colors = {"ss": "tab:green", "sa": "tab:orange",
              "as": "tab:red", "aa": "tab:purple"}
    seeds = d["seeds"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # ── (a) σ=σ_a: kanal büyüklükleri (bar=ort |f|, nokta=her seed) ──
    x = np.arange(len(chans))
    means = [np.mean([abs(d["a"][ch][i]) for i in range(len(seeds))]) for ch in chans]
    axa.bar(x, means, 0.6, color=[colors[c] for c in chans], alpha=0.55,
            edgecolor="k", linewidth=0.6)
    for k, ch in enumerate(chans):
        pts = [abs(d["a"][ch][i]) for i in range(len(seeds))]
        axa.scatter(np.full(len(pts), x[k]), pts, s=22, color=colors[ch],
                    edgecolor="k", linewidth=0.4, zorder=5)
    axa.set_yscale("log")
    axa.set_xticks(x); axa.set_xticklabels([labels[c] for c in chans])
    axa.set_ylabel(r"$|f|$  [units of target]")
    axa.set_title(f"(a)  channels at $\\sigma={d['sigma_a']*1e6:.0f}$ μm "
                  "— $f_{ss}$ smallest", fontsize=10)
    axa.grid(True, which="both", axis="y", alpha=0.3)
    # bilineerlik notu
    rr = d.get("bilinearity_full_over_sum", [])
    if rr:
        axa.text(0.02, 0.95, f"$f/\\!\\sum={min(rr):.3f}$–${max(rr):.3f}$",
                 transform=axa.transAxes, fontsize=8, va="top", color="dimgray")

    # ── (b) σ² ölçekleme: her kanal ──
    sig = np.array(d["b_sigmas"]) * 1e6         # μm
    for ch in chans:
        y = np.array([abs(v) for v in d["b"][ch]])
        axb.plot(sig, y, "o-", color=colors[ch], ms=5, label=labels[ch])
    # σ² kılavuz eğrisi (en büyük kanala demirle)
    ybig = np.array([abs(v) for v in d["b"][max(chans, key=lambda c: abs(d["b"][c][-1]))]])
    axb.plot(sig, ybig[-1] * (sig / sig[-1])**2, "k--", lw=1, alpha=0.6,
             label=r"$\propto\sigma^2$")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel(r"misalignment rms  $\sigma$  [μm]")
    axb.set_ylabel(r"$|f|$  [units of target]")
    axb.set_title("(b)  each channel scales as $\\sigma^2$", fontsize=10)
    axb.legend(fontsize=8, ncol=2); axb.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("fig_orbit_channels.png", dpi=150)
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
BBA_PASS   = [1, 2, 3, 4, 5]
BBA_F      = [72.8, 16.7, 28.4, 20.9, 10.5]
BBA_SYM_DX = [4.27, 2.90, 2.29, 1.92, 1.51]
BBA_SYM_DY = [3.27, 1.70, 0.91, 0.50, 0.28]
BBA_F_RAW  = 356.0
# geçiş-5 artığının kanal ayrışımı (× hedef): antisim domine, sim ihmal edilebilir
# (channel_split_out.json: f_anti=7.86×, f_sym=9e-5×)
BBA_P5_FANTI, BBA_P5_FSYM = 7.9, 0.0
# BBA + son yörünge düzeltmesi sonrası — 5-seed ensemble (sabit rcond=0.01)
BBA_OC_LO, BBA_OC_HI = 0.03, 0.22


def fig_bba_convergence():
    fig, ax = plt.subplots(figsize=(7.4, 5.3))

    # sol eksen: simetrik artık (μm) — düzenli düşüş
    ax.set_yscale("log")
    ax.plot(BBA_PASS, BBA_SYM_DX, "o-", color="tab:blue", ms=7,
            label="symmetric residual, horizontal  $|P_{\\rm sym}v_x|$")
    ax.plot(BBA_PASS, BBA_SYM_DY, "s-", color="tab:cyan", ms=7,
            label="symmetric residual, vertical  $|P_{\\rm sym}v_y|$")
    ax.set_xlabel("BBA pass")
    ax.set_ylabel("symmetric residual, rms  [μm]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(BBA_PASS)
    ax.set_ylim(0.2, 12)
    ax.set_xlim(-0.4, 6.0)
    ax.annotate("symmetric residual falls steadily\n"
                "$\\to$ BBA reaches the orbit-blind part",
                xy=(5, BBA_SYM_DY[-1]), xytext=(0.35, 0.35),
                fontsize=9, color="tab:blue", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:blue"))

    # sağ eksen: sahte-EDM f (× hedef) — plato/saçılma
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.plot(BBA_PASS, BBA_F, "D-", color="tab:red", ms=8, lw=2,
             label="false EDM  $|f|$ / target")
    ax2.plot(0, BBA_F_RAW, "D", color="tab:red", ms=8)
    ax2.annotate("raw\n356×", xy=(0, BBA_F_RAW), xytext=(-0.22, 500),
                 fontsize=8.5, color="tab:red", ha="center")
    ax2.set_ylabel("false EDM  $|f|$  [units of target]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(1e-2, 1e4)
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax2.text(5.3, 1.25, "target", fontsize=8.5)
    # ana mesaj: f antisim-artıkla sınırlı + kanal ayrışımı (pass-5)
    ax2.annotate("$|f|$ descends to $\\sim$10$\\times$ but is limited by the\n"
                 "orbit-VISIBLE antisymmetric residue:\n"
                 "pass-5 channel split  $f_{\\rm anti}\\!=\\!7.9\\times \\gg "
                 "f_{\\rm sym}\\!\\approx\\!0$",
                 xy=(5, BBA_F[-1]), xytext=(0.35, 1600),
                 fontsize=9, color="tab:red", ha="left",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
    # son yörünge düzeltmesi: antisimetriği sil → hedef-altı
    ax2.plot([5.5], [0.12], "*", color="tab:green", ms=18, zorder=6)
    ax2.annotate("+ one final orbit\ncorrection (5 seeds)\n"
                 f"$\\to$ {BBA_OC_LO}–{BBA_OC_HI}× target",
                 xy=(5.5, 0.12), xytext=(3.1, 0.02),
                 fontsize=9, color="tab:green", ha="left",
                 arrowprops=dict(arrowstyle="->", color="tab:green"))

    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax2.legend(ln1 + ln2, lb1 + lb2, loc="upper center",
               bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8,
               framealpha=0.95, columnspacing=1.0, handletextpad=0.4)
    ax.set_title("Beam-based alignment drives the symmetric misalignment down; the\n"
                 "false EDM is limited by the orbit-visible antisymmetric residue\n"
                 "(measured-matrix BBA, 1% grad. error $\\approx$5% $\\beta$-beat, 5 passes, C++)",
                 fontsize=10.5)
    fig.subplots_adjust(bottom=0.20)
    fig.tight_layout()
    fig.savefig("fig_orbit_bba_convergence.png", dpi=150)
    plt.close(fig)
    print("fig_orbit_bba_convergence.png yazıldı")


if __name__ == "__main__":
    fig_bba_convergence()
    fig_suppression()
    fig_modes()
    fig_breathing()
    fig_lockin()
    for fn in (fig_sigma, fig_crsep):
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print(f"{fn.__name__} atlandı ({e})")
    print("\nTüm analitik figürler üretildi. C++ gerektiren figürler için bkz. "
          "makale_orbit_bastirma.md §5 (kmod_drivers/fast_est.py vb.).")

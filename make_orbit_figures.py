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

def fig_suppression():
    sigma = np.logspace(0, 2, 200)          # 1–100 μm
    scale = (sigma / 10.0) ** 2
    raw, cwccw, corrected = 1000.0 * scale, 474.0 * scale, 62.0 * scale

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.loglog(sigma, raw, "-", color="#888", lw=2,
              label="raw false EDM (no correction)")
    ax.loglog(sigma, cwccw, "-", color="tab:orange", lw=2,
              label="+ CW/CCW difference (×3.4)")
    ax.loglog(sigma, corrected, "-", color="tab:blue", lw=2.5,
              label="+ orbit correction (×7.7)\n→ symmetric, orbit-blind floor")
    for y, c in ((1000, "#888"), (474, "tab:orange"), (62, "tab:blue")):
        ax.plot(10, y, "o", color=c, ms=7, zorder=5)

    ax.axhline(1.0, color="tab:red", ls="--", lw=1.5)
    ax.text(1.1, 1.25, "target: $10^{-29}\\,e\\!\\cdot\\!$cm  (1 nrad/s)",
            color="tab:red", fontsize=10)
    s_cross = 10.0 / np.sqrt(62.0)
    ax.axvline(s_cross, color="tab:blue", ls=":", lw=1.2)
    ax.annotate(f"$\\sigma_{{sym}} \\approx {s_cross:.1f}\\,\\mu$m needed\n"
                "(not verifiable from the orbit)",
                xy=(s_cross, 1.0), xytext=(1.8, 0.08), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="tab:blue"))
    ax.annotate("simulated points\n($\\sigma$ = 10 μm, C++ tracker)",
                xy=(10, 62), xytext=(22, 8), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k"))

    ax.set_xlabel("rms quadrupole misalignment $\\sigma$  [μm]")
    ax.set_ylabel("false EDM  [units of target, $10^{-29}\\,e\\!\\cdot\\!$cm]")
    sec = ax.secondary_yaxis("right",
                             functions=(lambda v: v * 1e-29, lambda v: v / 1e-29))
    sec.set_ylabel("equivalent EDM  [$e\\!\\cdot\\!$cm]")
    ax.set_title("Achievable false-EDM suppression by orbit-based correction\n"
                 "(false EDM scales as $\\sigma^2$; slope verified: $p = 2.00 \\pm 0.01$)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(1, 100); ax.set_ylim(3e-2, 2e5)
    fig.tight_layout()
    fig.savefig("fig_orbit_suppression.png")
    plt.close(fig)
    print("fig_orbit_suppression.png yazıldı")


# ═════════════════════════════════════════════════════
# FIG 2: mod yapısı — SV spektrumu (simetrik içerik) + G_k yasası
# ═════════════════════════════════════════════════════

def fig_modes():
    g0 = np.full(NQ, G_NOM)
    R, Q = R_perquad(g0)
    U, s, Vt = np.linalg.svd(R)
    P_sym, _ = sym_anti_projectors()
    fsym = np.array([np.linalg.norm(P_sym @ Vt[i]) ** 2 for i in range(NQ)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    um = 1e6
    ax1.scatter(dy * um, dy_fd_1 * um, s=28, color="tab:green", alpha=0.8,
                label=f"no breathing (feed-down only): corr = {cf:+.3f}")
    ax1.scatter(dy * um, dy_full_48 * um, s=28, marker="s", color="tab:purple",
                alpha=0.7, label=f"with breathing, 48 BPMs: corr = {c48:+.2f}")
    ax1.scatter(dy * um, dy_full_1 * um, s=30, marker="x", color="tab:red",
                label=f"with breathing, 1 BPM: corr = {c1:+.2f}")
    lim = 110
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-4 * lim, 4 * lim)
    ax1.set_xlabel("true quad offset $dy_j$  [μm]")
    ax1.set_ylabel("reconstructed offset  [μm]")
    ax1.set_title("Per-quad AC modulation readout fails")
    ax1.legend(fontsize=8.5, loc="upper left")

    ax2.bar([0, 1], [abs(A_all) * um, abs(A_own) * um],
            color=["tab:red", "tab:green"], width=0.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["full machine\n(existing orbit from\nall 48 offsets)",
                         "only own offset\n$dy_i$ (others ideal)"])
    ax2.set_ylabel("demodulated amplitude at BPM  [μm]")
    ax2.set_yscale("log")
    ratio = abs(A_all / A_own)
    ax2.set_title(f"The 'lever': modulating quad {i_big} moves the whole\n"
                  f"pre-existing orbit — {ratio:.0f}× its own signal")
    ax2.text(0.5, 0.62, "optics breathing:\ncoherent — does not\naverage away with\n"
             "more BPMs or SQUIDs", transform=ax2.transAxes,
             ha="center", fontsize=9, color="tab:red")

    fig.suptitle("Distributed-frequency K-modulation: optics breathing dominates "
                 "the per-quad amplitude", y=1.02, fontsize=12)
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

    sigma_g_list = [0.0, 0.005, 0.01, 0.05]
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
    ax.set_xticklabels([f"{sg*100:g}%" for sg in sigma_g_list])
    ax.set_xlabel("gradient / β-beat model error  $\\sigma_g$")
    ax.set_ylabel("reconstruction error  [μm]")
    for xi, c in zip(x, corrs):
        ax.text(xi, ax.get_ylim()[0] * 1.6, f"corr = {c:.2f}", ha="center",
                fontsize=9, color="tab:gray")
    ax.set_title("Single-frequency $\\Delta R$ inversion at the lock-in noise "
                 f"floor ({noise_floor*1e9:.0f} nm):\nβ-beat ≥ 0.5% drives the "
                 "symmetric error above the signal", fontsize=11)
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


if __name__ == "__main__":
    fig_suppression()
    fig_modes()
    fig_breathing()
    fig_lockin()
    try:
        fig_sigma()
    except FileNotFoundError:
        print("fig_sigma atlandı (paper_runs_results.json yok)")
    print("\nTüm analitik figürler üretildi. C++ gerektiren figürler için bkz. "
          "makale_orbit_bastirma.md §5 (kmod_drivers/fast_est.py vb.).")

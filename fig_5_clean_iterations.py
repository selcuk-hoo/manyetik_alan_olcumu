#!/usr/bin/env python3
"""fig_5_clean_iterations.py — ŞEKİL 5: CLEAN iterasyon basamakları.

Üst sol   : artık norm / başlangıç normu vs iterasyon (log ölçek);
            her seçilen k renkli işaretçiyle gösterilir.
Üst orta  : her iterasyonda seçilen k değerinin saçılım grafiği.
Alt panel : artık yörünge profili — başlangıç (tüm modlar), orta (kirleticiler
            soyulmuş), son (artık < 1 μm eşiği).

Çıktı: fig_5_clean_iterations.png
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_reconstruct import fodo_basis, clean_reconstruct

with open("params.json") as f:
    cfg = json.load(f)
N_Q     = 2 * int(cfg["nFODO"])
ANTISYM = cfg.get("smooth_antisym_fodo", True)
R0      = float(cfg.get("R0", 95.49))

R = np.load("R_dy_1.npy")

# ── Gerçek hizalama deseni ────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
TRUTH = {
    1: (30e-6,  0.80),
    2: (10e-6,  1.50),
    3: (25e-6,  0.30),
}
for k in range(4, 11):
    TRUTH[k] = (float(RNG.uniform(100e-6, 300e-6)), float(RNG.uniform(0, 2*math.pi)))

dy = np.zeros(N_Q)
for k, (A, phi) in TRUTH.items():
    F, _ = fodo_basis(N_Q, [k], ANTISYM)
    dy += A * math.cos(phi) * F[:, 0] + A * math.sin(phi) * F[:, 1]

y_meas = R @ dy
candidate_ks = list(range(1, 11))

# ── CLEAN çalıştır — history ile ─────────────────────────────────────────────
# clean_reconstruct history: [(iter, best_k, norm_ratio), ...]
accum, history, F_cache = clean_reconstruct([R], [y_meas], candidate_ks, ANTISYM,
                                            gain=0.2, max_iter=500, tol=1e-6)

iters   = [h[0] for h in history]
ks_seq  = [h[1] for h in history]
norms   = [h[2] for h in history]

# Artık yörüngeyi adım adım yeniden oluştur (snapshot için)
def residual_at_step(n_steps):
    r = y_meas.copy().astype(float)
    M_cache = {}
    for k in candidate_ks:
        F_k, _ = fodo_basis(N_Q, [k], ANTISYM)
        M_cache[k] = R @ F_k
    for it, best_k, _ in history[:n_steps]:
        M_bk = M_cache[best_k]
        a_k, _, _, _ = np.linalg.lstsq(M_bk, r, rcond=None)
        r -= 0.2 * (M_bk @ a_k)
    return r

n_total = len(history)
snap_start = 0
snap_mid   = next(i for i, (_, k, _) in enumerate(history) if k in (1, 2, 3))
snap_end   = n_total

r_start = residual_at_step(snap_start)
r_mid   = residual_at_step(snap_mid)
r_end   = residual_at_step(snap_end)

s_pos = np.arange(N_Q) * (2 * math.pi * R0 / N_Q)

# ── Renk haritası ─────────────────────────────────────────────────────────────
k_colors = {
    1: "tab:blue", 2: "tab:red", 3: "tab:green",
    4: "#9467bd", 5: "#8c564b", 6: "#e377c2",
    7: "#7f7f7f", 8: "#bcbd22", 9: "#17becf", 10: "tab:orange",
}

# ── Grafik ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs  = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.40)

# Üst sol: artık norm vs iterasyon
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(iters, norms, "k-", lw=0.8, alpha=0.5)
for it, k, nr in history:
    ax0.plot(it, nr, "o", color=k_colors[k], ms=4, alpha=0.7)
ax0.axhline(1e-3, color="gray", ls=":", lw=1, label="1‰ eşik")
ax0.set_yscale("log")
ax0.set_xlabel("İterasyon", fontsize=10)
ax0.set_ylabel("Artık norm / başlangıç", fontsize=10)
ax0.set_title("CLEAN yakınsama", fontsize=10)
patches = [mpatches.Patch(color=k_colors[k], label=f"k={k}") for k in candidate_ks]
ax0.legend(handles=patches, fontsize=7, ncol=2, loc="upper right")

# Üst orta: her iterasyonda seçilen k
ax1 = fig.add_subplot(gs[0, 1])
for it, k, _ in history:
    ax1.plot(it, k, "o", color=k_colors[k], ms=3.5, alpha=0.7)
ax1.axvline(snap_mid, color="k", lw=1, ls="--", alpha=0.5,
            label=f"İlk k∈{{1,2,3}} seçimi (adım {snap_mid})")
ax1.set_xlabel("İterasyon", fontsize=10)
ax1.set_ylabel("Seçilen k", fontsize=10)
ax1.set_yticks(candidate_ks)
ax1.set_title("Her adımda seçilen harmonik", fontsize=10)
ax1.legend(fontsize=8)

# Üst sağ: hedef modların birikimli katsayı yakınsaması
ax2 = fig.add_subplot(gs[0, 2])
cum_A = {k: [] for k in [1, 2, 3]}
r_tmp = y_meas.copy().astype(float)
M_cache_tmp = {}
for k in candidate_ks:
    F_k, _ = fodo_basis(N_Q, [k], ANTISYM)
    M_cache_tmp[k] = R @ F_k
running = {k: np.zeros(2) for k in candidate_ks}
for it, best_k, _ in history:
    M_bk = M_cache_tmp[best_k]
    a_k, _, _, _ = np.linalg.lstsq(M_bk, r_tmp, rcond=None)
    r_tmp -= 0.2 * (M_bk @ a_k)
    running[best_k] += 0.2 * a_k
    for kt in [1, 2, 3]:
        cum_A[kt].append(
            math.sqrt(running[kt][0]**2 + running[kt][1]**2) * 1e6
        )
for kt in [1, 2, 3]:
    A_true = TRUTH[kt][0] * 1e6
    ax2.plot(cum_A[kt], color=k_colors[kt], lw=1.5,
             label=fr"$k={kt}$ (gerçek: {A_true:.0f} μm)")
    ax2.axhline(A_true, color=k_colors[kt], lw=0.7, ls="--", alpha=0.5)
ax2.axvline(snap_mid, color="k", lw=1, ls="--", alpha=0.4)
ax2.set_xlabel("İterasyon", fontsize=10)
ax2.set_ylabel("Birikimli genlik [μm]", fontsize=10)
ax2.set_title("k=1,2,3 genlik yakınsaması\n(kesikli: gerçek değer)", fontsize=10)
ax2.legend(fontsize=8)

# Alt: artık yörünge profili üç aşamada
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(s_pos, r_start * 1e6, color="0.5", lw=1.2, alpha=0.9,
         label=f"Başlangıç  (norm={np.linalg.norm(r_start)*1e6:.0f} μm)")
ax3.plot(s_pos, r_mid * 1e6, color="tab:orange", lw=1.5,
         label=f"Adım {snap_mid}: k∈{{1,2,3}} henüz seçilmemiş  "
               f"(norm={np.linalg.norm(r_mid)*1e6:.1f} μm)")
ax3.plot(s_pos, r_end * 1e6, color="tab:blue", lw=2,
         label=f"Son  (norm={np.linalg.norm(r_end)*1e6:.3f} μm)")
ax3.axhline(0, color="k", lw=0.5)
ax3.set_xlabel("Halka konumu s [m]", fontsize=11)
ax3.set_ylabel("Artık yörünge [μm]", fontsize=11)
ax3.set_title("Artık yörünge — CLEAN ilerledikçe kirleticiler soyulur, "
              "k=1,2,3 temiz çıkar", fontsize=10)
ax3.legend(fontsize=9)

fig.suptitle("CLEAN algoritması iterasyon basamakları\n"
             "Büyük kirleticiler (k=4..10, 100–300 μm) önce ayrıştırılır; "
             "k=1,2,3 (10–30 μm) kirletici gölgesinde kalmadan belirlenir.",
             fontsize=11)
fig.savefig("fig_5_clean_iterations.png", dpi=140)
print(f"→ fig_5_clean_iterations.png kaydedildi  ({n_total} iterasyon)")
for kt in [1, 2, 3]:
    ac = accum[kt]
    Afit = math.sqrt(ac[0]**2 + ac[1]**2) * 1e6
    print(f"  k={kt}: A_fit={Afit:.2f} μm  (gerçek: {TRUTH[kt][0]*1e6:.0f} μm)")

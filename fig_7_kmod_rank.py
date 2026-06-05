#!/usr/bin/env python3
"""fig_7_kmod_rank.py — ŞEKİL 7: Kısmi k-modülasyonunun rank kısıtı.

Sol panel: k=2 genlik tahmin hatası (%) vs modüle edilen kuadrupol sayısı N_mod.
           Kirletici modlar mevcut (k=4..10, 100–300 μm); medyan ± IQR.
           Tek yörünge CLEAN referansı (~5%) yatay kesikli çizgi olarak.
Sağ panel: k=2 sinyal yakalama oranı vs N_mod.
           Yalnızca N_mod kuadrupolden gelen k=2 sinyalinin tam sinyale oranı.

Çıktı: fig_7_kmod_rank.png
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_reconstruct import fodo_basis

with open("params.json") as f:
    cfg = json.load(f)
N_Q     = 2 * int(cfg["nFODO"])
ANTISYM = cfg.get("smooth_antisym_fodo", True)

R = np.load("R_dy_1.npy")

# ── Gerçek hizalama deseni (fig_5 ile aynı) ───────────────────────────────────
RNG_TRUTH = np.random.default_rng(42)
TRUTH = {1: (30e-6, 0.80), 2: (10e-6, 1.50), 3: (25e-6, 0.30)}
for k in range(4, 11):
    TRUTH[k] = (float(RNG_TRUTH.uniform(100e-6, 300e-6)),
                float(RNG_TRUTH.uniform(0, 2 * math.pi)))

A_true_k2 = TRUTH[2][0] * 1e6   # 10 μm

F2, _ = fodo_basis(N_Q, [2], ANTISYM)

# k=2 kısmi sinyal yakalama (kirletici yok, sadece geometrik örnekleme etkisi)
def signal_capture(n_mod, n_trials=300, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    phi = TRUTH[2][1]
    dy_k2 = (TRUTH[2][0] * math.cos(phi) * F2[:, 0]
             + TRUTH[2][0] * math.sin(phi) * F2[:, 1])
    y_full = R @ dy_k2
    fracs = []
    for _ in range(n_trials):
        idx = rng.choice(N_Q, n_mod, replace=False)
        y_part = R[:, idx] @ dy_k2[idx]
        fracs.append(np.linalg.norm(y_part) / np.linalg.norm(y_full))
    return np.array(fracs)

# k=2 tahmin hatası (kirleticilerle) ─────────────────────────────────────────
def recovery_error(n_mod, n_trials=200, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    errs = []
    for trial in range(n_trials):
        # Her denemede kirletici fazları rastgele
        rng2 = np.random.default_rng(trial + 1000)
        dy_all = np.zeros(N_Q)
        for k, (A, phi) in TRUTH.items():
            phi2 = float(rng2.uniform(0, 2 * math.pi)) if k >= 4 else phi
            Fk, _ = fodo_basis(N_Q, [k], ANTISYM)
            dy_all += A * math.cos(phi2) * Fk[:, 0] + A * math.sin(phi2) * Fk[:, 1]

        idx = rng.choice(N_Q, n_mod, replace=False)
        y_part = R[:, idx] @ dy_all[idx]
        M_part = R[:, idx] @ F2[idx, :]   # 48×2
        a_est, _, _, _ = np.linalg.lstsq(M_part, y_part, rcond=None)
        A_est = math.sqrt(a_est[0] ** 2 + a_est[1] ** 2) * 1e6
        errs.append(abs(A_est - A_true_k2) / A_true_k2 * 100)
    return np.array(errs)

# ── Hesaplama ─────────────────────────────────────────────────────────────────
print("Hesaplanıyor ...", flush=True)
n_mods = np.arange(1, N_Q + 1)

err_med, err_q25, err_q75 = [], [], []
cap_med, cap_q25, cap_q75 = [], [], []

for n in n_mods:
    errs = recovery_error(n)
    err_med.append(np.median(errs))
    err_q25.append(np.quantile(errs, 0.25))
    err_q75.append(np.quantile(errs, 0.75))

    caps = signal_capture(n)
    cap_med.append(np.median(caps) * 100)
    cap_q25.append(np.quantile(caps, 0.25) * 100)
    cap_q75.append(np.quantile(caps, 0.75) * 100)
    if n % 8 == 0:
        print(f"  N_mod={n}: hata medyan={err_med[-1]:.0f}%  yakalama={cap_med[-1]:.1f}%",
              flush=True)

err_med = np.array(err_med)
err_q25 = np.array(err_q25)
err_q75 = np.array(err_q75)
cap_med = np.array(cap_med)
cap_q25 = np.array(cap_q25)
cap_q75 = np.array(cap_q75)

# ── Grafik ────────────────────────────────────────────────────────────────────
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

# Sol: k=2 tahmin hatası vs N_mod (log ölçek)
ax0.semilogy(n_mods, err_med, "tab:red", lw=2, label="Medyan hata")
ax0.fill_between(n_mods, err_q25, err_q75, color="tab:red", alpha=0.25,
                 label="IQR (%25–%75)")
# Tek yörünge CLEAN referans hatası (~5%)
ax0.axhline(5, color="tab:blue", lw=1.8, ls="--",
            label="Tek yörünge CLEAN (~5%)")
# N_mod=3 gerçek kmod_configs
ax0.axvline(3, color="k", lw=1.2, ls=":", alpha=0.7)
ax0.text(3.4, err_med[2] * 2.5, f"Gerçek k-mod\n$N_\\mathrm{{mod}}=3$\nhata≈{err_med[2]:.0f}%",
         fontsize=8, va="center", color="k")
# N_mod=48 ok birbirini bulur
ax0.annotate(f"Tam halka ($N_{{\\mathrm{{mod}}}}=48$)\nhata≈{err_med[-1]:.0f}%",
             xy=(48, max(err_med[-1], 0.3)),
             xytext=(38, 1.5),
             fontsize=8, ha="right",
             arrowprops=dict(arrowstyle="->", lw=0.8))
ax0.set_xlabel("Modüle edilen kuadrupol sayısı $N_\\mathrm{mod}$", fontsize=11)
ax0.set_ylabel("$|\\hat{A}_{k=2} - A_{k=2}|/A_{k=2}$ [%]", fontsize=11)
ax0.set_title("k=2 genlik tahmin hatası — kısmi k-mod\n"
              "(kirleticiler: $k=4\\ldots10$, $100$–$300\\,\\mu$m mevcut)",
              fontsize=10)
ax0.set_xlim(0, 49)
ax0.set_ylim(0.1, 3e4)
ax0.legend(fontsize=9)
ax0.grid(True, which="both", alpha=0.25)

# Sağ: k=2 sinyal yakalama oranı
ax1.plot(n_mods, cap_med, "tab:green", lw=2, label="Medyan yakalama")
ax1.fill_between(n_mods, cap_q25, cap_q75, color="tab:green", alpha=0.25,
                 label="IQR (%25–%75)")
ax1.axhline(100, color="k", lw=0.8, ls="--", alpha=0.5, label="Tam sinyal (%100)")
ax1.axvline(3, color="k", lw=1.2, ls=":", alpha=0.7)
ax1.text(3.5, 55, f"$N_\\mathrm{{mod}}=3$\n≈{cap_med[2]:.0f}% yakalama",
         fontsize=8, va="center")
ax1.set_xlabel("Modüle edilen kuadrupol sayısı $N_\\mathrm{mod}$", fontsize=11)
ax1.set_ylabel("$k=2$ sinyal yakalama oranı [%]", fontsize=11)
ax1.set_title("$k=2$ sinyal yakalama — geometrik örnekleme etkisi\n"
              r"$\|R_{:,\mathrm{sel}}\,F_{k=2,\mathrm{sel}}\|\,/\,\|R\,F_{k=2}\|$",
              fontsize=10)
ax1.set_xlim(0, 49)
ax1.set_ylim(0, 110)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

fig.suptitle("Kısmi k-modülasyonunun rank kısıtı: $N_\\mathrm{mod}<48$ kuadrupol modülasyonu\n"
             "k=2 sinyalinin yalnızca $N_\\mathrm{mod}/48$ fraksiyonunu örnekler; "
             "kirleticiler varlığında tahmin hatası $>1000\\%$'e fırlar.",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig_7_kmod_rank.png", dpi=140)
print("→ fig_7_kmod_rank.png kaydedildi")
print(f"  N_mod=3  (gerçek kmod_configs): hata medyan={err_med[2]:.0f}%  "
      f"sinyal yakalama={cap_med[2]:.1f}%")
print(f"  N_mod=48 (tam halka):           hata medyan={err_med[-1]:.1f}%  "
      f"sinyal yakalama={cap_med[-1]:.1f}%")
print(f"  Kıyaslama — tek yörünge CLEAN:  hata ≈5%  (gradyan modülasyonu yok)")

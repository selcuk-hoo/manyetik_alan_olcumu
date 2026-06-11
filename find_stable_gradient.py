#!/usr/bin/env python3
"""find_stable_gradient.py — Yeni topoloji için kararlı quad gradyan aralığı.

Tek hücre transfer matrisi (dikey düzlem, ince mercek yaklaşımı):

  Yeni topoloji:  QF → drift(d) → QD → drift(d) → ARC(2*L_def) → drift(2*d)
  Orijinal FODO:  ARC(L_def) → drift(d) → QF → drift(d) →
                  ARC(L_def) → drift(d) → QD → drift(d)

Kararlılık koşulu: |Tr(M_hücre)| < 2

Çıktı:
  - Her iki topoloji için kararlı g aralığı
  - Orijinal tasarım g₀=0.2 T/m'yi kararlı kılan yeni g değeri
  - Betatron faz ilerlemesi μ (hedef: 0 < μ < π)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Lattice parametreleri ────────────────────────────────────────────────────
R0      = 95.49        # m
nFODO   = 24
d       = 2.0833       # drift uzunluğu [m]
L_q     = 0.4          # quad uzunluğu [m]
g0      = 0.2          # mevcut gradyan [T/m]

# Fizik sabitleri (frozen-spin proton)
M_P_GEV = 0.938272046
G_P     = 1.792847356
p_magic = M_P_GEV / np.sqrt(G_P)   # GeV/c
C_LIGHT = 299792458.0
Q_E     = 1.602176565e-19
M_P_KG  = 1.672621777e-27
p_SI    = p_magic * 1e9 * Q_E / C_LIGHT   # kg·m/s (rMv = p/q)
Brho    = p_SI / Q_E   # manyetik rijidite [T·m]

# Arc uzunlukları
L_def = np.pi * R0 / nFODO          # yarı-hücre arc = π*R0/nFODO [m]
L_arc_per_cell = 2 * L_def          # tek deflektör arc uzunluğu [m]
print(f"L_def = {L_def:.3f} m,  2*L_def = {L_arc_per_cell:.3f} m")
print(f"Manyetik rijidite Bρ = {Brho:.4f} T·m")
print(f"Orijinal gradyan g0 = {g0} T/m  →  f = Bρ/(g0*L_q) = {Brho/(g0*L_q):.2f} m")


def transfer_matrix_thin(g, topo="yeni"):
    """Tek FODO hücresi transfer matrisi (dikey, ince mercek).

    g > 0: QF dikey odaklama, QD dikey defocusing
    topo = "yeni": QF-drift-QD-drift-ARC-drift
    topo = "orijinal": ARC-drift-QF-drift-ARC-drift-QD-drift
    """
    f = Brho / (g * L_q)   # ince mercek odak uzaklığı [m]

    def drift(l):
        return np.array([[1, l], [0, 1]])

    def qf():  # dikey odaklama
        return np.array([[1, 0], [-1/f, 1]])

    def qd():  # dikey defocusing
        return np.array([[1, 0], [+1/f, 1]])

    if topo == "yeni":
        # QF → drift(d) → QD → drift(d) → ARC(2*L_def) → drift(2*d)
        M = drift(2*d) @ drift(L_arc_per_cell) @ drift(d) @ qd() @ drift(d) @ qf()
    else:
        # ARC(L_def) → drift(d) → QF → drift(d) → ARC(L_def) → drift(d) → QD → drift(d)
        M = drift(d) @ qd() @ drift(d) @ drift(L_def) @ drift(d) @ qf() @ drift(d) @ drift(L_def)
    return M


# ── Gradyan taraması ─────────────────────────────────────────────────────────
g_vals = np.linspace(0.01, 2.0, 5000)
trace_yeni     = []
trace_orijinal = []

for g in g_vals:
    M_y = transfer_matrix_thin(g, "yeni")
    M_o = transfer_matrix_thin(g, "orijinal")
    trace_yeni.append(np.trace(M_y))
    trace_orijinal.append(np.trace(M_o))

trace_yeni     = np.array(trace_yeni)
trace_orijinal = np.array(trace_orijinal)

# Kararlılık: |Tr| < 2
stable_yeni     = np.abs(trace_yeni) < 2.0
stable_orijinal = np.abs(trace_orijinal) < 2.0

print("\n── Orijinal FODO kararlı aralık ──────────────────────────────")
g_orig_stable = g_vals[stable_orijinal]
if len(g_orig_stable):
    print(f"  g ∈ [{g_orig_stable[0]:.4f}, {g_orig_stable[-1]:.4f}] T/m")
    print(f"  Mevcut g0={g0} T/m kararlı: {stable_orijinal[np.argmin(np.abs(g_vals-g0))]}")
    M_orig = transfer_matrix_thin(g0, "orijinal")
    mu_orig = np.arccos(np.trace(M_orig)/2) * 180/np.pi
    print(f"  g0={g0}: Tr={np.trace(M_orig):.4f}, μ={mu_orig:.1f}°, "
          f"Q={mu_orig*nFODO/360:.3f}")

print("\n── Yeni topoloji kararlı aralık ──────────────────────────────")
g_yeni_stable = g_vals[stable_yeni]
if len(g_yeni_stable):
    print(f"  g ∈ [{g_yeni_stable[0]:.4f}, {g_yeni_stable[-1]:.4f}] T/m")
    # Orijinal ile aynı tonu veren g bul
    target_trace = np.trace(transfer_matrix_thin(g0, "orijinal"))
    idx = np.argmin(np.abs(trace_yeni - target_trace))
    if stable_yeni[idx]:
        g_match = g_vals[idx]
        M_match = transfer_matrix_thin(g_match, "yeni")
        mu_match = np.arccos(np.trace(M_match)/2) * 180/np.pi
        print(f"  Orijinal tonu (Tr={target_trace:.4f}) veren g = {g_match:.4f} T/m")
        print(f"  μ={mu_match:.1f}°, Q={mu_match*nFODO/360:.3f}")
else:
    print("  Kararlı bölge bulunamadı (g < 2 T/m'de)")

# Özel noktalar: ton hedefleri
print("\n── Hedef ton Q_y ≈ 2.68 için g değerleri ────────────────────")
Q_target = 2.68
mu_target = Q_target * 360 / nFODO  # derece
trace_target = 2 * np.cos(np.radians(mu_target))
print(f"  Q={Q_target} → μ={mu_target:.2f}° → Tr_hedef={trace_target:.4f}")

for topo, traces, label in [("orijinal", trace_orijinal, "Orijinal"),
                              ("yeni",     trace_yeni,     "Yeni topoloji")]:
    diffs = np.abs(traces - trace_target)
    idx = np.argmin(diffs)
    if abs(traces[idx] - trace_target) < 0.05 and stable_orijinal[idx] if topo=="orijinal" else stable_yeni[idx]:
        g_q = g_vals[idx]
        print(f"  {label}: g = {g_q:.4f} T/m  (Tr={traces[idx]:.4f})")
    else:
        print(f"  {label}: Q_y={Q_target} bu topolojide bu g aralığında bulunamadı")

# ── Figür ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, traces, label, color in [
        (axes[0], trace_orijinal, "Orijinal FODO", "C0"),
        (axes[1], trace_yeni,     "Yeni (QF-d-QD-d-ARC-d)", "C1")]:
    ax.plot(g_vals, traces, color, lw=1.5, label="Tr(M)")
    ax.axhline( 2, color="k", ls="--", lw=1, label="|Tr|=2 sınırı")
    ax.axhline(-2, color="k", ls="--", lw=1)
    ax.fill_between(g_vals, -2, 2, where=np.abs(traces)<2,
                    alpha=0.18, color="g", label="Kararlı bölge")
    ax.axvline(g0, color="r", ls=":", lw=1.5, label=f"g₀={g0} T/m")
    ax.set_xlabel("Quad gradyanı g [T/m]")
    ax.set_ylabel("Tr(M_hücre)")
    ax.set_ylim(-4, 4)
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # İkinci eksen: ton
    ax2 = ax.twinx()
    with np.errstate(invalid="ignore"):
        mu_deg = np.where(np.abs(traces) < 2,
                          np.degrees(np.arccos(np.clip(traces/2, -1, 1))), np.nan)
    q_vals = mu_deg * nFODO / 360
    ax2.plot(g_vals, q_vals, color, lw=1, ls="-.", alpha=0.6)
    ax2.axhline(2.68, color="purple", ls=":", lw=1, label="Q=2.68")
    ax2.set_ylabel("Q_y (ton)", color=color)
    ax2.set_ylim(0, 12)
    ax2.legend(fontsize=7, loc="lower right")

plt.suptitle("Tek hücre kararlılık analizi: Orijinal FODO vs Yeni topoloji\n"
             f"(ince mercek, d={d} m, L_q={L_q} m, L_arc={L_arc_per_cell:.2f} m)")
plt.tight_layout()
plt.savefig("find_stable_gradient.png", dpi=150)
print("\nFigür: find_stable_gradient.png")

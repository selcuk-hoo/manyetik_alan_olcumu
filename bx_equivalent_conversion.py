#!/usr/bin/env python3
"""
bx_equivalent_conversion.py  —  Quad kaçıklığı ↔ EŞDEĞER harici radyal B_x [nT]

DOĞRU YÖNTEM (spin-takibi tabanlı):
  Lokal quad alanı (G·δy ≈ 210 nT/μm) YANLIŞ ölçektir; çünkü quadlar halkanın
  yalnızca ~%3'ünü doldurur ve spin entegre alana tepki verir, lokal pike değil.

  İki bağımsız spin-takibi taraması doğrudan dönüşümü verir:
    (A) Harici radyal harmonik:  B_x(θ)=A_r cos(Nθ), A_r=1 nT  → dSy/dt_ext(N)
        [compare_field_harmonics.py — Omarov Fig.8 analoğu]
    (B) Quad kaçıklığı modu:     Δy_j = A·F_k[j], A=10 μm     → dSy/dt_quad(k)
        [false_edm_mode_scan.py]

  Her ikisi de AYNI gözlemleneni (dSy/dt) üretir. Spin-eşdeğer dönüşüm:
        c_k [nT/μm] = R_q(k) / R_B(k)
    R_q(k) = dSy/dt_quad(k) / A_quad   [rad/s per μm kaçıklık]
    R_B(k) = |dSy/dt_ext(k)| / A_r     [rad/s per nT harici alan]

  Bu, "k modlu kaçıklığın yarattığı sahte-EDM, kaç nT'lik harici k-harmonik
  radyal alana eşdeğer?" sorusunu spin-takibi gerçeğiyle yanıtlar.
"""
import numpy as np, os, sys, json, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Simülasyon çıktıları (Mac'te çalıştırılan koşulardan) ─────────────────────
# (A) Harici B_x=1 nT harmonik → dSy/dt [rad/s]   (compare_field_harmonics.py)
ext_field = {0: -1.546e-08, 1: -9.961e-12, 2: -2.639e-11, 3: +1.153e-11,
             4: +3.291e-12, 5: +1.636e-12, 6: +9.843e-13, 7: +5.542e-13,
             8: +2.764e-13, 9: +1.940e-13, 10: +1.003e-13, 11: +5.196e-14,
             12: +3.535e-14}
A_r_nT = 1.0   # harici alan genliği [nT]

# (B) Quad kaçıklığı A=10 μm → dSy/dt_strobe [rad/s]  (false_edm_mode_scan.py)
#     (yalnızca güvenilir 'strobe' sütunu; SG sütunu örnekleme-bağımlı)
quad_mis = {0: 1.798e-10, 1: 4.928e-10, 2: 1.437e-9, 3: -6.635e-10,
            4: -1.975e-10, 5: -9.920e-11, 6: -5.923e-11, 7: -4.728e-11}
A_quad_um = 10.0   # kaçıklık genliği [μm]

# JSON varsa üzerine yaz (Mac çıktısı senkronlanırsa)
if os.path.exists("field_harmonic_results.json"):
    try:
        with open("field_harmonic_results.json") as f:
            d = json.load(f)
        if "dSy_dt" in d and "N" in d:
            ext_field = {int(n): v for n, v in zip(d["N"], d["dSy_dt"])}
            A_r_nT = d.get("Br_nT", A_r_nT)
            print("(JSON'dan harici alan sonuçları yüklendi)")
    except Exception as e:
        print(f"(JSON okunamadı: {e})")

# ── Spin yanıt katsayıları ────────────────────────────────────────────────────
print("=" * 72)
print("SPIN YANIT KATSAYILARI (doğrudan takipten)")
print()
print(f"  {'k/N':>4}  {'R_B [rad/s/nT]':>16}  {'R_q [rad/s/μm]':>16}  "
      f"{'c_k = R_q/R_B [nT/μm]':>22}")
print("  " + "-"*64)

c_k = {}
for k in sorted(quad_mis):
    if k not in ext_field or ext_field[k] == 0:
        continue
    R_B = abs(ext_field[k]) / A_r_nT          # rad/s per nT
    R_q = abs(quad_mis[k]) / A_quad_um        # rad/s per μm
    c_k[k] = R_q / R_B                        # nT per μm
    star = "  ← N≈Qy" if k == 2 else ""
    print(f"  {k:>4}  {R_B:>16.3e}  {R_q:>16.3e}  {c_k[k]:>22.2f}{star}")

c_vals = np.array([c_k[k] for k in c_k if k >= 1])
print()
print(f"  c_k (k≥1) neredeyse SABİT: ortalama = {c_vals.mean():.2f} nT/μm, "
      f"std = {c_vals.std():.2f} ({c_vals.std()/c_vals.mean()*100:.0f}%)")
print(f"  Karşılaştırma — LOKAL quad alanı G·δy = 0.21 T/m × 1μm = 210 nT/μm")
print(f"  Oran: {210/c_vals.mean():.0f}×  (≈ 1/doluluk_oranı, %3.2 → 31×)")
print(f"  → Lokal pik DEĞİL, entegre (spin-eşdeğer) alan doğru ölçektir.")

# ── Ölçüm hassasiyeti: σ(A_k) → eşdeğer B_x ──────────────────────────────────
# σ(A_k) BPM ofsetinden (mode_tolerance_analysis.py, Monte Carlo, σ_off=100μm)
sigma_Ak_at_100um = {2: 0.857, 3: 3.263}   # μm  (k modu için 100μm ofset → σ(A_k))
# Doğrusal ölçekleme: σ(A_k) ∝ σ_offset
slope_sigma = {k: v/100.0 for k, v in sigma_Ak_at_100um.items()}  # μm(A) per μm(offset)

print()
print("=" * 72)
print("ÖLÇÜM HASSASİYETİ — eşdeğer harici B_x [nT]")
print("  Zincir: BPM ofset → yörünge → kaçıklık σ(A_k) → spin-eşdeğer B_x")
print()
print(f"  {'σ_offset':>10}  "
      f"{'k=2: σ(A) [μm]':>15}  {'B_x,eq [nT]':>12}  "
      f"{'k=3: σ(A) [μm]':>15}  {'B_x,eq [nT]':>12}")
print("  " + "-"*70)
for sig_off in [5, 10, 20, 50, 100, 200]:
    sA2 = slope_sigma[2] * sig_off
    sA3 = slope_sigma[3] * sig_off
    Bx2 = c_k[2] * sA2
    Bx3 = c_k[3] * sA3
    print(f"  {sig_off:>7} μm  {sA2:>15.3f}  {Bx2:>12.2f}  "
          f"{sA3:>15.3f}  {Bx3:>12.2f}")

# 1 nT için gereken σ_offset
print()
print("  1 nT eşdeğer hassasiyet için gereken BPM sistematik ofseti:")
for k in [2, 3]:
    # B_x = c_k × slope_sigma[k] × σ_off = 1 nT
    sig_off_1nT = 1.0 / (c_k[k] * slope_sigma[k])
    print(f"    k={k}:  σ_offset ≲ {sig_off_1nT:.1f} μm")

# ── Sahte-EDM (rad/s) cinsinden — doğrudan R_q ile (kappa proxy DEĞİL) ────────
print()
print("=" * 72)
print("SAHTE-EDM ARTIK BELİRSİZLİĞİ (rad/s) — doğrudan R_q(k) ile")
print("  σ(dSy/dt) = R_q(k) × σ(A_k)   [spin-takibi gerçeği]")
print()
print(f"  {'σ_offset':>10}  {'k=2 dSy/dt [rad/s]':>20}  {'k=3 dSy/dt [rad/s]':>20}")
print("  " + "-"*54)
for sig_off in [10, 20, 50, 100, 200]:
    sA2 = slope_sigma[2] * sig_off * 1e-6   # m
    sA3 = slope_sigma[3] * sig_off * 1e-6
    Rq2 = abs(quad_mis[2]) / (A_quad_um*1e-6)   # rad/s per m
    Rq3 = abs(quad_mis[3]) / (A_quad_um*1e-6)
    print(f"  {sig_off:>7} μm  {Rq2*sA2:>20.3e}  {Rq3*sA3:>20.3e}")

print()
print("  NOT: Bu değerler önceki mode_tolerance (kappa×Mk proxy) sonuçlarından")
print(f"  ~{(7e-6*166.9)/(abs(quad_mis[2])/(A_quad_um*1e-6)):.0f}× daha düşük; çünkü "
      f"doğrudan takip R_q kullanıldı.")

# ── Modların spin-eşdeğer alan profili (gerçekçi amplitüdler) ─────────────────
print()
print("=" * 72)
print("SİNYAL MODLARININ SPİN-EŞDEĞER HARİCİ B_x GENLİĞİ")
print()
amps_um = {2: 10.0, 3: 8.0}   # sinyal kaçıklık genlikleri
for k, A in amps_um.items():
    print(f"  k={k}: {A:.0f} μm kaçıklık  →  spin-eşdeğer {c_k[k]*A:.1f} nT "
          f"harici {k}-harmonik radyal alan")
print()
print(f"  (Lokal quad piki bu modlar için ~{0.21*10*1e3:.0f}-"
      f"{0.21*8*1e3:.0f} nT olurdu — ama spin entegre alana tepki verir.)")

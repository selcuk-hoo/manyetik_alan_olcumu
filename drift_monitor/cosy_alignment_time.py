#!/usr/bin/env python3
"""
cosy_alignment_time.py — Spin-tabanlı hizalama (radyal polarizasyon) ölçümü,
hedef sensitiviteye NE KADAR SÜREDE ulaşır? COSY polarimetre sayılarıyla.

Formül (cosy_polarimeter.md §4.3): sekuler hız (dS_y/dt) çözünürlüğü
    sigma = sqrt(12) / (A * P * sqrt(N) * T),   N = Ndot * T
 => sigma = sqrt(12) / (A * P * sqrt(Ndot) * T^{3/2})
 => hedef sigma için:  T = [ sqrt(12) / (A*P*sqrt(Ndot)*sigma) ]^{2/3}

GEÇERLİLİK: tek sürekli ölçüm; T koherans süresini (~1000 s) aşmamalı.
"""
import numpy as np

# --- COSY polarimetre parametreleri (cosy_polarimeter.md) ---
A    = 0.6        # etkin analiz gücü A_y
P    = 0.8        # demet polarizasyonu
Ndot = 7e4        # analiz edilen olay hızı [1/s] (dolum başı 7e7 olay / 1000 s)
T_coh = 1000.0    # spin-koherans (dolum) süresi [s]

C = np.sqrt(12) / (A * P * np.sqrt(Ndot))    # sigma = C / T^{3/2}

def T_for_sigma(sigma):
    return (C / sigma) ** (2.0/3.0)

print("sigma = %.4g / T^1.5   (T saniye, sigma rad/s)\n" % C)
print(f"{'hedef sigma [rad/s]':>22} {'gereken T':>14} {'koherans-içi mi?':>16}")
for sig in [1e-5, 3e-6, 1e-6, 1e-7, 1e-9]:
    T = T_for_sigma(sig)
    if T < 60:           tstr = f"{T:.1f} s"
    elif T < 3600:       tstr = f"{T/60:.1f} dk"
    elif T < 86400:      tstr = f"{T/3600:.1f} sa"
    else:                tstr = f"{T/86400:.1f} gün"
    inside = "EVET" if T <= T_coh else "HAYIR (çok-dolum)"
    print(f"{sig:>22.0e} {tstr:>14} {inside:>16}")

print("\nSağlama (T=1000s, EDM tek-dolum): sigma = %.2e rad/s  (cosy.md ~9e-7 ✓)"
      % (C/1000**1.5))

#!/usr/bin/env python3
"""test_realistic_beam.py — Test 8: Gerçek demet (büyük rastgele açılar) ile iptal.

Soru (kullanıcı):
  Gerçekte her parçacığın BÜYÜK rastgele injection açısı var (betatron ~mm-cm),
  iki demet AYRI dolumlar. Tek parçacık mirror eşi idealize. Gerçek demet
  ortalamasında (CW,n)−(CCW,f) sahte EDM'yi hâlâ iptal eder mi?

Fizik:
  Tek parçacık: f_CW(y,+y') = f_CCWf(y,−y') (mirror eşi → birebir aynı).
  Demet ortalaması: ⟨f⟩ simetrik dağılımda, eğer CW ve CCW dağılımları
  birbirinin aynasıysa eşit → farkta iptal. Ayrıca betatron sahte EDM'si
  faz üzerinden kısmen ortalanır. Pratik artık: istatistik (sonlu N) +
  dağılım uyumsuzluğu.

Yöntem (üniform 0.2 lattice, aynı misalignment):
  N parçacık, Gauss (σ_y, σ_y') ~ birkaç mm betatron. İki senaryo:
   (i) Bağımsız dolum: CW ve CCW flip ayrı rastgele örnekler (gerçekçi).
   (ii) Eşlenmiş (mirror): CCW flip = CW'nin (y, −y') aynası (ideal limit).
  Her demet için ⟨dSy/dt⟩ ve fark.

Çıktı:
  test_realistic_beam.png
  Terminal tablo
"""

import json, os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REF_SEED   = 7
A_MISALIGN = 1e-5
T2         = 5e-4
RSTEPS     = 3000
N_PART     = 20          # demet başına parçacık sayısı
SIGMA_Y    = 2e-3        # dikey konum RMS [m] = 2 mm
SIGMA_YP   = 66e-6       # dikey açı RMS [rad] ≈ σ_y/β (β≈30m) → betatron ~mm
BEAM_SEED  = 2024


def _suppress():
    fd = os.dup(1); n = os.open(os.devnull, os.O_WRONLY); os.dup2(n, 1); os.close(n); return fd
def _restore(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    label, direction, flip, yv, ypv, dy_list, t2, rsteps = task
    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields, _make_state
    from integrator import integrate_particle
    with open("params.json") as fh:
        config = json.load(fh)
    fields, _, beta0, R0, p_mag, _ = setup_fields(config)
    if flip:
        fields.quadG1 = -fields.quadG1; fields.quadG0 = -fields.quadG0
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, float)
    y_launch = _make_state([0.0, float(yv), 0.0, float(ypv)],
                           p_mag, float(direction), [0.0, 0.0, float(direction)])
    saved = _suppress()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(y_launch, 0.0, t2, dt, fields=fields,
                                              return_steps=rsteps, quad_dy=dy)
    finally:
        _restore(saved)
    return label, float(np.polyfit(np.asarray(poin_t, float),
                                   np.asarray(poin[:, 7], float), 1)[0])


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q = 2*int(fields.nFODO); direction = fields.direction
    rng_m = np.random.default_rng(REF_SEED)
    dy_ref = (rng_m.standard_normal(n_q) * A_MISALIGN).tolist()

    # Demet örnekleri
    rng = np.random.default_rng(BEAM_SEED)
    cw_y   = rng.normal(0, SIGMA_Y,  N_PART)
    cw_yp  = rng.normal(0, SIGMA_YP, N_PART)
    ccw_y  = rng.normal(0, SIGMA_Y,  N_PART)   # bağımsız dolum
    ccw_yp = rng.normal(0, SIGMA_YP, N_PART)

    print(f"Lattice üniform {fields.quadG1} T/m, aynı misalignment (RMS=10μm)")
    print(f"Demet: N={N_PART}/demet, σ_y={SIGMA_Y*1e3:.1f}mm, σ_y'={SIGMA_YP*1e6:.0f}μrad")

    tasks = []
    for i in range(N_PART):
        tasks.append((f"cw_{i}",   direction,  False, cw_y[i],  cw_yp[i],
                      dy_ref, T2, RSTEPS))
        # (i) bağımsız CCW flip
        tasks.append((f"ccwI_{i}", -direction, True,  ccw_y[i], ccw_yp[i],
                      dy_ref, T2, RSTEPS))
        # (ii) eşlenmiş CCW flip = CW'nin mirror'ı (y, −y')
        tasks.append((f"ccwM_{i}", -direction, True,  cw_y[i],  -cw_yp[i],
                      dy_ref, T2, RSTEPS))

    nw = mp.cpu_count()
    print(f"\n{len(tasks)} simülasyon ({nw} işçi)...")
    with mp.get_context("spawn").Pool(processes=nw) as pool:
        res = dict(pool.map(_worker, tasks))

    cw   = np.array([res[f"cw_{i}"]   for i in range(N_PART)])
    ccwI = np.array([res[f"ccwI_{i}"] for i in range(N_PART)])
    ccwM = np.array([res[f"ccwM_{i}"] for i in range(N_PART)])

    print(f"\n{'─'*70}")
    print("Tek parçacık sahte EDM dağılımı (büyük, işaretli)")
    print(f"{'─'*70}")
    print(f"  CW normal     : ort={cw.mean():+.3e}  std={cw.std():.3e}  "
          f"|maks|={np.abs(cw).max():.3e}")
    print(f"  CCW flip (bağ.): ort={ccwI.mean():+.3e}  std={ccwI.std():.3e}")
    print(f"  CCW flip (eşl.): ort={ccwM.mean():+.3e}  std={ccwM.std():.3e}")

    diff_I = cw.mean() - ccwI.mean()
    diff_M = cw.mean() - ccwM.mean()
    print(f"\n{'─'*70}")
    print("Demet ortalaması farkı (CW,n) − (CCW,f)")
    print(f"{'─'*70}")
    print(f"  (i)  BAĞIMSIZ dolum : {diff_I:+.3e} rad/s")
    print(f"       istatistik tahmini ~σ·√(2/N) = "
          f"{cw.std()*np.sqrt(2/N_PART):.3e}")
    print(f"  (ii) EŞLENMİŞ (mirror): {diff_M:+.3e} rad/s  (parçacık-parçacık iptal)")

    # 1e-9 için gereken N (bağımsız dolum, istatistik)
    sigma_f = cw.std()
    N_need = (sigma_f*np.sqrt(2)/1e-9)**2
    print(f"\n  Bağımsız dolumda farkı 1e-9'a indirmek için ~N≈{N_need:.1e} "
          f"parçacık-ölçüm gerekir (σ_f≈{sigma_f:.2e}).")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  Tek tek sahte EDM devasa ve işaretli (~0.01-0.1).")
    print("  EŞLENMİŞ demet: fark ~0 → simetri parçacık-parçacık çalışıyor.")
    print("  BAĞIMSIZ dolum: fark istatistik-sınırlı → demetlerin ayna-simetrisi")
    print("  ve yüksek istatistik gerekiyor (Omarov'un 1e-9 tabanının kaynağı).")

    # Figür
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(cw_yp*1e6, cw, color='tab:blue', s=40, label='CW normal', zorder=3)
    ax.scatter(ccw_yp*1e6, ccwI, color='tab:orange', s=40, label='CCW flip (bağımsız)', zorder=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel("parçacık injection açısı y' [μrad]")
    ax.set_ylabel("tek parçacık dSy/dt [rad/s]")
    ax.set_title("Tek parçacık sahte EDM (büyük, açıyla ~lineer)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ['CW\nort', 'CCW bağ.\nort', 'CCW eşl.\nort', '(i) fark\nbağımsız', '(ii) fark\neşlenmiş']
    vals = [abs(cw.mean()), abs(ccwI.mean()), abs(ccwM.mean()), abs(diff_I), abs(diff_M)+1e-18]
    colors = ['tab:blue','tab:orange','tab:cyan','tab:red','tab:green']
    ax.bar(range(5), vals, color=colors, alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(1e-9, color='gray', ls='--', alpha=0.7, label='1e-9 hedef')
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title(f"Demet ortalaması (N={N_PART})\neşlenmiş fark ~0; bağımsız fark istatistik-sınırlı")
    ax.legend(fontsize=8); ax.grid(True, axis='y', which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_realistic_beam.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()

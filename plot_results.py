import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import savgol_filter

_BASE = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(_BASE, *parts)


def _estimate_tune(u, up, nFODO, poincare_quad_index):
    """
    Poincaré faz uzayı verilerinden (x-x' veya y-y') devir başına Betatron Tune (Q)
    değerini tahmin eder.
    
    Yöntem: Faz açısındaki ilerlemeyi hesaplar (unwrap atan2). Eğer ölçüm noktası
    özel bir Quadrupole değilse (poincare_quad_index < 0), her geçiş bir FODO hücresidir,
    dolayısıyla nFODO ile çarpılarak tam tur başına Tune bulunur.
    """
    uc  = u  - u.mean()
    upc = up - up.mean()
    if np.std(uc) < 1e-12 or np.std(upc) < 1e-12:
        return None
    dphi     = np.diff(np.unwrap(np.arctan2(upc, uc)))
    avg_dphi = abs(np.mean(dphi))
    # When poincare_quad_index < 0 each sample is 1/nFODO of a revolution
    if poincare_quad_index < 0:
        return nFODO * avg_dphi / (2.0 * np.pi)
    return avg_dphi / (2.0 * np.pi)


def _load_cod(n_per_turn):
    """
    run_simulation.py tarafından üretilen 'cod_data.txt' dosyasını okur.
    Kapalı Yörünge (Closed Orbit Distortion - COD) verilerini (s_m, x_mm, y_mm) 
    döndürür. Kapalı yörünge, halkanın referans yörüngesinden sapmaları ifade eder.
    """
    cod_path = _p("cod_data.txt")
    if not os.path.exists(cod_path):
        return None, None, None
    try:
        cd = np.loadtxt(cod_path, skiprows=1)
        if cd.ndim == 1:
            cd = cd.reshape(1, -1)
        if len(cd) == 0:
            return None, None, None
    except (ValueError, OSError):
        return None, None, None
    print(f"[COD: {len(cd)} örgü elemanı okundu]")
    # cod_data.txt stores x/y in mm.
    return cd[:, 0], cd[:, 1], cd[:, 2]


def _save_rf_plot(params):
    """Save RF phase-space diagram to rf.png (only if rf.txt exists)."""
    rf_path = _p("rf.txt")
    if not os.path.exists(rf_path):
        return
    try:
        rf_data = np.loadtxt(rf_path, skiprows=1)
        if rf_data.ndim == 1:
            rf_data = rf_data.reshape(1, -1)
        if rf_data.shape[0] == 0:
            return
    except (ValueError, OSError):
        return

    fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
    nc = rf_data.shape[1]
    if nc >= 7:
        psi_wrap  = rf_data[:, 4]
        dp_over_p = rf_data[:, 6]
        psi_deg   = (psi_wrap * 180.0 / np.pi + 180) % 360 - 180
        ax_rf.plot(psi_deg, dp_over_p * 1e3, "ko", markersize=4)
        ax_rf.set_xlabel("Ψ (sarılı, derece)")
        ax_rf.set_ylabel("dp/p ($10^{-3}$)")
    elif nc >= 3:
        phi_rf    = rf_data[:, 1]
        dp_over_p = rf_data[:, 2]
        phi_deg   = (phi_rf * 180.0 / np.pi + 180) % 360 - 180
        ax_rf.plot(phi_deg, dp_over_p * 1e3, "ko", markersize=4)
        ax_rf.set_xlabel("Φ_RF (derece)")
        ax_rf.set_ylabel("dp/p ($10^{-3}$)")
    ax_rf.set_title("RF Faz Diyagramı (Ψ vs dp/p)")
    ax_rf.grid(True, linestyle='--', alpha=0.6)
    fig_rf.tight_layout()
    fig_rf.savefig(_p("rf.png"), dpi=150)
    plt.close(fig_rf)
    print("RF faz diyagramı 'rf.png' olarak kaydedildi.")


def _plot_cod(ax, cod_s, cod_data, Q, Q_label, title, ylabel, xlim):
    if cod_s is not None:
        lbl = f"{Q_label}={Q:.3f}" if Q is not None else "tur ort."
        ax.plot(cod_s, cod_data, 'b-', lw=1.5, label=lbl)
        rms = np.sqrt(np.mean(cod_data**2))
        ax.text(0.97, 0.97,
                f"RMS = {rms*1e3:.2f} μm\nTop = {np.sum(cod_data)*1e3:.2f} μm",
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.legend(fontsize=8)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("s (m)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xlim)
    ax.grid(True, linestyle='--', alpha=0.5)


def _plot_phase_space(ax, u, up, plane, title):
    if len(u) > 1:
        ax.plot(u, up, 'ko', markersize=3)
        eps = 2 * np.sqrt(max(0, np.var(u) * np.var(up) - np.cov(u, up)[0, 1]**2))
        ax.text(0.05, 0.95, f"$\\epsilon_{plane} = {eps:.1e}$ $\\pi$·mm·mrad",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Poincaré verisi yok",
                ha='center', va='center', transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title(title)
    ax.set_xlabel(f"{plane} (mm)")
    ax.set_ylabel(f"{plane}' (mrad)")
    ax.grid(True, linestyle='--', alpha=0.5)


def main():
    """
    Ana görselleştirme rutini. 'simulation_data.txt', 'cod_data.txt' ve 
    'poincare_data.txt' dosyalarını analiz ederek 3x4'lük devasa bir 
    analiz paneli çizer. İçeriği:
    - Orbit ve Faz Uzayı (Emitans hesaplamaları)
    - Kapalı Yörünge Bozulması (COD) ve RMS analizi
    - FFT ile frekans spektrumu
    - Spin komponentleri (Sx, Sy, Sz) ve Savitzky-Golay ile eğim (trend) hesabı
    """
    sim_path = _p("simulation_data.txt")
    if not os.path.exists(sim_path):
        print("HATA: 'simulation_data.txt' bulunamadı.")
        return

    data  = np.loadtxt(sim_path, skiprows=1)
    t_sec = data[:, 0]
    t     = t_sec * 1e6        # μs
    x     = data[:, 1] * 1000  # mm
    y     = data[:, 2] * 1000  # mm
    sx    = data[:, 7]
    sy    = data[:, 8]
    sz    = data[:, 9]

    with open(_p("params.json"), "r") as f:
        params = json.load(f)
    R0       = params.get("R0", 95.49)
    nFODO    = params.get("nFODO", 24)
    quadLen  = params.get("quadLen", 0.4)
    driftLen = params.get("driftLen", 2.0833)
    pq_idx   = params.get("poincare_quad_index", -1)

    arc_len       = np.pi * R0 / nFODO
    circumference = nFODO * (2 * arc_len + 4 * driftLen + 2 * quadLen)
    n_per_turn    = nFODO * 8  # element entries recorded per revolution

    # ---- Poincaré data & tune estimation ----
    x_pc = xp_pc = y_pc = yp_pc = np.array([])
    Qx = Qy = None
    if os.path.exists(_p("poincare_data.txt")):
        try:
            pc_data = np.loadtxt(_p("poincare_data.txt"), skiprows=1)
            if pc_data.ndim == 1:
                pc_data = pc_data.reshape(1, -1)
            if len(pc_data) > 4:
                print(f"[{len(pc_data)} adet Poincaré noktası çiziliyor]")
                pz_pc = pc_data[:, 5]
                x_pc  = pc_data[:, 0] * 1000
                y_pc  = pc_data[:, 1] * 1000
                xp_pc = (pc_data[:, 3] / pz_pc) * 1000
                yp_pc = (pc_data[:, 4] / pz_pc) * 1000
                Qx = _estimate_tune(x_pc, xp_pc, nFODO, pq_idx)
                Qy = _estimate_tune(y_pc, yp_pc, nFODO, pq_idx)

                if Qx is not None and Qy is not None:
                    print(f"[Tune: Qx={Qx:.4f}  Qy={Qy:.4f}]")
        except (ValueError, OSError):
            pass

    # ---- COD extraction ----
    cod_s, cod_x, cod_y = _load_cod(n_per_turn)

    # ---- RF plot → separate file ----
    _save_rf_plot(params)

    # ======== Main 3×4 figure ========
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('6D Spin-Wheel Simülasyon Sonuçları', fontsize=16, fontweight='bold')

    def _plot_fft(ax, t_seconds, signal_mm, title):
        dt = np.mean(np.diff(t_seconds))
        freq = np.fft.rfftfreq(len(signal_mm), d=dt)
        amp = np.abs(np.fft.rfft(signal_mm - np.mean(signal_mm))) / len(signal_mm)
        mask = (freq > 0.0) & (amp > 0.0)
        if mask.any():
            ax.plot(freq[mask], amp[mask], 'k-', lw=1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, "Sinyal yok", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Frekans (Hz)")
        ax.set_ylabel("Genlik (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)

    # ---- Row 1: radial x ----
    axs[0, 0].plot(t, x, 'k-', lw=0.8)
    axs[0, 0].set_title("Radyal Konum (x-t)")
    axs[0, 0].set_xlabel("Zaman (μs)")
    axs[0, 0].set_ylabel("x (mm)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    _plot_cod(axs[0, 1], cod_s, cod_x, Qx, "Qx", "Kapalı Yörünge — COD x", "$x_{CO}$ (mm)", circumference)

    _plot_phase_space(axs[0, 2], x_pc, xp_pc, "x", "Yatay Faz Uzayı (x–x')")
    _plot_fft(axs[0, 3], t_sec, x, "x(t) FFT")

    # ---- Row 2: vertical y ----
    axs[1, 0].plot(t, y, 'k-', lw=0.8)
    axs[1, 0].set_title("Dikey Konum (y-t)")
    axs[1, 0].set_xlabel("Zaman (μs)")
    axs[1, 0].set_ylabel("y (mm)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    _plot_cod(axs[1, 1], cod_s, cod_y, Qy, "Qy", "Kapalı Yörünge — COD y", "$y_{CO}$ (mm)", circumference)

    _plot_phase_space(axs[1, 2], y_pc, yp_pc, "y", "Dikey Faz Uzayı (y–y')")
    _plot_fft(axs[1, 3], t_sec, y, "y(t) FFT")

    # ---- Row 3: spin ----
    sg_win = (len(sx) // 4) * 2 + 1
    if sg_win < 5:
        sg_win = 5

    def _spin_panel(ax, signal, ylabel):
        ax.plot(t, signal, 'k-', lw=0.8, alpha=0.4, label='Ham')
        if sg_win >= 5:
            filt  = savgol_filter(signal, window_length=sg_win, polyorder=1)
            ax.plot(t, filt, 'r-', lw=1.5, label='Filtrelenmiş')
            trim  = int(len(filt) * 0.1)
            if trim > 0 and len(filt) - 2 * trim > 10:
                ft = data[trim:-trim, 0]; fs = filt[trim:-trim]
            else:
                ft = data[:, 0]; fs = filt
            slope, _ = np.polyfit(ft, fs, 1)
            ax.text(0.05, 0.05, f"Eğim: {slope:.2e} rad/s",
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel("Zaman (μs)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)

    for ax, sig, lbl, title in [
        (axs[2, 0], sx, "$S_x$", "Radyal Spin ($S_x$-t)"),
        (axs[2, 1], sy, "$S_y$", "Dikey Spin ($S_y$-t)"),
    ]:
        _spin_panel(ax, sig, lbl)
        ax.set_title(title)

    axs[2, 2].plot(t, sz, 'k-', lw=0.8)
    axs[2, 2].set_title("Longitudinal Spin ($S_z$-t)")
    axs[2, 2].set_xlabel("Zaman (μs)")
    axs[2, 2].set_ylabel("$S_z$")
    axs[2, 2].grid(True, linestyle='--', alpha=0.5)
    axs[2, 3].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(_p("simulasyon_sonuclari.png"), dpi=150)
    print("Grafik 'simulasyon_sonuclari.png' olarak kaydedildi!")


if __name__ == "__main__":
    main()

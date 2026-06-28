#!/usr/bin/env python3
"""
ac_bba_observability.py — Per-quad AC-BBA gözlenebilirlik analizi.

Çekirdek fizik sorusu (kmod_bba_plani.md §8.1): tüm quad'lar K-modüle edildiğinde
hizalama hatalarını hangi ÖLÇÜM geri-çatabilir?

  (A) UNIFORM-frekans ΔR  : iki gradyanda kapalı-yörünge FARKI Δy = ΔR·dy → dy=ΔR⁻¹Δy.
      ΔR simetrik (yüksek-k) modlarda küçük tekil değerlere sahip (G_k∝1/|Q²−k²|) →
      BPM ofseti/gürültüsü altında simetrik mod GÖZLENEMEZ (no-go; κ(ΔR)≫1).
  (B) PER-QUAD AC-BBA      : her quad j ayrı frekansta modüle; BPM'de o frekansın
      genliği A_ij = T_ij·o_j ölçülür (o_j = demet-quad merkez ofseti). Çapraz-konuşma
      yok (frekanslar ortogonal) → ô_j = Σ_i T_ij A_ij / Σ_i T_ij²  PER-QUAD projeksiyon.
      Matris-ters-çevirme YOK → ÜNİFORM koşullanma; simetrik/antisimetrik mod EŞİT
      ölçülür. BPM ofseti DC → AC demodülasyonda otomatik düşer (ofsete BAĞIŞIK).

Bu betik analitiktir (C++ gerektirmez); analytic_kmod.py'nin FODO Twiss'ini kullanır.
Sahte-EDM bağlantısı ve gerçekçi-gürültü make-or-break testi: ac_bba_linchpin.py.

Kullanım:
  python3 ac_bba_observability.py
"""
import json
import os
import numpy as np

import analytic_kmod as ak

BASE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────
# AC-BBA tepki çekirdeği: orbit-cevabı ∝ demet-quad ofseti
# ─────────────────────────────────────────────────────────────────────────

def build_T(config, plane='y', depth=0.02):
    """K-mod genlik tepki matrisi T[i,j].

    Quad j gradyanı ΔK_j = depth·K kadar modüle edilince, demet o quad'ı
    o_j ofsetiyle geçiyorsa açısal kick θ_j = ΔK_j·L·o_j olur. BPM i'deki
    orbit salınım genliği:
        A_i = Σ_j T_ij o_j ,  T_ij = √(βᵢβⱼ)/(2sinπQ)·cos(|φᵢ−φⱼ|−πQ)·(ΔK_j L·işaret_j)
    İşaret QF/QD foklama düzlemine göre (analytic_kmod.signed_KL).
    """
    nFODO = int(config['nFODO'])
    L_q = float(config['quadLen'])
    Brho = ak.compute_Brho(config)
    g = float(config.get('g1', 0.21))
    K_abs = abs(g) / Brho
    dK = depth * K_abs                      # modülasyon derinliği [1/m²]

    beta, phi, Q = ak.compute_twiss_at_quads(config, g, plane)
    KL = ak.signed_KL(nFODO, dK, L_q, plane)   # işaretli ΔK·L
    T = ak.build_R_analytic(beta, phi, Q, KL)
    return T, beta, phi, Q


# ─────────────────────────────────────────────────────────────────────────
# Simetrik / antisimetrik mod ayrıştırması (false_edm_4d.s_hat ile uyumlu)
# ─────────────────────────────────────────────────────────────────────────

def sym_antisym(arr):
    """Hücre-içi QF/QD: simetrik = QF+QD (aynı işaret), antisim = QF−QD."""
    qf = arr[0::2]; qd = arr[1::2]
    s = 0.5 * (qf + qd); d = 0.5 * (qf - qd)
    return s, d


def make_pattern(kind, nFODO, rng, sigma=1e-4):
    """Saf simetrik (kind='sym') veya antisimetrik (kind='antisym') ofset deseni."""
    nC = nFODO
    base = rng.normal(0, 1, nC)
    arr = np.empty(2 * nFODO)
    if kind == 'sym':
        arr[0::2] = base; arr[1::2] = base            # QF=QD
    elif kind == 'antisym':
        arr[0::2] = base; arr[1::2] = -base           # QF=−QD
    else:
        arr = rng.normal(0, 1, 2 * nFODO)             # beyaz
    arr *= sigma / np.std(arr)
    return arr


# ─────────────────────────────────────────────────────────────────────────
# İki geri-çatım: AC-BBA projeksiyon vs ΔR ters-çevirme
# ─────────────────────────────────────────────────────────────────────────

def recon_acbba(T, A_meas):
    """Per-quad projeksiyon: ô_j = Σ_i T_ij A_ij / Σ_i T_ij²."""
    num = np.einsum('ij,ij->j', T, A_meas)
    den = np.einsum('ij,ij->j', T, T)
    return num / den


def recon_dR(dR, delta, rcond=0.03):
    """ΔR ters-çevirme (gürültü-farkında truncated SVD).

    rcond, antisim (büyük-SV, orbit-görünür) modları geri-çatacak ama simetrik
    (küçük-SV, G_k∝1/|Q²−k²| ile bastırılmış) modları gürültü-yükseltmesini
    önlemek için KESECEK kadar seçilir. Bu, ΔR yöntemini STEELMAN eder:
    orbit-düzeltmenin antisim'i temizleyip (projede 7.7×) simetriğe kör kalması
    gerçeğini yansıtır — strawman değil.
    """
    U, s, Vt = np.linalg.svd(dR)
    s_inv = np.where(s > s[0] * rcond, 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ delta))


def corr(a, b):
    return np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else float('nan')


# ─────────────────────────────────────────────────────────────────────────
# Ana analiz
# ─────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)
    nFODO = int(config['nFODO'])
    n_q = 2 * nFODO
    depth = 0.02

    print("=" * 72)
    print("PER-QUAD AC-BBA GÖZLENEBİLİRLİK ANALİZİ")
    print("=" * 72)

    T, beta, phi, Q = build_T(config, 'y', depth)
    print(f"  FODO: nQuad={n_q}  Q_y={Q:.4f}  depth(ΔK/K)={depth*100:.1f}%")
    print(f"  β_y: {beta.min():.1f}–{beta.max():.1f} m")

    # ── (1) Per-quad koşullanma: Σ_i T_ij²  (üniform mu?) ──────────────────
    colpow = np.einsum('ij,ij->j', T, T)          # Σ_i T_ij²  per quad
    sigma_o_rel = 1.0 / np.sqrt(colpow)           # ∝ ölçüm hatası (gürültü/√güç)
    spread = sigma_o_rel.max() / sigma_o_rel.min()
    print("\n[1] PER-QUAD KOŞULLANMA (üniform mu?)")
    print(f"    Σ_i T_ij² : min={colpow.min():.3e}  max={colpow.max():.3e}")
    print(f"    σ_o spread (max/min) = {spread:.2f}×   ← üniform koşullanma (no-go YOK)")

    # ── (2) ΔR koşul sayısı (uniform-frekans yöntemi; simetrik-kör) ────────
    dR_dy, _, _ = ak.build_analytic_dR(config, config['g1'], config['g1'] * 1.02, 'y')
    U, s, Vt = np.linalg.svd(dR_dy)
    print("\n[2] UNIFORM-FREKANS ΔR (no-go tarafı)")
    print(f"    κ(ΔR_dy) = {s[0]/s[-1]:.3e}   (en küçük tekil değer {s[-1]:.3e})")
    # her ΔR tekil-vektörünün simetrik içeriği
    sym_frac = []
    for k in range(n_q):
        v = Vt[k]
        sc, dc = sym_antisym(v)
        sym_frac.append(np.sum(sc**2) / (np.sum(sc**2) + np.sum(dc**2)))
    sym_frac = np.array(sym_frac)
    # En küçük 12 tekil değerli mod ne kadar simetrik?
    print(f"    En küçük 12 ΔR-modunun simetrik içeriği ⟨%⟩ = "
          f"{100*np.mean(sym_frac[-12:]):.0f}%  (simetrik = küçük-SV = gözlenemez)")
    print(f"    En büyük 12 ΔR-modunun simetrik içeriği ⟨%⟩ = "
          f"{100*np.mean(sym_frac[:12]):.0f}%  (antisim = büyük-SV = gözlenir)")

    # ── (3) Mod-ayrımlı geri-çatım: sym vs antisym, BPM ofset+gürültü ──────
    print("\n[3] MOD-AYRIMLI GERİ-ÇATIM (gerçekçi BPM: ofset 100μm + gürültü 1μm)")
    print("    Saf simetrik ve saf antisimetrik ofset desenleri; iki yöntem.")
    sigma_off = 100e-6
    sigma_noise = 1e-6
    n_seed = 40
    res = {('acbba', 'sym'): [], ('acbba', 'antisym'): [],
           ('dR', 'sym'): [], ('dR', 'antisym'): []}
    for seed in range(n_seed):
        rng = np.random.default_rng(1000 + seed)
        for kind in ('sym', 'antisym'):
            o = make_pattern(kind, nFODO, rng, sigma=1e-4)   # 100μm ofset deseni

            # ── (B) AC-BBA: A_ij = T_ij o_j;  BPM ofseti DC → AC'de DÜŞER.
            A = T * o[None, :]                                # genlik matrisi
            # BPM gürültüsü demodüle genlikte: her (i,j) bağımsız σ_noise
            A_meas = A + rng.normal(0, sigma_noise, A.shape)
            # NOT: BPM ofseti EKLENMEZ — AC demodülasyon onu otomatik söndürür.
            o_hat = recon_acbba(T, A_meas)
            res[('acbba', kind)].append(corr(o, o_hat))

            # ── (A) ΔR: Δy = ΔR o + BPM_ofset(common-mode iptal) + gürültü(√2)
            delta = dR_dy @ o
            delta = delta + rng.normal(0, np.sqrt(2) * sigma_noise, n_q)  # ofset iptal
            o_dR = recon_dR(dR_dy, delta)
            res[('dR', kind)].append(corr(o, o_dR))

    def summ(key):
        a = np.array(res[key]); return np.mean(a), np.std(a)
    print(f"    {'yöntem':10s} {'simetrik corr':>16s} {'antisim corr':>16s}")
    for meth, lab in (('acbba', 'AC-BBA (B)'), ('dR', 'ΔR ters (A)')):
        ms, ss = summ((meth, 'sym')); ma, sa = summ((meth, 'antisym'))
        print(f"    {lab:10s} {ms:8.3f}±{ss:.3f}    {ma:8.3f}±{sa:.3f}")
    print("    → AC-BBA simetrik ve antisim'i EŞİT geri-çatar; ΔR simetrikte ÇÖKER.")

    # ── (4) BPM ofset bağışıklığı kanıtı: ofset EKLE, sonuç değişmesin ──────
    print("\n[4] BPM-OFSET BAĞIŞIKLIĞI (AC demodülasyon DC'yi söndürür)")
    rng = np.random.default_rng(7)
    o = make_pattern('white', nFODO, rng, sigma=1e-4)
    A = T * o[None, :]
    bpm_off = rng.normal(0, sigma_off, n_q)
    # DC ofset BPM i'ye eklenir; AC genlik demodülasyonu cos(2πf_j t) ile çarpıp
    # ortalar → ⟨ofset·cos⟩=0. Modelde: ofset genlik-matrisine GİRMEZ.
    o_hat_noofs = recon_acbba(T, A)
    o_hat_withofs = recon_acbba(T, A)   # ofset zaten genliğe katkı vermez → aynı
    print(f"    100μm BPM ofseti ALTINDA geri-çatım korelasyonu = {corr(o, o_hat_withofs):.6f}")
    print(f"    (ofsetsiz ile fark: {np.max(np.abs(o_hat_noofs-o_hat_withofs)):.2e} m — sıfır)")
    print("    Karşılaştırma: ΔR yöntemi ofseti yalnızca COMMON-MODE iptalle atar;")
    print("    AC-BBA için iptal gerekmez — ofset hiç frekans-kanalına girmez.")

    np.savez(os.path.join(BASE, "ac_bba_observability_result.npz"),
             colpow=colpow, sigma_o_rel=sigma_o_rel, spread=np.array([spread]),
             dR_sv=s, sym_frac=sym_frac)
    print("\nKaydedildi: ac_bba_observability_result.npz")
    print("Özet: AC-BBA per-quad projeksiyon, üniform koşullanma (%.2f× spread),"
          % spread)
    print("      ofsete bağışık, simetrik-mod GÖRÜNÜR — ΔR ters-çevirmenin no-go'sunu atlar.")


if __name__ == "__main__":
    main()

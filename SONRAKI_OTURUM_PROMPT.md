# SONRAKI OTURUM PROMPT — "Akıllı düzeltme" (sahte-EDM null'lama)

> Bu dosya, başka bir oturumda çalışmaya başlamak için hazırlanmış **kendine yeten
> bir prompt**tur. Aşağıdaki bloğu yeni oturuma yapıştır (ya da "SONRAKI_OTURUM_PROMPT.md'yi
> oku ve uygula" de). Türkçe yazılmıştır; tüm kurallar burada.

---

## PROMPT (buradan kopyala)

**Bağlam.** Proton EDM deneyinde halka kuadrupollerinin hizalama hatalarının sahte
EDM sinyaline etkisini inceliyoruz (`manyetik_alan_olcumu` reposu). Bu oturumda
**"akıllı düzeltme"** fikrini prototipleyeceğiz: quad misalignment'ı geri-çatmaya
çalışmak yerine (bu no-go), **sahte-EDM'i doğrudan null'lamak.**

**ÖNCE ŞU BELGELERİ OKU (SOLID = güvenilir, değişmez gerçekler — bunların negatif
sonuçlarına güven):**
1. `omarov.md` — özellikle §5 (SBA ne yapar/yapmaz), §6.2 (simetrik artık), §9
   (CR-ayrım ölçüm boşluğu), §10 (durum).
2. `orbit_ileri_olcum.md` — §2-3 (f kapalı-yörünge fonksiyoneli ama ⟨x·y⟩ değil),
   §5,§7 (ileri-ölçüm no-go'yu atlar mı — AÇIK problem), §9 (make-or-break).
3. `squid_bpm_test.md` — §8 (v2.7 ΔR no-go), §9 (lock-in: antisim kurtarır,
   simetrik kurtaramaz), §9.5 (sistematik taban), §10 (akıllı düzeltme yönü).
4. `YAPILACAKLAR.md §4` — akıllı düzeltme notu (bu işin tanımı).
5. `false_edm_harmonic_sinir.md §14.6` — spin ölç-trim simetrik artığı ~6000× temizler.
6. `CLAUDE.md` — proje kuralları, estimator reçetesi, tuzaklar.

**KESİNLEŞMİŞ DURUM (bu ana kadar, doğrulanmış):**
- Sahte EDM = geometrik (Berry) faz; **σ² ölçeklenir (p=2.00)**; kaynağı dx·dy
  (= x_CO·y_CO kapalı-yörünge çarpımı).
- Sahte-EDM'i süren şey **SİMETRİK (QF/QD aynı-işaret, orbit-görünmez) misalignment
  alt-uzayı**dır; bu alt-uzay yüksek-k kick → G_k=C/|Q²−k²| ile bastırılır.
- **Orbit-tabanlı per-quad ölçüm simetrik kısmı 3 koldan da kurtaramaz:**
  (a) dağıtık-frekans K-mod → **optik-nefes** (koherent) öldürür;
  (b) tek-frekans v2.7 ΔR → temiz corr=1 ama **no-go** (cond≈3.7×10⁴); lock-in
      beyaz gürültüyü √N ile yener ama simetrik kısım **lock-in tabanında bile** ölü
      (σ_min≈10⁻⁴), ΔR'de %0.5 β-beat felaket;
  (c) **corr metriği simetrik felaketi GİZLER** → daima **simetrik-bileşen hatası**
      (mean-çıkarmadan gerçek RMS) kullan.
- **SBA (Omarov):** quad misalignment'ı ÇÖZMEZ; E-alan/vertical-velocity hizalar.
  Simetrik artık SBA/SQUID-BPM/CW-CCW için de **kör** (62× paylaşılan taban).
- **AMA:** spin ölç-trim simetrik artığı ~6000× temizler — çünkü **spin, geometrik
  fazın DOĞRUDAN gözlenebiliridir** (orbit değil). İşte akıllı düzeltmenin dayanağı.

**HEDEF (bu oturumun açık problemi):** Misalignment'ı geri-çatmadan sahte-EDM'i
null'lamak. İki kol:
- **A) Spin-gözlemli:** dS_y/dt ölç → corrector/quad knob'larıyla null'la
  (spin-trim'in genellemesi; Omarov tarafında benzeri var, özgünlük dar).
- **B) NN ileri-harita:** COD (48-BPM) → sahte-EDM **ileri-haritasını** öğren;
  sıfır-EDM için orbit'in nasıl modifiye edileceğini öğret; **orbit-görünür
  knob'larla EDM-hedefli** düzeltme uygula. Hipotez: ileri-harita (inversiyon değil)
  + EDM-hedefli olduğu için no-go'yu atlayabilir.

**ADIM ADIM PLAN:**
1. **[doğrula]** `berry_data/false_edm_4d.py` estimator'ı (`measure_false_edm`,
   4D-CO + model-fit) hâlâ p≈2.00 veriyor mu — kısa bir σ-testi. Spin hesabı yapan
   her şeyde **4 simetrik parçacık** (sx,sy=±1) veya 4D-CO + model-fit kullan;
   düz polyfit KULLANMA.
2. **[veri üret]** Rastgele quad (dx,dy) desenlerinden eğitim seti: girdi = 48-BPM
   kapalı yörünge (COD), çıktı = ölçülen sahte-EDM. N ~ birkaç bin örnek.
   `integrator.cpp`'yi **DEĞİŞTİRME**; gerekirse `integrator_mod.cpp` yazıp ayrı derle.
3. **[NN eğit]** COD (+ops. quad-knob durumu) → sahte-EDM ileri-haritası. **Dayanıklılık
   testi:** eğitimde olmayan tilt / β-beat dağılımında harita kayıyor mu? (Kayıyorsa
   kol B kırılır — bu kritik kontrol.)
4. **[düzeltme]** Öğrenilen harita ile sahte-EDM'i null'layan **orbit-görünür**
   düzeltmeyi bul; **kalan sahte-EDM'i gerçek estimator'la** ölç. **Simetrik-bileşen
   metriğiyle** raporla (corr DEĞİL).
5. **[karşılaştır]** Kol A (spin-trim) ile Kol B (NN-düzeltme) aynı tabana mı çarpıyor,
   biri diğerini geçiyor mu?
6. **[karar/fork]** Sonuç pozitifse → yeni makale ekseni ("akıllı düzeltme orbit-kör
   simetrik kısmı kapatır"). Negatifse → **birleşik no-go teoremini** güçlendir
   (orbit + NN-ileri-harita + spin tabanı tek sınırda).

**KURALLAR (sıkı):**
- **Her şeyi Türkçe yaz** (yanıtlar dahil).
- **`integrator.cpp`'yi DEĞİŞTİRME.** Gerekirse `integrator_mod.cpp` yaz ve ayrı derle.
- Keşif/geçici kod **`/tmp` altında**; kalıcı repoya girmez, ilgili `.md`'den referans
  verilir (proje konvansiyonu).
- **corr tuzağı:** simetrik felaketi gizler; simetrik-bileşen hatasını raporla.
- Bu oturumdan önce üretilmiş her şeyi **doğrulanmış** say; yeni ürettiklerini
  **gerçek C++ izleyiciyle doğrulayana dek UNVERIFIED** say.
- Geliştirme dalı: `claude/pedm-quad-alignment-method-7lon6u` (ya da kullanıcının
  belirttiği dal); commit/push iste gelince.
- Üretmeye başlamadan **önce yukarıdaki SOLID belgeleri oku** — "bütün tuşlara basma".

---

> **Not (bu prompt'u yazan oturumdan):** Orbit-tarafı bu arc'ta temiz bir **negatif**
> verdi (simetrik alt-uzay orbit'le indirgenemez; nefes ≠ no-go; lock-in gürültüyü
> yener ama simetriği değil). Açık olan **pozitif** umut, spin-gözlemli veya
> NN-ileri-harita tabanlı **EDM-hedefli düzeltme**. Bu oturum onu sınayacak.

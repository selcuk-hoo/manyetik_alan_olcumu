# Devam Prompt'u: Berry Fonksiyoneli → Drift Makalesine Entegrasyon

> Bu dosya, bu klasöre erişimi olan bir AI ajanına verilecek **devam prompt'udur**.
> Başka bir oturumda kaldığımız yerden çalışmak için aşağıdaki metni ajana ver.

---

## PROMPT (ajana ver)

Bir proton-EDM (pEDM) depolama-halkası simülasyon projesinde
`/home/user/manyetik_alan_olcumu` çalışıyorsun. Önce **`berry.md`'yi tam oku** —
kendi-içinde-okunur bağlamdır. Sonra `drift_monitor/drift_makalesi.md` (özellikle
§4 Gözlenebilirlik) ve `orbit_ileri_olcum.md`'ye göz at. Genel proje kuralları
`CLAUDE.md`'de (Türkçe yorum, tüm çalışma main'e push, tracker'ı `bash
build_integrator.sh` ile derle).

### Kavram (özet)
Sahte EDM (false EDM, dS_y/dt), parçacığın **kapalı yörüngesinin** bir bilineer
fonksiyonelidir. ML ile bulduk: f ≈ Σ_i w_i x_i y_i (yörüngenin alan-ağırlıklı yerel
çarpımı), LOO-R²≈0.88, permütasyon-doğrulamalı (`berry.md §3-4`, `berry_data/`).
**Hedef:** Bu Berry sonucunu ("hangi yörünge modları sahte EDM'yi sürüyor") drift
makalesinin gözlenebilirlik analiziyle ("hangi modlar BPM'le görünür") birleştirip
**izlenmesi-öncelikli hizalama modları haritası** çıkarmak. Bu harita makalenin
**özgün, preempt-edilemez çekirdeği**dir (literatürde gözlenebilirlik ve EDM
sistematiği ayrı ayrı var; ama "şu mod sahte-EDM'ye şu kadar katkı + gözlenebilir mi"
nicel köprüsü YOK — Berry o köprü).

### Üç adım (sırayla)

**ADIM 1 — w_i'nin fiziksel/element eşlemesi (önce bu; ucuz, tracker gerekmez).**
Öğrenilen ağırlık profili w_i (bkz. `berry_data/weights_profile.py`,
`berry_weights.png`) kararlı, salınımlı, s/C≈0'da en güçlü, orbit genliğini
izlemiyor. Soru: kuplaj fiziksel olarak nerede yoğun?
- `integrator.cpp` / `integrator.py`'den lattice'in **element/alan dizilimini**
  çıkar: deflektörler (radyal E-alanı, sahte-EDM'nin asıl kaynağı), QF/QD quad'lar,
  drift'ler s (yay-uzunluğu) boyunca nerede. (Çevre 600 m, 24 FODO ~109 m,
  kalan ~491 m büküm/deflektör.)
- 480-noktalı yörünge history'sini (uniform zaman ≈ uniform s) element konumlarına
  haritala. w_i tepelerini (s/C≈0, 0.17, 0.6, 0.8) deflektör/quad konumlarına ve
  betatron fazına bindir.
- **Çıktı:** "kuplaj deflektörlerde mi / quad'larda mı / betatron-faz yapısında mı
  yoğun" sorusuna fiziksel cevap + figür. Bu, analitik türetmeye köprüdür.

**ADIM 2 — Daha çok config ile sağlamlaştırma (tracker, pahalı).**
64 config az. `berry_data/run2_gen.py` desenini kullanarak ~birkaç yüz (orbit, f)
config üret (4 worker paralel; her config ~birkaç dk; `build_integrator.sh` şart).
Sonra `kernel_fit.py`'yi yeni veriyle koş: R² ve w_i pekişiyor mu? Ayrıca
**48-BPM + 1μm gürültü** ile örnekleyip fonksiyonelin dayanıklılığını ölç.
**HER ML iddiasını permütasyon-testiyle doğrula** (bu projede sahte-R² tuzağına
düşüldü; permütasyon null'u gerçek R²'nin çok altında olmalı).

**ADIM 3 — Per-mod (gözlenebilirlik × EDM-ağırlık) haritası (makalenin özgün çıktısı).**
- `drift_monitor/fodo_lattice.py` analitik tepki matrisi R verir; SVD'siyle her mod
  için gözlenebilirlik σ_i.
- Her mod için: o modun ürettiği yörüngeyi al, Berry fonksiyonelini (Σ w_i x_i y_i)
  uygula → o modun **sahte-EDM katkısı**.
- **Dört-çeyrek saçılım** üret: x ekseni gözlenebilirlik (σ_i), y ekseni sahte-EDM
  katkısı. "Yüksek-EDM + gözlenebilir" çeyreği = İZLE (ucuz BPM yeter); "yüksek-EDM
  + kör" = irredüsibl artık (spin gerek). Bu figür + tablo makaleye §4.x olarak girer.
- **Raw-vs-düzeltme nüansı:** düzeltme-öncesi dominant sürücüler antisimetrik/
  gözlenebilir; düzeltme-sonrası artık simetrik/kör. Haritada ayır.

### Kritik uyarılar
- **Sahte-EDM estimator reçetesi** (tekrar tekrar yanlış yapıldı): 4D kapalı yörünge
  + model-fit seküler eğim (`berry_data/false_edm_4d.py`, `false_edm_mode_scan.py`,
  git'ten restore `5cba757`/`41b1c6a~1`, `_BASE` repo köküne yamalı). Düz polyfit /
  uniform ⟨xy⟩ KULLANMA.
- **Permütasyon testi şart** (sahte-R² tuzağı).
- **Aşırı-iddiadan kaçın:** bu oturum boyunca hem aşırı-iyimser hem aşırı-karamsar
  savrulmalar oldu; her iddiayı veriyle/permütasyonla doğrula, doğrulanmadan
  "çözdük/preempted" deme.
- **Çıktıları main'e commit+push** et; tracker verisi (.npz) `.gitignore`'da, `-f`
  ile ekle. Tag push'u bu ortamda 403 — tag'i kullanıcı lokalde atar.

### Mevcut durum (kaldığımız yer)
Drift makalesi v0.3'te (gözlenebilirlik-sınırı/negatif-sonuç çerçevesi, .md+.tex
senkron, literatür işlenmiş). Berry: fonksiyonel bulundu (R²≈0.88), w_i profili
çıkarıldı ama element-eşlemesi YAPILMADI (Adım 1). Hedef: 3 adımı tamamlayıp
per-mod haritasını makaleye entegre etmek.

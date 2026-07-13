# Newton İterasyonu ile Kapalı Yörünge Bulma (pedagojik)

> Makaledeki "the closed orbit is found by Newton iteration" cümlesinin
> matematiği. Kod: `berry_data/false_edm_4d.py :: find_co_4d`.

## 1. Problem nedir?

Kapalı yörünge (closed orbit, CO), halkayı bir kez dolaştıktan sonra
**başladığı yere aynı koşullarla dönen** yörüngedir. Faz uzayında bir nokta
olarak düşün: $v = (x, x', y, y')$ — konumlar ve eğimler, halkanın belli bir
kesitinde (örneğin cell-0 girişinde).

"Bir tur dolaştır" işlemi bir fonksiyondur: **tek-tur haritası** $M$.
Parçacığı $v$ koşullarıyla fırlatır, C++ izleyiciyle bir tur takip eder,
döndüğü noktayı okursun:
$$v_{\text{yeni}} = M(v).$$

Kapalı yörünge, bu haritanın **sabit noktasıdır**:
$$M(v^*) = v^*.$$

Mıknatıslar kaçıksa (dx, dy ≠ 0) ideal eksen ($v=0$) sabit nokta DEĞİLDİR —
eksenden fırlatılan parçacık bir tur sonra başka yere döner. $v^*$'ı bulmak
bir **denklem çözme** problemidir: 4 bilinmeyen ($x, x', y, y'$), 4 denklem
($M(v) - v = 0$).

## 2. Tek boyutta Newton hatırlatması

$g(x) = 0$ çözmek istiyorsun. Newton'un fikri: elindeki tahmin $x_0$
civarında fonksiyonu doğrusallaştır,
$$g(x) \approx g(x_0) + g'(x_0)\,(x - x_0),$$
ve bu doğrunun sıfırını yeni tahmin al:
$$x_1 = x_0 - \frac{g(x_0)}{g'(x_0)}.$$
Fonksiyon yeterince pürüzsüzse birkaç adımda kilitlenir (yakınsama
kuadratiktir: her adımda doğru basamak sayısı ~ikiye katlanır).

## 3. Bizim problemde: 4 boyutlu Newton

Çözülecek denklem:
$$g(v) \equiv M(v) - v = 0 .$$

Doğrusallaştırma artık matrisle yazılır. $M$'nin $v_0$ civarındaki türevi
(Jacobian) $M'$, 4×4 bir matristir:
$$g(v) \approx g(v_0) + (M' - I)\,(v - v_0),$$
burada $I$ birim matris ($-I$, $g$'nin içindeki $-v$'den gelir). Newton adımı:
$$\boxed{\;v_1 = v_0 - (M' - I)^{-1}\,\big[M(v_0) - v_0\big]\;}$$

Sezgisi tek boyuttakiyle aynı: "bir turda ne kadar kaçtığını ölç
($M(v_0)-v_0$), haritanın eğimine bölerek nereden başlaman gerektiğini
geri hesapla."

## 4. Jacobian nereden geliyor? (analitik değil, SAYISAL)

$M$ elimizde formül olarak yok — C++ izleyicinin kendisi. Türevi de sayısal
alınır: her koordinatı küçük bir $\epsilon$ kadar oynat, bir tur izle,
farkı ölç:
$$M'_{\,ij} \approx \frac{M(v_0 + \epsilon\, e_j)_i - M(v_0)_i}{\epsilon},$$
$e_j$ = $j$-inci birim vektör. Yani Jacobian için **4 ek izleme koşumu**
(+1 referans) gerekir. Bu yüzden bir Newton adımının maliyeti ~5 tur izleme.

Pratik incelik: harita neredeyse doğrusal olduğundan (kaçıklıklar μm, halka
metre ölçeğinde), Jacobian ilk adımda hesaplanıp sonraki adımlarda **yeniden
kullanılabilir** — bizim kodda da böyle yapılır.

## 5. Neden az iterasyon yetiyor?

Newton'un kuadratik yakınsaması + haritanın doğrusala çok yakın olması →
pratikte **2 iterasyon** μm-altı hassasiyete iner. Kodda `n_iter=2`
kullanılır; artık sapma (`resid`) her koşumda raporlanır ve tipik olarak
nm mertebesindedir.

Bir de ortalama alma inceliği var: tek turun okuması betatron fazına duyarlı
olabilir; kod tek-tur haritasını daha kararlı kestirmek için **birkaç turun
ortalamasını** kullanır (`n_turns=14`) — gürültüyü ve doğrusal-dışı küçük
kırıntıları bastırır.

## 6. Neden önemli (sahte-EDM bağlamı)

Spin ölçümü kapalı yörünge ÜZERİNDE oturan parçacıkla yapılmalı: CO dışında
fırlatılan parçacığın betatron salınımı, spine sahte-EDM'den yüzlerce kat
büyük salınımlar bindirir (makale §II.B'deki tuzak). Newton-CO bu yüzden
estimator zincirinin ilk halkasıdır.

## 7. Özet formül kutusu

| adım | işlem | maliyet |
|------|-------|---------|
| 1 | $g_0 = M(v_0) - v_0$ (bir tur izle) | 1 koşum |
| 2 | Jacobian: 4 koordinatı $\epsilon$ oynat, birer tur | 4 koşum |
| 3 | $v_1 = v_0 - (M'-I)^{-1} g_0$ (4×4 çözüm) | anlık |
| 4 | Gerekirse tekrarla (bizde 2 kez) | ~5 koşum/adım |

**Tek cümlelik özet:** Kapalı yörünge "bir turda kendine dönen nokta"dır;
Newton, "ne kadar kaçtın / harita ne kadar dik" oranıyla bu noktaya iki üç
adımda kilitlenen standart kök-bulucudur — türev formülle değil, izleyiciye
dört küçük deneme yaptırılarak sayısal ölçülür.

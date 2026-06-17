# Temel İşlevli Çözüm (MVP) — Sistem Mimarisi ve Teknik Tasarım

**Makine Öğrenmesi Destekli Lojistik Anahat Optimizasyonu Yarışması**
*Dataset A — Tahmin + Atama Hibrit Çözümü*

> Doküman Versiyonu 5.0 — negatif tahmin, hayalet yük ve kümülatif mesafe düzeltmeleri

---

## İçindekiler

1. [Yönetici Özeti](#1-yönetici-özeti)
2. [Problem Tanımı](#2-problem-tanımı)
3. [Veri Setinin Yapısı](#3-veri-setinin-yapısı)
4. [Sistem Mimarisi — Genel Bakış](#4-sistem-mimarisi--genel-bakış)
5. [Katman 1 — Tahmin (Makine Öğrenmesi)](#5-katman-1--tahmin-makine-öğrenmesi)
6. [Katman 2 — Atama (Uğrama + Spot Optimizasyonu)](#6-katman-2--atama-uğrama--spot-optimizasyonu)
7. [Mesafe Hesabı](#7-mesafe-hesabı)
8. [Çıktı ve Değerlendirme Uyumu](#8-çıktı-ve-değerlendirme-uyumu)
9. [Yarışma Kurallarına Uyum Matrisi](#9-yarışma-kurallarına-uyum-matrisi)
10. [Teknoloji Yığını](#10-teknoloji-yığını)
11. [Riskler ve Önlemler](#11-riskler-ve-önlemler)
12. [Sonraki Adımlar](#12-sonraki-adımlar)

---

## 1. Yönetici Özeti

Bu doküman, yarışmanın **Temel İşlevli Çözüm (MVP) aşaması** için tasarladığımız sistem mimarisini, veri akışını, katman kararlarını ve bu kararların gerekçelerini içerir. Aşamanın kapsamı **Dataset A** ile sınırlıdır: 18 transfer merkezi (TM), 89 sabit çıkış→varış (O→D) hattı ve 130 günlük geçmiş talep verisi.

Mimarimiz üç ana katmandan oluşur: (1) günlük desi talebini **yansız (unbiased)** tahmin eden bir **LightGBM tahmin katmanı**, (2) zorunlu kiralık filoyu yerleştirip kalan yükte uğrama (multi-stop) fırsatlarını değerlendiren bir **uğrama + Greedy+ atama katmanı**, (3) rota-tabanlı bir **ILP doğrulama valfi**. Katman sırası kritiktir: kiralık araçlar her zaman önce yerleştirilir, uğrama yalnızca spot'a düşen kalan yük üzerinde çalışır.

**Tasarım felsefesi:** gereksiz karmaşıklıktan kaçınmak, ancak yarışma kriterlerinden ödün vermemek. Konsolidasyon kapalı olsa da **uğrama serbesttir**; aynı aracın yol üstündeki bir TM'ye uğrayıp yük alması/bırakması (aktarmasız) maliyet avantajı sağladığında yapılır. Bu fırsat veride güçlüdür: 89 hattın 65'inde uğrama adayı vardır ve örnek bir senaryoda iki yarı-dolu Tır yerine tek uğramalı Tır %48 tasarruf sağlar.

---

## 2. Problem Tanımı

### 2.1 Operasyonel Bağlam

Problem, lojistik operasyonunun **middle-mile (linehaul)** bacağına odaklanır: transfer merkezleri arasındaki yüksek hacimli taşımaların planlanması. Sistemde iki tür araç birlikte çalışır: talep dalgalanmasından bağımsız sabit maliyetli **kiralık araçlar** ve talep zirvelerini karşılayan **spot araçlar**. Amaç, kiralık araçların doluluğunu maksimize ederken ihtiyaç duyulan spot araç kullanımını minimum maliyetle planlamaktır.

### 2.2 Amaç Fonksiyonu

Toplam operasyonel maliyeti minimize et:

$$Z = \sum_{k \in K}\left(C_{k}^{\text{sabit}} + C_{k}^{\text{km}} \cdot D_{k}\right) + \sum_{s \in S}\left(C_{s}^{\text{sabit}} + C_{s}^{\text{km}} \cdot D_{s}\right)$$

Burada **K** kullanılan kiralık araç kümesini, **S** atanan spot araç kümesini, **C** ilgili maliyet kalemlerini (sabit = günlük kira/spot sabit; km = kilometre başına maliyet) ve **D** aracın kat ettiği toplam mesafeyi temsil eder. Uğramalı rotalarda D, doğrudan O→D mesafesi değil **rotadaki tüm bacakların Haversine mesafelerinin toplamıdır** (bkz. 6.2). Kiralık aracın günlük kirası ve kilometre başına maliyeti formülde net biçimde ayrışır. Maliyet yapısı doğrudan `Araç_Kapasite_Maliyet` dosyasındaki kalemlerle örtüşür ve tüm takımlar için standarttır.

### 2.3 Karar Değişkenleri

- **x[hat, araç_tipi, gün]** = ilgili hat-günde kullanılan o tipteki araç sayısı (spot için)
- **y[kiralık_araç, gün]** = kiralık aracın o gün kullanılıp kullanılmadığı (zorunlu kullanım)
- **Atanan desi[araç]** = her araca yüklenen desi miktarı (kapasite kısıtına tabi)

### 2.4 Kısıtlar

| Kısıt | Açıklama |
|---|---|
| Araç kapasitesi | Bacak bazlı: her rota bacağındaki anlık yük kapasiteyi aşamaz |
| Kiralık öncelik | Kiralık araçlar zorunlu, doluluğa bakılmaksızın önce çıkar |
| Minimum %10 doluluk | Yalnızca spot araçlara uygulanır; kiralığa asla |
| Talep karşılama | Tahmin edilen günlük desi taşınmalıdır |
| Konsolidasyon yok | Aktarmalı yük birleştirme kapalıdır |
| Uğrama serbest | Aynı araç yol üstü TM'ye uğrayabilir (yalnız spot, aktarmasız) |
| Artık yük | %10 spot eşiği altı yük karşılanamaz, ayrı raporlanır |
| Dönüş yok | Araçların geri dönüşü maliyete katılmaz |
| Günlük bağımsızlık | Her gün başlangıç konumlarından bağımsız değerlendirilir |

---

## 3. Veri Setinin Yapısı

Mimari kararları doğrudan verinin yapısı belirlediği için, dört kaynak dosya aşağıda özetlenmiştir.

| Dosya | Boyut | İçerik |
|---|---|---|
| Desi_talep | 10.770 satır | 89 O→D hattı × 130 gün (01.01–10.05.2026), günlük toplam desi |
| Koordinatlar_v2 | 18 TM | Her transfer merkezinin enlem/boylam değeri |
| Araç_Kapasite_Maliyet | 4 tip | Tır, Kamyon, Hafif Kamyon, Kamyonet kapasite ve maliyetleri |
| Kiralık_Araçlar | 12 satır | Sabit rotalara bağlı 14 kiralık araç (8 Tır + 4 Kamyon) |

### Araç tipleri ve maliyet yapısı

| Araç | Kapasite (desi) | Kira/gün (TL) | Kira km (TL) | Spot km (TL) |
|---|---|---|---|---|
| Tır | 22.400 | 7.000 | 13 | 25 |
| Kamyon | 12.000 | 5.000 | 10 | 21 |
| Hafif Kamyon | 7.200 | 5.000 | 10 | 20 |
| Kamyonet | 5.600 | 3.750 | 6 | 18 |

**Kritik içgörü:** 400 km'lik örnek bir sefer için spot araçların desi başına maliyeti — Tır 0,97 TL, Kamyon 1,34 TL, Hafif Kamyon 2,33 TL, Kamyonet 2,13 TL. Yani **büyük yükte Tır açık ara en verimli araçtır**; ancak küçük artık yüklerde sabit maliyeti düşük Kamyonet öne geçebilir. Atama motorunun mantığı tam olarak bu dengeyi kullanır.

---

## 4. Sistem Mimarisi — Genel Bakış

Sistem üç katmanlı bir hat (pipeline) olarak çalışır. Katman sırası **operasyonel olarak zorunludur**: kiralık filo her zaman önce yerleştirilir, uğrama değerlendirmesi yalnızca kalan yük üzerinde yapılır.

### 4.1 Uçtan Uca Veri Akışı

1. **Ham veri → Feature mühendisliği:** geçmiş desi serisinden lag, hareketli ortalama ve takvim özellikleri türetilir.
2. **Tahmin katmanı:** tek global LightGBM modeli, 11–17 Mayıs haftası için yansız günlük desi tahmini üretir.
3. **Mesafe katmanı:** 18 TM koordinatından Haversine (kuş uçuşu) ile 18×18 mesafe matrisi hesaplanır.
4. **Kiralık yerleştirme:** hatta tanımlı kiralık araçlar zorunlu olarak, doluluğa bakılmaksızın plana eklenir.
5. **Uğrama katmanı:** yalnızca spot'a düşen kalan yük için yol üstü uğrama adayları üretilir ve maliyet kıyasıyla seçilir.
6. **Spot atama (Greedy+):** kalan yük en uygun spot araçlara atanır; %10 doluluk kuralı yalnız burada uygulanır.
7. **Doğrulama:** rota-tabanlı ILP valfi, uğrama rotalarını da içererek örnek günlerde optimum maliyeti kıyaslar.
8. **Çıktı:** example_solution formatında gün bazlı plan + toplam maliyet ve metodoloji notu.

> **Not:** Güncellenmiş atama akışının görsel diyagramı (kiralık önce → uğrama → spot → ILP) ayrıca sunulmuştur ve bu dokümanın eki niteliğindedir.

---

## 5. Katman 1 — Tahmin (Makine Öğrenmesi)

### 5.1 Seçim: Tek Global LightGBM Modeli

89 hattın tamamı için **hat kimliğini bir özellik (feature) olarak içeren tek bir LightGBM modeli** eğitilir. Hat-bazlı ayrı modeller (örn. her hatta ayrı Prophet) yerine bu yaklaşım seçilmiştir.

#### Neden tek model?

| Kriter | Tek global LightGBM |
|---|---|
| Bakım yükü | Tek pipeline, tek model — 89 ayrı modele karşı çok düşük |
| Az veri olan hatlar | Diğer hatlardan bilgi paylaşımıyla öğrenir, sağlam kalır |
| Doğruluk/emek oranı | En yüksek; feature emeği bir kez harcanır |
| Hata yüzeyi | Küçük; tek noktada izlenir ve doğrulanır |

### 5.2 Feature Mühendisliği

- **Lag özellikleri:** 1, 7 ve 14 gün önceki desi değerleri (haftalık çevrim yakalanır)
- **Hareketli ortalamalar:** 7 ve 14 günlük kayan ortalama ve standart sapma
- **Takvim özellikleri:** haftanın günü, ay, ayın günü, ay başı/sonu bayrakları
- **Tatil/bayram bayrakları:** resmi tatil ve dini bayram günleri (talep sıçramalarını açıklar)
- **Hat kimliği:** çıkış ve varış TM kategorik olarak modele verilir

### 5.3 Doğrulama Stratejisi

Zaman serisi yapısını koruyan **time-series cross-validation** (ileriye dönük katlamalar) ile hiperparametreler ayarlanır. Böylece model gelecekteki veriyi geçmişe bakarak değerlendirme hatasına düşmez. Metrik olarak MAE ve MAPE birlikte izlenir.

**Önemli — yansız tahmin:** Bu aşamada (Dataset A) taşınamayan yük için **herhangi bir SLA ceza katsayısı tanımlı değildir**, ve eleme kriterlerinden biri doğrudan tahminleme başarısıdır. Bu nedenle **varsayılan hedef yansız (unbiased)** tahmindir: agresif yukarı bias, jürinin tahmin doğruluğu değerlendirmesinde hata oranını artırır.

**İnce ayar valfi (opsiyonel asimetrik kayıp):** Değerlendirme aynı zamanda planın gerçek talebe karşı maliyetine bakar. Gerçek talep tahminin üstüne çıktığında karşılanamayan yük başarı ölçümünü düşürebilir. Bu riske karşı, LightGBM'de **çok hafif asimetrik kayıp (quantile regresyon, örn. ~0,55 quantile)** kullanılarak eksik tahminin (under-prediction) cezası fazla tahmine göre hafifçe artırılabilir. Bu, agresif bias değildir — nokta doğruluğunu büyük ölçüde korurken "eksik araç çıkarmaktansa aracın bir miktar daha dolu gitmesi" dengesini kurar. Asimetri derecesi bir hiperparametredir; yansız taban ile kıyaslanarak hem MAE/MAPE hem plan maliyeti üzerinde valide edilir ve yalnızca her ikisini de iyileştiriyorsa devreye alınır.

### 5.4 Çok-Adımlı (Multi-Step) Tahmin — Lag Tuzağı

**Kritik problem:** Feature setinde **lag_1** (1 gün önceki desi) bulunur. Eğitimde sorun yoktur, ancak tahmin haftası (11–17 Mayıs) çok-adımlıdır. 10 Mayıs'ta durup 11 Mayıs'ı tahmin ederken lag_1 = 10 Mayıs gerçek değeridir. Fakat 12 Mayıs'ı tahmin ederken lag_1 = 11 Mayıs olur ve **o gün henüz gerçekleşmemiştir**. Modele ham haliyle lag_1 verilirse NaN döner veya model hata üretir. Bu, naif bir tek-adım kurgunun en sık patladığı noktadır.

**Çözüm — recursive omurga + bilinen feature ayrımı:** Tahmin ufkundaki her gün için feature'lar iki gruba ayrılır:

- **Bilinen feature'lar:** takvim (haftanın günü, ay, tatil/bayram) ve 10 Mayıs ile öncesinden türeyen sabit istatistikler — tahmin anında kesin bilinir, doğrudan kullanılır.
- **Lag feature'ları (recursive):** 11 Mayıs tahmini üretilir; bu tahmin 12 Mayıs'ın lag_1'i olarak beslenir; 12 üretilir, 13'ün lag_1'i olur — böyle ilerlenir. Yani gelecekteki lag'ler gerçek değil, modelin kendi tahminiyle doldurulur.

**Hata birikimi önlemi:** Recursive yaklaşımda erken günün hatası sonraki günlere taşınır. Bunu sınırlamak için: (a) kısa ufuk (7 gün) zaten birikimi sınırlar; (b) lag ağırlığı, takvim ve hareketli ortalama gibi bilinen sinyallerle dengelenir; (c) doğrulama, **walk-forward (ileriye dönük) çok-adımlı simülasyonla** yapılır — yani CV sırasında da tam olarak tahmin haftasındaki gibi recursive üretim test edilir, böylece raporlanan başarı gerçek koşulu yansıtır. Bu kurgu, tek-adım CV'nin yarattığı yanıltıcı iyimserliği ortadan kaldırır.

### 5.5 Negatif Tahmin Koruması (Zero-Floor)

**Problem:** LightGBM regresyon modunda (MAE/MSE hedefleriyle) düşüş trendlerine veya tatil dip günlerine denk geldiğinde **negatif desi** üretebilir (örn. -150 desi). Atama motoruna negatif desi girerse araç sayısı mantığı çöker, hatta toplam maliyet eksiye düşerek matematiksel anomali yaratır.

**Çözüm — zero-floor:** Tahmin çıktısına **maks(0, tahmin)** uygulanır; negatif değerler sıfıra çekilir. Bu, lojistik olarak da doğrudur — negatif tahmin "o gün talep yok/çok az" demektir. Kritik sıralama: zero-floor, **kanonik yuvarlamadan önce** uygulanır; böylece hem Tahminlenen Talep dosyası hem atama motoru aynı pozitif kanonik değerden beslenir ve invariant korunur.

---

## 6. Katman 2 — Atama (Uğrama + Spot Optimizasyonu)

### 6.1 Problemin Yapısı ve Katman Sırası

Konsolidasyon ve dönüş yoktur, ancak **uğrama serbesttir**. Bu nedenle problem saf bağımsız bin-packing değil, **kiralık öncelikli, kalan yükte küçük bir multi-stop rota problemidir**. Atama dört mantıksal aşamada ilerler ve sıra zorunludur:

1. **Kiralık yerleştirme (zorunlu, ilk):** hatta tanımlı kiralık araçlar — doluluğa bakılmaksızın — plana eklenir. Parası ödenmiş ve çıkması zorunlu filo asla atlanmaz.
2. **Kalan yük:** kiralık kapasiteyi aşan kısım hesaplanır; uğrama ve spot yalnızca bu kalan üzerinde çalışır.
3. **Uğrama değerlendirmesi:** kalan yük için yol üstü aday rotalar üretilir, maliyet kıyasıyla seçilir.
4. **Spot atama + %10:** seçilen rotalar spot araçlara atanır; %10 doluluk yalnız burada uygulanır.

### 6.2 Uğrama Katmanı

Uğrama, aynı aracın O→D giderken yol üstündeki bir veya birden çok TM'ye uğrayıp yük alması/bırakmasıdır (aktarmasız — bu yüzden konsolidasyon değildir). Aday üretimi:

- **Yol üstü kriteri:** mesafe(O→…→D) ≤ mesafe(O→D) × 1,15 (sapma eşiği %15)
- **Çok-durak, coğrafi sıralı:** ara duraklar O'dan uzaklığa göre dizilir; veride 36 hatta iki-durak fırsatı bulunduğu için tek-durakla sınırlanmaz
- **Karar maliyetle:** eşik yalnızca aday kapısıdır; uğrama ancak ayrı araçlardan ucuzsa uygulanır, değilse reddedilir

> **Kesin kısıt — yalnızca spot:** Soru-Cevap oturumunda kiralık araçların "datasette tanımlı rotalar, araç sayıları ve araç tipleriyle sınırlı" olduğu açıkça belirtilmiştir. Bu nedenle **uğrama algoritması kesinlikle kiralık araçlara uygulanmaz**; kiralık araç katı biçimde tanımlı O→D rotasında çalışır. Uğrama yalnızca spot havuzu için çalıştırılır. Bu, kiralık aracı tanımlı rotasının dışına çıkarıp kural ihlali/diskalifikasyon riskini ortadan kaldırır.

#### Dinamik (bacak bazlı) kapasite — kritik düzeltme

Uğramalı rotada yük statik olarak toplanmaz. O→C→D rotasında C'de yükün bir kısmı iner, dolayısıyla doluluk **rota boyunca değişir**. Kapasite kontrolü her bacak için ayrı yapılır: O→C bacağında araçta "C'de inmeyip D'ye gidecek + C'de teslim edilecek" yük; C→D bacağında "C'de inmeyenler + C'de binenler" bulunur. Kısıt: **her bacaktaki anlık yük ≤ araç kapasitesi**. O→C ve C→D taleplerini doğrudan toplayıp kapasiteyle kıyaslamak hatalı olur.

#### Pik yük (bottleneck) araç seçimi

Bacak bazlı kapasite kontrolü kadar kritik bir nokta: araç O→D boyunca **aynı araçtır**, bacaktan bacağa değişmez. Bu yüzden araç tipi son bacağa göre değil, **rotanın herhangi bir bacağındaki maksimum anlık yüke (pik yük)** göre seçilmelidir. Kural: **gerekli kapasite ≥ maks(tüm bacaklardaki anlık yük)**.

**Örnek:** İstanbul→Kocaeli bacağında 12.000 desi, Kocaeli→Sivas bacağında 1.000 desi taşınıyorsa, son bacağa bakıp "Kamyonet yeter" denemez. Pik yük 12.000 olduğu için en az **Kamyon (kap. 12.000)** gerekir. Aksi halde araç İstanbul→Kocaeli bacağında kapasiteyi aşar ve plan geçersiz olur.

#### Kümülatif mesafe maliyeti

Uğramalı rotada maliyet hesabındaki **D (mesafe)** değişkeni, başlangıçtan varışa doğrudan Haversine mesafesi DEĞİLDİR. Araç tüm ara durakları fiilen kat ettiği için D, **rotadaki tüm bacakların Haversine mesafelerinin toplamıdır**: O→C→D için D = Haversine(O→C) + Haversine(C→D). Doğrudan Haversine(O→D) kullanmak, sapma arttıkça maliyeti eksik gösterir ve hem maliyet barajını hem uğrama kıyasını yanıltır. Sapma eşiği (%15) bu kümülatif mesafe üzerinden uygulanır.

### 6.3 Spot Seçim ve %10 Doluluk

Seçilen her rota için araç tipi, desi başına etkin maliyete göre belirlenir (büyük yükte Tır en verimli, küçük artıkta sabit maliyeti düşük Kamyonet öne geçebilir). **%10 doluluk kuralı yalnızca spot araçlara** ve atama anında uygulanır — global bir son-filtre değildir. Kiralık araçlar bu kurala tabi değildir; az dolu olsalar bile plandadır.

#### Hayalet yük (phantom cargo) — atanan desi ≠ kapasite

**Problem:** Kiralık araç doluluğa bakılmaksızın çıkar. Bir hatta tahmini talep 5.000 desi ve zorunlu 1 Kiralık Tır (kap. 22.400) varsa, araç çıkar. Ancak Araç Planlama dosyasına "Atanan Desi" olarak aracın kapasitesi (22.400) yazılırsa, sistemde var olmayan 17.400 desilik **hayalet yük** görünür ve "kanonik tahmin = atanan + karşılanamayan" invariantı patlar.

**Kural:** "Atanan Desi" her zaman **araca yüklenen gerçek yüktür, kapasite değil**. Örnekte atanan desi = 5.000 (doluluk %22,3), kapasite yalnızca üst sınır kontrolüdür. Maliyet zaten kapasiteden değil, sabit kira + km'den hesaplanır; araç boş bile gitse maliyeti vardır ama atanan desisi gerçek yüktür. Böylece invariant korunur: atanan(5.000) = tahmin(5.000).

#### Artık yük (orphaned demand) terminal kuralı

En küçük spot araç Kamyonet'tir (kapasite 5.600 desi); %10 eşiği **560 desi**. Bir hat-günde kiralık ve uğrama sonrası kalan yük bu eşiğin altındaysa (örn. 450 desi), hiçbir spot araç %10 doluluğa ulaşamaz. Bu durumda motor sonsuz döngüye girmemeli; net terminal kural uygulanır:

**Kural:** Kalan talep, hattaki en küçük spot aracın %10 kapasitesini (560 desi) dolduramıyorsa spot araç atanmaz; bu desi **"Karşılanamayan Artık Yük"** olarak işaretlenir ve ayrı bir tabloda raporlanır. Talepler her gün bağımsız değerlendirildiği için bu yük **bir sonraki güne devredilmez**.

**Karşılanamayanlar tablosu:** Artık yük, ana araç planını kirletmeden ayrı bir **unfulfilled** tablosunda tutulur. Bu yapısal ayrım, otomatik değerlendirme scriptinin araç planını okurken bu satırları araç ataması sanmasını önler ve jüri savunmasında şeffaflık sağlar ("bu yük bir hata değil, %10 kuralı gereği bilinçli olarak bırakıldı"). Kolonlar: `Tarih, Çıkış TM, Varış TM, Karşılanamayan Desi, Sebep ("%10 spot eşiği altında")`.

### 6.4 Rota-Tabanlı ILP Doğrulama Valfi

Greedy+ artık uğrama rotaları ürettiği için, yalnız O→D değişkenlerine sahip bir ILP **optimum kanıtı olma özelliğini kaybeder** — Greedy+'tan daha kötü sonuç verirdi. Bu yüzden valf **rota-tabanlı (set-partitioning) formülasyonla** kurulur: önce geçerli aday rotalar üretilir (doğrudan O→D + geçerli uğrama rotaları, bacak kapasitesine uygun), sonra ILP her talebi tam bir kez karşılayacak en ucuz rota kombinasyonunu seçer. Bu formülasyon uğramayı da kapsadığı için gerçek optimum kanıtı sağlar.

**Çıktı:** "Greedy+ çözümümüz rota-tabanlı exact optimumla %X fark gösteriyor." Bu, jüri savunmasında hem optimumda olduğumuzu kanıtlar hem de günlük işi hafif tutar.

### 6.5 Karşılaştırma

| Boyut | Saf Greedy | Greedy+ uğramalı | Rota-tabanlı ILP |
|---|---|---|---|
| Uğrama desteği | Yok | Var | Var |
| Optimuma yakınlık | Zayıf | Yüksek | %100 garanti |
| Bağımlılık | Yok | Yok | Solver |
| Rol | — | Ana motor | Doğrulama valfi |

---

## 7. Mesafe Hesabı

Soru-Cevap oturumunda kesinleştiği üzere mesafeler **Haversine (kuş uçuşu)** formülüyle, herhangi bir düzeltme katsayısı olmadan hesaplanır. Gerçek karayolu mesafesi kullanılmaz; bu, tüm takımlar için standardı korur. 18 TM için bir kez 18×18 mesafe matrisi üretilir ve atama motoru bunu sabit girdi olarak kullanır.

**Örnek mesafeler:** İstanbul→Yalova 47 km, İstanbul→Eskişehir 189 km, Mersin→Şanlıurfa 372 km, İstanbul→Mardin 1.091 km.

**Uğrama kazanç örneği:** İstanbul→Sivas (300 desi) ve Kocaeli→Sivas (200 desi) talepleri için iki ayrı yarı-dolu spot Tır ≈ 56.209 TL tutar. Kocaeli yol üstünde olduğundan (İst→Koc 78 km), tek Tır İstanbul'dan çıkıp Kocaeli'ye uğrayarak 500 desiyi tek seferde taşır ≈ 29.079 TL. Tasarruf **%48** — uğrama katmanının değerinin somut kanıtı.

---

## 8. Çıktı ve Değerlendirme Uyumu

### 8.1 Teslim Çıktıları ve Birebir Kolon Formatı

**Kritik:** Format uyumu eleme ön koşuludur ve değerlendirme otomatik script ile yapılabilir. Bu nedenle kolon başlıkları, organizasyonun Soru-Cevap oturumunda verdiği örnek tabloya birebir uyacak biçimde yazılır.

#### Tahminlenen Talep dosyası

| Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi |
|---|---|---|---|
| 2026-05-11 | İstanbul | Ankara TM | 1250.5 |

#### Araç Planlama dosyası

| Tarih | Araç Tipi | Çıkış TM | Varış TM | Atanan Desi | Maliyet |
|---|---|---|---|---|---|
| 2026-05-11 | Kiralık TIR | İstanbul | Ankara TM | 4800 | 2340 |

**Sağlamlık — kolon eşleme (mapping) katmanı:** Değerlendirme tarafındaki olası yazım varyantlarına (örn. "Çakış/Çıkış", "Mallyet/Maliyet") karşı çıktı üreticisine küçük bir mapping katmanı eklenir: bizim ürettiğimiz standart başlıklar, beklenen başlığın varyantlarıyla eşleştirilebilir biçimde tutulur. Böylece organizasyon tarafındaki bir tipografik fark teslimimizi bozmaz.

- **Karşılanamayanlar (unfulfilled) dosyası:** Tarih, Çıkış TM, Varış TM, Karşılanamayan Desi, Sebep — %10 eşiği altında kalan artık yükün ayrı raporu
- **Toplam maliyet:** plan genelinin toplam operasyonel maliyeti
- **Çalışan kaynak kod:** Python/notebook — kod olmadan teslim değerlendirilmez
- **Metodoloji notu:** varsayımlar ve yaklaşımı açıklayan kısa doküman (opsiyonel ama önerilir)

### 8.2 Float/Integer Tutarlılığı — Kanonik Yuvarlama

**Problem:** LightGBM `.predict()` float döner (örn. 1250,56 desi). Atama motoru bu talebi araçlara dağıtırken farklı yuvarlama yaparsa, Tahminlenen Talep dosyasındaki O→D toplamı (1250,56) ile Araç Planlama dosyasındaki atanan desi toplamı (1251 veya 1250) uyuşmaz. Otomatik değerlendirme scripti bunu "tahmin edilen yük ile taşınan yük eşleşmiyor" olarak okuyup pipeline'ı kırabilir.

**Çözüm — tek kanonik kaynak:** Tahmin üretilir üretilmez sabit bir ondalık basamağa (örn. 2 hane) yuvarlanır ve **bu kanonik değer hem Tahminlenen Talep dosyasına hem de atama motorunun girdisine tek kaynak olur**. Atama, ham float'tan değil, yalnızca bu yuvarlanmış değerden beslenir. Böylece iki dosya aynı sayıdan türer.

**İnvariant kontrolü:** Her hat-gün için "atanan desi toplamı + karşılanamayan desi = kanonik tahmin" eşitliği bir doğrulama adımıyla zorlanır. Eşitlik bozulursa pipeline uyarı verir ve teslim öncesi yakalanır. Bu, format elemesine karşı son güvenlik katmanıdır.

### 8.3 Değerlendirme Mantığı

Değerlendirme, **sizin tahmin ettiğiniz desi üzerinden kurduğunuz araç planının gerçek talebe karşı yarattığı maliyete** göre yapılır. Yani tahmin doğruluğu ve optimizasyon başarısı birlikte sonucu belirler. Eleme kriteri hem spot maliyeti hem tahminleme başarısıdır; bu aşamada referans maliyet barajının altında kalmak esastır.

---

## 9. Yarışma Kurallarına Uyum Matrisi

Soru-Cevap oturumunda kesinleşen kurallar ve mimarideki karşılıkları:

| Kural (Soru-Cevap) | Mimarideki karşılığı |
|---|---|
| Talep tahmini yapılacak | LightGBM tahmin katmanı |
| Konsolidasyon kapalı | Hatlar bağımsız çözülür, birleştirme yok |
| Mesafe = Haversine kuş uçuşu | 18×18 Haversine matrisi, katsayısız |
| %10 doluluk kuralı | Atama motorunun spot adımı |
| Kiralık öncelikli ve zorunlu | Atama motorunun 1. adımı |
| Dönüş hesaba katılmaz | Maliyet sadece tek yön sefer |
| Her gün bağımsız, başlangıç konumundan | Günlük bağımsız çözüm döngüsü |
| Yeni hat tanımlanamaz | Yalnızca veri setindeki 89 hat kullanılır |
| Maliyet = kira + km başına | Amaç fonksiyonu her iki kalemi içerir |
| Uğrama serbest, konsolidasyon değil | Tek-yön uğrama maliyet avantajındaysa serbest |
| Kiralık tanımlı rotayla sınırlı | Uğrama yalnız spota; kiralık katı O→D rotasında |
| %10 altı yük karşılanamaz | Artık yük unfulfilled tablosunda raporlanır |

---

## 10. Teknoloji Yığını

| Bileşen | Teknoloji |
|---|---|
| Dil | Python 3 |
| Veri işleme | pandas, numpy |
| Tahmin modeli | LightGBM, scikit-learn (CV ve metrikler) |
| Mesafe | Haversine (numpy ile vektörize) |
| Atama motoru | Saf Python (Greedy+) |
| Doğrulama valfi | PuLP veya OR-Tools (yalnızca örnek günler) |
| Çıktı | openpyxl (xlsx üretimi) |

---

## 11. Riskler ve Önlemler

| Risk | Etki | Önlem |
|---|---|---|
| Düşük tahmin → taşınamayan yük | Maliyet riski | Yansız tahmin + atama verimliliği |
| Uğramada statik kapasite hatası | Geçersiz plan | Bacak bazlı dinamik kapasite kontrolü |
| ILP'nin uğramadan habersizliği | Yanlış optimum kanıtı | Rota-tabanlı ILP formülasyonu |
| Negatif tahmin (-desi) | Motor çökmesi/anomali | Zero-floor: maks(0, tahmin) |
| Kiralıkta hayalet yük | İnvariant patlaması | Atanan desi = gerçek yük, kapasite değil |
| Uğramada eksik mesafe | Yanlış maliyet | Kümülatif bacak mesafesi toplamı |
| Multi-step lag tuzağı (NaN) | Tahmin çökmesi | Recursive omurga + walk-forward CV |
| Uğramada son-bacak araç seçimi | Geçersiz plan | Pik yüke göre araç boyutlandırma |
| Float toplam uyuşmazlığı | Pipeline kırılması | Kanonik yuvarlama + invariant kontrolü |
| %10 altı artık yük (sonsuz döngü) | Hata/çökme | Terminal kural + unfulfilled tablosu |
| Kiralık araca uğrama uygulanması | Diskalifikasyon | Uğrama yalnız spot havuzunda |
| Kolon adı sapması | Format elemesi | Birebir başlık + mapping katmanı |
| Format uyumsuzluğu | Eleme (ön koşul) | example_solution'a birebir uyum |
| Optimumdan sapma şüphesi | Savunma zayıflığı | ILP valfi ile kanıt |
| Veri sızıntısı (CV) | Yanıltıcı başarı | Time-series CV, ileriye dönük katlama |

---

## 12. Sonraki Adımlar

1. Feature pipeline'ının kurulması ve geçmiş veride doğrulanması
2. LightGBM modelinin eğitimi ve time-series CV ile ayarı
3. Haversine mesafe matrisinin üretilmesi
4. Kiralık yerleştirme + uğrama katmanı + Greedy+ spot atamanın yazılması (sıra korunarak)
5. Rota-tabanlı ILP doğrulama valfinin örnek günlerde çalıştırılması
6. example_solution formatında çıktı üretimi ve toplam maliyet raporu
7. Metodoloji notunun yazılması ve kod deposunun hazırlanması

> *Bu doküman MVP aşaması için kararlaştırılan mimariyi ve gerekçelerini kapsar. Gelişmiş Çözüm (Dataset B) aşamasına geçişte PyVRP/ALNS, konsolidasyon operatörleri ve stokastik tahmin yöntemleri devreye alınacaktır.*

# Yüz Biyometrisi İçin Federe Öğrenme (Extreme Non-IID)

Bu proje, yüz tanıma görevleri için özel olarak tasarlanmış, **aşırı Non-IID** veri dağılımları altında çalışan bir **Federe Öğrenme (Federated Learning)** çerçevesidir. Proje, kimlik tabanlı veri ayrımı (identity-based partitioning), gelişmiş birleştirme algoritmaları ve gizlilik koruma yöntemlerini içerir.

---

## 📊 Algoritma İzlenebilirlik Raporu (Kod Haritası)

Bu tablo, projede kullanılan temel algoritmaların ve tekniklerin hangi dosyalarda ve hangi satır aralıklarında uygulandığını detaylandırmaktadır.

| Algoritma / Teknik       | Uygulandığı Dosya | Satır Aralığı    | Açıklama                                                                                                            |
| :----------------------- | :---------------- | :--------------- | :------------------------------------------------------------------------------------------------------------------ |
| **FedAvg**               | [aggregation.py]  | 22-26            | İstemci güncellemelerinin (deltaların) örnek sayısına göre ağırlıklı ortalaması.                                    |
| **FedProx**              | [client.py]       | 102-106          | Yerel sapmayı (drift) sınırlamak için loss fonksiyonuna eklenen Proximal Terim.                                     |
| **FedProx (Agg)**        | [aggregation.py]  | 22-26            | FedProx, sunucu tarafında FedAvg ile aynı birleştirme mantığını kullanır.                                           |
| **SCAFFOLD**             | [client.py]       | 117-122, 191-203 | Control Variates (Kontrol Değişkenleri) ile gradyan düzeltmesi ve yerel drift güncellemesi.                         |
| **SCAFFOLD (Global)**    | [server.py]       | 191-199          | Sunucu tarafında küresel kontrol değişkeninin (c_global) güncellenmesi.                                             |
| **FedNova**              | [aggregation.py]  | 29-37            | Yerel iterasyon (local steps) farklılıklarını normalize eden ağırlıklı birleştirme.                                 |
| **Proposed (Cosine)**    | [aggregation.py]  | 62-67, 71-72     | Bir önceki raunt güncellemesi ile mevcut istemci güncellemesi arasındaki kosinüs benzerliğine göre ağırlıklandırma. |
| **Proposed (Norm)**      | [aggregation.py]  | 56-60, 73-74     | İstemci güncellemelerinin L2 normuna (sapma miktarı) göre ters orantılı ağırlıklandırma.                            |
| **Proposed (Combined)**  | [aggregation.py]  | 75-79            | Cosine similarity ve Gradient Norm metriklerinin hibrit (birleştirilmiş) kullanımı.                                 |
| **ArcFace Loss**         | [model.py]        | 36-79            | Açısal pay (Angular Margin) ekleyerek sınıf içi benzerliği artıran kayıp fonksiyonu.                                |
| **CosFace Loss**         | [model.py]        | 81-111           | Kosinüs payı (Cosine Margin) tabanlı büyük marjlı sınıflandırma.                                                    |
| **Differential Privacy** | [privacy.py]      | 81-103           | Gradyan kırpma (clipping) ve Gauss gürültüsü ekleyerek veri gizliliğini sağlama.                                    |
| **DLG Attack**           | [privacy.py]      | 7-79             | Deep Leakage from Gradients (Gradyanlardan Veri Sızıntısı) saldırı simülasyonu.                                     |
| **Margin Warmup**        | [client.py]       | 64-72            | ArcFace marjının ilk turlarda kademeli artırılarak eğitimin stabilize edilmesi.                                     |
| **Drift Norm Takibi**    | [client.py]       | 206-210          | Her istemcinin küresel modelden ne kadar saptığının L2 normu ile hesaplanması.                                      |

---

## 🚀 Temel Özellikler

- **Aşırı Non-IID (Identity-Based):** CelebA veri seti, kimlikler (identities) cihazlar arasında asla paylaşılmayacak şekilde bölünmüştür.
- **Backbone & Local Head Mimarisi:** ResNet-18 omurgası (backbone) federatif olarak eğitilirken, ArcFace katmanları her istemcide yerel (local) tutulur ve sunucu tarafından kalıcılığı (persistence) sağlanır.
- **Dinamik Ölçekleme (Scale Warmup):** ArcFace ölçek parametresi (`s`), ilk rauntlarda kademeli artırılarak gradyan patlamaları önlenir.
- **Kapsamlı Metrikler:** Sadece kayıp (loss) değil, aynı zamanda Top-1/Top-5 doğruluk, model sapması (drift) ve "unseen" (hiç görülmemiş) test setinde Face Verification başarımı ölçülür.

---

## 📂 Dosya Yapısı

- `main.py`: Deneylerin ana giriş noktası. Algoritmaları yarıştırır ve görselleştirir.
- `server.py`: Küresel model yönetimi, istemci seçimi ve SCAFFOLD küresel güncellemeleri.
- `client.py`: Yerel eğitim döngüsü, FedProx/SCAFFOLD yerel mantığı ve DP uygulaması.
- `aggregation.py`: **Tüm birleştirme algoritmalarının merkezi.** FedAvg, FedNova ve Önerilen (Proposed) yöntemler burada bulunur.
- `model.py`: `FaceResNet18`, `ArcFaceLoss` ve `CosFaceLoss` tanımları.
- `privacy.py`: Diferansiyel Gizlilik (DP) ve DLG saldırı araçları.
- `dataset.py`: CelebA Identity-Based veri yükleyici.

---

## 🛠️ Kullanım

### 1. Hazırlık

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

### 2. Veri Bölümleme

Veri setini 50 istemciye kimlik tabanlı olarak ayırmak için:

```bash
python partition_data.py
```

### 3. Eğitim ve Kıyaslama

Farklı algoritmaları test etmek için `main.py` dosyasını çalıştırın:

```bash
python main.py
```

Eğitim sonunda `results_plot.png` ve her algoritma için `live_plot_{algo}.png` dosyaları otomatik olarak oluşturulacaktır.

---

## 📈 Ölçüm ve Değerlendirme

Sistem her rauntta şu metrikleri raporlar:

- **Avg Loss:** Seçilen istemcilerin ortalama eğitim kaybı.
- **Avg Acc:** İstemci tarafındaki yerel Top-1 doğruluk oranı.
- **Test Acc (Verification):** Hiç görülmemiş test kimlikleri üzerinde modelin yüz doğrulama başarımı (Cosine Similarity tabanlı).
- **Avg Drift:** İstemci modellerinin küresel modelden uzaklaşma miktarı (L2).

---

> [!NOTE]
> Bu proje, akademik araştırma amaçlı olup federatif öğrenmede model sapmasını (client drift) azaltmaya yönelik yeni yöntemlerin test edilmesi için geliştirilmiştir.


# Yüz Biyometrisi İçin Federe Öğrenme (Extreme Non-IID)

## Genel Bakış

Bu depo (repository), yüz tanıma görevleri için özel olarak tasarlanmış bir **Federe Öğrenme (FL)** sisteminin PyTorch uygulamasını içerir. Sistem, her istemcinin (cihazın) yalnızca belirli kişilerin fotoğraflarına erişebildiği gerçekçi bir senaryoyu simüle etmek amacıyla, **kimlik tabanlı aşırı Non-IID (dağılımı özdeş olmayan)** şeklinde bölümlere ayrılmış **CelebA veri setini** kullanır.

Ana model, bir özellik çıkarıcı (backbone) olarak kullanılan **ResNet-18** tabanlıdır ve istemcilerde yerel olarak **ArcFace** kayıp fonksiyonu (loss function) ile eğitilir. Proje, aşırı Non-IID veri dağılımının neden olduğu ağırlık sapmalarını (istemci kayması - client drift) azaltmak için çeşitli birleştirme (aggregation) stratejilerini incelemektedir.

## Temel Özellikler

- **Aşırı Non-IID Veri Bölümleme:** İstemcilere, gerçek dünya koşullarını simüle etmek için birbirini dışlayan (mutually exclusive) kimlikler atanmıştır.
- **Yerel ArcFace Eğitimi ve Kalıcılık (Persistence):** Her istemci, sunucuyla _paylaşılmayan_ yerel bir ArcFace sınıflandırma katmanı eğitir. Kritik bir güncelleme ile bu yerel ağırlıklar artık sunucu tarafında istemci bazlı olarak saklanmakta ve her rauntta sıfırlanmak yerine kaldığı yerden devam etmektedir. Bu, modelin gerçek anlamda yakınsamasını sağlar.
- **Gelişmiş Veri Artırma:** Eğitim sırasında `RandomHorizontalFlip` ve `ColorJitter` uygulanarak modelin yüz tanıma konusundaki genel performans ve dayanıklılığı artırılmıştır.
- **Hata Oranı ve Doğruluk Takibi:** Sistem artık sadece "Loss" değerini değil, aynı zamanda "% Error Rate" ve "% Accuracy" değerlerini de takip ederek grafikler oluşturur.
- **GPU Optimizasyonu:** Kodlar NVIDIA GPU (RTX serisi) üzerinde en yüksek verimlilikle çalışacak şekilde ayarlanmıştır. CUDA desteği otomatik olarak algılanır ve eğitim sürecini hızlandırır.
- **Çeşitli Birleştirme Algoritmaları:**
  - `fedavg`: Standart Federe Ortalama (Standard Federated Averaging).
  - `fedprox`: Yerel güncellemeleri kısıtlamak için Federe Proksimal optimizasyon.
  - `proposed_cosine`: Kosinüs benzerliğine dayalı özel birleştirme ağırlıklandırması.
  - `proposed_norm`: Güncelleme normlarına dayalı özel birleştirme ağırlıklandırması.
  - `proposed_combined`: Hibrit bir özel birleştirme yaklaşımı.
- **Gizlilik Koruma:** İstemci verilerini çıkarım saldırılarına karşı korumak için gürültü ekleyen ve gradyanları kırpan Diferansiyel Gizlilik (Differential Privacy - DP) desteği.

## Dizin Yapısı

- `main.py`: FL deneylerini çalıştırmak ve farklı algoritmaları kıyaslamak için giriş noktasıdır. Yakınsama grafiklerini oluşturur.
- `server.py`: İstemci örnekleme, küresel model yönetimi ve birleştirmeden sorumlu `FLServer` sınıfını içerir.
- `client.py`: Yerel eğitim (ResNet-18 omurga + ArcFace) ve diferansiyel gizlilik uygulamaktan sorumlu `FLClient` sınıfını içerir.
- `dataset.py`: Bölümlenmiş CelebA görüntülerini yüklemek ve kimlik etiketlerini eşlemek için PyTorch `Dataset` sınıfını (`CelebA_IdentityBased_Dataset`) içerir.
- `model.py`: `FaceResNet18` ve `ArcFaceLoss` dahil olmak üzere sinir ağı tanımlarını içerir.
- `aggregation.py`: Çeşitli sunucu tarafı birleştirme stratejilerinin uygulamalarını içerir.
- `privacy.py`: Model güncellemelerine Diferansiyel Gizlilik (DP) gürültüsü uygulamak için yardımcı araçlar.
- `partition_data.py`: CelebA veri setini aşırı Non-IID alt kümelere ayırmak ve `fl_partition.json` dosyasını oluşturmak için kullanılan betik.
- `download_identity.py`: Gerekli kimlik meta verilerini indirmek için yardımcı betik.
- `fl_partition.json`: İstemcileri belirli görüntülerle eşleyen veri bölümleme haritası.

## Gereksinimler

- Python 3.8+
- PyTorch & Torchvision
- PIL (Pillow)
- Matplotlib

### Donanım ve CUDA Notu (RTX 50-Serisi Kullanıcıları İçin)

Eğer RTX 5060 Ti veya Blackwell mimarili (`sm_120`) bir kart kullanıyorsanız, mevcut PyTorch sürümleri henüz bu kartları desteklemiyor olabilir. Bu durumda kod otomatik olarak **CPU** moduna geçecek ve stabiliteyi koruyacaktır.

**Çözüm:**
- Gelecek PyTorch güncellemelerini takip edin.
- Şu an için CPU üzerinden eğitim yapabilirsiniz (ancak daha yavaştır).
- `torch.AcceleratorError: CUDA error: no kernel image is available` hatası alıyorsanız, bu durum mimari uyumsuzluğundan kaynaklanmaktadır.

## Kullanım

### 1. Veri Setini Hazırlayın

CelebA veri setinin indirildiğinden ve dışa aktarıldığından emin olun. Dizin yapısı şu şekilde olmalıdır:

```text
img_align_celeba/
    img_align_celeba/
        000001.jpg
        ...
identity_CelebA.txt
```

İstemci veri dağılımını oluşturmak için bölümleme betiğini çalıştırın (eğer `fl_partition.json` zaten mevcut değilse):

```bash
python partition_data.py
```

### 2. Deneyleri Çalıştırın

Farklı federe öğrenme algoritmalarını kıyaslamak için ana betiği çalıştırın:

```bash
python main.py
```

Bu betik şunları yapacaktır:

1. Küresel modeli başlatır.
2. Birden fazla iletişim turu (varsayılan: 20 tur) boyunca 50 istemci üzerinde simülasyon yapar.
3. Her algoritma için ortalama eğitim kaybını (loss) değerlendirir ve kaydeder.
4. Kök dizinde karşılaştırmalı bir grafik olan `results_plot.png` dosyasını oluşturur.

## Sistem Mimarisi Detayları

1. **Başlatma (Initialization)**: Sunucu, küresel bir ResNet-18 modeli başlatır.
2. **İstemci Seçimi (Client Selection)**: Her iletişim turu için istemcilerin bir kısmı (örneğin %10'u) rastgele seçilir.
3. **Yerel Eğitim (Local Training)**: İstemciler küresel omurgayı alır, kendi yerel ArcFace katmanlarını ekler ve kendi yerel Non-IID verileri üzerinde eğitirler.
4. **Model Güncellemeleri**: İstemciler sunucuya yalnızca omurga ağırlık farklarını (deltaları) geri gönderir. Bu aşamada diferansiyel gizlilik uygulanabilir.
5. **Birleştirme (Aggregation)**: Sunucu, seçilen algoritmayı kullanarak deltaları birleştirir ve küresel modeli günceller.
6. **Yineleme (Iteration)**: Belirtilen iletişim turu sayısı kadar 2. ve 5. adımlar tekrarlanır.


| Doküman Maddesi | Projede Karşılık Gelen Özellik / Dosya | Durum |
| :--- | :--- | :--- |
| **1. Identity-based extreme non-IID** | `dataset.py` & `partition_data.py`: Kimlikler cihazlar arasında asla paylaşılmıyor. | ✅ Tamamlandı |
| **2. Veri Seti (CelebA)** | `dataset.py`: CelebA (~200k imge) entegrasyonu sağlandı. | ✅ Tamamlandı |
| **3. 50 Client & %80-%20 Split** | `fl_partition.json`: 50 client ve %20 "unseen" (görülmemiş) test ayrımı yapıldı. | ✅ Tamamlandı |
| **4. ResNet-18 & ArcFace** | `model.py`: ResNet-18 backbone + ArcFace Loss entegre edildi. | ✅ Tamamlandı |
| **5. Baseline FL Yöntemleri** | `aggregation.py`: FedAvg ve FedProx implemente edildi. | 🏗️ FedNova ve SCAFFOLD eklenecek |
| **6. Asıl Katkı (Cosine & Norm)** | `aggregation.py`: proposed_cosine, proposed_norm, proposed_combined eklendi. | ✅ Tamamlandı |
| **7. Gizlilik (DLG Attack)** | `privacy.py`: Deep Leakage from Gradients sınıfı ve mantığı oluşturuldu. | ✅ Hazır |
| **8. Savunma (Clipping & DP)** | `client.py` & `privacy.py`: Gradient Clipping ve Differential Privacy eklendi. | ✅ Tamamlandı |
| **9. Ölçümler (Acc, Conv, Drift)** | `main.py` & `test/evaluator.py`: Accuracy, Convergence ve ROC-AUC ölçülüyor. | ⏳ Drift analizi eklenecek |

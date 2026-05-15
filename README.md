
# Yüz Biyometrisi İçin Federe Öğrenme (Extreme Non-IID)

Bu çalışma, yüz tanıma sistemlerinde karşılaşılan **Aşırı Non-IID** (bağımsız ve özdeş olmayan dağılım) sorununu Federe Öğrenme (FL) çerçevesinde ele alarak farklı birleştirme algoritmalarını kıyaslamaktadır. Proje, kimlik tabanlı veri ayrımı, gelişmiş birleştirme stratejileri ve gizlilik koruma yöntemlerini kapsamlı bir şekilde analiz eder.

---

## 📝 1. Özet (Abstract)

Yapılan deneysel analizler sonucunda **FedProx**, proksimal terim regülasyonu sayesinde model sapmasını (Drift) diğer algoritmalara oranla 3 kat daha düşük seviyelerde (35-70 bandı) tutarak ve **%97.29** ile en yüksek doğruluğa ulaşarak en başarılı ve stabil algoritma olarak öne çıkmıştır. Tarafımızca geliştirilen **Proposed Combined** yöntemi ise gradyan yönü ve büyüklüğünü temel alan hibrit yaklaşımıyla %95.40 doğruluğa çok daha pürüzsüz bir eğriyle ulaşarak inovatif bir alternatif sunmuştur. 

Buna karşın, FedAvg ve FedNova yüksek doğruluğa ulaşmalarına rağmen kontrolsüz model sapmalarıyla (200+ Drift) kararsız bir profil çizmiş; SCAFFOLD ise başarılı drift kontrolüne rağmen Loss grafiklerindeki sert sıçramalar nedeniyle daha gürültülü bir seyir izlemiştir. Sonuç olarak, aşırı Non-IID koşullarda hem yüksek doğruluk hem de model bütünlüğünü koruyan FedProx ve Proposed Combined yaklaşımları, yüz biyometrisi gibi hassas görevler için en ideal çözümler olarak belirlenmiştir.

---

## 🏗️ 2. Metodoloji ve Teknik Altyapı

### 2.1. Veri Dağıtımı (Extreme Non-IID)
Sıradan FL projelerinin aksine, bu projede veri setindeki her bir "Person ID", yalnızca belirli bir istemciye atanmıştır. Bu, istemciler arasındaki gradyanların birbirinden çok uzak olmasına neden olan **"Weight Divergence"** (Ağırlık Sapması) sorununu tetiklemektedir.

### 2.2. Model Mimarisi: Backbone & Local Head
Eğitimde **ResNet-18** omurgası küresel (global) olarak paylaşılırken, yüz tanıma başarısını artıran **ArcFace** katmanı her istemcide yerel (local) tutulmuştur. Bu sayede kişisel özniteliklerin sunucuya sızması engellenmiş ve gizlilik katmanı güçlendirilmiştir.

---

## 📊 3. Uygulanan Algoritmalar ve Kod Referansı

| Algoritma / Teknik | Fonksiyonel Katkı | Kod Referansı |
| :--- | :--- | :--- |
| **FedAvg** | İstemci modellerinin düz bir ortalamasını alarak birleştirir. | [aggregation.py](file:///c:/Users/root/project/federetad_learning/aggregation.py) [22-26] |
| **FedProx** | Proximal terim ile yerel güncellemelerin küresel modelden kopmasını engeller. | [client.py](file:///c:/Users/root/project/federetad_learning/client.py) [102-106] |
| **SCAFFOLD** | Kontrol değişkenleri kullanarak istemci gradyanlarındaki sapmayı (drift) düzeltir. | [server.py](file:///c:/Users/root/project/federetad_learning/server.py) [191-199] |
| **FedNova** | Farklı yerel iterasyon sayılarına sahip istemcileri normalize ederek birleştirir. | [aggregation.py](file:///c:/Users/root/project/federetad_learning/aggregation.py) [29-37] |
| **Proposed (Combined)** | Gradient Norm ve Cosine Similarity metriklerini birleştirerek ağırlıklandırma yapar. | [aggregation.py](file:///c:/Users/root/project/federetad_learning/aggregation.py) [75-79] |
| **Differential Privacy** | Gradyanlara gürültü ekleyerek DLG saldırılarına karşı koruma sağlar. | [privacy.py](file:///c:/Users/root/project/federetad_learning/privacy.py) [81-103] |
| **ArcFace Loss** | Açısal pay ekleyerek sınıf içi benzerliği artıran kayıp fonksiyonu. | [model.py](file:///c:/Users/root/project/federetad_learning/model.py) [36-79] |
| **DLG Attack** | Gradyanlardan veri sızıntısı saldırı simülasyonu. | [privacy.py](file:///c:/Users/root/project/federetad_learning/privacy.py) [7-79] |

---

## 📈 4. Deneysel Sonuçlar ve Analiz

### 4.1. FedAvg: Kaotik Yakınsama
- **Gözlem:** Sert zikzaklar ve ani kayıp (Loss) sıçramaları (özellikle 10. rauntta).
- **Analiz:** FedAvg, aşırı Non-IID dağılımdaki gradyan çatışmasını düzeltecek bir mekanizmaya sahip değildir. Model sapması (Drift) 190.03 seviyelerine çıkarak bir "dağılma krizi" yaşatmıştır.

### 4.2. FedNova: Agresif Başarı, Yüksek Varyans
- **Gözlem:** %95.33 tepe doğruluğa ulaşsa da, Drift grafiğinde 213.42 seviyelerinde devasa sıçramalar görülmüştür.
- **Analiz:** Normalizasyon çabasına rağmen yerel modeller globalden sert bir şekilde kopmaktadır.

### 4.3. FedProx: Regülasyon Lideri (🏆 En İyi Performans)
- **Gözlem:** **%97.29 Accuracy** ve sadece **35-70** bandında kalan Drift.
- **Analiz:** Proksimal terim (proximal term), yerel güncellemelerin küresel modelden kopmasını en efektid şekilde dizginlemiş, en dengeli ve güvenli optimizasyon profilini çizmiştir.

### 4.4. Proposed Combined: İnovatif Hibrit Yaklaşım
- **Gözlem:** %95.40 doğruluk ve pürüzsüz bir öğrenme eğrisi.
- **Analiz:** Hem gradyan yönü (Cosine Similarity) hem de büyüklüğünü (Norm) temel alan ağırlıklandırma mekanizması, istemcilerin global modelden kopmasını engellemede üstün başarı göstermiştir.

### 4.5. SCAFFOLD: Yönsel Denge
- **Gözlem:** Drift değerleri 60-85 bandında (maksimum 98) baskılanmıştır.
- **Analiz:** Kontrol değişkenleri gradyan heterojenliğini başarıyla dengelemiş olsa da, eğitim sırasında kayıp (loss) değerinde dalgalanmalar yaşanmıştır.

---

## 🏁 5. Sonuç

Aşırı Non-IID koşullarda hem yüksek doğruluk hem de model bütünlüğünü koruyan **FedProx** ve **Proposed Combined** yaklaşımları, yüz biyometrisi gibi hassas görevler için en ideal çözümlerdir. FedProx mutlak doğruluk ve stabilite lideri iken, Proposed Combined akıllı ağırlıklandırma mekanizmasıyla güçlü bir alternatif sunmaktadır.

---

### 👤 Geliştirici Bilgileri
- **İsim:** BARTU BAŞARAN
- **Öğrenci No:** 259120012
- **GitHub:** [baartu/federer_learning](https://github.com/baartu/federer_learning)

---
> [!NOTE]
> Bu proje, akademik araştırma amaçlı olup federatif öğrenmede model sapmasını azaltmaya yönelik yeni yöntemlerin test edilmesi için geliştirilmiştir.

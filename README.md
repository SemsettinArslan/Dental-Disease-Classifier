# DentalGlow AI - Yapay Zeka Destekli Diş Hastalığı Sınıflandırma Sistemi

DentalGlow AI; diş hekimliği ve ağız sağlığı alanında sıkça karşılaşılan 6 farklı dental durumu/hastalığı, ağız içi fotoğraflarını analiz ederek sınıflandıran derin öğrenme tabanlı bir web uygulaması prototipidir. 

Sistem; NVIDIA GPU hızlandırmalı **PyTorch** arka planı ve modern, responsive, koyu tema cam tasarımlı (glassmorphic) **Flask** web arayüzünden oluşmaktadır.

---

## 🚀 Öne Çıkan Özellikler
- **NVIDIA GPU (CUDA) Desteği**: PyTorch CUDA 12.1 ile RTX GPU üzerinde milisaniyeler içinde tahmin.
- **Yüksek Doğruluk Oranı**: Doğrulama (Validation) seti üzerinde **%94.27** başarı oranı.
- **Detaylı Analiz Paneli**: Sınıf bazlı Precision, Recall, F1-Score metrikleri ve interaktif hata matrisi.
- **Kullanıcı Dostu Web Arayüzü**: Sürükle-bırak görsel yükleme, olasılık grafik animasyonları, Türkçe teşhis tanımları ve hekim önerileri.
- **Bilinçli Sınıflandırma**: 
  - Diş Taşı (Calculus)
  - Diş Çürüğü (Caries)
  - Diş Eti İltihabı (Gingivitis)
  - Doğuştan Diş Eksikliği (Hypodontia)
  - Ağız Yarası (Mouth Ulcer)
  - Diş Renklenmesi (Tooth Discoloration)

---

## 📊 Model Performansı

Model, transfer learning yöntemiyle pre-trained **ResNet-34** modeli kullanılarak eğitilmiştir. Eğitimde dengeli veri dağılımı için veri artırımı (Data Augmentation) ve hızlı eğitim için Mixed Precision (`torch.amp`) tekniklerinden yararlanılmıştır.

### Genel Doğrulama Metrikleri (`metrics.json`)

| Sınıf / Hastalık | Precision (Kesinlik) | Recall (Duyarlılık) | F1-Score | Örnek Sayısı |
| :--- | :---: | :---: | :---: | :---: |
| **Diş Taşı (Calculus)** | %74.15 | %78.35 | 0.762 | 194 |
| **Diş Çürüğü (Caries)** | %99.72 | %99.72 | 0.997 | 357 |
| **Diş Eti İltihabı (Gingivitis)** | %87.21 | %85.23 | 0.862 | 352 |
| **Diş Eksikliği (Hypodontia)** | %99.46 | %97.86 | 0.987 | 187 |
| **Ağız Yarası (Mouth Ulcer)** | %100.00 | %100.00 | 1.000 | 381 |
| **Diş Renklenmesi (Tooth Discoloration)** | %99.64 | %99.64 | 0.996 | 275 |
| **Genel Başarı (Accuracy)** | **%94.27** | | | **1,746** |


---

## 📁 Proje Yapısı

```text
├── data/                    # Ham veri klasörü 
├── dataset/                 # Bölünmüş ve işlenmiş veri
│   ├── train/               # %70 Eğitim verisi
│   ├── val/                 # %15 Doğrulama verisi
│   └── test/                # %15 Test verisi (Okul/Jüri sunumu ve testler için)
├── templates/
│   └── index.html           # Dashboard HTML/CSS/JS (Web Arayüzü)
├── app.py                   # Flask Web Sunucu Uygulaması
├── prepare_dataset.py       # Veriyi bölme ve hazırlama betiği
├── train.py                 # PyTorch model eğitim betiği
├── eval_metrics.py          # Model performansını ölçme ve JSON çıktısı alma betiği
├── predict.py               # Komut satırı (CLI) tahmin aracı
├── classes.txt              # Model sınıfları listesi
├── metrics.json             # Doğrulama sonuçları JSON verisi
├── learning_curves.png      # Eğitim Kayıp/Doğruluk grafiği
├── confusion_matrix.png     # Hata Matrisi görseli
├── .gitignore               # Git dışı bırakılacaklar listesi
└── README.md                # Proje genel dökümanı (Şu an okuduğunuz dosya)
```

---

## 🛠️ Kurulum Adımları

Projeyi kendi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edin:

### 1. Gereksinimlerin Yüklenmesi
Python ve CUDA uyumlu PyTorch kurulumunu gerçekleştirin. (Örn. CUDA 12.1 için):

```bash
# Sanal ortam oluşturun (Önerilen)
python -m venv venv
venv\Scripts\activate

# PyTorch (CUDA 12.1) ve diğer kütüphaneleri yükleyin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flask tqdm Pillow matplotlib numpy
```

### 2. Veri Setinin Hazırlanması
Ham verileri projenin `data/` klasörünün altına ilgili sınıf isimleriyle yerleştirin. Ardından verileri eğitim, doğrulama ve test olarak bölmek için:

```bash
python prepare_dataset.py
```

### 3. Modelin Eğitilmesi
Modeli CUDA GPU hızlandırmasıyla eğitmek için:

```bash
python train.py --epochs 10 --batch_size 32
```
*Eğitim tamamlandığında model ağırlıkları `best_model.pth` olarak kaydedilir ve grafikler kök dizine çizilir.*

### 4. Metriklerin Hesaplanması
Tabloları ve hata analizlerini web arayüzüne taşımak için doğrulama setini test edin:

```bash
python eval_metrics.py
```

---

## 💻 Çalıştırma ve Kullanım

### Komut Satırı Üzerinden Tahmin (CLI)
Tek bir görseli hızlıca test etmek için:

```bash
python predict.py --image "dataset/val/Calculus/(100).jpg"
```

### Web Arayüzü Üzerinden Tahmin (Arayüz)
Aesthetic Dashboard'u ayağa kaldırmak için:

```bash
python app.py
```
Sunucu başladığında tarayıcınızdan **`http://localhost:5000`** adresine giderek sistemi interaktif olarak kullanabilirsiniz.

---

## ⚠️ Yasal Uyarı / Disclaimer
Bu proje, tıbbi teşhis doğruluğu taahhüt etmeyen deneysel bir yapay zeka çalışmasıdır. Sunulan analizler ve öneriler hekim muayenesi yerine geçmez; sadece bilgilendirme amaçlıdır. Sağlık problemleriniz için her zaman uzman bir Diş Hekimine başvurmalısınız.

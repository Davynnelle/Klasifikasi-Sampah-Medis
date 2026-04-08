💉 Klasifikasi Limbah Biomedis – Biomedical Waste Classification 🗑️

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![EfficientNetB0](https://img.shields.io/badge/Model-EfficientNetB0-green)
![GPU](https://img.shields.io/badge/GPU-Tesla_T4-purple?logo=nvidia)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Proyek klasifikasi gambar limbah biomedis berbasis **Transfer Learning** menggunakan **EfficientNetB0** untuk mengklasifikasikan gambar limbah medis dari lingkungan klinis nyata ke dalam **20 kategori** menggunakan arsitektur **EfficientNetB0** dengan strategi *two-phase fine-tuning* dan akurasi **98,05%** pada test set.

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Arsitektur Model](#-arsitektur-model)
- [Hasil Training](#-hasil-training)
- [Evaluasi Model](#-evaluasi-model)
- [Insight Bisnis](#-insight-bisnis)
- [Konversi Model](#-konversi-model)
- [Teknologi](#-teknologi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Struktur Proyek](#-struktur-proyek)
- [Hyperparameter](#-hyperparameter)

---

## 📌 Deskripsi Proyek

Pengelolaan limbah medis adalah masalah krusial dalam sektor kesehatan. Kesalahan klasifikasi dapat menyebabkan:

* Risiko infeksi & kontaminasi
* Pelanggaran regulasi
* Inefisiensi operasional

Proyek ini membangun model klasifikasi berbasis Computer Vision **end-to-end** untuk mengidentifikasi jenis limbah biomedis menggunakan pendekatan **Transfer Learning** dengan **two-phase training**:

| Fase | Strategi | Learning Rate | Hasil Val Accuracy |
|---|---|---|---|
| **Fase 1** | Feature Extraction (base frozen) | `1e-3` | **98.36%** (epoch 11) |
| **Fase 2** | Fine-tuning 30 layer terakhir | `1e-5` | **98.91%** (epoch 28) |

Dataset yang digunakan adalah direktori **Raw** yang berisi gambar asli dari lingkungan klinis nyata (tanpa preprocessing sebelumnya) dengan variasi latar belakang, pencahayaan, dan noise yang tinggi. Model akhir diekspor ke tiga format deployment: **SavedModel**, **TF-Lite**, dan **TensorFlow.js**.

---

## 📦 Dataset

| Keterangan | Detail |
|---|---|
| Sumber | [Biomedical Waste Classification Images Dataset](https://www.kaggle.com/datasets/mario78/biomedical-waste-classification-images-dataset) — Kaggle, oleh mario78 |
| Direktori | `Raw` (gambar asli tanpa preprocessing) |
| Total kelas tersedia | 23 kelas |
| Kelas yang digunakan | 20 kelas (1 kelas dikecualikan) |
| Gambar per kelas | 640 gambar |
| Total gambar | **12.800 gambar** |
| Resolusi | Tidak seragam — Min: **100px**, Max: **3.456px**, Mean: **~727px** |
| Lisensi dataset | MIT |

### Kelas yang Dikecualikan
Kelas `uncategorized_or_overlapping` (1.520 gambar) dikecualikan karena:
- Label ambigu: gambar tidak memiliki kategori yang jelas.
- Risiko bias: jumlah gambar 2.4× lebih banyak dari kelas lain.
- Noise label:  gambar yang seharusnya masuk kelas lain berpotensi salah dilabeli.

### 20 Kelas yang Digunakan

| No | Nama Kelas | Keterangan |
|---|---|---|
| 1 | `ampoules_full` | Ampul obat (penuh) |
| 2 | `ampuoles_broken` | Ampul obat (pecah) |
| 3 | `blood_soaked_bandages` | Perban tercemar darah |
| 4 | `disinfectant_bottles` | Botol disinfektan |
| 5 | `episiotomy_scissors` | Gunting episiotomi |
| 6 | `expired_tablets` | Tablet kadaluarsa |
| 7 | `forceps` | Forsep/klem bedah |
| 8 | `general_organic_waste` | Limbah organik umum |
| 9 | `hemostats` | Hemostat/penjepit pembuluh |
| 10 | `human_organs` | Jaringan/organ manusia |
| 11 | `iv_bottles` | Botol cairan infus |
| 12 | `mayo_scissors` | Gunting mayo |
| 13 | `scalpels` | Pisau bedah (scalpel) |
| 14 | `stitch_removal_scissors` | Gunting lepas jahitan |
| 15 | `syrup_bottles` | Botol sirup obat |
| 16 | `tweezers` | Pinset |
| 17 | `used_masks` | Masker bekas pakai |
| 18 | `used_medical_gloves` | Sarung tangan medis bekas |
| 19 | `used_medical_paper` | Kertas/tisu medis bekas |
| 20 | `used_syringes` | Jarum suntik bekas |

### Split Dataset

| Split | Jumlah | Proporsi |
|---|---|---|
| Train | 10.240 | 80% |
| Validation | 1.280 | 10% |
| Test | 1.280 | 10% |

> Dataset memiliki distribusi kelas yang **seimbang** (640 gambar per kelas), sehingga tidak diperlukan teknik oversampling.

**Karakteristik Resolusi Dataset (Raw):**

- Width: Min 100px → Max 3.456px → Mean ≈ 727px
- Height: Min 100px → Max 3.456px → Mean ≈ 703px
- Kesimpulan: resolusi **tidak seragam** → diselesaikan dengan resize ke 224×224 saat preprocessing.

---

## 🔧 Pipeline

```
Dataset Kaggle (23 kelas × 640 gambar)
      ↓
Dataset Raw (menjadi 20 kelas dengan 12.800 gambar, resolusi tidak seragam)
      ↓
Eksplorasi & EDA
  ├── Analisis distribusi kelas (balanced: 640/kelas)
  ├── Analisis resolusi (Min: 100px → Max: 3.456px)
  └── Visualisasi sampel per kelas
      ↓
Split Dataset (80/10/10)
      ↓
ImageDataGenerator
  ├── target_size = (224, 224)      ← resize otomatis
  ├── Train: augmentasi             ← rotation, flip, zoom, dll
  └── Val/Test: tanpa augmentasi
      ↓
Lambda(preprocess_input)            ← normalisasi internal EfficientNetB0
      ↓
Training Fase 1 (base frozen, LR=1e-3, max 15 epoch)
dengan Val Accuracy terbaik: 98,36% (epoch 11)
      ↓
Training Fase 2 (fine-tuning 30 layer terakhir, LR=1e-5, max 30 epoch)
dengan Val Accuracy terbaik: 98,91% (epoch 28)
      ↓
Evaluasi: Accuracy, Classification Report, Confusion Matrix
      ↓
Export: SavedModel · TF-Lite · TensorFlow.js
      ↓
Inference (TF-Lite dan SavedModel)
```

---

## 🏗️ Arsitektur Model

```
Input (224 × 224 × 3)
      ↓
Lambda: preprocess_input (normalisasi EfficientNet)
      ↓
EfficientNetB0 (pretrained ImageNet, 238 layer)
  ├── Fase 1: semua layer frozen
  └── Fase 2: 30 layer terakhir dibuka
      ↓
GlobalAveragePooling2D
      ↓
BatchNormalization
      ↓
Dense(512, relu) → Dropout(0.4)
      ↓
Dense(256, relu) → Dropout(0.3)
      ↓
Dense(20, softmax)  
```

### Mengapa Tidak Pakai `rescale=1./255`?

EfficientNetB0 sudah memiliki preprocessing bawaan yang mengharapkan input **0–255**. Menggunakan `rescale=1./255` di `ImageDataGenerator` menyebabkan **double normalisasi** yang merusak distribusi nilai piksel dan terbukti membuat model hanya melakukan *random guessing* (akurasi ~5%).

## 🚀 Strategi Training

### Two-Phase Training

Proyek ini menggunakan strategi pelatihan dua fase untuk memaksimalkan transfer learning secara bertahap:

**Fase 1 — Feature Extraction (Base Frozen)**

- Semua layer EfficientNetB0 dibekukan (`trainable=False`)
- Hanya custom head yang dilatih
- Learning Rate: `1e-3`
- Max Epoch: 15
- Val Accuracy terbaik: **98,36%** (epoch 11)

**Fase 2 — Fine-Tuning (30 Layer Terakhir Dibuka)**

- 30 layer terakhir EfficientNetB0 dibuka (`trainable=True`)
- Learning Rate diturunkan ke `1e-5` untuk mencegah *catastrophic forgetting*
- Max Epoch: 30
- Val Accuracy terbaik: **98,91%** (epoch 28)

### Augmentasi Data (Train Set Only)

```python
rotation_range    = 20
width_shift_range = 0.2
height_shift_range= 0.2
shear_range       = 0.15
zoom_range        = 0.2
horizontal_flip   = True
brightness_range  = [0.8, 1.2]
fill_mode         = 'nearest'
```

---

## 📈 Hasil Training

### Performa per Fase

| Fase | Val Accuracy Terbaik | Epoch |
|---|---|---|
| Fase 1 (frozen) | **98.36%** | 11 / 15 |
| Fase 2 (fine-tuning) | **98.91%** | 28 / 30 |

### Evaluasi Akhir

| Set | Accuracy |
|---|---|
| Train Set | > 95% |
| Validation Set | ~98.91% |
| **Test Set** | **98.05%** |

---

## 📊 Evaluasi Model

### Kelas dengan Performa Sempurna (F1 = 1.00)

- `forceps`
- `iv_bottles`
- `tweezers`
- `used_masks`

Kelas-kelas ini memiliki bentuk dan tekstur yang sangat khas sehingga mudah dibedakan model.

### Kelas yang Paling Rentan Salah Klasifikasi

| Kelas | Precision | Recall | Penyebab |
|---|---|---|---|
| `mayo_scissors` | 0.92 | 0.95 | Kemiripan visual dengan `episiotomy_scissors` dan `stitch_removal_scissors` |
| `ampoules_full` | 0.94 | 0.98 | Bentuk silindris mirip `disinfectant_bottles` |
| `ampuoles_broken` | 0.98 | 0.92 | Bentuk pecahan tidak konsisten |

---

## 💡 Insight Bisnis

### Konteks Penerapan

Model ini berpotensi digunakan sebagai komponen awal dalam sistem pengelolaan limbah medis berbasis visi komputer di fasilitas kesehatan seperti rumah sakit, klinik, dan laboratorium untuk mendukung identifikasi limbah secara otomatis melalui kamera secara real-time.

### Implikasi Praktis

**1. Reduksi Risiko Kontaminasi**

Kesalahan penanganan limbah tajam seperti `scalpels`, `episiotomy_scissors`, dan `hemostats` berisiko menyebabkan luka tusuk (*needlestick injury*) pada tenaga medis. Model dengan F1-score tinggi pada kelas-kelas ini membantu mitigasi risiko tersebut.

**2. Efisiensi Operasional**

Identifikasi otomatis mengurangi waktu pemilahan manual di fasilitas kesehatan yang memiliki volume limbah tinggi.

**3. Kepatuhan Regulasi**

Klasifikasi yang akurat mendukung kepatuhan terhadap regulasi pengelolaan limbah medis yang ketat.

### Rekomendasi Pengembangan

1. 🔧 **Perbaiki klasifikasi gunting bedah**

   Tambah augmentasi yang lebih agresif (rotasi 360°, variasi pencahayaan ekstrem) khusus untuk `mayo_scissors`, `episiotomy_scissors`, dan `stitch_removal_scissors` untuk mengurangi kebingungan antar kelas yang serupa secara visual.

3. 📱 **Deployment mobile**

   Model TF-Lite (5,11 MB) cukup ringan untuk diintegrasikan ke dalam aplikasi Android atau iOS yang digunakan oleh petugas kebersihan medis di lapangan.

5. 🎥 **Real-time classification**

   Model dapat diintegrasikan dengan sistem kamera pada tempat pembuangan sementara (TPS medis) untuk melakukan klasifikasi limbah secara otomatis saat limbah dibuang.

7. 🔄 **Continuous learning**

   Mengumpulkan gambar limbah dari fasilitas kesehatan yang berbeda dapat membantu proses fine-tuning berkala, karena jenis alat medis dan kemasan dapat bervariasi antar fasilitas kesehatan.

---

## 📥 Konversi Model

Model diekspor ke 3 format untuk berbagai kebutuhan deployment:

| Format | Kegunaan | Lokasi |
|---|---|---|
| **SavedModel** | TensorFlow Serving / deployment server | `submission/saved_model/` |
| **TF-Lite** | Mobile / edge deployment | `submission/tflite/model.tflite` |
| **TensorFlow.js** | Deployment di browser | `submission/tfjs_model/` |

---

## 🛠️ Teknologi

```
Deep Learning    : TensorFlow / Keras
Algoritma        : EfficientNetB0 (pretrained ImageNet).
Augmentasi       : ImageDataGenerator (Keras).
Evaluasi         : scikit-learn (classification_report, confusion_matrix).
Visualisasi      : Matplotlib, Seaborn.
Dataset          : Kaggle API.
Inference        : TF-Lite Interpreter.
Environment      : Google Colab (GPU: Tesla T4).
```

---

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/Davynnelle/Klasifikasi-Sampah-Medis.git
cd Klasifikasi-Sampah-Medis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Siapkan Kaggle API

Download `kaggle.json` dari akun Kaggle kamu:
- Buka [kaggle.com](https://kaggle.com) → Settings → API → **Create New Token**
- Simpan file `kaggle.json`

### 4. Jalankan notebook

Buka `notebook.ipynb` di Google Colab dan jalankan sel secara berurutan.

> ⚠️ **Pastikan GPU aktif** sebelum training: `Runtime → Change runtime type → T4 GPU`

> ⚠️ **Upload `kaggle.json`** saat cell `files.upload()` dijalankan

> Quick Demo (No Training Required)
Gunakan aplikasi yang sudah dideploy: [MedWaste Classifier](https://klasifikasi-sampah-medis-ai.streamlit.app/)

---

## 📁 Struktur Proyek

```
submission/
│
├── Biomedical_Waste_Classification_Dhea_Yuza_Fadiya.ipynb              ← notebook utama
│
├── saved_model/                                                        ← TensorFlow SavedModel
│   ├── saved_model.pb
│   └── variables/
│
├── tflite/                                                             ← TF-Lite untuk mobile/edge
│   ├── model.tflite
│   └── label.txt                                                       ← 20 nama kelas
│
├── tfjs_model/                                                         ← TensorFlow.js untuk browser
│   ├── model.json
│   └── group1-shard1of1.bin
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Hyperparameter

| Parameter | Nilai |
|---|---|
| IMG_SIZE | 224 × 224 |
| BATCH_SIZE | 32 |
| FASE 1 - Learning Rate | `1e-3` |
| FASE 1 - Max Epochs | 15 |
| FASE 2 - Learning Rate | `1e-5` |
| FASE 2 - Max Epochs | 30 |
| Fine-tuning layers | 30 layer terakhir EfficientNetB0 |
| EarlyStopping patience | 7 |
| ReduceLROnPlateau factor | 0.3 |
| Dropout (Dense 512) | 0.4 |
| Dropout (Dense 256) | 0.3 |

---

📄 *Note: This project is based on a final submission from the course "Belajar Fundamental Deep Learning" on Dicoding Indonesia. The goal was to implement and expand on the core concepts introduced in the course.*

---

> 💬 *"Dari 12.800 gambar limbah biomedis di 20 kategori, model EfficientNetB0 berhasil mengklasifikasikan dengan akurasi **98,05%** pada test set — berpotensi untuk deployment sistem identifikasi limbah medis otomatis di fasilitas kesehatan."*

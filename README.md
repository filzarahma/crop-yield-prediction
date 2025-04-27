# Laporan Proyek Machine Learning - Filza Rahma Muflihah

## Domain Proyek
Pertanian adalah sektor vital dalam penyediaan pangan dan penghidupan masyarakat. Salah satu tantangan utama dalam sektor ini adalah ketidakpastian hasil panen (crop yield), yang dipengaruhi oleh banyak faktor seperti curah hujan, kualitas tanah, ukuran lahan, intensitas sinar matahari, dan penggunaan pupuk. Dengan memanfaatkan machine learning, kita dapat membangun model prediktif untuk memperkirakan hasil panen berdasarkan faktor-faktor ini, sehingga membantu petani dan pengambil kebijakan dalam perencanaan dan pengelolaan pertanian yang lebih efektif.

**Mengapa masalah ini penting?**
- Membantu petani mengoptimalkan penggunaan sumber daya (air, pupuk, lahan).
- Mengurangi risiko gagal panen.
- Membantu pemerintah dalam membuat kebijakan pangan berbasis data.
- Mengurangi ketidakpastian hasil pertanian.

**Bagaimana masalah ini diselesaikan?**
Dengan membangun model regresi berbasis machine learning untuk memprediksi hasil panen berdasarkan atribut lingkungan.

**Penelitian Terkait:**
Studi oleh [Kang et al. (2020)](https://iopscience.iop.org/article/10.1088/1748-9326/ab7df9/meta) menunjukkan bahwa penggunaan banyak variabel lingkungan (cuaca, satelit, tanah, fenologi) bersama dengan algoritma machine learning canggih seperti XGBoost mampu meningkatkan akurasi prediksi hasil panen hingga 5% dibandingkan baseline model. Mereka juga menemukan bahwa XGBoost mengungguli model lain (termasuk deep learning seperti LSTM dan CNN) dalam hal akurasi dan stabilitas untuk prediksi hasil jagung di Midwest Amerika Serikat.

## Business Understanding

### Problem Statements
- Bagaimana memprediksi hasil panen berdasarkan faktor-faktor seperti curah hujan, kualitas tanah, ukuran lahan, intensitas sinar matahari, dan jumlah pupuk?
- Algoritma machine learning apa yang paling efektif dalam memprediksi hasil panen di dataset ini?

### Goals
- Mengembangkan model regresi untuk memprediksi hasil panen dengan error seminimal mungkin.
- Membandingkan performa beberapa model regresi berdasarkan metrik MSE dan R².

### Solution Statements
Solution: Membangun dan membandingkan model Linear Regression, Decision Tree, Ridge Regression, Lasso Regression, dan XGBoost.
Evaluasi Solusi: Menggunakan metrik Mean Squared Error (MSE) dan R-squared (R²) untuk mengukur performa model.

## Data Understanding
### Informasi Dataset
- Jumlah data: 3000 data.
- Data kondisi: Tidak ada missing value, semua bertipe numerik (int64).
- Sumber data: [Kaggle](https://www.kaggle.com/datasets/govindaramsriram/crop-yield-of-a-farm/data). 

### Fitur Dataset

| No	| Nama Fitur	| Deskripsi |
| --- | ----------- | ----------|
| 1	| rainfall_mm	| Curah hujan dalam milimeter |
| 2	| soil_quality_index |	Indeks kualitas tanah |
| 3	| farm_size_hectares |	Ukuran lahan dalam hektar |
| 4	| sunlight_hours |	Jumlah jam paparan sinar matahari per hari |
| 5	| fertilizer_kg	| Jumlah pupuk yang digunakan dalam kilogram |
| 6	| crop_yield |	Hasil panen (target yang ingin diprediksi) |

### Exploratory Data Analysis
#### Analisis Univariat
![image](https://github.com/user-attachments/assets/f9e591dc-21f9-4082-846d-5d1afa07cb3d)

#### Analisis Bivariat
![image](https://github.com/user-attachments/assets/af2ab444-d67a-4bb1-8cd5-6c1836c27859)

#### Analisis Korelasi
![image](https://github.com/user-attachments/assets/270a3494-2d04-4230-bc2e-1c5e22144888)

### Data Preparation
#### Tahapan Data Preparation
- Memisahkan fitur (X) dan target (y).
- Train-test split: 80% data untuk training, 20% untuk testing.
- Melakukan feature scaling menggunakan StandardScaler.

#### Alasan Tahapan
- Membagi data diperlukan untuk menguji generalisasi model.
- Feature scaling untuk melakukan normalisasi data agar data yang digunakan tidak memiliki sebaran (range) yang beragam.

### Modeling
#### Model dan Parameter
Setiap model dibangun menggunakan pipeline yang berisi satu langkah, yaitu algoritma regresi itu sendiri. Berikut adalah parameter utama yang digunakan untuk setiap model:
- **Linear Regression**: Tidak ada parameter khusus yang diatur dalam implementasi ini, menggunakan parameter default dari scikit-learn.
- **Decision Tree**: Parameter `random_state=42` digunakan untuk memastikan hasil yang dapat direproduksi. Parameter lain menggunakan nilai default.
- **XGBoost**: Parameter `n_estimators=100` mengatur jumlah pohon yang dibangun, `random_state=42` untuk reproduktifitas, dan `verbosity=0` untuk mengurangi keluaran verbose.
- **Lasso Regression**: Parameter `alpha=1.0` adalah koefisien regularisasi L1. `random_state=42` digunakan untuk reproduktifitas.
- **Ridge Regression**: Parameter `alpha=1.0` adalah koefisien regularisasi L2. `random_state=42` digunakan untuk reproduktifitas.

#### Tahapan Utama 
1. **Definisi Model**: Membuat instance dari setiap algoritma regresi dengan parameter yang telah ditentukan.
2. **Pembentukan Pipeline**: Menggabungkan setiap model ke dalam pipeline scikit-learn. Meskipun dalam kasus ini hanya ada satu langkah (model), penggunaan pipeline adalah praktik yang baik untuk alur kerja yang lebih kompleks di masa depan.
3. **Pelatihan Model**: Melatih setiap pipeline menggunakan data latih (train). Metode `fit()` digunakan untuk tujuan ini.
4. **Evaluasi Model**: Mengevaluasi kinerja setiap model yang telah dilatih menggunakan data latih dan data uji (test). Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE) dan R-squared (koefisien determinasi). Metode `predict()` digunakan untuk mendapatkan prediksi, dan kemudian metrik dihitung menggunakan fungsi-fungsi dari scikit-learn.
   
#### Kelebihan dan Kekurangan Model
| Model | Kelebihan | Kekurangan |
|-------|-----------|------------|
|Linear Regression| Interpretatif, cepat.| Tidak bagus untuk hubungan non-linear, sensitif terhadap outlier.|
|Decision Tree|Menangani non-linearitas, interpretatif.|Overfitting pada data kecil.|
|XGBoost|Akurasi tinggi, menangani missing value.|Kompleksitas tinggi, tuning sulit.|
|Lasso Regression|Seleksi fitur otomatis (regulasi L1).|Bisa mengabaikan fitur penting.|
|Ridge Regression|Mengurangi overfitting (regulasi L2).|Sensitif terhadap korelasi antar fitur.|

### Evaluation
#### Metrik Evaluasi yang Digunakan
- **Mean Squared Error (MSE):**
  
  $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \begin{itemize}
       \item $y_i$: nilai sebenarnya ke-$i$
       \item $\hat{y}_i$: nilai prediksi ke-$i$
       \item $n$: jumlah total data
   \end{itemize}
  $$
  
  Mengukur rata-rata kuadrat error. Semakin kecil MSE, semakin baik model.
- **R-squared (R²):**
  
  $$
  R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
 \begin{description}
     \item[$\text{SS}_{\text{res}}$] Sum of Squares of Residuals
     \item[$\text{SS}_{\text{tot}}$] Total Sum of Squares
 \end{description}
$$

Mengukur seberapa besar variasi target yang dapat dijelaskan oleh model. Nilai R² mendekati 1 menunjukkan model sangat baik.

#### Hasil Evaluasi

  







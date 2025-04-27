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
Berdasarkan histogram dari berbagai fitur dan target variabel, berikut adalah deskripsi mengenai sebaran datanya:
1. Rainfall (mm):
- Sebaran curah hujan terlihat cukup merata di sepanjang rentang nilai dari sekitar 500 mm hingga 2000 mm.
- Tidak terlihat adanya puncak (mode) yang sangat dominan, melainkan frekuensi yang relatif serupa di berbagai interval curah hujan.
2. Soil Quality Index:
- Indeks kualitas tanah menunjukkan sebaran yang diskrit dan terkumpul pada nilai-nilai integer dari 1 hingga 10.
- Terlihat adanya frekuensi yang bervariasi antar indeks kualitas tanah. Beberapa nilai seperti 2, 5, dan 8 memiliki frekuensi yang lebih tinggi dibandingkan nilai lainnya.
3. Farm Size (hectares):
- Ukuran lahan pertanian juga menunjukkan sebaran yang relatif merata dari sekitar 0 hingga 1000 hektar.
- Mirip dengan curah hujan, tidak ada satu ukuran lahan yang secara signifikan lebih dominan dibandingkan yang lain.
4. Sunlight Hours:
- Jumlah jam sinar matahari menunjukkan sebaran yang diskrit dan terbatas pada nilai-nilai integer dari 4 hingga 12, yang kemungkinan merepresentasikan bulan dalam setahun atau kategori durasi sinar matahari.
- Terdapat variasi frekuensi antar jumlah jam sinar matahari, dengan beberapa nilai seperti 6 dan 9 memiliki frekuensi yang lebih tinggi.
5. Fertilizer (kg):
- Jumlah pupuk yang digunakan memiliki sebaran yang cukup luas dari 0 hingga 3000 kg.
- Terlihat adanya beberapa puncak frekuensi di sekitar nilai 500, 1500, dan 2500 kg, namun secara keseluruhan sebarannya cukup bervariasi.
6. Crop Yield:
- Hasil panen (crop yield) menunjukkan sebaran yang cenderung unimodal (memiliki satu puncak utama) di sekitar nilai 300 hingga 400.
- Sebagian besar data hasil panen terkonsentrasi di rentang antara sekitar 100 hingga 600.

![image](https://github.com/user-attachments/assets/a4bd1273-2bbb-4cf6-99f6-95d0ab1ae058)
Boxplot menunjukkan tidak ada outlier pada fitur dataset.

#### Analisis Korelasi
![image](https://github.com/user-attachments/assets/af2ab444-d67a-4bb1-8cd5-6c1836c27859)
![image](https://github.com/user-attachments/assets/270a3494-2d04-4230-bc2e-1c5e22144888)
Ukuran lahan pertanian memiliki korelasi positif yang sangat tinggi berdasarkan scatter plot di atas. Semakin luas lahan pertanian semakin tinggi hasil panen yang didapatkan. Sedangkan fitur yang lain memiliki korelasi yang sangat lemah terhadap hasil panen.

## Data Preparation
### Tahapan Data Preparation
- Memisahkan fitur (X) dan target (y).
- Train-test split: 80% data untuk training, 20% untuk testing.
- Melakukan feature scaling menggunakan StandardScaler.

### Alasan Tahapan
- Membagi data diperlukan untuk menguji generalisasi model.
- Feature scaling untuk melakukan normalisasi data agar data yang digunakan tidak memiliki sebaran (range) yang beragam.

## Modeling
### Model dan Parameter
Setiap model dibangun menggunakan pipeline yang berisi satu langkah, yaitu algoritma regresi itu sendiri. Berikut adalah parameter utama yang digunakan untuk setiap model:
- **Linear Regression**: Tidak ada parameter khusus yang diatur dalam implementasi ini, menggunakan parameter default dari scikit-learn.
- **Decision Tree**: Parameter `random_state=42` digunakan untuk memastikan hasil yang dapat direproduksi. Parameter lain menggunakan nilai default.
- **XGBoost**: Parameter `n_estimators=100` mengatur jumlah pohon yang dibangun, `random_state=42` untuk reproduktifitas, dan `verbosity=0` untuk mengurangi keluaran verbose.
- **Lasso Regression**: Parameter `alpha=1.0` adalah koefisien regularisasi L1. `random_state=42` digunakan untuk reproduktifitas.
- **Ridge Regression**: Parameter `alpha=1.0` adalah koefisien regularisasi L2. `random_state=42` digunakan untuk reproduktifitas.

### Tahapan Utama 
1. **Definisi Model**: Membuat instance dari setiap algoritma regresi dengan parameter yang telah ditentukan.
2. **Pembentukan Pipeline**: Menggabungkan setiap model ke dalam pipeline scikit-learn. Meskipun dalam kasus ini hanya ada satu langkah (model), penggunaan pipeline adalah praktik yang baik untuk alur kerja yang lebih kompleks di masa depan.
3. **Pelatihan Model**: Melatih setiap pipeline menggunakan data latih (train). Metode `fit()` digunakan untuk tujuan ini.
4. **Evaluasi Model**: Mengevaluasi kinerja setiap model yang telah dilatih menggunakan data latih dan data uji (test). Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE) dan R-squared (koefisien determinasi). Metode `predict()` digunakan untuk mendapatkan prediksi, dan kemudian metrik dihitung menggunakan fungsi-fungsi dari scikit-learn.
   
### Kelebihan dan Kekurangan Model
| Model | Kelebihan | Kekurangan |
|-------|-----------|------------|
|Linear Regression| Interpretatif, cepat.| Tidak bagus untuk hubungan non-linear, sensitif terhadap outlier.|
|Decision Tree|Menangani non-linearitas, interpretatif.|Overfitting pada data kecil.|
|XGBoost|Akurasi tinggi, menangani missing value.|Kompleksitas tinggi, tuning sulit.|
|Lasso Regression|Seleksi fitur otomatis (regulasi L1).|Bisa mengabaikan fitur penting.|
|Ridge Regression|Mengurangi overfitting (regulasi L2).|Sensitif terhadap korelasi antar fitur.|

## Evaluation
### Metrik Evaluasi yang Digunakan
- **Mean Squared Error (MSE):**
![image](https://github.com/user-attachments/assets/74e20a23-8cbb-4438-b732-d9e1d36b65fc)
Mengukur rata-rata kuadrat error. Semakin kecil MSE, semakin baik model.

- **R-squared (R²):**
![image](https://github.com/user-attachments/assets/a803e3da-eee3-4be2-acb6-89de7be8b484)
Mengukur seberapa besar variasi target yang dapat dijelaskan oleh model. Nilai R² mendekati 1 menunjukkan model sangat baik.

### Hasil Evaluasi
| Model              | Train MSE  | Train R²  | Test MSE   | Test R²   |
|--------------------|------------|-----------|------------|-----------|
| Linear Regression  | 0.085804   | 0.999996  | 0.081761   | 0.999996  |
| Decision Tree      | 0.000000   | 1.000000  | 137.530000 | 0.993578  |
| XGBoost            | 1.182992   | 0.999943  | 23.437340  | 0.998906  |
| Lasso Regression   | 0.269812   | 0.999987  | 0.261464   | 0.999988  |
| Ridge Regression   | 0.085804   | 0.999996  | 0.081751   | 0.999996  |

### Visualisasi Performa Model
![image](https://github.com/user-attachments/assets/a3102f85-3b53-4a62-b37c-997317935dce)

## Kesimpulan
Berdasarkan hasil pemodelan dengan berbagai algoritma machine learning, dapat disimpulkan bahwa faktor-faktor seperti curah hujan, kualitas tanah, ukuran lahan, intensitas sinar matahari, dan jumlah pupuk memiliki hubungan yang sangat kuat dengan hasil panen. Hal ini ditunjukkan oleh nilai R-squared yang sangat tinggi pada data latih dan uji untuk beberapa model, yang mendekati 1. Nilai R-squared yang tinggi mengindikasikan bahwa sebagian besar variabilitas dalam hasil panen dapat dijelaskan oleh fitur-fitur input tersebut.

Mengenai algoritma machine learning yang paling efektif dalam memprediksi hasil panen pada dataset ini, Linear Regression dan Ridge Regression menunjukkan kinerja terbaik dan hampir identik. Kedua model ini menghasilkan nilai Mean Squared Error (MSE) yang sangat rendah (sekitar 0.08 pada data uji) dan nilai R-squared yang sangat tinggi (0.999996 pada data uji). Ini menunjukkan bahwa kedua model ini mampu memprediksi hasil panen dengan akurasi yang sangat tinggi dan memiliki kemampuan generalisasi yang baik terhadap data yang belum pernah dilihat.

Meskipun Lasso Regression juga menunjukkan kinerja yang baik dengan nilai MSE 0.261464 dan R-squared 0.999988 pada data uji, kinerjanya sedikit di bawah Linear Regression dan Ridge Regression. Sebaliknya, Decision Tree mengalami overfitting yang signifikan. Meskipun memiliki kinerja sempurna pada data latih (MSE 0 dan R-squared 1.0), kinerjanya jauh lebih buruk pada data uji (MSE 137.53 dan R-squared 0.993578). Hal ini mengindikasikan bahwa model Decision Tree terlalu kompleks dan menghafal pola pada data latih, sehingga tidak mampu menggeneralisasi dengan baik pada data baru. XGBoost menunjukkan kinerja yang cukup baik dengan MSE 23.437340 dan R-squared 0.998906 pada data uji, namun masih di bawah performa Linear Regression dan Ridge Regression dalam kasus ini.

Oleh karena itu, dengan mempertimbangkan kinerja pada data uji (kemampuan generalisasi), Linear Regression dan Ridge Regression adalah algoritma yang paling efektif untuk memprediksi hasil panen berdasarkan faktor-faktor yang diberikan dalam dataset ini. Ridge Regression mungkin sedikit lebih unggul karena kemampuannya dalam menangani potensi multicollinearity antar fitur, meskipun dalam kasus ini dampaknya terlihat minimal berdasarkan hasil yang sangat serupa dengan Linear Regression.

  







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
![image](https://github.com/user-attachments/assets/f9e591dc-21f9-4082-846d-5d1afa07cb3d)




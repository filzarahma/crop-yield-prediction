# Laporan Proyek Machine Learning - Filza Rahma Muflihah

## Domain Proyek
Pertanian adalah sektor vital dalam penyediaan pangan dan penghidupan masyarakat. Salah satu tantangan utama dalam sektor ini adalah ketidakpastian hasil panen (crop yield), yang dipengaruhi oleh banyak faktor seperti curah hujan, kualitas tanah, ukuran lahan, intensitas sinar matahari, dan penggunaan pupuk. Dengan memanfaatkan machine learning, kita dapat membangun model prediktif untuk memperkirakan hasil panen berdasarkan faktor-faktor ini, sehingga membantu petani dan pengambil kebijakan dalam perencanaan dan pengelolaan pertanian yang lebih efektif.

**Mengapa masalah ini penting?**
- Membantu petani mengoptimalkan penggunaan sumber daya (air, pupuk, lahan).
- Membantu pemerintah dalam membuat kebijakan pangan berbasis data.
- Mengurangi ketidakpastian hasil pertanian.

## Business Understanding
### Problem Statements
- Bagaimana memprediksi hasil panen (crop yield) berdasarkan data cuaca, kondisi tanah, ukuran lahan, paparan sinar matahari, dan penggunaan pupuk?
- Seberapa besar akurasi model regresi sederhana seperti Linear Regression dalam memprediksi crop yield?

### Solution Statements
- Melakukan eksperimen untuk model prediksi dengan menggunakan Linear Regression, Decission Tree, XGBoost, Lasso Regression, Ridge Regression.
- Melakukan data scaling untuk meningkatkan performa model.
- Metrik evaluasi: Mean Squared Error (MSE) dan R2 Score.

## Data Understanding
Dataset yang digunakan terdiri dari 3000 observasi dengan 6 fitur:

| No	| Nama Fitur	| Deskripsi |
| --- | ----------- | ----------|
| 1	| rainfall_mm	| Curah hujan dalam milimeter |
| 2	| soil_quality_index |	Indeks kualitas tanah |
| 3	| farm_size_hectares |	Ukuran lahan dalam hektar |
| 4	| sunlight_hours |	Jumlah jam paparan sinar matahari per hari |
| 5	| fertilizer_kg	| Jumlah pupuk yang digunakan dalam kilogram |
| 6	| crop_yield |	Hasil panen (target yang ingin diprediksi) |
Dataset ini bersih, tidak terdapat missing values.

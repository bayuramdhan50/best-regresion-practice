# Dokumentasi Machine Learning

## Pengantar

Repositori ini berisi implementasi berbagai model machine learning yang dikembangkan untuk tujuan pembelajaran dan analisis. Setiap model dikategorikan dalam folder terpisah dan dilengkapi dengan dokumentasi komprehensif.

## Struktur Repositori

```
best-regresion/
│
├── regresi/
│   ├── main_regresi.py           # Program utama untuk analisis regresi
│
├── klasifikasi/
│   ├── main_classification.py    # Program utama untuk model klasifikasi
│   ├── iris_classification.csv   # Dataset klasifikasi bunga Iris
│   ├── credit_approval.csv       # Dataset persetujuan kredit
│   ├── medical_diagnosis.csv     # Dataset diagnosis medis
│   └── .gitignore                # Konfigurasi git untuk mengabaikan file output
│
├── clustering/
│   ├── main_clustering.py        # Program utama untuk model clustering
│   ├── customer_segmentation.csv # Dataset segmentasi pelanggan
│   ├── mall_customers.csv        # Dataset pelanggan mall
│   ├── product_categories.csv    # Dataset kategori produk
│   └── .gitignore                # Konfigurasi git untuk mengabaikan file output
│
└── dokumentasi/
    ├── dokumentasi_regresi.md    # Dokumentasi detail model regresi
    ├── dokumentasi_klasifikasi.md# Dokumentasi detail model klasifikasi
    └── dokumentasi_clustering.md # Dokumentasi detail model clustering
```

## Jenis Model

### 1. Model Regresi
Model regresi digunakan untuk memprediksi nilai numerik kontinu. Implementasi mencakup regresi linear dan polinomial dengan fokus pada perbandingan performa model dengan berbagai derajat polinomial.

[Dokumentasi Lengkap Model Regresi](dokumentasi/dokumentasi_regresi.md)

### 2. Model Klasifikasi
Model klasifikasi digunakan untuk memprediksi kategori atau kelas dari data. Implementasi mencakup berbagai algoritma klasifikasi populer seperti Decision Tree, Random Forest, SVM, Neural Network, dan lainnya.

[Dokumentasi Lengkap Model Klasifikasi](dokumentasi/dokumentasi_klasifikasi.md)

### 3. Model Clustering
Model clustering digunakan untuk mengelompokkan data menjadi segmen-segmen berdasarkan kesamaan. Implementasi mencakup algoritma clustering seperti K-Means, Hierarchical Clustering, Gaussian Mixture Model, dan DBSCAN.

[Dokumentasi Lengkap Model Clustering](dokumentasi/dokumentasi_clustering.md)

## Cara Penggunaan

### Model Regresi
```
cd regresi
python main_regresi.py
```
Program akan meminta input untuk derajat polinomial minimum dan maksimum, lalu melakukan analisis dan visualisasi hasil.

### Model Klasifikasi
```
cd klasifikasi
python main_classification.py
```
Program akan menawarkan pilihan dataset (otomatis atau manual) dan melakukan analisis klasifikasi lengkap.

### Model Clustering
```
cd clustering
python main_clustering.py
```
Program akan menawarkan pilihan dataset (otomatis atau manual) dan melakukan analisis clustering, termasuk penentuan jumlah cluster optimal.

## Fitur Umum

Semua model memiliki beberapa fitur umum:

1. **Visualisasi Komprehensif**: Setiap model menghasilkan berbagai plot dan visualisasi untuk membantu pemahaman hasil.

2. **Evaluasi Performa**: Penerapan berbagai metrik evaluasi yang sesuai untuk masing-masing jenis model.

3. **Pemilihan Model Terbaik**: Implementasi algoritma untuk membandingkan dan menentukan model optimal.

4. **Pelaporan**: Pembuatan laporan terstruktur untuk dokumentasi hasil.

5. **Pengorganisasian File**: Semua output disimpan dalam folder terpisah dengan penamaan konsisten.

## Kebutuhan Program

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost (untuk model klasifikasi)

## Pengembangan Masa Depan

Beberapa arah pengembangan yang direncanakan:

1. Implementasi validasi silang untuk evaluasi model yang lebih robust
2. Penambahan algoritma machine learning lanjutan
3. Implementasi pipeline preprocessing data yang lebih komprehensif
4. Optimasi parameter otomatis dan teknik ensemble
5. Integrasi dengan framework deep learning untuk perbandingan

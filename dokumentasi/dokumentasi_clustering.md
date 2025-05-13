# Dokumentasi Model Clustering

## Deskripsi Umum
Model clustering adalah algoritma machine learning unsupervised yang mengelompokkan data ke dalam cluster berdasarkan kesamaan atau jarak antar titik data. Program clustering dalam folder `clustering` mengimplementasikan berbagai algoritma clustering dan menganalisis hasilnya untuk membantu pengguna menemukan pola dan struktur dalam data mereka.

## File Utama
- `main_clustering.py`: Program utama yang melaksanakan analisis clustering
- Dataset yang disediakan:
  - `customer_segmentation.csv`: Dataset untuk segmentasi pelanggan
  - `mall_customers.csv`: Dataset pelanggan mall
  - `product_categories.csv`: Dataset untuk pengelompokan produk

## Algoritma yang Diimplementasikan
Program mengimplementasikan beberapa algoritma clustering populer:

1. **K-Means**:
   - Algoritma clustering berbasis centroid
   - Membagi data menjadi K cluster dengan jarak titik ke centroid minimal
   - Kelebihan: Simpel, cepat, dan efisien untuk dataset besar
   - Keterbatasan: Perlu menentukan jumlah cluster (K) sebelumnya, sensitif terhadap inisialisasi, cenderung menemukan cluster berbentuk bulat
   - Parameter utama: n_clusters, init, max_iter, n_init

2. **Hierarchical Clustering (Agglomerative)**:
   - Algoritma clustering hierarkis yang dimulai dengan menganggap setiap titik sebagai cluster, lalu menggabungkan cluster secara bertahap
   - Kelebihan: Tidak perlu menentukan jumlah cluster sebelumnya, menghasilkan dendrogram untuk visualisasi hierarki
   - Keterbatasan: Komputasional intensif untuk dataset besar, sulit diinterpretasi dengan data dimensi tinggi
   - Parameter utama: n_clusters, linkage (single, complete, average, ward)

3. **Gaussian Mixture Model (GMM)**:
   - Algoritma berbasis model probabilistik yang mengasumsikan data berasal dari campuran distribusi Gaussian
   - Kelebihan: Fleksibel untuk berbagai bentuk cluster, memberikan probabilitas keanggotaan cluster
   - Keterbatasan: Dapat konvergen ke solusi suboptimal, sensitif terhadap inisialisasi
   - Parameter utama: n_components, covariance_type, init_params

4. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - Algoritma berbasis densitas yang menemukan daerah dengan kepadatan tinggi dipisahkan oleh daerah kepadatan rendah
   - Kelebihan: Tidak perlu menentukan jumlah cluster sebelumnya, dapat menemukan cluster berbentuk acak, mendeteksi outlier/noise
   - Keterbatasan: Sensitif terhadap parameter densitas, kesulitan dengan cluster densitas bervariasi
   - Parameter utama: eps (radius tetangga), min_samples (minimum titik untuk cluster)

## Alur Kerja Program
Program clustering ini mengikuti alur kerja sebagai berikut:

1. **Persiapan Data**:
   - Memuat dataset dari file CSV
   - Menyajikan statistik deskriptif dan informasi dataset
   - Memungkinkan pemilihan dataset secara manual atau otomatis

2. **Preprocessing Data**:
   - Menghapus kolom ID yang tidak relevan
   - One-hot encoding untuk fitur kategorikal (jika ada)
   - Standarisasi fitur numerik untuk menyamakan skala
   - Menyimpan struktur data asli untuk analisis lanjutan

3. **Penentuan Jumlah Cluster Optimal**:
   - Implementasi berbagai metode untuk menentukan jumlah cluster terbaik:
     - Metode Elbow (WCSS - Within Cluster Sum of Squares)
     - Silhouette Score
     - Davies-Bouldin Index
     - Calinski-Harabasz Index
   - Visualisasi hasil setiap metode untuk membantu pengambilan keputusan
   - Opsi manual override untuk jumlah cluster

4. **Clustering dengan Berbagai Algoritma**:
   - Menjalankan K-Means, Hierarchical, GMM, dan DBSCAN
   - Mengevaluasi hasil setiap algoritma dengan Silhouette Score
   - Menentukan algoritma terbaik berdasarkan skor tertinggi

5. **Analisis dan Visualisasi**:
   - Reduksi dimensi menggunakan PCA untuk visualisasi 2D
   - Visualisasi hasil clustering untuk setiap algoritma
   - Analisis kontribusi fitur terhadap komponen utama (loadings PCA)
   - Karakterisasi cluster berdasarkan statistik deskriptif

6. **Pelaporan**:
   - Menyimpan profil detail setiap cluster
   - Membuat laporan lengkap hasil clustering
   - Menyimpan semua visualisasi dalam format gambar

## Metode Penentuan Jumlah Cluster
Program menggunakan empat metode berbeda untuk menentukan jumlah cluster optimal:

1. **Metode Elbow (WCSS)**:
   - Mengukur jumlah varians dalam data yang dijelaskan oleh clustering
   - Mencari "siku" dalam grafik dimana penambahan cluster baru memberikan manfaat marginal kecil
   - Semakin rendah nilainya, semakin baik clustering menjelaskan data

2. **Silhouette Score**:
   - Mengukur seberapa mirip objek dengan clusternya sendiri dibandingkan dengan cluster lain
   - Rentang nilai: -1 hingga 1, dimana nilai yang lebih tinggi menunjukkan definisi cluster yang lebih baik
   - Silhouette Score tinggi: objek sangat cocok dengan clusternya, sangat berbeda dari cluster lain

3. **Davies-Bouldin Index**:
   - Mengukur rata-rata "kesamaan" antara setiap cluster dengan cluster yang paling mirip
   - Nilai yang lebih rendah menunjukkan separasi cluster yang lebih baik
   - Lebih cocok untuk data dengan cluster berbentuk hipersfir

4. **Calinski-Harabasz Index**:
   - Juga dikenal sebagai Variance Ratio Criterion
   - Rasio dispersi antar-cluster dengan dispersi intra-cluster
   - Nilai yang lebih tinggi menunjukkan cluster yang lebih terdefinisi dengan baik

## Visualisasi
Program menghasilkan berbagai visualisasi untuk membantu interpretasi hasil:

1. **Matriks Korelasi**: Visualisasi korelasi antara fitur
2. **Grafik Penentuan Jumlah Cluster**: Visualisasi metode Elbow, Silhouette, Davies-Bouldin, dan Calinski-Harabasz
3. **Hasil Clustering 2D**: Visualisasi hasil clustering setelah PCA untuk setiap algoritma
4. **Kontribusi Fitur PCA**: Visualisasi kontribusi setiap fitur terhadap komponen utama
5. **Karakteristik Cluster**: Heatmap yang menampilkan nilai rata-rata fitur untuk setiap cluster

## Analisis Hasil
Program menyediakan analisis komprehensif dari hasil clustering:

1. **Statistik Cluster**:
   - Jumlah dan persentase titik data dalam setiap cluster
   - Nilai rata-rata dan standar deviasi setiap fitur dalam cluster

2. **Karakterisasi Cluster**:
   - Identifikasi fitur utama yang mendefinisikan setiap cluster
   - Perbandingan nilai-nilai fitur antar cluster

3. **Pelaporan Terstruktur**:
   - Laporan lengkap disimpan dalam file teks
   - Profil detail cluster disimpan dalam format CSV

## Penggunaan Program
1. Jalankan script `main_clustering.py`
2. Pilih mode pemilihan dataset (otomatis/manual)
3. Pilih dataset yang ingin dianalisis
4. Program akan menjalankan analisis dan menampilkan visualisasi
5. Secara opsional, tentukan jumlah cluster secara manual
6. Lihat hasil visualisasi dan laporan di folder `clustering`

## Interpretasi Hasil
Panduan singkat untuk menginterpretasi hasil clustering:

1. **Identifikasi Jumlah Cluster Optimal**:
   - Menggunakan perpaduan metode Elbow, Silhouette, Davies-Bouldin, dan Calinski-Harabasz
   - Mempertimbangkan interpretabilitas domain dan tujuan bisnis

2. **Evaluasi Kualitas Cluster**:
   - Silhouette Score tinggi menunjukkan cluster terdefinisi dengan baik
   - Distribusi ukuran cluster seimbang menunjukkan segmentasi yang baik
   - Variasi yang tinggi antara cluster, variasi rendah dalam cluster menunjukkan clustering optimal

3. **Karakterisasi Cluster**:
   - Mengidentifikasi fitur diskriminatif utama untuk setiap cluster
   - Memberi nama pada cluster berdasarkan karakteristiknya

4. **Aplikasi Bisnis**:
   - Untuk segmentasi pelanggan: strategi pemasaran berbeda untuk setiap segment
   - Untuk pengelompokan produk: pengaturan inventaris dan strategi promosi

## Potensi Pengembangan
Beberapa cara untuk mengembangkan model ini:
- Implementasi algoritma clustering tambahan seperti Spectral Clustering atau OPTICS
- Penambahan metode validasi eksternal untuk dataset berlabel
- Implementasi clustering berbasis anomali untuk deteksi outlier
- Pengembangan modul untuk clustering dinamis dan analisis perubahan cluster seiring waktu
- Integrasi dengan teknik visualisasi dimensi tinggi seperti t-SNE atau UMAP

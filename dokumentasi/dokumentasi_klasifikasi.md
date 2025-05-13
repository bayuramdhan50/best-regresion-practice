# Dokumentasi Model Klasifikasi

## Deskripsi Umum
Model klasifikasi adalah algoritma machine learning yang digunakan untuk memprediksi kategori atau kelas dari data berdasarkan fitur input. Program klasifikasi dalam folder `klasifikasi` mengimplementasikan berbagai algoritma klasifikasi dan membandingkan performanya untuk membantu pengguna menemukan model terbaik untuk dataset mereka.

## File Utama
- `main_classification.py`: Program utama yang melaksanakan analisis klasifikasi
- Dataset yang disediakan:
  - `iris_classification.csv`: Dataset klasifikasi bunga iris (multi-kelas)
  - `credit_approval.csv`: Dataset persetujuan kredit (biner)
  - `medical_diagnosis.csv`: Dataset diagnosis medis (biner)

## Algoritma yang Diimplementasikan
Program mengimplementasikan berbagai algoritma klasifikasi:

1. **Decision Tree**:
   - Algoritma berbasis pohon keputusan
   - Mudah diinterpretasi dan divisualisasikan
   - Parameter utama: kedalaman pohon, minimum sampel untuk split

2. **Random Forest**:
   - Ensemble dari banyak pohon keputusan
   - Lebih tahan terhadap overfitting dibandingkan decision tree tunggal
   - Parameter utama: jumlah estimator, kedalaman pohon maksimum

3. **Gradient Boosting**:
   - Algoritma boosting yang membangun pohon secara sekuensial
   - Performa yang kuat tetapi bisa lambat untuk dilatih
   - Parameter utama: jumlah estimator, learning rate, kedalaman pohon

4. **AdaBoost**:
   - Algoritma boosting yang memberikan bobot pada sampel yang salah diklasifikasi
   - Parameter utama: jumlah estimator, learning rate

5. **Support Vector Machine (SVM)**:
   - Linier dan non-linier (RBF kernel)
   - Efektif pada dimensi tinggi dan dataset kecil hingga menengah
   - Parameter utama: C (regularisasi), gamma, kernel

6. **K-Nearest Neighbors (KNN)**:
   - Algoritma berbasis instans yang mengklasifikasikan berdasarkan tetangga terdekat
   - Sederhana tetapi bisa komputasional intensif pada dataset besar
   - Parameter utama: jumlah tetangga (k), bobot, algoritma pencarian

7. **Naive Bayes**:
   - Algoritma probabilistik berdasarkan teorema Bayes
   - Cepat dan efisien untuk dataset besar
   - Parameter utama: smoothing parameter

8. **Neural Network (MLP)**:
   - Multi-layer perceptron
   - Mampu menangkap pola kompleks non-linier
   - Parameter utama: hidden layer size, activation function, learning rate

9. **XGBoost**:
   - Implementasi gradient boosting yang dioptimalkan
   - Performa tinggi dengan kecepatan lebih baik
   - Parameter utama: jumlah estimator, learning rate, kedalaman pohon

## Alur Kerja Program
Program klasifikasi ini mengikuti alur kerja sebagai berikut:

1. **Persiapan Data**:
   - Memuat dataset dari file CSV
   - Identifikasi otomatis kolom target
   - Menyajikan statistik deskriptif dan informasi dataset

2. **Preprocessing Data**:
   - Pemisahan fitur (X) dan target (y)
   - Split data menjadi training dan testing (75% training, 25% testing)
   - Standarisasi fitur numerik
   - Penanganan fitur kategorikal (jika ada)

3. **Evaluasi Model**:
   - Melatih dan mengevaluasi semua algoritma klasifikasi
   - Menghitung metrik performa untuk setiap model
   - Membandingkan performa model dan menentukan yang terbaik

4. **Fine-tuning Model Terbaik**:
   - Menggunakan GridSearchCV untuk pencarian hiperparameter
   - Validasi silang stratified 5-fold
   - Evaluasi performa model yang telah di-tuning

5. **Analisis Fitur Penting**:
   - Menampilkan dan memvisualisasikan kepentingan fitur
   - Menggunakan metode yang sesuai berdasarkan tipe model

6. **Visualisasi dan Pelaporan**:
   - Membuat visualisasi perbandingan model
   - Membuat matriks konfusi
   - Kurva ROC untuk kasus klasifikasi biner
   - Menyimpan laporan klasifikasi lengkap

## Metrik Evaluasi
Program mengevaluasi model klasifikasi menggunakan berbagai metrik:

1. **Akurasi**:
   - Persentase prediksi benar dari total prediksi
   - Metrik sederhana tetapi bisa menyesatkan pada dataset tidak seimbang

2. **Presisi**:
   - Rasio prediksi positif benar terhadap total prediksi positif
   - Mengukur seberapa "tepat" model ketika memprediksi positif

3. **Recall (Sensitivitas)**:
   - Rasio prediksi positif benar terhadap total positif aktual
   - Mengukur kemampuan model mendeteksi kelas positif

4. **F1 Score**:
   - Rata-rata harmonis dari presisi dan recall
   - Menyeimbangkan presisi dan recall, penting untuk dataset tidak seimbang

5. **ROC AUC Score**:
   - Area di bawah kurva Receiver Operating Characteristic
   - Mengukur kemampuan model membedakan kelas (untuk klasifikasi biner)

## Visualisasi
Program menghasilkan berbagai visualisasi:

1. **Distribusi Kelas**: Menampilkan distribusi variabel target
2. **Matriks Korelasi**: Visualisasi korelasi antara fitur numerik
3. **Perbandingan Model**: Grafik akurasi dan F1-score semua model
4. **Kurva ROC**: Untuk kasus klasifikasi biner
5. **Matriks Konfusi**: Visualisasi prediksi vs nilai aktual
6. **Fitur Penting**: Visualisasi kontribusi setiap fitur terhadap prediksi

## Fitur Khusus
1. **Deteksi Otomatis Kasus Multi-Kelas**: Program secara otomatis menyesuaikan metrik dan metode evaluasi berdasarkan jumlah kelas
2. **Pemilihan Dataset Fleksibel**: User dapat memilih dataset secara otomatis atau manual
3. **Pelaporan Komprehensif**: Menyimpan hasil analisis dalam file teks terstruktur
4. **Pengelolaan Folder**: Menyimpan output dalam folder terpisah untuk kerapihan

## Penggunaan Program
1. Jalankan script `main_classification.py`
2. Pilih mode pemilihan dataset (otomatis/manual)
3. Pilih dataset yang ingin dianalisis
4. Program akan menjalankan semua analisis secara otomatis
5. Lihat hasil visualisasi dan laporan di folder `klasifikasi`

## Potensi Pengembangan
Beberapa cara untuk mengembangkan model ini:
- Implementasi teknik sampling untuk menangani dataset tidak seimbang (SMOTE, undersampling, dll)
- Penambahan algoritma klasifikasi lain seperti LightGBM atau CatBoost
- Implementasi metode seleksi fitur automatik
- Penambahan modul untuk interpretabilitas model (SHAP values, LIME)

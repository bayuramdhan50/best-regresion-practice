# Dokumentasi Model Regresi

## Deskripsi Umum
Model regresi adalah algoritma machine learning yang digunakan untuk memprediksi nilai numerik kontinu berdasarkan satu atau lebih variabel input. Program utama dalam folder `regresi` adalah implementasi regresi polinomial yang digunakan untuk memprediksi nilai target dengan mencari hubungan matematis terbaik antara fitur input dan variabel target.

## File Utama
- `main_regresi.py`: Program utama yang melaksanakan analisis regresi polinomial

## Alur Kerja Program
Program regresi ini mengikuti alur kerja sebagai berikut:

1. **Persiapan Data**:
   - Membaca dataset (baik dari file CSV atau data contoh)
   - Memisahkan fitur (X) dan variabel target (y)

2. **Pemilihan Derajat Polinomial**:
   - User dapat menentukan rentang derajat polinomial minimum dan maksimum
   - Program akan menguji model untuk setiap derajat dalam rentang tersebut

3. **Pelatihan Model**:
   - Untuk setiap derajat polinomial, program membuat fitur polinomial dari data input
   - Model regresi linier dilatih menggunakan fitur polinomial

4. **Evaluasi Model**:
   - Program menghitung beberapa metrik performa untuk setiap model:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - Mean Absolute Percentage Error (MAPE)
     - Koefisien Determinasi (R²)

5. **Visualisasi**:
   - Menampilkan tabel perbandingan performa setiap model
   - User dapat memilih model untuk visualisasi:
     1. Model terbaik berdasarkan R²
     2. Model dengan derajat polinomial tertentu
     3. Perbandingan beberapa model

## Dataset
Program dirancang untuk bekerja dengan berbagai jenis dataset. Saat ini menggunakan contoh dataset sederhana:
- **Dataset Default**: Data hubungan antara usia dan tekanan darah sistolik

## Parameter Model
- **Derajat Polinomial**: Menentukan kompleksitas model, dengan nilai yang lebih tinggi menghasilkan model yang lebih kompleks
- **Normalisasi**: Program melakukan normalisasi data untuk visualisasi yang lebih baik

## Metrik Evaluasi
Program menggunakan berbagai metrik untuk mengevaluasi performa model:

1. **Mean Squared Error (MSE)**:
   - Mengukur rata-rata kuadrat perbedaan antara nilai prediksi dan nilai sebenarnya
   - Nilai yang lebih rendah menunjukkan performa yang lebih baik
   - Sensitif terhadap outlier

2. **Root Mean Squared Error (RMSE)**:
   - Akar kuadrat dari MSE, memiliki unit yang sama dengan variabel target
   - Nilai yang lebih rendah menunjukkan performa yang lebih baik

3. **Mean Absolute Error (MAE)**:
   - Mengukur rata-rata nilai absolut perbedaan antara nilai prediksi dan nilai sebenarnya
   - Lebih tahan terhadap outlier dibandingkan MSE/RMSE

4. **Mean Absolute Percentage Error (MAPE)**:
   - Mengukur persentase rata-rata perbedaan antara nilai prediksi dan nilai sebenarnya
   - Membantu memahami kesalahan relatif terhadap skala data

5. **Koefisien Determinasi (R²)**:
   - Mengukur proporsi variasi dalam variabel target yang dapat dijelaskan oleh model
   - Nilai mendekati 1 menunjukkan model yang lebih baik
   - Nilai dapat negatif jika model lebih buruk dari rata-rata sederhana

## Visualisasi
Program menyediakan tiga opsi visualisasi:

1. **Model Terbaik**: Menampilkan visualisasi dari model dengan nilai R² tertinggi
2. **Model Dengan Derajat Tertentu**: Menampilkan visualisasi model dengan derajat polinomial yang dipilih user
3. **Perbandingan Model**: Membandingkan beberapa model dengan derajat polinomial yang berbeda

Setiap visualisasi mencakup:
- Plot data asli
- Plot data dengan kurva prediksi model

## Potensi Pengembangan
Beberapa cara untuk mengembangkan model ini:
- Implementasi k-fold cross validation untuk evaluasi model yang lebih robust
- Penambahan teknik regularisasi untuk menghindari overfitting pada derajat polinomial tinggi
- Penambahan metode preprocessing data seperti penanganan outlier dan imputasi nilai yang hilang
- Implementasi algoritma regresi lain seperti Ridge, Lasso, atau ElasticNet

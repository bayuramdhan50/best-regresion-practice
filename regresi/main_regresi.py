# Import library
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Ganti bagian ini jika kamu punya data sendiri
data = {
    'Usia': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Tekanan_Darah_Sistolik': [120, 122, 124, 128, 130, 135, 138, 142, 145, 150]
}
df = pd.DataFrame(data)

# Input untuk pengaturan derajat polinomial
min_degree = int(input("Masukkan derajat polinomial minimum (misalnya 1): ") or "1")
max_degree = int(input("Masukkan derajat polinomial maksimum (misalnya 10): ") or "10")

# Variabel input dan target tanpa normalisasi dulu
X = df[['Usia']]
y = df['Tekanan_Darah_Sistolik']

# Uji beberapa model polinomial
results = []
print(f"\nMengevaluasi model polinomial dari derajat {min_degree} hingga {max_degree}...\n")

for degree in range(min_degree, max_degree + 1):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    r2 = r2_score(y, y_pred)

    results.append({
        'degree': degree,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    })

# Tampilkan hasil evaluasi
results_df = pd.DataFrame(results)
pd.set_option('display.float_format', '{:.6f}'.format)  # Mengatur format tampilan angka
print("Evaluasi Model Regresi (Linear & Polynomial):\n")
print(results_df)

# Tentukan model terbaik berdasarkan R2 dan error paling kecil
best_model = results_df.sort_values(by='r2', ascending=False).iloc[0]
print("\nModel Terbaik:")
print(best_model)

# Menentukan derajat polinomial yang ingin divisualisasikan
print("\nModel terbaik berdasarkan R2 adalah derajat", int(best_model['degree']))
print("\nPilih opsi visualisasi:")
print("1. Model terbaik (R2 tertinggi)")
print("2. Pilih derajat polinomial tertentu")
print("3. Bandingkan beberapa model polinomial")
visual_choice = input("Masukkan pilihan (1/2/3): ").strip()

# Normalisasi data untuk visualisasi (digunakan untuk semua jenis plot)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
y_normalized = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

if visual_choice == "1":
    # Visualisasi model terbaik
    selected_degree = int(best_model['degree'])
    print(f"\nMenampilkan visualisasi untuk model terbaik (Derajat {selected_degree})...")
    compare_models = False
    
elif visual_choice == "2":
    # Visualisasi model dengan derajat tertentu
    while True:
        try:
            selected_degree = int(input(f"\nMasukkan derajat polinomial yang ingin divisualisasikan ({min_degree}-{max_degree}): "))
            if selected_degree < min_degree or selected_degree > max_degree:
                print(f"Error: Masukkan derajat antara {min_degree} dan {max_degree}")
                continue
            break
        except ValueError:
            print("Error: Masukkan angka bulat")

    print(f"\nMenampilkan visualisasi untuk model polinomial derajat {selected_degree}...")
    compare_models = False
    
else:
    # Visualisasi perbandingan beberapa model
    compare_models = True
    print("\nMembandingkan beberapa model polinomial...")
    
    # Pilih model yang akan dibandingkan
    print("\nPilih derajat polinomial yang ingin dibandingkan (pisahkan dengan koma, maksimal 5 model)")
    print(f"Contoh: 1,3,5 (untuk membandingkan model derajat 1, 3, dan 5)")
    
    while True:
        degrees_input = input(f"Masukkan derajat ({min_degree}-{max_degree}): ").strip()
        try:
            selected_degrees = [int(d) for d in degrees_input.split(',')]
            
            # Validasi input
            if any(d < min_degree or d > max_degree for d in selected_degrees):
                print(f"Error: Semua derajat harus antara {min_degree} dan {max_degree}")
                continue
                
            if len(selected_degrees) > 5:
                print("Error: Maksimal 5 model yang dapat dibandingkan")
                continue
                
            break
        except ValueError:
            print("Error: Format tidak valid. Masukkan angka yang dipisahkan koma")

    print(f"\nMembandingkan model polinomial derajat: {', '.join(map(str, selected_degrees))}...")
# Bagian visualisasi
print("Menyiapkan visualisasi...")

if not compare_models:
    # Visualisasi satu model saja
    poly_selected = PolynomialFeatures(degree=selected_degree)
    X_poly_selected = poly_selected.fit_transform(X_normalized.reshape(-1, 1))
    model_selected = LinearRegression().fit(X_poly_selected, y_normalized)
    y_pred_selected = model_selected.predict(X_poly_selected)

    plt.figure(figsize=(12, 6))

    # Plot 1: Data asli (tidak dinormalisasi)
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, label='Data Asli')
    plt.title('Data Asli (Tidak Dinormalisasi)')
    plt.xlabel('Usia')
    plt.ylabel('Tekanan Darah Sistolik')
    plt.legend()

    # Plot 2: Model yang dipilih dengan data yang dinormalisasi
    plt.subplot(1, 2, 2)
    plt.scatter(X_normalized, y_normalized, label='Data Dinormalisasi')
    plt.plot(X_normalized, y_pred_selected, color='red', label=f'Polynomial Degree {selected_degree}')
    plt.title(f'Model Regresi Polinomial (Derajat {selected_degree})')
    plt.xlabel('Usia (Normalized)')
    plt.ylabel('Tekanan Darah Sistolik (Normalized)')
    plt.legend()

else:
    # Visualisasi perbandingan beberapa model
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Data asli (tidak dinormalisasi)
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, label='Data Asli')
    plt.title('Data Asli (Tidak Dinormalisasi)')
    plt.xlabel('Usia')
    plt.ylabel('Tekanan Darah Sistolik')
    plt.legend()
    
    # Plot 2: Perbandingan model polinomial
    plt.subplot(1, 2, 2)
    plt.scatter(X_normalized, y_normalized, label='Data', color='blue', alpha=0.7)
    
    # Generate colors for each model
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    
    # Plot each selected model
    for i, degree in enumerate(selected_degrees):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_normalized.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y_normalized)
        
        # Create more points for smoother curve
        X_smooth = np.linspace(0, 1, 100).reshape(-1, 1)
        X_smooth_poly = poly.transform(X_smooth)
        y_smooth = model.predict(X_smooth_poly)
        
        plt.plot(X_smooth, y_smooth, color=colors[i % len(colors)], 
                 label=f'Degree {degree}', linewidth=2)
    
    plt.title('Perbandingan Model Polinomial')
    plt.xlabel('Usia (Normalized)')
    plt.ylabel('Tekanan Darah Sistolik (Normalized)')
    plt.legend()

plt.tight_layout()
print("Menampilkan visualisasi...\n")
plt.show()

print("Program selesai.")

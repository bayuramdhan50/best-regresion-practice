# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import warnings
warnings.filterwarnings('ignore')

# Fungsi untuk memuat dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset berhasil dimuat dari {file_path}")
        print(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom")
        print("\nInformasi dataset:")
        print(df.info())
        print("\nStatistik deskriptif:")
        print(df.describe())
        print("\nSample data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error saat membaca dataset: {e}")
        return None

# Fungsi untuk preprocessing data
def preprocess_data(df):
    # Hapus kolom ID jika ada
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    elif 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    elif 'ProductID' in df.columns:
        df = df.drop('ProductID', axis=1)
    
    # Cek data kategorikal dan numerik
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # One-hot encoding untuk data kategorikal
    if categorical_cols:
        print(f"\nMelakukan one-hot encoding pada {len(categorical_cols)} kolom kategorikal")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Standarisasi data numerik
    print(f"\nMelakukan standarisasi pada {len(numeric_cols)} kolom numerik")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Buat DataFrame dengan nama kolom
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df_scaled, df.columns.tolist()

# Fungsi untuk menentukan jumlah cluster optimal dengan metode elbow
def find_optimal_clusters(data, max_clusters=10):
    wcss = []  # Within-Cluster Sum of Square
    silhouette_avg = []  # Silhouette Score
    db_score = []  # Davies-Bouldin Score
    ch_score = []  # Calinski-Harabasz Score
    
    K = range(2, max_clusters+1)
    
    for k in K:
        print(f"Mengevaluasi dengan {k} cluster...")
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        
        wcss.append(kmeans.inertia_)
        
        labels = kmeans.labels_
        
        # Hitung silhouette score
        try:
            s_score = silhouette_score(data, labels)
            silhouette_avg.append(s_score)
        except:
            silhouette_avg.append(0)
        
        # Hitung Davies-Bouldin score
        try:
            db = davies_bouldin_score(data, labels)
            db_score.append(db)
        except:
            db_score.append(0)
        
        # Hitung Calinski-Harabasz score
        try:
            ch = calinski_harabasz_score(data, labels)
            ch_score.append(ch)
        except:
            ch_score.append(0)
    
    # Plot WCSS (Elbow Method)
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('WCSS')
    plt.title('Metode Elbow untuk Jumlah Cluster Optimal')
    plt.grid(True)
    
    # Plot Silhouette Score
    plt.subplot(2, 2, 2)
    plt.plot(K, silhouette_avg, 'rx-')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score untuk Jumlah Cluster Optimal')
    plt.grid(True)
    
    # Plot Davies-Bouldin Score
    plt.subplot(2, 2, 3)
    plt.plot(K, db_score, 'gx-')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score untuk Jumlah Cluster Optimal')
    plt.grid(True)
    
    # Plot Calinski-Harabasz Score
    plt.subplot(2, 2, 4)
    plt.plot(K, ch_score, 'yx-')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score untuk Jumlah Cluster Optimal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('clustering/optimal_clusters.png')
    print("Visualisasi metode penentuan jumlah cluster optimal disimpan sebagai 'clustering/optimal_clusters.png'")
    
    # Temukan jumlah cluster optimal berdasarkan silhouette score
    optimal_k = K[silhouette_avg.index(max(silhouette_avg))]
    print(f"\nJumlah cluster optimal berdasarkan Silhouette Score: {optimal_k}")
    
    # Berikan semua skor untuk masing-masing k
    results = pd.DataFrame({
        'Jumlah Cluster': K,
        'WCSS': wcss,
        'Silhouette Score': silhouette_avg,
        'Davies-Bouldin Score': db_score,
        'Calinski-Harabasz Score': ch_score
    })
    
    print("\nSkor untuk setiap jumlah cluster:")
    print(results)
    
    return optimal_k, results

# Fungsi untuk melakukan clustering dengan beberapa algoritma
def perform_clustering(data, optimal_k):
    # KMeans Clustering
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    
    # Hierarchical Clustering (Agglomerative)
    agglomerative = AgglomerativeClustering(n_clusters=optimal_k)
    agg_labels = agglomerative.fit_predict(data)
    
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    
    # DBSCAN (tidak memerlukan jumlah cluster yang ditentukan sebelumnya)
    # Metode ini memerlukan parameter eps dan min_samples yang tepat
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data)
    
    # Hitung skor silhouette untuk setiap metode
    scores = {}
    
    try:
        scores['KMeans'] = silhouette_score(data, kmeans_labels)
    except:
        scores['KMeans'] = 0
        
    try:
        scores['Agglomerative'] = silhouette_score(data, agg_labels)
    except:
        scores['Agglomerative'] = 0
        
    try:
        scores['GMM'] = silhouette_score(data, gmm_labels)
    except:
        scores['GMM'] = 0
    
    try:
        # DBSCAN mungkin menghasilkan label -1 (noise), jadi kita perlu menangani ini
        if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
            scores['DBSCAN'] = silhouette_score(data, dbscan_labels)
        else:
            scores['DBSCAN'] = 0
    except:
        scores['DBSCAN'] = 0
    
    print("\nSilhouette Score untuk setiap metode clustering:")
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    
    # Pilih metode dengan skor tertinggi
    best_method = max(scores, key=scores.get)
    print(f"\nMetode clustering terbaik: {best_method}")
    
    # Dapatkan label dari metode terbaik
    if best_method == 'KMeans':
        best_labels = kmeans_labels
        best_model = kmeans
    elif best_method == 'Agglomerative':
        best_labels = agg_labels
        best_model = agglomerative
    elif best_method == 'GMM':
        best_labels = gmm_labels
        best_model = gmm
    else:
        best_labels = dbscan_labels
        best_model = dbscan
    
    return best_method, best_model, best_labels, {
        'KMeans': kmeans_labels,
        'Agglomerative': agg_labels,
        'GMM': gmm_labels,
        'DBSCAN': dbscan_labels
    }

# Fungsi untuk visualisasi hasil clustering dengan PCA
def visualize_clusters(data, labels_dict, feature_names, original_df=None):
    # Reduce dimensionality untuk visualisasi
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Buat DataFrame untuk plotting
    df_reduced = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    
    # Plot untuk setiap metode clustering
    plt.figure(figsize=(20, 15))
    
    plot_idx = 1
    for method, labels in labels_dict.items():
        # Tambahkan labels ke DataFrame
        df_reduced[f'Cluster_{method}'] = labels
        
        plt.subplot(2, 2, plot_idx)
        
        # Jika DBSCAN dan ada noise points (label = -1)
        if method == 'DBSCAN' and -1 in labels:
            # Plot noise points terlebih dahulu
            noise_mask = labels == -1
            plt.scatter(df_reduced.loc[noise_mask, 'PC1'], 
                       df_reduced.loc[noise_mask, 'PC2'],
                       c='black', marker='x', label='Noise')
            
            # Plot cluster points
            for cluster in sorted(set(labels)):
                if cluster != -1:  # Skip noise points
                    mask = labels == cluster
                    plt.scatter(df_reduced.loc[mask, 'PC1'], 
                               df_reduced.loc[mask, 'PC2'],
                               label=f'Cluster {cluster}')
        else:
            # Plot untuk metode lain
            for cluster in sorted(set(labels)):
                mask = labels == cluster
                plt.scatter(df_reduced.loc[mask, 'PC1'], 
                           df_reduced.loc[mask, 'PC2'],
                           label=f'Cluster {cluster}')
        
        plt.title(f'Clustering dengan {method}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('clustering/cluster_visualization.png')
    print("Visualisasi hasil clustering disimpan sebagai 'clustering/cluster_visualization.png'")
    
    # Visualisasi komponen PCA
    plt.figure(figsize=(12, 10))
    
    # Komponen loading
    loadings = pca.components_.T
    
    # Plot loading untuk PC1
    plt.subplot(2, 1, 1)
    plt.bar(feature_names, loadings[:, 0])
    plt.xticks(rotation=90)
    plt.title('Feature Contribution to Principal Component 1')
    plt.tight_layout()
    
    # Plot loading untuk PC2
    plt.subplot(2, 1, 2)
    plt.bar(feature_names, loadings[:, 1])
    plt.xticks(rotation=90)
    plt.title('Feature Contribution to Principal Component 2')
    plt.tight_layout()
    
    plt.savefig('clustering/pca_components.png')
    print("Visualisasi komponen PCA disimpan sebagai 'clustering/pca_components.png'")
    
    # Jika original_df disediakan, kita bisa melakukan analisis cluster
    if original_df is not None:
        # Tambahkan label cluster dari metode terbaik ke DataFrame asli
        best_method = max(labels_dict, key=lambda k: silhouette_score(data, labels_dict[k]) if len(set(labels_dict[k])) > 1 and -1 not in labels_dict[k] else 0)
        original_df['Cluster'] = labels_dict[best_method]
        
        # Analisis karakteristik setiap cluster
        plt.figure(figsize=(15, 10))
        
        # Hitung rata-rata fitur untuk setiap cluster
        cluster_means = original_df.groupby('Cluster').mean()
        
        # Heatmap perbandingan cluster
        sns.heatmap(cluster_means, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Karakteristik Rata-rata untuk Setiap Cluster')
        plt.tight_layout()
        plt.savefig('clustering/cluster_characteristics.png')
        print("Visualisasi karakteristik cluster disimpan sebagai 'clustering/cluster_characteristics.png'")
        
        # Simpan profil cluster ke file
        cluster_profile = original_df.groupby('Cluster').agg(['mean', 'std'])
        cluster_profile.to_csv('clustering/cluster_profiles.csv')
        print("Profil detail cluster disimpan sebagai 'clustering/cluster_profiles.csv'")
        
        return original_df, cluster_profile
    
    return df_reduced

# Fungsi utama
def main():
    print("="*80)
    print("ANALISIS CLUSTERING")
    print("="*80)
    
    # Buat folder output jika belum ada
    if not os.path.exists('clustering'):
        os.makedirs('clustering')
    
    # Daftar dataset yang tersedia
    datasets = [
        'clustering/customer_segmentation.csv',
        'clustering/mall_customers.csv',
        'clustering/product_categories.csv'
    ]
    
    print("\nDataset yang tersedia:")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset.split('/')[-1]}")
    
    # Pilihan otomatis atau manual
    auto_choice = input("\nGunakan pilihan otomatis? (y/n): ")
    
    if auto_choice.lower() == 'y':
        # Gunakan dataset customer segmentation secara default
        file_path = datasets[0]
        print(f"\nMenggunakan dataset: {file_path}")
    else:
        # Pemilihan dataset manual
        choice = input("\nPilih dataset (1-3) atau ketik nama file langsung: ")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                file_path = datasets[idx]
            else:
                file_path = choice
        except ValueError:
            file_path = choice
    
    # Load dataset
    df = load_dataset(file_path)
    
    if df is None:
        print("Program berhenti karena dataset tidak dapat dimuat.")
        return
    
    # Simpan copy dari data asli sebelum preprocessing
    df_original = df.copy()
    
    # Preprocessing data
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)
    df_scaled, feature_names = preprocess_data(df)
    
    # Exploratory Data Analysis
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Heatmap korelasi
    plt.figure(figsize=(12, 10))
    correlation = df_scaled.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title('Matriks Korelasi')
    plt.tight_layout()
    plt.savefig('clustering/correlation_matrix.png')
    print("Visualisasi matriks korelasi disimpan sebagai 'clustering/correlation_matrix.png'")
    
    # Cari jumlah cluster optimal
    print("\n" + "="*80)
    print("PENENTUAN JUMLAH CLUSTER OPTIMAL")
    print("="*80)
    
    max_clusters = min(10, df.shape[0] // 5)  # Maksimal 10 cluster atau 1/5 dari jumlah data
    optimal_k, cluster_scores = find_optimal_clusters(df_scaled, max_clusters)
    
    # Input jumlah cluster manual (opsional)
    custom_k = input(f"\nIngin menentukan jumlah cluster secara manual? (y/n, default: {optimal_k}): ")
    
    if custom_k.lower() == 'y':
        while True:
            try:
                k_input = int(input(f"Masukkan jumlah cluster (2-{max_clusters}): "))
                if 2 <= k_input <= max_clusters:
                    optimal_k = k_input
                    break
                else:
                    print(f"Error: Masukkan angka antara 2 dan {max_clusters}")
            except ValueError:
                print("Error: Masukkan angka bulat")
    
    # Lakukan clustering
    print("\n" + "="*80)
    print(f"CLUSTERING DENGAN {optimal_k} CLUSTER")
    print("="*80)
    
    best_method, best_model, best_labels, all_labels = perform_clustering(df_scaled, optimal_k)
    
    # Visualisasi hasil clustering
    print("\n" + "="*80)
    print("VISUALISASI HASIL CLUSTERING")
    print("="*80)
    
    df_result, cluster_profile = visualize_clusters(df_scaled, all_labels, feature_names, df_original)
    
    # Analisis karakteristik cluster
    print("\n" + "="*80)
    print("ANALISIS KARAKTERISTIK CLUSTER")
    print("="*80)
    
    # Tampilkan statistik untuk setiap cluster
    cluster_sizes = df_result['Cluster'].value_counts().sort_index()
    print("\nJumlah data dalam setiap cluster:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} data ({size / len(df_result) * 100:.2f}%)")
    
    print("\nRata-rata nilai untuk setiap fitur dalam setiap cluster:")
    print(cluster_profile.xs('mean', axis=1, level=1))
    
    # Simpan hasil clustering
    df_result.to_csv('clustering/clustering_results.csv', index=False)
    print("\nHasil clustering disimpan sebagai 'clustering/clustering_results.csv'")
    
    # Simpan laporan
    with open('clustering/laporan_clustering.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("LAPORAN CLUSTERING\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {file_path}\n\n")
        f.write(f"Jumlah cluster optimal: {optimal_k}\n")
        f.write(f"Metode clustering terbaik: {best_method}\n\n")
        f.write("Jumlah data dalam setiap cluster:\n")
        for cluster, size in cluster_sizes.items():
            f.write(f"Cluster {cluster}: {size} data ({size / len(df_result) * 100:.2f}%)\n")
        
        f.write("\nKarakteristik utama setiap cluster:\n")
        for cluster in sorted(df_result['Cluster'].unique()):
            f.write(f"\nCluster {cluster}:\n")
            # Dapatkan 5 fitur teratas yang mencirikan cluster ini (nilai tertinggi)
            cluster_mean = cluster_profile.xs('mean', axis=1, level=1).loc[cluster]
            top_features = cluster_mean.sort_values(ascending=False).head(5)
            for feature, value in top_features.items():
                f.write(f"  - {feature}: {value:.4f}\n")
    
    print("\nLaporan lengkap disimpan sebagai 'clustering/laporan_clustering.txt'")
    print("\nProgram selesai!")

if __name__ == "__main__":
    main()

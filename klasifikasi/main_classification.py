# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline

# Model klasifikasi 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
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
        
        # Menentukan kolom target berdasarkan dataset
        target_col = get_target_column(file_path, df)
        if target_col:
            print(f"\nKolom target: {target_col}")
            print("\nDistribusi kelas:")
            print(df[target_col].value_counts())
        else:
            print("\nTidak dapat menentukan kolom target!")
        
        print("\nSample data:")
        print(df.head())
        return df, target_col
    except Exception as e:
        print(f"Error saat membaca dataset: {e}")
        return None, None
        
# Fungsi untuk menentukan kolom target berdasarkan dataset
def get_target_column(file_path, df):
    # Berdasarkan nama file
    file_name = file_path.split('\\')[-1].lower()
    
    if 'medical_diagnosis' in file_name:
        return 'Diagnosis'
    elif 'iris' in file_name:
        return 'species'
    elif 'credit' in file_name:
        return 'Loan_Amount_Approved'
    
    # Jika tidak ditemukan, cari kolom yang mungkin target
    potential_targets = ['class', 'target', 'label', 'diagnosis', 'species', 'result', 'outcome', 'y']
    for col in potential_targets:
        if col in df.columns:
            return col
    
    # Gunakan kolom terakhir sebagai default
    return df.columns[-1]

# Fungsi untuk preprocessing data
def preprocess_data(df, target_col):
    # Pisahkan fitur dan target
    if 'ID' in df.columns:
        X = df.drop(['ID', target_col], axis=1)
    else:
        X = df.drop([target_col], axis=1)
    y = df[target_col]

    # Konversi label kategorikal ke numerik jika perlu (LabelEncoder untuk target)
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Label target dikonversi: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Untuk fitur kategorikal (selain target), lakukan one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print(f"\nData dibagi menjadi {X_train.shape[0]} sampel train dan {X_test.shape[0]} sampel test")

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Deteksi kolom numerik
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        print(f"\nMelakukan standarisasi pada {len(numeric_cols)} kolom numerik")
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test

# Fungsi untuk evaluasi model
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Fit model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Cek jumlah kelas untuk metrik
    n_classes = len(np.unique(y_test))
    multi_class = n_classes > 2
    
    # Probabilitas prediksi (untuk ROC)
    y_prob_test = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            # Untuk kasus biner
            if proba.shape[1] == 2:
                y_prob_test = proba[:, 1]
            else:
                # Untuk kasus multi-class, tidak bisa langsung menggunakan ROC curve biasa
                y_prob_test = proba
        except:
            y_prob_test = None
    elif hasattr(model, "decision_function"):
        try:
            y_prob_test = model.decision_function(X_test)
        except:
            y_prob_test = None
    
    # Evaluasi metrik
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Untuk masalah multi-class, gunakan average='macro' atau 'weighted'
    if multi_class:
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    else:
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
    
    # ROC AUC score (hanya untuk kasus biner)
    test_roc_auc = None
    if not multi_class and y_prob_test is not None:
        try:
            test_roc_auc = roc_auc_score(y_test, y_prob_test)
        except:
            test_roc_auc = None
    
    return {
        'model_name': model_name,
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'y_pred': y_pred_test,
        'y_prob': y_prob_test,
        'multi_class': multi_class
    }

# Fungsi untuk membandingkan model
def compare_models(X_train, X_test, y_train, y_test):
    models = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('SVM (Linear)', SVC(kernel='linear')),
        ('SVM (RBF)', SVC(kernel='rbf', probability=True)),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Neural Network', MLPClassifier(max_iter=1000)),
        ('XGBoost', XGBClassifier())
    ]
    
    results = []
    
    print("\nEvaluasi berbagai model klasifikasi:")
    for name, model in models:
        print(f"\nMemproses model: {name}")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        print(f"  Akurasi training: {result['train_accuracy']*100:.2f}%")
        print(f"  Akurasi testing: {result['test_accuracy']*100:.2f}%")
        print(f"  F1 Score: {result['test_f1']:.4f}")
    
    return results

# Fungsi untuk fine-tuning model terbaik
def tune_best_model(best_model_name, X_train, X_test, y_train, y_test):
    # Deteksi multi-class atau tidak
    n_classes = len(np.unique(y_train))
    multi_class = n_classes > 2
    
    if 'Decision Tree' in best_model_name:
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = DecisionTreeClassifier()
    
    elif 'Random Forest' in best_model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier()
    
    elif 'Gradient Boosting' in best_model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = GradientBoostingClassifier()
    
    elif 'AdaBoost' in best_model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
        base_model = AdaBoostClassifier()
    
    elif 'SVM' in best_model_name:
        if 'Linear' in best_model_name:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear']
            }
        else:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'poly']
            }
        
        # Untuk multi-class, tambahkan parameter decision_function_shape
        if multi_class:
            param_grid['decision_function_shape'] = ['ovo', 'ovr']
            
        base_model = SVC(probability=True)
    
    elif 'K-Nearest Neighbors' in best_model_name:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        base_model = KNeighborsClassifier()
    
    elif 'Neural Network' in best_model_name:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000]
        }
        base_model = MLPClassifier()
    
    elif 'XGBoost' in best_model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        base_model = XGBClassifier()
    
    else:
        # Default to Naive Bayes
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
        base_model = GaussianNB()
    
    print(f"\nMelakukan fine-tuning untuk model {best_model_name}...")
    print("Parameter yang dioptimalkan:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # GridSearchCV dengan stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Scoring berdasarkan jumlah kelas
    scoring = 'f1_weighted' if multi_class else 'f1'
    
    grid_search = GridSearchCV(
        base_model, param_grid, cv=skf, 
        scoring=scoring, n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nHasil Fine-tuning:")
    print(f"Parameter terbaik: {grid_search.best_params_}")
    print(f"F1-score terbaik pada CV: {grid_search.best_score_:.4f}")
    
    # Evaluasi model dengan parameter terbaik
    best_model = grid_search.best_estimator_
    result = evaluate_model(best_model, X_train, X_test, y_train, y_test, f"{best_model_name} (Tuned)")
    
    print("\nPerforma model setelah tuning:")
    print(f"  Akurasi training: {result['train_accuracy']*100:.2f}%")
    print(f"  Akurasi testing: {result['test_accuracy']*100:.2f}%")
    print(f"  Precision: {result['test_precision']:.4f}")
    print(f"  Recall: {result['test_recall']:.4f}")
    print(f"  F1 Score: {result['test_f1']:.4f}")
    if result['test_roc_auc']:
        print(f"  ROC AUC: {result['test_roc_auc']:.4f}")
    
    return result

# Fungsi untuk analisis fitur penting
def feature_importance_analysis(X_train, y_train, best_model, feature_names):
    # Deteksi multi-class atau tidak
    n_classes = len(np.unique(y_train))
    multi_class = n_classes > 2
    
    if hasattr(best_model, 'feature_importances_'):
        # Model berbasis tree
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nAnalisis Fitur Penting:")
        for i in range(len(feature_names)):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot fitur penting
        plt.figure(figsize=(10, 6))
        plt.title('Fitur Penting')
        plt.bar(range(len(feature_names)), importances[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('klasifikasi/feature_importance.png')
        print("Visualisasi fitur penting disimpan sebagai 'klasifikasi/feature_importance.png'")
    
    else:
        print("\nModel tidak mendukung analisis fitur penting secara langsung.")
        
        # Gunakan permutation importance untuk model lain
        from sklearn.inspection import permutation_importance
        
        # Gunakan lebih sedikit pengulangan untuk dataset besar
        n_repeats = 5 if X_train.shape[0] > 1000 else 10
        
        print(f"\nMenghitung Permutation Feature Importance (n_repeats={n_repeats})...")
        perm_importance = permutation_importance(best_model, X_train, y_train, n_repeats=n_repeats, random_state=42)
        feature_importance = perm_importance.importances_mean
        indices = np.argsort(feature_importance)[::-1]
        
        print("\nPermutation Feature Importance:")
        for i in range(len(feature_names)):
            print(f"  {i+1}. {feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")
        
        # Plot permutation importance
        plt.figure(figsize=(10, 6))
        plt.title('Permutation Feature Importance')
        plt.bar(range(len(feature_names)), feature_importance[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('klasifikasi/permutation_importance.png')
        print("Visualisasi permutation importance disimpan sebagai 'klasifikasi/permutation_importance.png'")

# Fungsi untuk visualisasi model dan hasil
def visualize_results(results, X_test, y_test):
    # Plot perbandingan akurasi
    plt.figure(figsize=(12, 6))
    model_names = [result['model_name'] for result in results]
    test_accuracy = [result['test_accuracy'] for result in results]
    test_f1 = [result['test_f1'] for result in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, test_accuracy, width, label='Akurasi')
    plt.bar(x + width/2, test_f1, width, label='F1 Score')
    
    plt.xlabel('Model')
    plt.ylabel('Skor')
    plt.title('Perbandingan Performa Model')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('klasifikasi/model_comparison.png')
    print("Visualisasi perbandingan model disimpan sebagai 'klasifikasi/model_comparison.png'")
    
    # Plot ROC Curve untuk model yang mendukung predict_proba (hanya untuk kasus biner)
    any_roc = False
    for result in results:
        if result['test_roc_auc'] is not None and result['y_prob'] is not None and not result.get('multi_class', False):
            any_roc = True
            break
    
    if any_roc:
        plt.figure(figsize=(10, 8))
        for result in results:
            if result['test_roc_auc'] is not None and result['y_prob'] is not None and not result.get('multi_class', False):
                fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['test_roc_auc']:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig('klasifikasi/roc_curve.png')
        print("Visualisasi ROC curve disimpan sebagai 'klasifikasi/roc_curve.png'")
    else:
        print("ROC curve tidak tersedia (membutuhkan kasus biner dan model yang mendukung predict_proba)")
    
    # Confusion Matrix untuk model terbaik
    best_model_result = max(results, key=lambda x: x['test_f1'])
    
    # Deteksi apakah multi-kelas atau biner
    is_multi_class = best_model_result.get('multi_class', False)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_model_result['y_pred'])
    
    if is_multi_class and cm.shape[0] > 2:
        # Untuk visualisasi matriks yang lebih besar, gunakan format yang berbeda
        fmt = 'd' if np.max(cm) < 100 else '.1f'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted (0=Negatif, 1=Positif)')
        plt.ylabel('Actual (0=Negatif, 1=Positif)')
    
    plt.title(f'Confusion Matrix - {best_model_result["model_name"]}')
    plt.savefig('klasifikasi/confusion_matrix.png')
    print("Visualisasi confusion matrix disimpan sebagai 'klasifikasi/confusion_matrix.png'")
    
    return best_model_result

# Fungsi untuk menyalin dataset ke folder klasifikasi
def copy_datasets_to_folder():
    import shutil
    import os
    
    # Daftar dataset yang akan disalin
    datasets = [
        'medical_diagnosis.csv',
        'iris_classification.csv',
        'credit_approval.csv'
    ]
    
    # Buat folder klasifikasi jika belum ada
    if not os.path.exists('klasifikasi'):
        os.makedirs('klasifikasi')
    
    # Salin file dataset
    for dataset in datasets:
        try:
            source = dataset
            destination = os.path.join('klasifikasi', dataset)
            shutil.copy2(source, destination)
            print(f"Dataset {dataset} berhasil disalin ke folder klasifikasi")
        except Exception as e:
            print(f"Error saat menyalin {dataset}: {e}")

# Fungsi utama
def main():
    print("="*80)
    print("ANALISIS MODEL KLASIFIKASI TERBAIK")
    print("="*80)
    
    # Salin dataset ke folder klasifikasi
    copy_datasets_to_folder()
    
    # Daftar dataset yang tersedia
    datasets = [
        'klasifikasi/iris_classification.csv',
        'klasifikasi/credit_approval.csv',
        'klasifikasi/medical_diagnosis.csv'
    ]
    
    print("\nDataset yang tersedia:")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset.split('/')[-1]}")
    
    # Pilihan otomatis atau manual
    auto_choice = input("\nGunakan pilihan otomatis? (y/n): ")
    
    if auto_choice.lower() == 'y':
        # Gunakan dataset Iris secara default
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
    df, target_col = load_dataset(file_path)
    
    if df is None or target_col is None:
        print("Program berhenti karena dataset tidak dapat dimuat.")
        return
    
    # Eksplorasi dan visualisasi data
    print("\n" + "="*80)
    print("EKSPLORASI DATA")
    print("="*80)
    
    # Plot distribusi kelas
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title(f'Distribusi Kelas {target_col}')
    plt.savefig('klasifikasi/class_distribution.png')
    print("Visualisasi distribusi kelas disimpan sebagai 'klasifikasi/class_distribution.png'")
    
    # Heatmap korelasi
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.shape[1] > 1:  # Pastikan ada lebih dari 1 kolom numerik
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriks Korelasi')
        plt.tight_layout()
        plt.savefig('klasifikasi/correlation_matrix.png')
        print("Visualisasi matriks korelasi disimpan sebagai 'klasifikasi/correlation_matrix.png'")
    else:
        print("Tidak cukup kolom numerik untuk matriks korelasi")
    
    # Preprocessing data
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
    
    # Bandingkan berbagai model
    print("\n" + "="*80)
    print("EVALUASI MODEL")
    print("="*80)
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Visualisasi hasil
    print("\n" + "="*80)
    print("VISUALISASI HASIL")
    print("="*80)
    best_model_result = visualize_results(results, X_test, y_test)
    
    # Fine-tuning model terbaik
    print("\n" + "="*80)
    print("FINE-TUNING MODEL TERBAIK")
    print("="*80)
    best_model_name = best_model_result['model_name']
    print(f"Model terbaik berdasarkan F1 Score: {best_model_name}")
    
    tuned_result = tune_best_model(best_model_name, X_train, X_test, y_train, y_test)
    
    # Analisis fitur penting
    print("\n" + "="*80)
    print("ANALISIS FITUR PENTING")
    print("="*80)
    feature_names = X_train.columns
    feature_importance_analysis(X_train, y_train, tuned_result['model'], feature_names)
    
    # Klasifikasi report lengkap
    print("\n" + "="*80)
    print("LAPORAN KLASIFIKASI LENGKAP")
    print("="*80)
    print(classification_report(y_test, tuned_result['y_pred']))
    
    # Pernyataan kesimpulan
    print("\n" + "="*80)
    print("KESIMPULAN")
    print("="*80)
    print(f"Model klasifikasi terbaik untuk dataset {file_path} adalah: {tuned_result['model_name']}")
    print(f"Dengan metrik performa:")
    print(f"  Akurasi: {tuned_result['test_accuracy']*100:.2f}%")
    print(f"  Precision: {tuned_result['test_precision']:.4f}")
    print(f"  Recall: {tuned_result['test_recall']:.4f}")
    print(f"  F1 Score: {tuned_result['test_f1']:.4f}")
    if tuned_result['test_roc_auc']:
        print(f"  ROC AUC: {tuned_result['test_roc_auc']:.4f}")
    
    # Simpan hasil laporan ke file
    with open('klasifikasi/laporan_klasifikasi.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("LAPORAN KLASIFIKASI\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {file_path}\n\n")
        f.write("Model terbaik: " + tuned_result['model_name'] + "\n")
        f.write("Metrik performa:\n")
        f.write(f"  Akurasi: {tuned_result['test_accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {tuned_result['test_precision']:.4f}\n")
        f.write(f"  Recall: {tuned_result['test_recall']:.4f}\n")
        f.write(f"  F1 Score: {tuned_result['test_f1']:.4f}\n")
        if tuned_result['test_roc_auc']:
            f.write(f"  ROC AUC: {tuned_result['test_roc_auc']:.4f}\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, tuned_result['y_pred']))
    
    print("\nLaporan lengkap disimpan sebagai 'klasifikasi/laporan_klasifikasi.txt'")
    print("\nProgram selesai!")

if __name__ == "__main__":
    main()

import os
import sys
import joblib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    load_data,
    ganti_tanda_tanya,
    cek_missing_value,
    handle_missing_value,
    cek_duplikasi,
    cek_outlier,
    handle_outlier,
    encode_kategorikal,
    pisahkan_fitur_target
)
from transform import minmax_scaling
from resampling import adasyn_tomek
from feature_selection import select_from_model, rfecv_selection
from model import train_decision_tree
from nb_model import train_naive_bayes
from svm_model import train_svm
from evaluation import evaluasi_model
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'adult.csv'
OUTPUT_DIR = BASE_DIR / 'output'
TARGET_COL = 'income'
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("1. INPUT DATA")
    print("=" * 60)
    df = load_data(str(DATA_PATH))
    df = ganti_tanda_tanya(df)
    df = df.drop(columns=['fnlwgt'])

    print("\n" + "=" * 60)
    print("2. PREPROCESSING")
    print("=" * 60)

    df = cek_missing_value(df)
    df = handle_missing_value(df)
    df = cek_duplikasi(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    df = cek_outlier(df, numeric_cols)
    df = handle_outlier(df, numeric_cols)
    df, encoders = encode_kategorikal(df)
    X, y = pisahkan_fitur_target(df, TARGET_COL)

    print("\n" + "=" * 60)
    print("3. SPLIT DATA (80% training, 20% testing)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\n" + "=" * 60)
    print("4. TRANSFORMASI MIN-MAX")
    print("=" * 60)
    X_train_scaled, X_test_scaled, scaler = minmax_scaling(X_train, X_test)

    print("\n" + "=" * 60)
    print("5. RESAMPLING (ADASYN + TOMEK LINKS)")
    print("=" * 60)
    X_train_res, y_train_res = adasyn_tomek(X_train_scaled, y_train)

    print("\n" + "=" * 60)
    print("6. SELEKSI FITUR")
    print("=" * 60)

    X_train_sfm, X_test_sfm, sfm = select_from_model(
        X_train_res, y_train_res, X_test_scaled
    )
    X_train_rfecv, X_test_rfecv, rfecv = rfecv_selection(
        X_train_res, y_train_res, X_test_scaled
    )

    print("\n" + "=" * 60)
    print("7. KLASIFIKASI (3 MODEL)")
    print("=" * 60)

    print("\n--- Model 1: Decision Tree ---")
    model_dt = train_decision_tree(X_train_sfm, y_train_res)

    print("\n--- Model 2: Naive Bayes ---")
    model_nb = train_naive_bayes(X_train_sfm, y_train_res)

    print("\n--- Model 3: SVM ---")
    model_svm = train_svm(X_train_sfm, y_train_res)

    print("\n" + "=" * 60)
    print("8. EVALUASI")
    print("=" * 60)

    print("\n### Evaluasi Decision Tree ###")
    hasil_dt = evaluasi_model(
        model_dt, X_test_sfm, y_test,
        output_dir=str(OUTPUT_DIR),
        nama_model='DecisionTree'
    )

    print("\n### Evaluasi Naive Bayes ###")
    hasil_nb = evaluasi_model(
        model_nb, X_test_sfm, y_test,
        output_dir=str(OUTPUT_DIR),
        nama_model='NaiveBayes'
    )

    print("\n### Evaluasi SVM ###")
    hasil_svm = evaluasi_model(
        model_svm, X_test_sfm, y_test,
        output_dir=str(OUTPUT_DIR),
        nama_model='SVM'
    )

    print("\n" + "=" * 60)
    print("9. PERBANDINGAN 3 MODEL")
    print("=" * 60)

    print(f"{'Metode':<20} {'Akurasi':<10} {'Presisi':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    print(f"{'Decision Tree':<20} {hasil_dt['accuracy']:<10.4f} {hasil_dt['precision']:<10.4f} {hasil_dt['recall']:<10.4f} {hasil_dt['f1_score']:<10.4f}")
    print(f"{'Naive Bayes':<20} {hasil_nb['accuracy']:<10.4f} {hasil_nb['precision']:<10.4f} {hasil_nb['recall']:<10.4f} {hasil_nb['f1_score']:<10.4f}")
    print(f"{'SVM':<20} {hasil_svm['accuracy']:<10.4f} {hasil_svm['precision']:<10.4f} {hasil_svm['recall']:<10.4f} {hasil_svm['f1_score']:<10.4f}")

    print("\n" + "=" * 60)
    print("10. KESIMPULAN")
    print("=" * 60)

    hasil = {
        'Decision Tree': hasil_dt,
        'Naive Bayes': hasil_nb,
        'SVM': hasil_svm
    }

    model_terbaik = max(hasil, key=lambda x: hasil[x]['f1_score'])
    print(f"Model terbaik: {model_terbaik}")
    print(f"F1-Score: {hasil[model_terbaik]['f1_score']:.4f}")
    print(f"Akurasi : {hasil[model_terbaik]['accuracy']:.4f}")

    print("\nOutput tersimpan di:", OUTPUT_DIR)
    print("Pipeline selesai.")

    print("\n" + "=" * 60)
    print("11. SIMPAN MODEL")
    print("=" * 60)

    MODEL_DIR = BASE_DIR / 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(encoders, MODEL_DIR / 'encoders.pkl')

    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

    joblib.dump(sfm, MODEL_DIR / 'selector_sfm.pkl')

    joblib.dump(model_dt, MODEL_DIR / 'decision_tree.pkl')

    joblib.dump(model_nb, MODEL_DIR / 'naive_bayes.pkl')

    joblib.dump(model_svm, MODEL_DIR / 'svm.pkl')

    print("Model tersimpan di:", MODEL_DIR)
    print("Pipeline selesai.")

if __name__ == '__main__':
    main()
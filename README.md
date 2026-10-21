# Machine Learning Classification Project

Implementasi klasifikasi dataset **Adult Census Income** menggunakan beberapa metode Machine Learning, lengkap dengan pipeline preprocessing, transformasi, resampling, seleksi fitur, dan UI interaktif berbasis Streamlit.

---

## Deskripsi

Project ini memprediksi apakah seseorang memiliki penghasilan **lebih dari $50K/tahun** berdasarkan data sensus. Terdapat tiga model klasifikasi yang dibandingkan performanya:

1. **Decision Tree**
2. **Naive Bayes**
3. **Support Vector Machine (SVM)**

Alur pengerjaan mengikuti pipeline berikut:

```
Input
  ↓
Preprocessing
  ↓
Split Data
  ↓
┌───────────────┬───────────────┐
│ Data Testing  │ Data Training │
│      ↓        │       ↓       │
│ Transformation│ Transformation│
│      ↓        │       ↓       │
│    Fitur      │ Ekstraksi Fitur│
│               │       ↓       │
│               │    Fitur      │
└───────────────┴───────────────┘
                  ↓
               Training
                  ↓
               Testing
                  ↓
               Evaluasi
```

---

## Dataset

| Item | Keterangan |
|---|---|
| **Nama** | Adult Census Income |
| **Sumber** | [Kaggle — UCI ML](https://www.kaggle.com/datasets/uciml/adult-census-income) |
| **Jumlah Baris** | 32.561 |
| **Jumlah Kolom** | 15 |
| **Target** | `income` (`<=50K` / `>50K`) |
| **Tipe Fitur** | Kategorikal + Numerik |
| **Karakteristik** | Imbalanced (mayoritas `<=50K`) |

**Deskripsi:**  
Dataset ini digunakan untuk memprediksi apakah seseorang memiliki penghasilan **lebih dari $50K/tahun** berdasarkan data sensus. Fitur mencakup usia, pekerjaan, pendidikan, status pernikahan, jam kerja, dll.

---

## Metode

| Tahap | Metode | Keterangan |
|---|---|---|
| **Preprocessing** | Missing value, duplikasi, outlier, encoding | Deteksi & handling |
| **Transformasi** | Min-Max Scaler | Rentang 0–1 |
| **Split Data** | 80% training, 20% testing | Stratified |
| **Resampling** | ADASYN + Tomek Links | Hanya data training |
| **Seleksi Fitur 1** | SelectFromModel (Decision Tree) | Embedded method |
| **Seleksi Fitur 2** | RFECV (Recursive Feature Elimination with CV) | Wrapper method |
| **Klasifikasi 1** | Decision Tree | `criterion='gini'` |
| **Klasifikasi 2** | Naive Bayes | `GaussianNB` |
| **Klasifikasi 3** | SVM | `kernel='rbf'` |
| **Evaluasi** | Confusion Matrix | Akurasi, Presisi, Recall, F1-Score |
| **UI/UX** | Streamlit | Prediksi interaktif |

---

## Struktur Folder

```
Machine-Learning-Project/
│
├── data/
│   └── adult.csv                  # Dataset (tidak di-push ke GitHub)
│
├── models/
│   ├── scaler.pkl                 # Min-Max Scaler
│   ├── selector_sfm.pkl           # SelectFromModel
│   ├── decision_tree.pkl          # Model Decision Tree
│   ├── naive_bayes.pkl            # Model Naive Bayes
│   ├── svm.pkl                    # Model SVM
│   └── encoders.pkl               # LabelEncoder kategorikal
│
├── output/
│   ├── confusion_matrix_DecisionTree.png
│   ├── confusion_matrix_NaiveBayes.png
│   ├── confusion_matrix_SVM.png
│   ├── classification_report_DecisionTree.txt
│   ├── classification_report_NaiveBayes.txt
│   └── classification_report_SVM.txt
│
├── src/
│   ├── preprocessing.py           # Load, missing value, duplikasi, outlier, encoding
│   ├── transform.py               # Min-Max Scaling
│   ├── resampling.py              # ADASYN + Tomek Links
│   ├── feature_selection.py       # SelectFromModel + RFECV
│   ├── model.py                   # Decision Tree
│   ├── nb_model.py                # Naive Bayes
│   ├── svm_model.py               # SVM
│   ├── evaluation.py              # Confusion Matrix + metrik
│   └── main.py                    # Full pipeline
│
├── app.py                         # UI Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `adult.csv` dari [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income), lalu letakkan di folder `data/`.

### 4. Jalankan Pipeline Utama (Training 3 Model)
```bash
python src/main.py
```
*(Ini akan menyimpan model ke folder `models/` dan hasil evaluasi ke `output/`)*

### 5. Jalankan UI Streamlit
```bash
python -m streamlit run app.py
```
*(Buka browser di `http://localhost:8501` untuk mencoba prediksi interaktif)*

---

## Hasil Evaluasi

| Metode | Akurasi | Presisi | Recall | F1-Score |
|---|---|---|---|---|
| **Decision Tree** | **0.7707** | 0.7756 | **0.7707** | **0.7730** |
| Naive Bayes | 0.7133 | 0.8013 | 0.7133 | 0.7333 |
| SVM | 0.7466 | **0.8418** | 0.7466 | 0.7646 |

**Kesimpulan:**  
Model **Decision Tree** memberikan performa terbaik secara keseluruhan (F1-Score 0.7730). Namun, SVM unggul dalam presisi dan recall kelas minoritas. Naive Bayes memiliki recall tertinggi untuk kelas minoritas tetapi presisi terendah.

---

## Confusion Matrix

### Decision Tree
```
[[4147  793]
 [ 699  869]]
```

### Naive Bayes
```
[[3394 1546]
 [ 320 1248]]
```

### SVM
```
[[3455 1485]
 [ 164 1404]]
```

---

## Analisis Singkat

- **Akurasi ~77%** → model Decision Tree cukup baik untuk prediksi income.
- **Kelas mayoritas (`<=50K`)** → F1-Score 0.85 (bagus).
- **Kelas minoritas (`>50K`)** → F1-Score 0.54 (kurang bagus).
- **Resampling** → Membantu meningkatkan sensitivitas model terhadap kelas minoritas.
- **UI Streamlit** → Memudahkan pengguna memprediksi income berdasarkan input data.

---

## Requirements

```txt
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
streamlit
joblib
```

---

## Referensi

- [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html)
- [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
- [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    """Load dataset Adult Census Income."""
    df = pd.read_csv(path)
    print("Shape Awal:", df.shape)
    return df

def ganti_tanda_tanya(df):
    """Ganti '?' jadi NaN."""
    df = df.replace('?', np.nan)
    return df

def cek_missing_value(df):
    """Cek dan tampilkan missing value."""
    print("\n=== Missing Value ===")
    print(df.isnull().sum())
    return df

def handle_missing_value(df):
    """Handling missing value: numerik -> median, kategorikal -> modus."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    print("\nSetelah handling missing value:")
    print(df.isnull().sum())
    return df

def cek_duplikasi(df):
    """Cek dan hapus duplikasi."""
    print("\n=== Duplikasi ===")
    print("Jumlah duplikasi:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Setelah drop duplikasi:", df.shape)
    return df

def cek_outlier(df, numeric_cols):
    """Cek outlier dengan IQR."""
    print("\n=== Outlier (IQR) ===")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outlier")
    return df

def handle_outlier(df, numeric_cols):
    """Handling outlier dengan capping (IQR)."""
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    print("\nOutlier sudah di-capping.")
    return df

def encode_kategorikal(df):
    """Encoding kategorikal dengan Label Encoding."""
    encoders = {}
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    for col in df.select_dtypes(include=['string']).columns:
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    print("\nEncoding selesai.")
    return df, encoders

def pisahkan_fitur_target(df, target_col='income'):
    """Pisahkan fitur dan target."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    print("\nShape X:", X.shape)
    print("Shape y:", y.shape)
    return X, y
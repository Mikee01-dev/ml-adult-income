import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks

def adasyn_tomek(X_train, y_train):
    """
    Resampling: ADASYN + Tomek Links.
    Hanya dilakukan pada data training.
    """
    print("\n=== Resampling: ADASYN + Tomek Links ===")

    print("\nSebelum resampling:")
    print(pd.Series(y_train).value_counts())

    print("\n--- ADASYN (Oversampling) ---")
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)
    print("Setelah ADASYN:")
    print(pd.Series(y_res).value_counts())

    print("\n--- Tomek Links (Undersampling) ---")
    tomek = TomekLinks()
    X_res, y_res = tomek.fit_resample(X_res, y_res)
    print("Setelah Tomek Links:")
    print(pd.Series(y_res).value_counts())

    print("\nResampling selesai.")
    print("Shape X_res:", X_res.shape)
    print("Shape y_res:", y_res.shape)

    return X_res, y_res
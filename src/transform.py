from sklearn.preprocessing import MinMaxScaler

def minmax_scaling(X_train, X_test):
    """
    Transformasi Min-Max: fit di training, transform ke testing.
    
    Parameter:
        X_train: fitur training
        X_test: fitur testing
    
    Return:
        X_train_scaled: fitur training setelah scaling
        X_test_scaled: fitur testing setelah scaling
        scaler: objek MinMaxScaler (buat referensi)
    """
    print("\n=== Transformasi Min-Max ===")
    print("Sebelum scaling:")
    print("X_train min:", X_train.min().min(), "| max:", X_train.max().max())
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nSetelah scaling:")
    print("X_train_scaled min:", X_train_scaled.min(), "| max:", X_train_scaled.max())
    print("X_test_scaled min:", X_test_scaled.min(), "| max:", X_test_scaled.max())
    print("Transformasi Min-Max selesai.")
    
    return X_train_scaled, X_test_scaled, scaler
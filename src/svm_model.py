from sklearn.svm import SVC

def train_svm(X_train, y_train, random_state=42):
    """Training SVM."""
    print("\n=== Training SVM ===")
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Training SVM selesai.")
    return model
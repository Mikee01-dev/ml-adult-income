from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    """Training Naive Bayes."""
    print("\n=== Training Naive Bayes ===")
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Training Naive Bayes selesai.")
    return model
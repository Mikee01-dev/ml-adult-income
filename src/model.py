from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, random_state=42):
    """
    Training Decision Tree.
    
    Parameter:
        X_train: fitur training (sudah diseleksi)
        y_train: target training (sudah di-resampling)
        random_state: seed biar konsisten
    
    Return:
        model: DecisionTreeClassifier yang sudah dilatih
    """
    print("\n=== Training Decision Tree ===")
    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    print("Training selesai.")
    print("Jumlah node:", model.tree_.node_count)
    print("Kedalaman pohon:", model.tree_.max_depth)

    return model
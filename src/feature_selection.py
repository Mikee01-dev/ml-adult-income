from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

def select_from_model(X_train, y_train, X_test):
    """
    Seleksi fitur dengan SelectFromModel (Decision Tree).
    """
    print("\n--- SelectFromModel (Decision Tree) ---")
    dt = DecisionTreeClassifier(random_state=42)
    sfm = SelectFromModel(dt, threshold='median')
    X_train_sel = sfm.fit_transform(X_train, y_train)
    X_test_sel = sfm.transform(X_test)
    print("Shape setelah SelectFromModel:", X_train_sel.shape)
    return X_train_sel, X_test_sel, sfm


def rfecv_selection(X_train, y_train, X_test):
    """
    Seleksi fitur dengan RFECV.
    """
    print("\n--- RFECV ---")
    dt = DecisionTreeClassifier(random_state=42)
    rfecv = RFECV(
        estimator=dt,
        step=1,
        cv=StratifiedKFold(5),
        scoring='f1',
        min_features_to_select=1,
        n_jobs=-1
    )
    X_train_sel = rfecv.fit_transform(X_train, y_train)
    X_test_sel = rfecv.transform(X_test)
    print("Shape setelah RFECV:", X_train_sel.shape)
    print("Jumlah fitur terpilih:", rfecv.n_features_)
    return X_train_sel, X_test_sel, rfecv
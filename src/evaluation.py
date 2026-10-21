import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def evaluasi_model(model, X_test, y_test, output_dir='../output', nama_model='model'):
    """
    Evaluasi model: Confusion Matrix + Akurasi, Presisi, Recall, F1-Score.
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== Confusion Matrix ({nama_model}) ===")
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    report = classification_report(y_test, y_pred)
    print(f"\n=== Classification Report ({nama_model}) ===")
    print(report)

    with open(f'{output_dir}/classification_report_{nama_model}.txt', 'w') as f:
        f.write(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {nama_model}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_{nama_model}.png')
    plt.close()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
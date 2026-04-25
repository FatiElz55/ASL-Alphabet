import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


print("\n" + "=" * 55)
print("  STEP 1 — Loading data")
print("=" * 55)

X       = np.load("X.npy")
y       = np.load("y.npy")
classes = np.load("classes.npy", allow_pickle=True)

print(f"\nX shape    : {X.shape}")
print(f"y shape    : {y.shape}")
print(f"Classes    : {list(classes)}")
print(f"Num classes: {len(classes)}")


print("\n" + "=" * 55)
print("  STEP 2 — Train / test split  (80 / 20, stratified)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


print("\n" + "=" * 55)
print("  STEP 3 — Training SVM  (RBF kernel, C=10, gamma=scale)")
print("=" * 55)
print("  (This may take a minute on large datasets...)")


model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,   
        random_state=42,
    )),
])
model.fit(X_train, y_train)


print("\n" + "=" * 55)
print("  STEP 4 — Evaluation")
print("=" * 55)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy : {acc * 100:.2f} %\n")
print(classification_report(y_test, y_pred, target_names=classes))


fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=classes,
    xticks_rotation=45,
    ax=ax,
    colorbar=False,
)
ax.set_title(f"SVM — Confusion Matrix — Test accuracy: {acc * 100:.1f}%", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png", dpi=120, bbox_inches="tight")
plt.show()
print("Confusion matrix saved → confusion_matrix_svm.png")


print("\n" + "=" * 55)
print("  STEP 5 — Saving model")
print("=" * 55)

MODEL_PATH = "model_svm.pkl"
joblib.dump({"model": model, "classes": classes}, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

print("\n" + "=" * 55)
print("  train_model_svm.py DONE")
print("  To use this model in app.py, change MODEL_PATH to 'model_svm.pkl'")
print("=" * 55)
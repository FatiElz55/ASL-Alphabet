"""
train_model.py
==============
Trains a Random Forest classifier on the MediaPipe landmark vectors produced
by data_cleaning.py and saves the ready-to-use model.

Inputs  : X.npy, y.npy, classes.npy  (produced by data_cleaning.py)
Output  : model.pkl                   (loaded by app.py at runtime)

Run:
    python train_model.py
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3 — Training Random Forest")
print("=" * 55)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4 — Evaluation")
print("=" * 55)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy : {acc * 100:.2f} %\n")
print(classification_report(y_test, y_pred, target_names=classes))


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=classes,
    xticks_rotation=45,
    ax=ax,
    colorbar=False,
)
ax.set_title(f"Confusion Matrix — Test accuracy: {acc * 100:.1f}%", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("Confusion matrix saved → confusion_matrix.png")


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5 — Saving model")
print("=" * 55)

MODEL_PATH = "model.pkl"
joblib.dump({"model": model, "classes": classes}, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

print("\n" + "=" * 55)
print("  train_model.py DONE")
print("  Next step → python app.py")
print("=" * 55)

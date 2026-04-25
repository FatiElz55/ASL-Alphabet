import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

MODELS = {
    "Random Forest": {"path": "model.pkl",     "color": "#a6e3a1"},
    "SVM":           {"path": "model_svm.pkl", "color": "#89b4fa"},
}

print("\n" + "=" * 60)
print("  Loading dataset")
print("=" * 60)

X       = np.load("X.npy")
y       = np.load("y.npy")
classes = np.load("classes.npy", allow_pickle=True)

print(f"  X shape    : {X.shape}")
print(f"  Classes    : {len(classes)}  →  {list(classes)}")

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print(f"  Test samples: {len(X_test)}")

results = {}   # name → {acc, precision, recall, f1, y_pred}

for name, cfg in MODELS.items():
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"\n  [{name}]  ⚠  '{path}' not found — skipped.")
        continue

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {name}")
    print("=" * 60)

    payload = joblib.load(path)
    clf     = payload["model"]
    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n  Accuracy  : {acc  * 100:.2f} %")
    print(f"  Precision : {prec * 100:.2f} %  (macro)")
    print(f"  Recall    : {rec  * 100:.2f} %  (macro)")
    print(f"  F1-score  : {f1   * 100:.2f} %  (macro)")
    print(f"\n{classification_report(y_test, y_pred, target_names=classes, zero_division=0)}")

    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    results[name] = {
        "acc":     acc,
        "prec":    prec,
        "rec":     rec,
        "f1":      f1,
        "y_pred":  y_pred,
        "color":   cfg["color"],
        "pc_prec": pc_prec,   # shape (n_classes,)
        "pc_rec":  pc_rec,
        "pc_f1":   pc_f1,
    }

if not results:
    print("\nNo model files found. Train at least one model first.")
    raise SystemExit(1)

n_models = len(results)

model_names = list(results.keys())
n_classes   = len(classes)

COL_HEADERS = ["Precision", "Recall", "F1-score", "Support"]
BG          = "#1e1e2e"
PANEL_BG    = "#2a2a3e"
HDR_BG      = "#3a3a5e"
TEXT_FG     = "#cdd6f4"
MUTED       = "#888888"

_, pc_support = precision_recall_fscore_support(
    y_test, list(results.values())[0]["y_pred"],
    labels=list(range(n_classes)), zero_division=0
)[:2]   # we only need support; recompute properly below
_, _, _, pc_support = precision_recall_fscore_support(
    y_test, list(results.values())[0]["y_pred"],
    labels=list(range(n_classes)), zero_division=0
)

fig_t, axes_t = plt.subplots(1, n_models, figsize=(6.5 * n_models, 0.38 * n_classes + 3))
fig_t.patch.set_facecolor(BG)

if n_models == 1:
    axes_t = [axes_t]

for ax, name in zip(axes_t, model_names):
    res   = results[name]
    color = res["color"]

    ax.set_facecolor(BG)
    ax.axis("off")

    pc_prec = res["pc_prec"]
    pc_rec  = res["pc_rec"]
    pc_f1   = res["pc_f1"]

    macro_prec = res["prec"]
    macro_rec  = res["rec"]
    macro_f1   = res["f1"]
    total_sup  = int(pc_support.sum())

    row_labels = list(classes) + ["macro avg"]
    cell_data  = [
        [f"{pc_prec[i]:.2f}", f"{pc_rec[i]:.2f}", f"{pc_f1[i]:.2f}", str(int(pc_support[i]))]
        for i in range(n_classes)
    ] + [[f"{macro_prec:.2f}", f"{macro_rec:.2f}", f"{macro_f1:.2f}", str(total_sup)]]

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=COL_HEADERS,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if row == 0:                          # header row
            cell.set_facecolor(HDR_BG)
            cell.set_text_props(color=color, fontweight="bold")
        elif row == len(row_labels):          # macro avg row
            cell.set_facecolor(HDR_BG)
            cell.set_text_props(color=TEXT_FG, fontweight="bold")
        elif col == -1:                       # row-label column (class names)
            cell.set_facecolor(PANEL_BG)
            cell.set_text_props(color=color, fontweight="bold")
        else:
            cell.set_facecolor(PANEL_BG)
            cell.set_text_props(color=TEXT_FG)

    ax.set_title(
        f"{name}\nAccuracy: {res['acc'] * 100:.2f} %",
        color=color, fontsize=12, fontweight="bold", pad=10,
    )

plt.suptitle(
    "Classification Report Comparison — All Models",
    color=TEXT_FG, fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("comparison_reports.png", dpi=130, bbox_inches="tight",
            facecolor=fig_t.get_facecolor())
plt.show()
print("Saved → comparison_reports.png")

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
header = f"  {'Model':<18}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}"
print(header)
print("  " + "-" * (len(header) - 2))
for name, res in results.items():
    print(
        f"  {name:<18}  {res['acc']*100:>9.2f}%  "
        f"{res['prec']*100:>9.2f}%  "
        f"{res['rec']*100:>9.2f}%  "
        f"{res['f1']*100:>9.2f}%"
    )
    
print("=" * 60)
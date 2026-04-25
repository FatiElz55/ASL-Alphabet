import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from hand_detector import HandDetector

DATASET_DIR    = r"C:\Users\Mery\OneDrive\Desktop\vpo\data\asl_alphabet_train\asl_alphabet_train"
CLEAN_DIR      = "dataset_clean"
MAX_PER_CLASS  = 1000   # set to None to use all images
MIN_IMG_SIZE   = 50     # minimum width/height in pixels
OUTPUT_X       = "X.npy"
OUTPUT_Y       = "y.npy"
OUTPUT_CLASSES = "classes.npy"


print("\n" + "=" * 55)
print("  STEP 1 — Discovering classes")
print("=" * 55)

classes = sorted([
    c for c in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, c))
])

print(f"\nClasses found : {len(classes)}")
print(f"Labels        : {classes}\n")


print("=" * 55)
print("  STEP 2 — Cleaning images")
print("=" * 55)

total_kept = total_rejected = 0

for cls in tqdm(classes, desc="Cleaning"):
    src_dir = os.path.join(DATASET_DIR, cls)
    dst_dir = os.path.join(CLEAN_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)

    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if MAX_PER_CLASS:
        files = files[:MAX_PER_CLASS]

    kept = rejected = 0
    for fname in files:
        src = os.path.join(src_dir, fname)
        img = cv2.imread(src)

        if img is None:
            rejected += 1
            continue

        h, w = img.shape[:2]
        if h < MIN_IMG_SIZE or w < MIN_IMG_SIZE:
            rejected += 1
            continue

        mean_val = float(np.mean(img))
        if mean_val < 5 or mean_val > 250:
            rejected += 1
            continue

        if img.ndim < 3 or img.shape[2] != 3:
            rejected += 1
            continue

        shutil.copy2(src, os.path.join(dst_dir, fname))
        kept += 1

    total_kept     += kept
    total_rejected += rejected

print(f"\nKept     : {total_kept}")
print(f"Rejected : {total_rejected}")
print(f"Clean dir: {CLEAN_DIR}/")


print("\n" + "=" * 55)
print("  STEP 3 — Extracting MediaPipe landmarks")
print("=" * 55)

detector = HandDetector(
    max_hands=1,
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
    min_tracking_confidence=0.25,
)

X:        list[np.ndarray] = []
y_labels: list[str]        = []
skipped = 0

for cls in tqdm(classes, desc="Landmarks"):
    folder = os.path.join(CLEAN_DIR, cls)
    files  = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            skipped += 1
            continue

        vector, _ = detector.get_landmarks(img)
        if vector is None:
            skipped += 1
            continue

        X.append(vector)
        y_labels.append(cls)

print(f"\nVectors extracted : {len(X)}")
print(f"Images skipped    : {skipped} (no hand detected)")


print("\n" + "=" * 55)
print("  STEP 4 — Encoding labels & saving")
print("=" * 55)

classes_array = np.array(sorted(set(y_labels)))
label_to_idx  = {lbl: i for i, lbl in enumerate(classes_array)}

X_arr = np.array(X, dtype=np.float32)
y_arr = np.array([label_to_idx[lbl] for lbl in y_labels], dtype=np.int32)

np.save(OUTPUT_X,       X_arr)
np.save(OUTPUT_Y,       y_arr)
np.save(OUTPUT_CLASSES, classes_array)

print(f"X.npy       → shape {X_arr.shape}")
print(f"y.npy       → shape {y_arr.shape}")
print(f"classes.npy → {list(classes_array)}")


print("\n" + "=" * 55)
print("  STEP 5 — Sanity check")
print("=" * 55)

if X_arr.size == 0:
    print("\nERROR: No landmark vectors were extracted.")
    print("Check that DATASET_DIR points to the folder containing letter sub-folders (A, B, C…).")
else:
    print(f"\nNaN values      : {np.isnan(X_arr).sum()}")
    print(f"Infinite values : {np.isinf(X_arr).sum()}")
    print(f"Min / Max       : {X_arr.min():.4f} / {X_arr.max():.4f}")

    if np.isnan(X_arr).sum() == 0 and np.isinf(X_arr).sum() == 0:
        print("\nData is clean — no NaN or infinite values detected.")
    else:
        print("\nWARNING: invalid values detected — check your images!")

print("\nClass distribution:")
for i, cls in enumerate(classes_array):
    n   = int((y_arr == i).sum())
    bar = "█" * (n // 50)
    print(f"  {cls:10s} : {n:5d}  {bar}")

print("\n" + "=" * 55)
print("  data_cleaning.py DONE")
print("  Next step → python train_model_svm.py")
print("=" * 55)
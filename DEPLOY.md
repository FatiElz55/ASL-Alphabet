# Deployment notes (Vercel & size limits)

## What was making the bundle huge

1. **Duplicate OpenCV** — If `requirements.txt` listed `opencv-python` (or `opencv-python-headless`) *and* `mediapipe`, you pay for two OpenCV stacks. `mediapipe` already requires `opencv-contrib-python` for `import cv2`. The runtime `requirements.txt` should **not** include a separate opencv line.

2. **MediaPipe** pulls `matplotlib` and `opencv-contrib-python` (by design). That stack, plus `scikit-learn` → `scipy`, is still large. There is no supported way to make MediaPipe “small” on pip.

3. **Repo clutter** — Use `.vercelignore` (and `.gitignore`) so you do not upload datasets, `*.npy`, training scripts, docs, and extra `model.pkl` if you only serve `model_svm.pkl`.

## Vercel Python limit (500 MB)

After deduplication, total installed size can **still** exceed 500 MB because of MediaPipe + SciPy + OpenCV. If the build still fails:

- **Option A (simplest):** Host the FastAPI app on a platform with a **larger** image limit (e.g. **Railway**, **Render** with a Docker image, **Fly.io**, **Google Cloud Run**).
- **Option B (architectural):** Run **MediaPipe in the browser** (JavaScript) and send only the 63-dimension landmark vector to the server. The server would then need only `numpy` + `joblib` + `scikit-learn` (or an ONNX model + `onnxruntime`) and **no** MediaPipe or OpenCV — a much smaller bundle.

## Required files in production

- `app.py`, `hand_detector.py`
- `templates/index.html`
- `model_svm.pkl`
- `requirements.txt` (as in this repo)

## Python version

Vercel defaults to 3.12 if unset. A `.python-version` file in the repo (e.g. `3.12`) keeps environments aligned.

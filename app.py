import os
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

CONFIDENCE_THRESH = 0.40

MODEL_NAME = "SVM"
MODEL_PATH = "model_svm.pkl"

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"

_model_lock = threading.Lock()
_clf = None
_classes: list | None = None
_active_model_name: str | None = None


def load_model(path: str):
    payload = joblib.load(path)
    return payload["model"], list(payload["classes"])


def first_available_model() -> tuple[str | None, str | None]:
    if os.path.exists(MODEL_PATH):
        return MODEL_NAME, MODEL_PATH
    return None, None


def _ensure_globals():
    global _clf, _classes, _active_model_name
    if _clf is not None:
        return
    init_name, init_path = first_available_model()
    if init_name is None:
        raise RuntimeError(
            "No model file found. Run train_model_svm.py to create model_svm.pkl."
        )
    _clf, _classes = load_model(init_path)
    _active_model_name = init_name


class PredictRequest(BaseModel):
    vector: list[float] = Field(..., description="Flattened 21*(x,y,z) = 63 floats")


class PredictResponse(BaseModel):
    letter: str
    confidence: float


app = FastAPI()


@app.get("/")
async def index():
    if not INDEX_PATH.is_file():
        return HTMLResponse("<p>Missing templates/index.html</p>", status_code=500)
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    _ensure_globals()

    vec = np.asarray(req.vector, dtype=np.float32)
    if vec.shape != (63,):
        return PredictResponse(letter="", confidence=0.0)

    with _model_lock:
        clf = _clf
        classes = _classes

    proba = clf.predict_proba([vec])[0]
    top = int(np.argmax(proba))
    confidence = float(proba[top])
    if confidence >= CONFIDENCE_THRESH:
        letter = str(classes[top])
    else:
        letter = ""

    return PredictResponse(letter=letter, confidence=confidence)


if __name__ == "__main__":
    import uvicorn

    try:
        _ensure_globals()
    except RuntimeError as exc:
        print("ERROR:", exc)
        sys.exit(1)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
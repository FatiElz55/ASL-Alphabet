import asyncio
import base64
import os
import sys
import threading
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from hand_detector import HandDetector

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
HOLD_FRAMES_NEEDED = 25
CONFIDENCE_THRESH = 0.40
# Lower quality = less bandwidth. This does not affect recognition.
PREVIEW_JPEG_QUALITY = 70

MODEL_NAME = "SVM"
MODEL_PATH = "model_svm.pkl"

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"

_model_lock = threading.Lock()
_clf = None
_classes: list | None = None
_active_model_name: str | None = None

_detector: HandDetector | None = None
_detect_lock = threading.Lock()


def load_model(path: str):
    payload = joblib.load(path)
    return payload["model"], list(payload["classes"])


def first_available_model() -> tuple[str | None, str | None]:
    if os.path.exists(MODEL_PATH):
        return MODEL_NAME, MODEL_PATH
    return None, None


def _ensure_globals():
    global _clf, _classes, _active_model_name, _detector
    if _clf is not None:
        return
    init_name, init_path = first_available_model()
    if init_name is None:
        raise RuntimeError(
            "No model file found. Run train_model_svm.py to create model_svm.pkl."
        )
    _clf, _classes = load_model(init_path)
    _active_model_name = init_name
    _detector = HandDetector(
        max_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )


def encode_preview_jpeg(bgr: np.ndarray) -> str:
    small = cv2.resize(bgr, (620, 465), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(
        ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY]
    )
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


class SessionState:
    def __init__(self):
        self.text = ""
        self.current_letter = ""
        self.hold_count = 0


def process_frame_bgr(frame_bgr: np.ndarray, session: SessionState) -> dict:
    """Run detection + classifier; update session hold/text; return UI fields."""
    _ensure_globals()
    frame = cv2.flip(frame_bgr, 1)
    with _detect_lock:
        vector, annotated = _detector.get_landmarks(frame)

    letter = ""
    confidence = 0.0

    if vector is not None:
        with _model_lock:
            clf = _clf
            classes = _classes
        proba = clf.predict_proba([vector])[0]
        top = int(np.argmax(proba))
        confidence = float(proba[top])
        if confidence >= CONFIDENCE_THRESH:
            letter = str(classes[top])

    if letter and letter == session.current_letter:
        session.hold_count += 1
    else:
        session.current_letter = letter
        session.hold_count = 0

    if session.hold_count >= HOLD_FRAMES_NEEDED:
        if letter and letter.lower() != "nothing":
            _append_letter(session, letter)
        session.hold_count = 0

    display_letter = letter if letter else "?"
    cv2.putText(
        annotated,
        display_letter,
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (124, 106, 247),
        3,
        cv2.LINE_AA,
    )
    conf_txt = f"{confidence * 100:.0f}%" if letter else ""
    cv2.putText(
        annotated,
        conf_txt,
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    preview_b64 = encode_preview_jpeg(annotated)
    return {
        "type": "state",
        "letter": letter,
        "confidence": confidence,
        "hold_count": session.hold_count,
        "hold_needed": HOLD_FRAMES_NEEDED,
        "text": session.text,
        "preview": preview_b64,
        "active_model": _active_model_name,
    }


def _append_letter(session: SessionState, letter: str):
    if letter.lower() == "space":
        session.text += " "
    elif letter.lower() == "del":
        session.text = session.text[:-1]
    else:
        session.text += letter


app = FastAPI()


@app.get("/")
async def index():
    if not INDEX_PATH.is_file():
        return HTMLResponse("<p>Missing templates/index.html</p>", status_code=500)
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws_sign(websocket: WebSocket):
    await websocket.accept()
    try:
        _ensure_globals()
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    session = SessionState()

    await websocket.send_json(
        {
            "type": "hello",
            "active_model": _active_model_name,
        }
    )

    loop = asyncio.get_running_loop()

    def decode_jpeg(b64: str) -> np.ndarray | None:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "frame":
                b64 = msg.get("image") or ""
                frame = await loop.run_in_executor(None, decode_jpeg, b64)
                if frame is None:
                    await websocket.send_json(
                        {
                            "type": "state",
                            "letter": "",
                            "confidence": 0.0,
                            "hold_count": session.hold_count,
                            "hold_needed": HOLD_FRAMES_NEEDED,
                            "text": session.text,
                            "preview": "",
                            "active_model": _active_model_name,
                        }
                    )
                    continue

                if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                def run_detection():
                    return process_frame_bgr(frame, session)

                payload = await loop.run_in_executor(None, run_detection)
                await websocket.send_json(payload)

            elif mtype == "action":
                act = msg.get("action")
                if act == "space":
                    session.text += " "
                elif act == "backspace":
                    session.text = session.text[:-1]
                elif act == "clear":
                    session.text = ""
                await websocket.send_json(
                    {
                        "type": "state",
                        "letter": session.current_letter,
                        "confidence": 0.0,
                        "hold_count": session.hold_count,
                        "hold_needed": HOLD_FRAMES_NEEDED,
                        "text": session.text,
                        "preview": "",
                        "active_model": _active_model_name,
                    }
                )

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    try:
        _ensure_globals()
    except RuntimeError as exc:
        print("ERROR:", exc)
        sys.exit(1)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

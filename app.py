"""
app.py
======
Real-time ASL alphabet sign language detector with a Tkinter text-writing
interface.

Requirements:
  model.pkl must exist (run train_model.py first).

Controls:
  - Hold a sign steady for ~1 second to add the letter to the text.
  - Use the on-screen buttons for Space / Backspace / Clear / Copy.

Run:
    python app.py
"""

import sys
import tkinter as tk
from tkinter import font as tkfont
import threading
import time

import cv2
import numpy as np
import joblib
from PIL import Image, ImageTk

from hand_detector import HandDetector

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH          = "model.pkl"
CAMERA_INDEX        = 0
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
HOLD_FRAMES_NEEDED  = 25    # frames a letter must be stable before it's added
CONFIDENCE_THRESH   = 0.40  # minimum prediction confidence to show a letter
PREVIEW_W           = 620   # displayed webcam width (px)
PREVIEW_H           = 465   # displayed webcam height (px)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
try:
    payload = joblib.load(MODEL_PATH)
    clf     = payload["model"]
    classes = payload["classes"]
except FileNotFoundError:
    print(f"ERROR: '{MODEL_PATH}' not found. Run train_model.py first.")
    sys.exit(1)


# ─────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────
class ASLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASL Sign Language — Text Writer")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")

        # State
        self._text          = ""
        self._current_letter = ""
        self._hold_count    = 0
        self._confidence    = 0.0
        self._running       = True

        self._build_ui()
        self._detector = HandDetector(
            max_hands=1,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self._cap = self._open_camera()

        # Camera thread
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI layout ──────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = 12
        BG  = "#1e1e2e"
        PANEL_BG = "#2a2a3e"
        ACCENT   = "#7c6af7"
        TEXT_FG  = "#cdd6f4"

        # ── Left panel: webcam ──────────────────────────────────────
        left = tk.Frame(self, bg=BG, padx=PAD, pady=PAD)
        left.grid(row=0, column=0, sticky="nsew")

        tk.Label(
            left, text="Webcam", bg=BG, fg=TEXT_FG,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        self._canvas = tk.Canvas(
            left, width=PREVIEW_W, height=PREVIEW_H,
            bg="#000000", highlightthickness=0,
        )
        self._canvas.pack(pady=(6, 0))

        # Progress bar (hold indicator)
        prog_frame = tk.Frame(left, bg=BG)
        prog_frame.pack(fill="x", pady=(8, 0))

        tk.Label(
            prog_frame, text="Hold:", bg=BG, fg=TEXT_FG,
            font=("Segoe UI", 9),
        ).pack(side="left")

        self._prog_canvas = tk.Canvas(
            prog_frame, height=18, bg="#3a3a5e",
            highlightthickness=1, highlightbackground="#555",
        )
        self._prog_canvas.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._prog_bar = self._prog_canvas.create_rectangle(
            0, 0, 0, 18, fill=ACCENT, outline=""
        )

        # ── Right panel: detection + text ───────────────────────────
        right = tk.Frame(self, bg=BG, padx=PAD, pady=PAD)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, PAD))

        # ── Detected letter ─────────────────────────────────────────
        det_frame = tk.Frame(right, bg=PANEL_BG, padx=16, pady=16)
        det_frame.pack(fill="x")

        tk.Label(
            det_frame, text="Detected", bg=PANEL_BG, fg="#888",
            font=("Segoe UI", 9),
        ).pack()

        self._letter_var = tk.StringVar(value="—")
        tk.Label(
            det_frame, textvariable=self._letter_var, bg=PANEL_BG, fg=ACCENT,
            font=("Segoe UI", 72, "bold"),
        ).pack()

        self._conf_var = tk.StringVar(value="")
        tk.Label(
            det_frame, textvariable=self._conf_var, bg=PANEL_BG, fg="#888",
            font=("Segoe UI", 10),
        ).pack()

        # ── Written text ─────────────────────────────────────────────
        tk.Label(
            right, text="Written text", bg=BG, fg=TEXT_FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(18, 4))

        text_frame = tk.Frame(right, bg=PANEL_BG, padx=8, pady=8)
        text_frame.pack(fill="both", expand=True)

        mono = tkfont.Font(family="Consolas", size=16)
        self._textbox = tk.Text(
            text_frame, wrap="word", height=8, width=28,
            font=mono, bg="#13131f", fg=TEXT_FG,
            insertbackground=TEXT_FG, relief="flat",
            bd=0, state="disabled",
        )
        self._textbox.pack(fill="both", expand=True)

        # ── Buttons ───────────────────────────────────────────────────
        btn_cfg = dict(
            bg=PANEL_BG, fg=TEXT_FG,
            activebackground=ACCENT, activeforeground="#fff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=12, pady=8, cursor="hand2",
        )

        btn_frame = tk.Frame(right, bg=BG)
        btn_frame.pack(fill="x", pady=(10, 0))

        tk.Button(
            btn_frame, text="SPACE", command=self._add_space, **btn_cfg,
        ).grid(row=0, column=0, padx=3, pady=3, sticky="ew")

        tk.Button(
            btn_frame, text="⌫  DEL", command=self._backspace, **btn_cfg,
        ).grid(row=0, column=1, padx=3, pady=3, sticky="ew")

        tk.Button(
            btn_frame, text="CLEAR", command=self._clear,
            **{**btn_cfg, "fg": "#f38ba8"},
        ).grid(row=1, column=0, padx=3, pady=3, sticky="ew")

        tk.Button(
            btn_frame, text="COPY", command=self._copy, **btn_cfg,
        ).grid(row=1, column=1, padx=3, pady=3, sticky="ew")

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # Keyboard shortcuts: Space = space, Delete/BackSpace = delete last char
        self.bind("<space>",     lambda e: self._add_space())
        self.bind("<Delete>",    lambda e: self._backspace())
        self.bind("<BackSpace>", lambda e: self._backspace())

    # ── Camera helpers ──────────────────────────────────────────────────────

    def _open_camera(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        for idx in (CAMERA_INDEX, 1, 2):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if idx != CAMERA_INDEX:
                    print(f"Using camera index {idx}")
                return cap
        raise RuntimeError(
            "Could not open webcam.\n"
            "Windows: Settings → Privacy & security → Camera → allow desktop apps."
        )

    # ── Camera loop (runs in background thread) ─────────────────────────────

    def _camera_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)  # mirror for natural feel
            vector, annotated = self._detector.get_landmarks(frame)

            letter = ""
            confidence = 0.0

            if vector is not None:
                proba  = clf.predict_proba([vector])[0]
                top    = int(np.argmax(proba))
                confidence = float(proba[top])
                if confidence >= CONFIDENCE_THRESH:
                    letter = str(classes[top])

            # Hold logic
            if letter and letter == self._current_letter:
                self._hold_count += 1
            else:
                self._current_letter = letter
                self._hold_count     = 0

            if self._hold_count >= HOLD_FRAMES_NEEDED:
                if letter and letter.lower() != "nothing":
                    self.after(0, self._append_letter, letter)
                self._hold_count = 0

            self._confidence = confidence

            # Overlay: letter + confidence on frame
            display_letter = letter if letter else "?"
            cv2.putText(
                annotated, display_letter,
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (124, 106, 247), 3, cv2.LINE_AA,
            )
            conf_txt = f"{confidence * 100:.0f}%" if letter else ""
            cv2.putText(
                annotated, conf_txt,
                (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA,
            )

            # Push frame to UI thread
            rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb).resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)
            imtk = ImageTk.PhotoImage(pil)
            self.after(0, self._update_frame, imtk, letter, confidence)

    # ── UI update helpers (called on main thread via after()) ────────────────

    def _update_frame(self, imtk, letter: str, confidence: float):
        self._canvas.imtk = imtk   # prevent garbage collection
        self._canvas.create_image(0, 0, anchor="nw", image=imtk)

        # Letter display
        self._letter_var.set(letter if letter else "—")
        self._conf_var.set(f"{confidence * 100:.0f} %" if letter else "")

        # Progress bar
        prog_w = self._prog_canvas.winfo_width()
        if prog_w > 1:
            fill = int(prog_w * min(self._hold_count / HOLD_FRAMES_NEEDED, 1.0))
            self._prog_canvas.coords(self._prog_bar, 0, 0, fill, 18)

    def _append_letter(self, letter: str):
        if letter.lower() == "space":
            self._text += " "
        elif letter.lower() == "del":
            self._text = self._text[:-1]
        else:
            self._text += letter
        self._refresh_textbox()

    def _refresh_textbox(self):
        self._textbox.config(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("end", self._text)
        self._textbox.see("end")
        self._textbox.config(state="disabled")

    # ── Button callbacks ────────────────────────────────────────────────────

    def _add_space(self):
        self._text += " "
        self._refresh_textbox()

    def _backspace(self):
        self._text = self._text[:-1]
        self._refresh_textbox()

    def _clear(self):
        self._text = ""
        self._refresh_textbox()

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._text)

    # ── Shutdown ────────────────────────────────────────────────────────────

    def _on_close(self):
        self._running = False
        time.sleep(0.1)
        self._cap.release()
        self.destroy()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ASLApp()
    app.mainloop()

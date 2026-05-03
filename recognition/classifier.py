# ─────────────────────────────────────────────
# recognition/classifier.py
# Real-time Gesture Classifier
# ─────────────────────────────────────────────

import os
import sys
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURE_LABELS, GESTURES, NUM_CLASSES,
    BEST_MODEL_PATH, MODEL_PATH,
    CONFIDENCE_THRESHOLD, PREDICTION_SMOOTHING,
    FRAMES_PER_SAMPLE,
)

import tensorflow as tf


class Prediction:
    """
    Result of one classification call.

    Attributes:
        label      : str   — e.g. "FURAHI"
        swahili    : str   — e.g. "Furahi"
        english    : str   — e.g. "Happy"
        confidence : float — softmax probability [0, 1]
        top_n      : list  — [(label, confidence), ...]
        is_valid   : bool  — confidence >= CONFIDENCE_THRESHOLD
    """

    def __init__(self, label: str, confidence: float, top_n: list):
        self.label      = label
        self.swahili    = GESTURES[label][0]
        self.english    = GESTURES[label][1]
        self.confidence = confidence
        self.top_n      = top_n
        self.is_valid   = confidence >= CONFIDENCE_THRESHOLD

    def __repr__(self):
        return (
            f"Prediction(label={self.label!r}, "
            f"confidence={self.confidence:.2%}, "
            f"valid={self.is_valid})"
        )


class GestureClassifier:
    """
    Wraps the trained CNN for real-time prediction.

    Usage:
        clf = GestureClassifier()
        pred = clf.predict(sequence)   # (60, 64, 64, 3) float32
        if pred.is_valid:
            print(pred.swahili, pred.english, pred.confidence)
    """

    def __init__(self, weights_path: str = None):
        if weights_path is None:
            if os.path.exists(BEST_MODEL_PATH):
                weights_path = BEST_MODEL_PATH
            elif os.path.exists(MODEL_PATH):
                weights_path = MODEL_PATH
            else:
                raise FileNotFoundError(
                    "No trained model found.\n"
                    "Run:  python model/train.py  first.\n"
                    f"Expected: {BEST_MODEL_PATH}"
                )

        print(f"[INFO] Loading model: {weights_path}")
        self._model = tf.keras.models.load_model(weights_path)
        print("[INFO] Model ready.")

        self._smooth_buf: deque = deque(maxlen=PREDICTION_SMOOTHING)

    def predict(self, sequence: np.ndarray, top_n: int = 3) -> Prediction:
        """
        Classify a gesture from a (60, 64, 64, 3) float32 sequence.
        Applies rolling temporal smoothing to reduce flickering.
        """
        if sequence.shape[0] != FRAMES_PER_SAMPLE:
            raise ValueError(
                f"Expected {FRAMES_PER_SAMPLE} frames, got {sequence.shape[0]}"
            )

        batch  = np.expand_dims(sequence, axis=0)   # (1, 60, 64, 64, 3)
        probs  = self._model.predict(batch, verbose=0)[0]

        self._smooth_buf.append(probs)
        smoothed = np.mean(self._smooth_buf, axis=0)

        best_idx   = int(np.argmax(smoothed))
        best_conf  = float(smoothed[best_idx])
        best_label = GESTURE_LABELS[best_idx]

        top_indices = np.argsort(smoothed)[::-1][:top_n]
        top_results = [
            (GESTURE_LABELS[i], float(smoothed[i]))
            for i in top_indices
        ]

        return Prediction(label=best_label, confidence=best_conf,
                          top_n=top_results)

    def reset_smoothing(self):
        self._smooth_buf.clear()

    def summary(self):
        self._model.summary()

    @property
    def input_shape(self):
        return self._model.input_shape[1:]


if __name__ == "__main__":
    import cv2
    from recognition.detector import HandDetector, FrameBuffer
    from config import CAMERA_INDEX

    try:
        clf = GestureClassifier()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    buf = FrameBuffer(capacity=FRAMES_PER_SAMPLE)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    with HandDetector() as detector:
        print("Running — press Q to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_data, annotated = detector.process(frame)
            h, w  = annotated.shape[:2]

            if frame_data is not None:
                buf.add(frame_data)
                if buf.is_full():
                    pred = clf.predict(buf.get())
                    if pred.is_valid:
                        cv2.putText(
                            annotated,
                            f"{pred.label}  ({pred.english})",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 212, 170), 2, cv2.LINE_AA,
                        )
                        cv2.putText(
                            annotated,
                            f"Confidence: {pred.confidence:.1%}",
                            (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (180, 180, 180), 1, cv2.LINE_AA,
                        )
                        for i, (lbl, c) in enumerate(pred.top_n):
                            cv2.putText(
                                annotated,
                                f"  {i+1}. {lbl}  {c:.1%}",
                                (12, 100 + i * 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (160, 160, 160), 1, cv2.LINE_AA,
                            )
                    else:
                        cv2.putText(
                            annotated, "Waiting for gesture...",
                            (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (120, 120, 120), 1, cv2.LINE_AA,
                        )

            cv2.putText(
                annotated,
                f"Buffer: {len(buf)}/{FRAMES_PER_SAMPLE}",
                (w - 210, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA,
            )
            cv2.imshow("SLRS — Classifier", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

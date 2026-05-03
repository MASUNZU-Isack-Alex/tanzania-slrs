# ─────────────────────────────────────────────
# recognition/detector.py
# Hand Detection + Frame Buffer
#
# Design:
#   • MediaPipe Hands detects whether a hand is present in each frame.
#   • When a hand IS present, the FULL resized frame (not just a hand crop)
#     is returned and added to the buffer.  The full frame includes the
#     signer's face so the CNN can learn facial expression cues.
#   • When NO hand is detected, None is returned — the buffer is not updated,
#     preventing idle background frames from polluting the gesture sequence.
#   • FrameBuffer uses collections.deque(maxlen) for O(1) append/eviction.
# ─────────────────────────────────────────────

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MP_MAX_HANDS, MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    ROI_PADDING, IMG_HEIGHT, IMG_WIDTH,
)


class HandDetector:
    """
    Wraps MediaPipe Hands.

    Returns the full preprocessed frame (not a hand crop) so the model
    sees both hand shape and facial expression simultaneously.

    Usage:
        with HandDetector() as detector:
            frame_data, annotated = detector.process(bgr_frame)
            # frame_data : (64, 64, 3) float32 RGB — or None if no hand
            # annotated  : BGR frame with landmarks + bounding box
    """

    _mp_hands   = mp.solutions.hands
    _mp_drawing = mp.solutions.drawing_utils

    _LANDMARK_SPEC = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 212, 170), thickness=2, circle_radius=3
    )
    _CONNECTION_SPEC = mp.solutions.drawing_utils.DrawingSpec(
        color=(59, 130, 246), thickness=2
    )

    def __init__(self):
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )

    def process(self, frame: np.ndarray):
        """
        Process one BGR frame.

        Returns:
            frame_data  (np.ndarray | None)
                (IMG_H, IMG_W, 3) float32 RGB [0,1] — full frame.
                None when no hand detected.
            annotated   (np.ndarray)
                BGR frame with landmarks and bounding box drawn.
        """
        h, w   = frame.shape[:2]
        annotated = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            self._draw_no_hand(annotated)
            # We MUST return the preprocessed frame even if no hand is detected.
            # Otherwise, dropped frames will squash the gesture temporally, breaking
            # the Conv1D model which expects real-time (30fps) velocity exactly as
            # recorded during training.
            return self._preprocess_full_frame(frame), annotated

        # Draw all detected hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._LANDMARK_SPEC,
                self._CONNECTION_SPEC,
            )

            # Draw bounding box (visual only — not used for crop)
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x_min = max(int(min(xs)) - ROI_PADDING, 0)
            y_min = max(int(min(ys)) - ROI_PADDING, 0)
            x_max = min(int(max(xs)) + ROI_PADDING, w)
            y_max = min(int(max(ys)) + ROI_PADDING, h)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                          (0, 212, 170), 2)

        # Return full frame — face + hands both visible to the model
        frame_data = self._preprocess_full_frame(frame)
        return frame_data, annotated

    def _preprocess_full_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize full BGR frame → RGB float32 [0,1] with light blur.
        Must match exactly what preprocess.py does so training data
        and live inference data look identical to the model.
        """
        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        floated = rgb.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(floated, (3, 3), 0)
        return blurred

    @staticmethod
    def _draw_no_hand(frame: np.ndarray):
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "No hand detected",
            (w // 2 - 90, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (80, 80, 80), 1, cv2.LINE_AA,
        )

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FrameBuffer:
    """
    Rolling buffer of the last N preprocessed frames.
    Uses deque(maxlen=N) — O(1) append, automatic oldest-frame eviction.
    """

    def __init__(self, capacity: int):
        self._cap    = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, frame: np.ndarray):
        self._buffer.append(frame)

    def is_full(self) -> bool:
        return len(self._buffer) == self._cap

    def get(self) -> np.ndarray:
        """Return (N, H, W, C) float32."""
        return np.array(self._buffer, dtype=np.float32)

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)


if __name__ == "__main__":
    from config import CAMERA_INDEX, FRAMES_PER_SAMPLE

    cap = cv2.VideoCapture(CAMERA_INDEX)
    buf = FrameBuffer(capacity=FRAMES_PER_SAMPLE)

    with HandDetector() as detector:
        print("HandDetector test — press Q to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_data, annotated = detector.process(frame)

            if frame_data is not None:
                buf.add(frame_data)
                preview = cv2.resize(
                    cv2.cvtColor(
                        (frame_data * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    ),
                    (192, 192),
                )
                cv2.imshow("Model input (full frame 64x64)", preview)

            cv2.putText(
                annotated,
                f"Buffer: {len(buf)}/{FRAMES_PER_SAMPLE}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 212, 170), 2,
            )
            cv2.imshow("Detector", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

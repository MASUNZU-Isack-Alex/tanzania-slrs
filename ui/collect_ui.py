# ─────────────────────────────────────────────
# ui/collect_ui.py
# SLRS — Data Collection Interface
#
# HOW TO USE:
#   streamlit run ui/collect_ui.py
#
# Positioning rules for clean training data:
#   • Centre yourself in the frame — both hands AND face visible at all times
#   • Plain background behind you
#   • Good, steady lighting — no harsh shadows on your face
#   • Perform each sign at a natural conversational speed
#   • Keep conditions consistent across all sessions
# ─────────────────────────────────────────────

import os
import sys
import time
import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURE_LABELS, GESTURES, RAW_DIR,
    SAMPLES_PER_GESTURE, FRAMES_PER_SAMPLE,
    COLLECTION_COUNTDOWN, IMG_HEIGHT, IMG_WIDTH,
    CAMERA_INDEX,
)

st.set_page_config(
    page_title="SLRS - Data Collection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session():
    defaults = {"selected_gesture": GESTURE_LABELS[0]}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


def count_samples(label: str) -> int:
    path = os.path.join(RAW_DIR, label)
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(".npy")])


def make_gesture_dirs():
    for label in GESTURE_LABELS:
        os.makedirs(os.path.join(RAW_DIR, label), exist_ok=True)


def draw_overlay(frame, gesture_label, sample_idx, state,
                 countdown=None, frame_count=0):
    h, w = frame.shape[:2]
    swahili, english = GESTURES[gesture_label]

    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"{swahili}  ({english})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(frame, f"Sample #{sample_idx + 1}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)

    if state == "COUNTDOWN":
        cv2.putText(frame, f"Get ready... {countdown}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    elif state == "RECORDING":
        cv2.putText(
            frame,
            f"RECORDING  [{frame_count}/{FRAMES_PER_SAMPLE}]",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        # Red border during recording
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 3)
        # Progress bar
        progress_w = int((frame_count / FRAMES_PER_SAMPLE) * w)
        cv2.rectangle(frame, (0, h - 6), (progress_w, h), (0, 100, 255), -1)
    elif state == "DONE":
        cv2.putText(frame, "Saved!", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return frame


def record_sample(gesture_label, sample_idx, cam_placeholder, status_placeholder):
    """
    Record one sample. Saves full BGR uint8 frames — preprocess.py
    handles normalisation later. Full frame preserves both hands and face.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        status_placeholder.error(
            f"Cannot open camera index {CAMERA_INDEX}. "
            "Check DroidCam is connected and running."
        )
        return False
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        # Countdown phase
        for count in range(COLLECTION_COUNTDOWN, 0, -1):
            status_placeholder.warning(f"⏳ Starting in **{count}**… Get ready!")
            deadline = time.time() + 1.0
            while time.time() < deadline:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Camera read failed.")
                    return False
                frame   = cv2.flip(frame, 1)
                display = draw_overlay(frame.copy(), gesture_label,
                                       sample_idx, "COUNTDOWN", countdown=count)
                # Render countdown every frame or just read fast
                cam_placeholder.image(
                    cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True,
                )

        # Recording phase
        status_placeholder.error("🔴 **RECORDING** — Perform the gesture now!")
        frames = []

        while len(frames) < FRAMES_PER_SAMPLE:
            ret, frame = cap.read()
            if not ret:
                break
            frame   = cv2.flip(frame, 1)
            display = draw_overlay(frame.copy(), gesture_label, sample_idx,
                                   "RECORDING", frame_count=len(frames) + 1)
            cam_placeholder.image(
                cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True,
            )
            resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frames.append(resized)

        # Pad if camera dropped frames
        while len(frames) < FRAMES_PER_SAMPLE:
            frames.append(
                frames[-1] if frames
                else np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            )

        arr  = np.array(frames, dtype=np.uint8)
        path = os.path.join(RAW_DIR, gesture_label, f"{sample_idx:04d}.npy")
        os.makedirs(os.path.join(RAW_DIR, gesture_label), exist_ok=True)
        np.save(path, arr)

        # Show done frame
        ret, frame = cap.read()
        if ret:
            frame   = cv2.flip(frame, 1)
            display = draw_overlay(frame, gesture_label, sample_idx, "DONE")
            cam_placeholder.image(
                cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True,
            )

        status_placeholder.success(
            f"✅ Sample #{sample_idx + 1} saved — "
            f"{FRAMES_PER_SAMPLE} frames @ {IMG_WIDTH}×{IMG_HEIGHT}"
        )
        time.sleep(0.8)
        return True

    finally:
        cap.release()


def main():
    make_gesture_dirs()

    st.title("📹 Data Collection — SLRS")
    st.caption(
        "Record gesture samples for training.  "
        "**Keep your face AND both hands clearly in frame for every sample.**"
    )

    with st.sidebar:
        st.markdown("### Select Gesture")
        gesture = st.radio(
            "Gesture to record:",
            GESTURE_LABELS,
            format_func=lambda x: f"{GESTURES[x][0]} — {GESTURES[x][1]}",
            key="gesture_radio",
        )
        st.session_state.selected_gesture = gesture

        st.divider()
        st.markdown("### Progress")
        total_collected = 0
        total_needed    = SAMPLES_PER_GESTURE * len(GESTURE_LABELS)

        for label in GESTURE_LABELS:
            count  = count_samples(label)
            total_collected += count
            icon   = "✅" if count >= SAMPLES_PER_GESTURE else "⬜"
            st.markdown(
                f"{icon} **{GESTURES[label][0]}** — {count}/{SAMPLES_PER_GESTURE}"
            )

        st.divider()
        st.metric("Total", f"{total_collected} / {total_needed}")
        if total_needed > 0:
            st.progress(total_collected / total_needed)

    gesture  = st.session_state.selected_gesture
    sw, en   = GESTURES[gesture]
    current  = count_samples(gesture)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Gesture",  sw)
    with c2: st.metric("Recorded", f"{current} / {SAMPLES_PER_GESTURE}")
    with c3: st.metric("Remaining", max(0, SAMPLES_PER_GESTURE - current))

    st.divider()

    cam_col, info_col = st.columns([3, 1])

    with cam_col:
        cam_placeholder = st.empty()
        cam_placeholder.info(
            f"Ready to record **{sw}** ({en}).\n\n"
            "Click **Record Sample** when you are in position."
        )

    with info_col:
        st.markdown("**Settings**")
        st.text(f"Frames : {FRAMES_PER_SAMPLE}")
        st.text(f"Duration : ~{FRAMES_PER_SAMPLE/30:.1f}s")
        st.text(f"Countdown : {COLLECTION_COUNTDOWN}s")
        st.text(f"Size : {IMG_WIDTH}×{IMG_HEIGHT}")
        st.text(f"Camera : index {CAMERA_INDEX}")
        st.divider()
        st.markdown("**Tips**")
        st.markdown(
            "- Face **and** hands in frame\n"
            "- Plain background\n"
            "- Steady, even lighting\n"
            "- Natural speed\n"
            "- Same conditions each session"
        )
        if current >= SAMPLES_PER_GESTURE:
            st.success("All samples collected!")

    status_placeholder = st.empty()

    if current >= SAMPLES_PER_GESTURE:
        st.warning(
            f"All {SAMPLES_PER_GESTURE} samples done for **{sw}**. "
            "Select another gesture."
        )
    else:
        btn_col, cap_col = st.columns([2, 1])
        with btn_col:
            if st.button("🔴 Record Sample",
                         use_container_width=True, type="primary"):
                success = record_sample(
                    gesture, current, cam_placeholder, status_placeholder
                )
                if success:
                    st.rerun()
        with cap_col:
            st.caption(f"Will record sample #{current + 1}")

    with st.expander("ℹ️ Why does the model need your face?", expanded=False):
        st.markdown("""
Tanzanian Sign Language uses **facial expressions as part of the grammar**,
not just decoration.  Signs like **Furahi** (happy) and **Huzuni** (sad)
from Chapter 8 of TSL Book One are distinguished by both the hand movement
AND the signer's facial expression.

By capturing the full frame — face and hands together — you give the model
all the information a human interpreter would use.  Cropping to just the
hand throws away half the signal.

**Practical checklist before each recording session:**
1. DroidCam connected and showing a clear, sharp feed
2. Your face is centred and fully visible
3. Both hands have room to move without leaving frame
4. Lighting is even — no strong shadows on face or hands
5. Background is plain and consistent
        """)


if __name__ == "__main__":
    main()

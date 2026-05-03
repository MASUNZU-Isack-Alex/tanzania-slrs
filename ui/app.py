# ─────────────────────────────────────────────
# ui/app.py
# SLRS — Main Recognition Application
#
# HOW TO USE:
#   streamlit run ui/app.py
# ─────────────────────────────────────────────

import os
import sys
import time
import threading
import cv2
import numpy as np
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURE_LABELS, GESTURES, CAMERA_INDEX,
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    FRAMES_PER_SAMPLE, CONFIDENCE_THRESHOLD,
    APP_TITLE, APP_SUBTITLE, BEST_MODEL_PATH, MODEL_PATH,
)
from recognition.detector import HandDetector, FrameBuffer
from recognition.classifier import GestureClassifier

st.set_page_config(
    page_title="SLRS - Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title    { font-size:24px; font-weight:bold; margin-bottom:2px; }
.sub-title     { font-size:13px; color:#888; margin-bottom:20px; }
.result-label  { font-size:36px; font-weight:bold; color:#1a73e8;
                 text-align:center; padding:10px 0; }
.result-meaning{ font-size:14px; color:#555; text-align:center; }
.history-item  { padding:6px 0; border-bottom:1px solid #eee; font-size:13px; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "running":          False,
        "output_text":      "",
        "history":          [],
        "session_count":    0,
        "session_correct":  0,
        "last_label":       None,
        "last_added_label": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner=False)
def load_classifier():
    try:
        return GestureClassifier(), None
    except FileNotFoundError as e:
        return None, str(e)


def speak_text(text: str):
    def _speak():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()


def main():
    with st.sidebar:
        st.markdown("### SLRS")
        st.caption("Tanzanian Sign Language Recognition")
        st.divider()
        st.markdown("**Gestures (15)**")
        for label in GESTURE_LABELS:
            sw, en = GESTURES[label]
            st.markdown(f"- **{sw}** — {en}")
        st.divider()
        st.markdown("**Settings**")
        st.text(f"Confidence : {CONFIDENCE_THRESHOLD:.0%}")
        st.text(f"Frames/clip: {FRAMES_PER_SAMPLE}")
        st.text(f"Camera     : index {CAMERA_INDEX}")

    st.markdown(f'<div class="main-title">{APP_TITLE}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{APP_SUBTITLE}</div>',
                unsafe_allow_html=True)

    clf, err = load_classifier()

    if err:
        st.error("**Model not found.** Train the model first.")
        st.info(
            "Steps:\n"
            "1. `streamlit run ui/collect_ui.py`\n"
            "2. `python data/preprocess.py`\n"
            "3. `python model/train.py`\n"
            "4. Come back here."
        )
        st.stop()

    cam_col, result_col = st.columns([3, 2], gap="large")

    with cam_col:
        st.markdown("#### 📹 Camera Feed")
        cam_placeholder  = st.empty()
        c1, c2, c3       = st.columns(3)
        start_btn        = c1.button("▶ Start",  use_container_width=True)
        stop_btn         = c2.button("⏹ Stop",   use_container_width=True)
        clear_btn        = c3.button("🔄 Reset",  use_container_width=True)
        conf_placeholder = st.empty()

    with result_col:
        st.markdown("#### 🖐 Recognition Result")
        result_placeholder = st.empty()
        st.markdown("#### 📊 Top Predictions")
        pred_placeholder   = st.empty()
        st.divider()
        st.markdown("#### 📝 Output Text")
        output_placeholder = st.empty()
        s1, s2 = st.columns(2)
        speak_btn     = s1.button("🔊 Speak",     use_container_width=True)
        clear_out_btn = s2.button("✕ Clear Text", use_container_width=True)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected",       st.session_state.session_count)
    total = st.session_state.session_count
    m2.metric("High Confidence",
              f"{st.session_state.session_correct/total*100:.0f}%"
              if total > 0 else "0%")
    m3.metric("Words",
              len(st.session_state.output_text.split())
              if st.session_state.output_text else 0)
    m4.metric("Buffer", f"{FRAMES_PER_SAMPLE} frames")

    with st.expander("📋 Recognition History", expanded=False):
        if st.session_state.history:
            for item in reversed(st.session_state.history[-20:]):
                st.markdown(
                    f'<div class="history-item">'
                    f'<b>{item["time"]}</b> — '
                    f'<b style="color:#1a73e8">{item["label"]}</b> '
                    f'({item["swahili"]}) — {item["conf"]:.0%}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No gestures recognised yet.")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    if clear_btn:
        st.session_state.update({
            "history": [], "session_count": 0,
            "session_correct": 0, "last_label": None,
        })
    if speak_btn and st.session_state.output_text:
        speak_text(st.session_state.output_text)
    if clear_out_btn:
        st.session_state.output_text      = ""
        st.session_state.last_added_label = None

    def _output_box(text):
        content = text.replace("\n", "<br>") if text else "Waiting for gestures…"
        return (
            '<div style="min-height:80px;max-height:120px;overflow-y:auto;'
            'background:rgba(0,0,0,0.05);padding:12px;border-radius:8px;'
            'border:1px solid rgba(0,0,0,0.1);font-size:16px;">'
            f'{content}</div>'
        )

    output_placeholder.markdown(
        _output_box(st.session_state.output_text), unsafe_allow_html=True
    )

    if not st.session_state.running:
        cam_placeholder.info("Click **Start** to begin recognition.")
        result_placeholder.markdown(
            '<div class="result-label">—</div>'
            '<div class="result-meaning">Waiting to start…</div>',
            unsafe_allow_html=True,
        )
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        st.error(f"Cannot open camera {CAMERA_INDEX}. Check DroidCam.")
        st.session_state.running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    buf         = FrameBuffer(capacity=FRAMES_PER_SAMPLE)
    pred_result = None
    cooldown_frames = 0
    frame_counter = 0

    with HandDetector() as detector:
        for _ in range(3000):
            if not st.session_state.running:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_data, annotated = detector.process(frame)
            frame_counter += 1

            if frame_data is not None:
                buf.add(frame_data)
                if cooldown_frames > 0:
                    cooldown_frames -= 1

            if buf.is_full() and frame_counter % 3 == 0:
                pred_result = clf.predict(buf.get())

                if pred_result and pred_result.is_valid:
                    result_placeholder.markdown(
                        f'<div class="result-label">'
                        f'{pred_result.swahili.upper()}</div>'
                        f'<div class="result-meaning">'
                        f'{pred_result.english}</div>',
                        unsafe_allow_html=True,
                    )

                    top_md = ""
                    for lbl, conf in pred_result.top_n:
                        top_md += f"**{GESTURES[lbl][0]}** — {conf:.0%}\n\n"
                    pred_placeholder.markdown(top_md)

                    if pred_result.confidence >= CONFIDENCE_THRESHOLD:
                        label = pred_result.label
                        
                        if cooldown_frames == 0:
                            st.session_state.last_label    = label
                            st.session_state.session_count += 1
                            if pred_result.confidence >= 0.80:
                                st.session_state.session_correct += 1

                            if label != st.session_state.last_added_label:
                                st.session_state.output_text = (
                                    st.session_state.output_text
                                    + " " + pred_result.swahili
                                ).strip()
                                st.session_state.last_added_label = label
                                st.session_state.history.append({
                                    "time":    datetime.now().strftime("%H:%M:%S"),
                                    "label":   label,
                                    "swahili": pred_result.swahili,
                                    "english": pred_result.english,
                                    "conf":    pred_result.confidence,
                                })
                                if len(st.session_state.history) > 100:
                                    st.session_state.history.pop(0)

                            cooldown_frames = 20  # Wait before predicting same/next gesture to prevent spam

            if frame_counter % 2 == 0:
                cam_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True,
                )

                if pred_result:
                    conf_placeholder.progress(
                        float(pred_result.confidence),
                        text=f"Confidence: {pred_result.confidence:.0%}",
                    )

                # Only update the text box occasionally to save UI rendering time
                output_placeholder.markdown(
                    _output_box(st.session_state.output_text),
                    unsafe_allow_html=True,
                )

    cap.release()
    st.session_state.running = False


if __name__ == "__main__":
    main()

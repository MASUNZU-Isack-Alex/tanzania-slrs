# ─────────────────────────────────────────────
# ui/train_ui.py
# SLRS — Training Dashboard
#
# HOW TO USE:
#   streamlit run ui/train_ui.py
# ─────────────────────────────────────────────

import os
import sys
import subprocess
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURE_LABELS, NUM_CLASSES, DATASET_DIR, RAW_DIR,
    MODEL_PATH, BEST_MODEL_PATH, LOGS_DIR,
    BATCH_SIZE, EPOCHS, INPUT_SHAPE, FRAMES_PER_SAMPLE,
    SAMPLES_PER_GESTURE, BASE_DIR,
)

st.set_page_config(
    page_title="SLRS - Training",
    page_icon="🧠",
    layout="wide",
)


def count_raw():
    counts, total = {}, 0
    for label in GESTURE_LABELS:
        path = os.path.join(RAW_DIR, label)
        n    = len([f for f in os.listdir(path) if f.endswith(".npy")]) \
               if os.path.exists(path) else 0
        counts[label] = n
        total        += n
    return counts, total


def count_dataset():
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        d = os.path.join(DATASET_DIR, split)
        if os.path.exists(d):
            for ld in os.listdir(d):
                lp = os.path.join(d, ld)
                if os.path.isdir(lp):
                    counts[split] += len(
                        [f for f in os.listdir(lp) if f.endswith(".npy")]
                    )
    return counts


def main():
    st.title("🧠 Training Dashboard — SLRS")

    # ── Raw data ──────────────────────────────────────────────────────────────
    st.markdown("### 📁 Raw Data")
    raw_counts, raw_total = count_raw()
    total_needed          = SAMPLES_PER_GESTURE * len(GESTURE_LABELS)

    c1, c2, c3 = st.columns(3)
    c1.metric("Collected",  raw_total)
    c2.metric("Target",     total_needed)
    c3.metric("Complete",
              f"{sum(1 for v in raw_counts.values() if v >= SAMPLES_PER_GESTURE)}"
              f" / {len(GESTURE_LABELS)}")

    missing = [
        f"{lbl}: {raw_counts[lbl]}/{SAMPLES_PER_GESTURE}"
        for lbl in GESTURE_LABELS
        if raw_counts[lbl] < SAMPLES_PER_GESTURE
    ]
    if missing:
        st.warning("⚠  Still needed:\n\n" + "  •  ".join(missing))

    with st.expander("Per-gesture breakdown"):
        for label in GESTURE_LABELS:
            n  = raw_counts[label]
            ok = "✅" if n >= SAMPLES_PER_GESTURE else "⬜"
            st.text(f"{ok}  {label:12s}  {n:3d} / {SAMPLES_PER_GESTURE}")

    st.divider()

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Preprocessed Dataset")
    dc    = count_dataset()
    dtot  = sum(dc.values())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Train",      dc["train"])
    d2.metric("Validation", dc["val"])
    d3.metric("Test",       dc["test"])
    d4.metric("Total",      dtot)

    if dtot == 0 and raw_total > 0:
        st.warning("Raw data exists but not preprocessed yet. Run Step 1 below.")
    elif dtot == 0:
        st.info("No data yet. Collect samples first.")

    st.divider()

    # ── Actions ───────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Actions")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("**Step 1 — Preprocess**")
        if st.button("🔄 Run Preprocessing", use_container_width=True):
            if raw_total == 0:
                st.error("No raw data. Collect samples first.")
            else:
                with st.spinner("Preprocessing…"):
                    r = subprocess.run(
                        [sys.executable,
                         os.path.join(BASE_DIR, "data", "preprocess.py")],
                        capture_output=True, text=True, cwd=BASE_DIR,
                    )
                    if r.returncode == 0:
                        st.success("Done!")
                        st.code(r.stdout[-1000:])
                        st.rerun()
                    else:
                        st.error("Failed.")
                        st.code(r.stderr[-500:])

    with a2:
        st.markdown("**Step 2 — Train**")
        if st.button("🚀 Start Training",
                     use_container_width=True, type="primary"):
            if dtot == 0:
                st.error("No preprocessed data.")
            else:
                with st.spinner("Training… may take several minutes."):
                    r = subprocess.run(
                        [sys.executable,
                         os.path.join(BASE_DIR, "model", "train.py")],
                        capture_output=True, text=True, cwd=BASE_DIR,
                    )
                    if r.returncode == 0:
                        st.success("Training complete!")
                        st.code(r.stdout[-2000:])
                        st.rerun()
                    else:
                        st.error("Training failed.")
                        st.code(r.stderr[-500:])

    with a3:
        st.markdown("**Step 3 — Recognise**")
        st.info("`streamlit run ui/app.py`")

    st.divider()

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 Model")
    m1, m2 = st.columns(2)
    with m1:
        if os.path.exists(BEST_MODEL_PATH):
            sz = os.path.getsize(BEST_MODEL_PATH) / 1024 / 1024
            st.success(f"Best model found ({sz:.1f} MB)")
            st.text(BEST_MODEL_PATH)
        else:
            st.warning("No best model yet.")
    with m2:
        if os.path.exists(MODEL_PATH):
            sz = os.path.getsize(MODEL_PATH) / 1024 / 1024
            st.success(f"Latest model ({sz:.1f} MB)")
            st.text(MODEL_PATH)
        else:
            st.warning("No latest model yet.")

    st.divider()

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Training Results")
    h_img = os.path.join(LOGS_DIR, "training_history.png")
    c_img = os.path.join(LOGS_DIR, "confusion_matrix.png")

    if os.path.exists(h_img) or os.path.exists(c_img):
        p1, p2 = st.columns(2)
        with p1:
            if os.path.exists(h_img):
                st.markdown("**Training History**")
                st.image(Image.open(h_img), use_container_width=True)
        with p2:
            if os.path.exists(c_img):
                st.markdown("**Confusion Matrix**")
                st.image(Image.open(c_img), use_container_width=True)

        csv = os.path.join(LOGS_DIR, "training_log.csv")
        if os.path.exists(csv):
            import pandas as pd
            with st.expander("Training Log"):
                st.dataframe(pd.read_csv(csv), use_container_width=True)
    else:
        st.info("No training results yet.")

    with st.expander("📋 Configuration"):
        st.text(f"Batch size     : {BATCH_SIZE}  (CPU-safe)")
        st.text(f"Epochs         : {EPOCHS}")
        st.text(f"Classes        : {NUM_CLASSES}")
        st.text(f"Input shape    : {INPUT_SHAPE}")
        st.text(f"Frames/sample  : {FRAMES_PER_SAMPLE}  (~{FRAMES_PER_SAMPLE/30:.1f}s)")
        st.text(f"Samples/gesture: {SAMPLES_PER_GESTURE}")
        st.text(f"Camera index   : 1  (DroidCam)")


if __name__ == "__main__":
    main()

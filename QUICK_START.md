# SLRS — Quick Start

## Setup

```bash
cd slrs
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
 
---

## Step 1 — Collect data (2 gestures first to test)

Edit `config.py` - add gestures you want to use
```python
GESTURES = {
    "HABARI": ("Habari", "Hello / How are you"),
    "FURAHI": ("Furahi", "Happy"),
}
```

Position yourself so **face AND both hands** are visible. Start DroidCam. **jhdjsd hchjs**  

```bash
streamlit run ui/collect_ui.py
```

Record 20 samples per gesture. Each sample is 2 seconds (60 frames).

---

## Step 2 — Preprocess

```bash
python data/preprocess.py
```

---

## Step 3 — Train

```bash
python model/train.py
```

**Healthy epoch-0 check:**
- 2 gestures → val_accuracy ≈ 50 %
- 15 gestures → val_accuracy ≈ 6–10 %

If you see 100 % at epoch 0 → only one gesture has data → re-collect.

---

## Step 4 — Test recognition

```bash
streamlit run ui/app.py
```

If it works well with 2 gestures, uncomment all 15 in `config.py`,
collect the remaining 13, and repeat steps 2–4.

---

## Key settings (all in config.py)

| Setting | Value | Why |
|---------|-------|-----|
| `CAMERA_INDEX` | 1 | DroidCam (built-in webcam = 0) |
| `FRAMES_PER_SAMPLE` | 60 | 2 seconds per clip |
| `BATCH_SIZE` | 4 | Prevents OOM on CPU-only machines |
| `IMG_HEIGHT/WIDTH` | 64 | Full frame resized — face + hands |

---

## All commands

```bash
streamlit run ui/collect_ui.py   # record samples
python data/preprocess.py         # normalise + split
python model/train.py             # train model
streamlit run ui/app.py           # live recognition
streamlit run ui/train_ui.py      # training dashboard
python recognition/detector.py    # test camera + hand detection
python recognition/classifier.py  # test full pipeline
```

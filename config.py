# ─────────────────────────────────────────────
# config.py 
# ─────────────────────────────────────────────

import os

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
FRAMES_DIR      = os.path.join(DATA_DIR, "frames")
DATASET_DIR     = os.path.join(DATA_DIR, "dataset")
MODEL_DIR       = os.path.join(BASE_DIR, "model")
WEIGHTS_DIR     = os.path.join(MODEL_DIR, "saved")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

for _dir in [RAW_DIR, FRAMES_DIR, DATASET_DIR, WEIGHTS_DIR, LOGS_DIR,
             os.path.join(DATASET_DIR, "train"),
             os.path.join(DATASET_DIR, "val"),
             os.path.join(DATASET_DIR, "test")]:
    os.makedirs(_dir, exist_ok=True)

# ── Gesture Vocabulary ─────────────────────────────────────────────────────────
# 15 signs from "Lugha ya Alama ya Tanzania — Book One" (Deaf Child Worldwide)

GESTURES = {
    "HABARI":    ("Habari",    "Hello / How are you"),   # Ch.1
    "ASANTE":    ("Asante",    "Thank You"),              # Ch.1
    "TAFADHALI": ("Tafadhali", "Please"),                 # Ch.1
    "SAMAHANI":  ("Samahani",  "Sorry / Excuse me"),      # Ch.1
    "MAMA":      ("Mama",      "Mother"),                 # Ch.2
    #"BABA":      ("Baba",      "Father"),                 # Ch.2
    #"RAFIKI":    ("Rafiki",    "Friend"),                 # Ch.2
    #"KULA":      ("Kula",      "Eat"),                    # Ch.5
    #"KIMBIA":    ("Kimbia",    "Run"),                    # Ch.4
    #"NJOO":      ("Njoo",      "Come"),                   # Ch.4
    #"SIMAMA":    ("Simama",    "Stand"),                  # Ch.4
    #"MAJI":      ("Maji",      "Water"),                  # Ch.4
    #"CHAKULA":   ("Chakula",   "Food"),                   # Ch.4
    #"FURAHI":    ("Furahi",    "Happy"),                  # Ch.8 — uses face
    #"HUZUNI":    ("Huzuni",    "Sad"),                    # Ch.8 — uses face
}

GESTURE_LABELS = list(GESTURES.keys())
NUM_CLASSES    = len(GESTURE_LABELS)   # 15

# ── Camera ─────────────────────────────────────────────────────────────────────
# DroidCam is index 1 on this machine (built-in webcam is index 0)
CAMERA_INDEX   = 1
DISPLAY_WIDTH  = 640
DISPLAY_HEIGHT = 480

# ── Data Collection ────────────────────────────────────────────────────────────
SAMPLES_PER_GESTURE  = 40   # video clips per gesture
FRAMES_PER_SAMPLE    = 60   # frames per clip = 2 seconds at 30 fps
COLLECTION_FPS       = 30
COLLECTION_COUNTDOWN = 5    # seconds of countdown before recording

# ── Frame / Image ──────────────────────────────────────────────────────────────
# Full frame (not hand crop) so the model sees BOTH hands AND face.
# Face is for FURAHI, HUZUNI, SAMAHANI.
IMG_HEIGHT   = 64
IMG_WIDTH    = 64
IMG_CHANNELS = 3   # RGB

# ── Dataset Split ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── MediaPipe ──────────────────────────────────────────────────────────────────
# Used as a gate: only buffer frames when a hand is visible.
# The model still receives the full frame — hand detection is not used for crop.
MP_MAX_HANDS            = 2
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE  = 0.5
ROI_PADDING             = 20   # px — bounding box drawn on screen only

# ── Model Hyperparameters ──────────────────────────────────────────────────────
# BATCH_SIZE = 4  keeps peak RAM under ~1 GB on CPU-only machines.
# Increasing it will cause OOM (ResourceExhaustedError) if you have < 8 GB RAM.
BATCH_SIZE    = 4
EPOCHS        = 30
LEARNING_RATE = 0.001
DROPOUT_RATE  = 0.4

# Input shape: (FRAMES_PER_SAMPLE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
#              = (60, 64, 64, 3)
INPUT_SHAPE = (FRAMES_PER_SAMPLE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ── Model Saving ───────────────────────────────────────────────────────────────
MODEL_NAME      = "slrs_model.h5"
MODEL_PATH      = os.path.join(WEIGHTS_DIR, MODEL_NAME)
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.h5")

# ── Inference ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # min softmax probability to show a result
PREDICTION_SMOOTHING = 15     # rolling-average window size (increased for stability at 30fps)

# ── UI ─────────────────────────────────────────────────────────────────────────
APP_TITLE    = "SLRS — Tanzanian Sign Language Recognition"
APP_SUBTITLE = "Gesture. Interpret. Connect."

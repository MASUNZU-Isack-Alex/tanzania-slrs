# ─────────────────────────────────────────────
# model/train.py
# Model Training Script
#
# HOW TO USE:
#   python model/train.py
# ─────────────────────────────────────────────

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GESTURE_LABELS, NUM_CLASSES, DATASET_DIR,
    MODEL_PATH, BEST_MODEL_PATH, LOGS_DIR,
    BATCH_SIZE, EPOCHS, INPUT_SHAPE,
)
from model.cnn_model import build_model

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)


def load_split(split: str):
    X, y = [], []
    for class_idx, label in enumerate(GESTURE_LABELS):
        label_dir = os.path.join(DATASET_DIR, split, label)
        if not os.path.exists(label_dir):
            continue
        files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
        for fname in tqdm(files, desc=f"  {split}/{label}", leave=False):
            seq = np.load(os.path.join(label_dir, fname))
            X.append(seq)
            y.append(class_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def get_callbacks():
    return [
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(
            os.path.join(LOGS_DIR, "training_log.csv"),
            append=False,
        ),
    ]


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("SLRS — Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val",   linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val",   linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(LOGS_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [INFO] Plot saved → {path}")


def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GESTURE_LABELS,
                yticklabels=GESTURE_LABELS, ax=ax)
    ax.set_title("SLRS — Confusion Matrix (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    path = os.path.join(LOGS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [INFO] Confusion matrix saved → {path}")


def check_health(history):
    """Warn if epoch-0 val_accuracy ≈ 1.0 — indicates single-class data."""
    v0 = history.history["val_accuracy"][0]
    expected = 1.0 / NUM_CLASSES
    if v0 >= 0.99:
        print(
            f"\n  ⚠  WARNING: val_accuracy at epoch 0 = {v0:.2%}."
            "\n     This means only one gesture class has data."
            "\n     Collect samples for ALL active gestures and retrain."
        )
    else:
        print(
            f"\n  ✓  Epoch-0 val_accuracy = {v0:.2%}  "
            f"(random baseline ≈ {expected:.2%}) — looks healthy."
        )


def run():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  SLRS — Model Training                          ║")
    print("╚══════════════════════════════════════════════════╝\n")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU(s)     : {[g.name for g in gpus] if gpus else 'None (CPU)'}\n")

    print("  Loading data...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}\n")

    if len(X_train) == 0:
        print("[ERROR] No training data found.")
        print("  1. streamlit run ui/collect_ui.py")
        print("  2. python data/preprocess.py")
        print("  3. python model/train.py")
        sys.exit(1)

    print("  Building model...")
    model = build_model()
    model.summary()

    print(f"\n  Training — up to {EPOCHS} epochs, batch size {BATCH_SIZE}...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1,
    )

    check_health(history)

    model.save(MODEL_PATH)
    print(f"\n  [INFO] Final model → {MODEL_PATH}")

    print("\n── Test Evaluation ───────────────────────────────────")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=GESTURE_LABELS))

    print("  Generating plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)

    print(f"\n✓  Training complete!")
    print(f"   Best model : {BEST_MODEL_PATH}")
    print(f"   Logs       : {LOGS_DIR}")
    print("   Next       : streamlit run ui/app.py\n")


if __name__ == "__main__":
    run()

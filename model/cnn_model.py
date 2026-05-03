# ─────────────────────────────────────────────
# model/cnn_model.py
# CNN Model Architecture — SLRS Final
#
# Architecture (CPU-safe, fits in ~1 GB RAM):
#   TimeDistributed CNN  →  spatial features per frame (hand shape + face)
#   Conv1D temporal      →  short-range motion patterns across frames
#   GlobalAveragePooling →  aggregate over time
#   Dense head           →  classify 15 gesture classes
#
# Filter counts are halved vs the original design (16 / 32 / 64 instead of
# 32 / 64 / 128) so the peak intermediate tensor during backpropagation stays
# within CPU RAM at BATCH_SIZE=4 and FRAMES_PER_SAMPLE=60.
#
# The model still captures:
#   • Hand shape and position     (TimeDistributed Conv blocks)
#   • Facial expression           (full frame input — face always visible)
#   • Motion patterns             (Conv1D temporal layer, kernel=3)
# ─────────────────────────────────────────────

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, LEARNING_RATE

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers


def build_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile the SLRS CNN model.

    Input : (60, 64, 64, 3)  — 60 full RGB frames, 64×64 pixels
    Output: (15,)             — softmax probabilities over 15 gesture classes

    Full frame input is critical: the model sees both the signer's hands
    AND their face in every frame, allowing it to learn facial expression
    cues alongside hand configuration — essential for signs like FURAHI
    (happy) and HUZUNI (sad) from TSL Book One Chapter 8.
    """

    inputs = layers.Input(shape=input_shape, name="frame_sequence")

    # ── Block 1 — 16 filters ─────────────────────────────────────────────────
    x = layers.TimeDistributed(
        layers.Conv2D(16, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4)),
        name="td_conv1"
    )(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="td_bn1")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)), name="td_pool1")(x)
    # 64×64 → 32×32

    # ── Block 2 — 32 filters ─────────────────────────────────────────────────
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4)),
        name="td_conv2"
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="td_bn2")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)), name="td_pool2")(x)
    # 32×32 → 16×16

    # ── Block 3 — 64 filters ─────────────────────────────────────────────────
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4)),
        name="td_conv3"
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name="td_bn3")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)), name="td_pool3")(x)
    # 16×16 → 8×8

    # ── Flatten per-frame spatial features ───────────────────────────────────
    # Shape: (batch, 60, 8×8×64) = (batch, 60, 4096)
    x = layers.TimeDistributed(layers.Flatten(), name="td_flatten")(x)

    # ── Conv1D temporal layer ─────────────────────────────────────────────────
    # Slides a 3-frame window to learn motion patterns:
    # beginning / middle / end of each sign movement.
    # Helps distinguish motion verbs: KIMBIA, NJOO, SIMAMA, KULA.
    x = layers.Conv1D(
        64, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="temporal_conv"
    )(x)
    x = layers.BatchNormalization(name="temporal_bn")(x)

    # ── Aggregate across time ─────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)

    # ── Dense classification head ─────────────────────────────────────────────
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout2")(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SLRS_CNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_model(weights_path: str) -> tf.keras.Model:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
    model = tf.keras.models.load_model(weights_path)
    print(f"[INFO] Model loaded: {weights_path}")
    return model


if __name__ == "__main__":
    import numpy as np
    model = build_model()
    model.summary()
    dummy = np.random.rand(2, *INPUT_SHAPE).astype(np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"\nInput  : {dummy.shape}")
    print(f"Output : {out.shape}  (should be (2, {NUM_CLASSES}))")
    print(f"Sum    : {out[0].sum():.4f}  (should be 1.0)")

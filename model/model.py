#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Sign Language Detection — Model Definition
#
#  Architecture: Stacked LSTM → Dense → Softmax
#  Input : (batch, SEQUENCE_LENGTH, KEYPOINT_SIZE)
#  Output: (batch, num_classes)
# ─────────────────────────────────────────────

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (SEQUENCE_LENGTH, KEYPOINT_SIZE, SIGNS,
                    LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE,
                    LEARNING_RATE, MODEL_PATH, LOG_DIR)


# ── Build ──────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Stacked LSTM network for temporal sign-language classification.

    Layer layout
    ────────────
    Input → LSTM(64, return_seq) → LSTM(128, return_seq) → LSTM(64)
          → Dense(64, relu) → Dropout → Dense(32, relu) → Dropout
          → Dense(num_classes, softmax)
    """
    inp = layers.Input(shape=(SEQUENCE_LENGTH, KEYPOINT_SIZE),
                       name="keypoints")
    x = inp

    # Stacked LSTM layers
    for i, units in enumerate(LSTM_UNITS):
        return_sequences = i < len(LSTM_UNITS) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=0.2,
            recurrent_dropout=0.1,
            name=f"lstm_{i+1}",
        )(x)

    # Dense head
    for i, units in enumerate(DENSE_UNITS):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(DROPOUT_RATE, name=f"drop_{i+1}")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="SignLanguageLSTM")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Callbacks ──────────────────────────────────
def get_callbacks():
    return [
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
        ),
    ]


# ── Summary helper ─────────────────────────────
if __name__ == "__main__":
    m = build_model(len(SIGNS))
    m.summary()

"""
cnn_model.py - Conv1D Feature Extractor / Classifier for network flow data.

Architecture:
  Input(n_features, 1)
  -> Conv1D(32, k=3, ReLU) -> MaxPool1D(2) -> Dropout(0.3)
  -> Conv1D(64, k=3, ReLU) -> MaxPool1D(2) -> Dropout(0.3)
  -> GlobalAveragePooling1D
  -> Dense(64, ReLU, name='embedding') -> Dropout(0.3)
  -> Dense(n_classes, softmax)          [Baseline B: full classifier]

Two modes:
  1. build_cnn_classifier  - full model with softmax head (Baseline B).
  2. build_cnn_embedding_extractor - frozen model up to the embedding layer,
     used to generate features for XGBoost (Proposed Hybrid).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks


def build_cnn_classifier(n_features: int, n_classes: int,
                         config: dict) -> Model:
    """
    Build the full CNN classifier with softmax output.

    Parameters
    ----------
    n_features : int
        Number of input features (flow-level).
    n_classes : int
        Number of target classes.
    config : dict
        CNN hyperparams from configs/config.yaml["cnn"].

    Returns
    -------
    keras.Model compiled and ready to train.
    """
    cnn_cfg = config["cnn"]

    inp = layers.Input(shape=(n_features, 1), name="input")

    # Block 1
    x = layers.Conv1D(cnn_cfg["conv1_filters"],
                      kernel_size=cnn_cfg["kernel_size"],
                      activation="relu", padding="same",
                      name="conv1")(inp)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(cnn_cfg["dropout_rate"], name="drop1")(x)

    # Block 2
    x = layers.Conv1D(cnn_cfg["conv2_filters"],
                      kernel_size=cnn_cfg["kernel_size"],
                      activation="relu", padding="same",
                      name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(cnn_cfg["dropout_rate"], name="drop2")(x)

    # Global pooling -> embedding
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(cnn_cfg["embedding_dim"], activation="relu",
                     name="embedding")(x)
    x = layers.Dropout(cnn_cfg["dropout_rate"], name="drop_emb")(x)

    # Classification head
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cnn_cfg["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_embedding_extractor(trained_model: Model) -> Model:
    """
    Create a frozen embedding model from a trained CNN classifier.

    Strips the softmax head and freezes all layers so no gradient updates
    occur during inference - two-stage, NOT end-to-end.

    Parameters
    ----------
    trained_model : keras.Model
        A trained CNN classifier (output of build_cnn_classifier after fit).

    Returns
    -------
    keras.Model outputting the 'embedding' layer activations (shape: [batch, embedding_dim]).
    """
    embedding_layer = trained_model.get_layer("embedding")
    embedding_model = Model(
        inputs=trained_model.input,
        outputs=embedding_layer.output,
        name="CNN_Embedding_Extractor",
    )
    # Freeze ALL layers - this is a feature extractor, not a trainable model
    for layer in embedding_model.layers:
        layer.trainable = False

    return embedding_model


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              n_classes: int, config: dict,
              models_dir: str) -> tuple[Model, callbacks.History]:
    """
    Train the CNN classifier and save the model.

    Parameters
    ----------
    X_train, y_train : training data (2D arrays - will be reshaped to 3D).
    X_val, y_val     : validation data.
    n_classes        : number of classes.
    config           : full config dict.
    models_dir       : where to save the .keras model.

    Returns
    -------
    (trained_model, history)
    """
    cnn_cfg = config["cnn"]
    seed = config["random_seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]

    # Reshape for Conv1D: (samples, features, 1)
    X_train_3d = X_train.reshape(-1, n_features, 1)
    X_val_3d = X_val.reshape(-1, n_features, 1)

    model = build_cnn_classifier(n_features, n_classes, config)
    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cnn_cfg["patience"],
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=cnn_cfg["epochs"],
        batch_size=cnn_cfg["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    # Save model
    model_path = os.path.join(models_dir, "cnn_classifier.keras")
    model.save(model_path)
    print(f"[cnn] Model saved to {model_path}")

    return model, history


# ---------------------------------------------------------------------------
# CLI entry point (standalone training)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = os.path.join(project_root, config["data"]["processed_dir"])
    models_dir = os.path.join(project_root, "models")

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    train_cnn(X_train, y_train, X_val, y_val,
              config["n_classes"], config, models_dir)

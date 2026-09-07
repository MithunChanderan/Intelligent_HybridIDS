"""
embeddings.py - Generate CNN embeddings for the XGBoost hybrid stage.

Workflow:
  1. Load the trained CNN classifier from models/.
  2. Build the frozen embedding extractor (strips softmax, freezes weights).
  3. Generate embeddings for train/val/test sets.
  4. Save as .npy files in data/processed/.
"""

import os
import numpy as np
import yaml
from tensorflow import keras

from cnn_model import build_cnn_embedding_extractor


def generate_embeddings(project_root: str, config: dict = None):
    """
    Load trained CNN -> freeze -> extract embeddings -> save.

    Parameters
    ----------
    project_root : str
        Path to hybrid-ids/ root.
    config : dict, optional
        Pre-loaded config. If None, reads from configs/config.yaml.

    Returns
    -------
    dict with keys: emb_train, emb_val, emb_test (np.ndarray each).
    """
    if config is None:
        config_path = os.path.join(project_root, "configs", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    processed_dir = os.path.join(project_root, config["data"]["processed_dir"])
    models_dir = os.path.join(project_root, "models")

    # Load trained CNN
    cnn_path = os.path.join(models_dir, "cnn_classifier.keras")
    print(f"[embeddings] Loading trained CNN from {cnn_path}")
    trained_model = keras.models.load_model(cnn_path)

    # Build frozen embedding extractor
    emb_model = build_cnn_embedding_extractor(trained_model)

    # Verify freeze
    trainable_count = sum(
        1 for layer in emb_model.layers if layer.trainable
    )
    print(f"[embeddings] Embedding model layers: {len(emb_model.layers)}, "
          f"trainable: {trainable_count} (should be 0)")
    assert trainable_count == 0, "CNN must be fully frozen for two-stage!"

    # Load processed data
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))

    n_features = X_train.shape[1]
    batch_size = config["cnn"]["batch_size"]

    # Reshape for Conv1D input
    embeddings = {}
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        X_3d = X.reshape(-1, n_features, 1)
        emb = emb_model.predict(X_3d, batch_size=batch_size, verbose=0)
        embeddings[f"emb_{name}"] = emb
        print(f"[embeddings] {name}: {X.shape} -> {emb.shape}")

    # Save embeddings
    for key, arr in embeddings.items():
        path = os.path.join(processed_dir, f"{key}.npy")
        np.save(path, arr)
    print(f"[embeddings] Saved embeddings to {processed_dir}")

    return embeddings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_embeddings(project_root)

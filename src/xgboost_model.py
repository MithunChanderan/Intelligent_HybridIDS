"""
xgboost_model.py - Multi-class XGBoost classifier.

Two use cases:
  Baseline A:      train on raw preprocessed features.
  Proposed Hybrid: train on CNN embeddings.

Both share the same training logic; only the input data and save path differ.
"""

import os
import time
import numpy as np
import yaml
import xgboost as xgb


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   config: dict, model_save_path: str,
                   experiment_name: str = "xgboost") -> tuple[xgb.XGBClassifier, float]:
    """
    Train a multi-class XGBoost model.

    Parameters
    ----------
    X_train, y_train : training data.
    X_val, y_val     : validation data (used for early stopping).
    config           : full config dict (reads xgboost + random_seed keys).
    model_save_path  : full path to save the model .json file.
    experiment_name  : label for logging.

    Returns
    -------
    (trained_model, training_time_seconds)
    """
    xgb_cfg = config["xgboost"]
    seed = config["random_seed"]
    n_classes = config["n_classes"]

    print(f"[{experiment_name}] Training XGBoost on {X_train.shape} ...")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        eval_metric=xgb_cfg["eval_metric"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        tree_method=xgb_cfg["tree_method"],
        random_state=seed,
        n_jobs=-1,
        verbosity=1,
    )

    start = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    train_time = time.perf_counter() - start

    print(f"[{experiment_name}] Training complete in {train_time:.2f}s, "
          f"best iteration: {model.best_iteration}")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    print(f"[{experiment_name}] Model saved to {model_save_path}")

    return model, train_time


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

    train_xgboost(
        X_train, y_train, X_val, y_val, config,
        os.path.join(models_dir, "xgboost_raw.json"),
        experiment_name="Baseline_A_XGBoost_Raw"
    )

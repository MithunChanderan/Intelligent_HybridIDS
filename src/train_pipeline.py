"""
train_pipeline.py - Orchestrates the full Hybrid IDS training pipeline.

Stages:
  0. Preprocessing   (if data/processed/ doesn't exist)
  1. Baseline A       - XGBoost on raw preprocessed features
  2. Baseline B       - CNN-only classifier (softmax head)
  3. Proposed Hybrid  - Frozen CNN -> embeddings -> XGBoost

Each stage produces metrics; at the end a comparison_table.csv ranks them.

Usage:
  python src/train_pipeline.py                    # full dataset
  python src/train_pipeline.py --sample-frac 0.01 # 1% sample for testing
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np

# ---------------------------------------------------------------------------
# Ensure src/ is on the path so intra-package imports work when run as script
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str = None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ===================================================================
# Stage 0 - Preprocessing
# ===================================================================

def stage_preprocessing(config: dict, config_path: str):
    from preprocessing import run_preprocessing
    processed_dir = os.path.join(PROJECT_ROOT, config["data"]["processed_dir"])
    if os.path.isfile(os.path.join(processed_dir, "X_train.npy")):
        print("[pipeline] Preprocessed data found - skipping Stage 0.")
        # Reload config (may have been updated by a previous run)
        return load_config(config_path)
    print("\n" + "=" * 60)
    print("  Stage 0 - Preprocessing")
    print("=" * 60)
    _, _, _, _, config = run_preprocessing(PROJECT_ROOT, config_path)
    return config


# ===================================================================
# Stage 1 - Baseline A: XGBoost on Raw Features
# ===================================================================

def stage_baseline_a(X_train, y_train, X_val, y_val, X_test, y_test,
                     config, class_names, results_dir, models_dir):
    from xgboost_model import train_xgboost
    from evaluation import evaluate_experiment

    print("\n" + "=" * 60)
    print("  Stage 1 - Baseline A: XGBoost on Raw Features")
    print("=" * 60)

    model_path = os.path.join(models_dir, "xgboost_raw.json")
    model, train_time = train_xgboost(
        X_train, y_train, X_val, y_val, config,
        model_path, experiment_name="Baseline_A"
    )

    y_pred = model.predict(X_test)

    def predict_fn(X):
        return model.predict(X)

    metrics = evaluate_experiment(
        "Baseline_A_XGBoost_Raw",
        y_test, y_pred, class_names,
        train_time, predict_fn, X_test,
        model_path, results_dir,
    )
    return metrics


# ===================================================================
# Stage 2 - Baseline B: CNN-only Classifier
# ===================================================================

def stage_baseline_b(X_train, y_train, X_val, y_val, X_test, y_test,
                     config, class_names, results_dir, models_dir):
    from cnn_model import train_cnn
    from evaluation import evaluate_experiment
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("  Stage 2 - Baseline B: CNN-only Classifier")
    print("=" * 60)

    n_classes = config["n_classes"]
    n_features = X_train.shape[1]

    # Train CNN once with wall-clock timing
    start = time.perf_counter()
    model, history = train_cnn(
        X_train, y_train, X_val, y_val,
        n_classes, config, models_dir,
    )
    cnn_train_time = time.perf_counter() - start

    X_test_3d = X_test.reshape(-1, n_features, 1)
    y_pred_proba = model.predict(X_test_3d, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    def predict_fn(X):
        X_3d = X.reshape(-1, n_features, 1)
        return np.argmax(model.predict(X_3d, verbose=0), axis=1)

    model_path = os.path.join(models_dir, "cnn_classifier.keras")
    metrics = evaluate_experiment(
        "Baseline_B_CNN_Only",
        y_test, y_pred, class_names,
        cnn_train_time, predict_fn, X_test,
        model_path, results_dir,
    )
    return metrics


# ===================================================================
# Stage 3 - Proposed Hybrid: CNN embeddings -> XGBoost
# ===================================================================

def stage_hybrid(X_train, y_train, X_val, y_val, X_test, y_test,
                 config, class_names, results_dir, models_dir):
    from embeddings import generate_embeddings
    from xgboost_model import train_xgboost
    from evaluation import evaluate_experiment

    print("\n" + "=" * 60)
    print("  Stage 3 - Proposed Hybrid: CNN Embeddings -> XGBoost")
    print("=" * 60)

    processed_dir = os.path.join(PROJECT_ROOT, config["data"]["processed_dir"])

    # Generate embeddings from frozen CNN
    embs = generate_embeddings(PROJECT_ROOT, config)
    emb_train = embs["emb_train"]
    emb_val = embs["emb_val"]
    emb_test = embs["emb_test"]

    # Train XGBoost on embeddings
    model_path = os.path.join(models_dir, "xgboost_hybrid.json")
    model, train_time = train_xgboost(
        emb_train, y_train, emb_val, y_val, config,
        model_path, experiment_name="Hybrid"
    )

    y_pred = model.predict(emb_test)

    # For latency measurement, include embedding extraction + XGBoost predict
    from tensorflow import keras
    from cnn_model import build_cnn_embedding_extractor

    cnn_model = keras.models.load_model(
        os.path.join(models_dir, "cnn_classifier.keras")
    )
    emb_model = build_cnn_embedding_extractor(cnn_model)
    n_features = X_test.shape[1]

    def predict_fn(X):
        X_3d = X.reshape(-1, n_features, 1)
        emb = emb_model.predict(X_3d, verbose=0)
        return model.predict(emb)

    metrics = evaluate_experiment(
        "Proposed_Hybrid_CNN_XGBoost",
        y_test, y_pred, class_names,
        train_time, predict_fn, X_test,
        model_path, results_dir,
    )
    return metrics


# ===================================================================
# Main Orchestrator
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Hybrid IDS - Training Pipeline"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="Override sample fraction (e.g. 0.01 for 1%% of data)"
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip preprocessing even if processed data doesn't exist"
    )
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = load_config(config_path)

    # Override sample frac from CLI if provided
    if args.sample_frac is not None:
        config["data"]["sample_frac"] = args.sample_frac
        save_config(config, config_path)
        print(f"[pipeline] Overriding sample_frac to {args.sample_frac}")

    # Set global seed
    seed = config["random_seed"]
    np.random.seed(seed)

    processed_dir = os.path.join(PROJECT_ROOT, config["data"]["processed_dir"])
    models_dir = os.path.join(PROJECT_ROOT, "models")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ---- Stage 0: Preprocessing ----
    if not args.skip_preprocessing:
        config = stage_preprocessing(config, config_path)
    # Reload config to get feature_order and class_mapping
    config = load_config(config_path)

    # Load processed data
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    class_mapping = config.get("class_mapping", {})
    # YAML may load keys as int or str - normalize to int for sorting
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    class_names = [class_mapping[k] for k in sorted(class_mapping.keys())]
    if not class_names:
        class_names = [str(i) for i in range(config["n_classes"])]

    print(f"\n[pipeline] Data loaded:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes ({len(class_names)}): {class_names}")

    all_metrics = []

    # ---- Stage 1: Baseline A ----
    metrics_a = stage_baseline_a(
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, class_names, results_dir, models_dir
    )
    all_metrics.append(metrics_a)

    # ---- Stage 2: Baseline B ----
    metrics_b = stage_baseline_b(
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, class_names, results_dir, models_dir
    )
    all_metrics.append(metrics_b)

    # ---- Stage 3: Proposed Hybrid ----
    metrics_h = stage_hybrid(
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, class_names, results_dir, models_dir
    )
    all_metrics.append(metrics_h)

    # ---- Comparison table ----
    from evaluation import build_comparison_table
    build_comparison_table(all_metrics, results_dir)

    print("\n" + "=" * 60)
    print("  Pipeline Complete - Phase 1-3 (50%)")
    print("=" * 60)
    print(f"  Results:  {results_dir}/")
    print(f"  Models:   {models_dir}/")
    print(f"  Config:   {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

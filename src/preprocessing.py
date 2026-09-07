"""
preprocessing.py - CIC-IDS2017 Data Cleaning, Encoding, Scaling & Splitting.

Design decisions documented inline:
- Inf handling: DROP rows. CIC-IDS2017 has inf values in Flow Bytes/s and
  Flow Packets/s due to zero-duration flows (division by zero in CICFlowMeter).
  These are artefacts, not meaningful data - imputation would inject noise.
- NaN handling: DROP rows (after inf->NaN conversion). Affected ~0.1% of rows.
- Label encoding: sklearn LabelEncoder fitted dynamically on observed labels.
- Scaler: StandardScaler fitted on TRAINING data only - applied to val/test.
"""

import os
import sys
import glob
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    """Load the YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _save_config(config: dict, config_path: str) -> None:
    """Write the config dict back to YAML (preserves runtime updates)."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def load_raw_data(raw_dir: str, sample_frac: float = 1.0,
                  seed: int = 42) -> pd.DataFrame:
    """Load and concatenate all CSVs in raw_dir (searches subdirectories too)."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    # Also search subdirectories (e.g. data/raw/MachineLearningCVE/*.csv)
    csv_files += sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"),
                                  recursive=True))
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in csv_files:
        real = os.path.realpath(f)
        if real not in seen:
            seen.add(real)
            unique_files.append(f)
    csv_files = unique_files
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir} (or subdirectories). "
            "Place CIC-IDS2017 MachineLearningCSV files there first."
        )
    print(f"[preprocessing] Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
        frames.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows, "
              f"{df.shape[1]} cols")

    data = pd.concat(frames, ignore_index=True)
    print(f"[preprocessing] Total rows after concat: {len(data):,}")

    # Optional sampling for dev/debug runs
    if 0 < sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=seed).reset_index(
            drop=True
        )
        print(f"[preprocessing] Sampled {sample_frac*100:.1f}% -> "
              f"{len(data):,} rows")
    return data


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (CIC-IDS2017 leading-space issue)."""
    df.columns = df.columns.str.strip()
    return df


def handle_inf_nan(df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """
    Replace infinities with NaN, then handle NaN.

    method="drop" - drop rows containing any NaN (default & recommended).
    method="impute" - fill NaN with column median.
    """
    n_before = len(df)
    # Identify numeric columns only for inf replacement
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_inf = n_before - df.dropna(subset=numeric_cols).shape[0]
    print(f"[preprocessing] Rows with inf/NaN: {n_inf:,} "
          f"({n_inf/n_before*100:.2f}%)")

    if method == "drop":
        df = df.dropna().reset_index(drop=True)
        print(f"[preprocessing] After dropping inf/NaN: {len(df):,} rows")
    elif method == "impute":
        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        print(f"[preprocessing] Imputed NaN with column medians")
    else:
        raise ValueError(f"Unknown inf/nan handling method: {method}")
    return df


def drop_non_feature_columns(df: pd.DataFrame,
                              drop_cols: list) -> pd.DataFrame:
    """Drop identifier / non-generalizable columns."""
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"[preprocessing] Dropped columns: {existing}")
    return df


def encode_labels(df: pd.DataFrame,
                  label_col: str = "Label") -> tuple[pd.DataFrame,
                                                      LabelEncoder]:
    """
    Encode the multi-class Label column using LabelEncoder.
    Dynamically discovers all present labels - no hardcoded list.
    Sanitizes non-ASCII characters in labels (common in CIC-IDS2017).
    """
    if label_col not in df.columns:
        raise KeyError(
            f"'{label_col}' column not found. Available: {list(df.columns)}"
        )
    # Clean label strings: strip whitespace and remove non-ASCII chars
    df[label_col] = (df[label_col].astype(str)
                     .str.strip()
                     .str.encode('ascii', errors='ignore')
                     .str.decode('ascii'))
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    class_list = [str(c) for c in le.classes_]
    print(f"[preprocessing] Encoded {len(le.classes_)} classes: "
          f"{class_list}")
    return df, le


def split_data(X: np.ndarray, y: np.ndarray,
               train_ratio: float, val_ratio: float,
               test_ratio: float, seed: int) -> dict:
    """
    Stratified train/val/test split.
    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=y
    )
    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=y_temp
    )
    print(f"[preprocessing] Split sizes - train: {len(X_train):,}, "
          f"val: {len(X_val):,}, test: {len(X_test):,}")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def scale_features(splits: dict) -> tuple[dict, StandardScaler]:
    """
    Fit StandardScaler on training data ONLY, transform val/test.
    NO DATA LEAKAGE: scaler sees only training statistics.
    """
    scaler = StandardScaler()
    splits["X_train"] = scaler.fit_transform(splits["X_train"])
    splits["X_val"] = scaler.transform(splits["X_val"])
    splits["X_test"] = scaler.transform(splits["X_test"])
    print("[preprocessing] StandardScaler fitted on training data, "
          "applied to val/test")
    return splits, scaler


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_preprocessing(project_root: str, config_path: str = None):
    """
    Full preprocessing pipeline.
    Returns (splits_dict, label_encoder, scaler, feature_names, config).
    """
    if config_path is None:
        config_path = os.path.join(project_root, "configs", "config.yaml")
    config = _load_config(config_path)
    seed = config["random_seed"]
    data_cfg = config["data"]
    pre_cfg = config["preprocessing"]

    raw_dir = os.path.join(project_root, data_cfg["raw_dir"])
    processed_dir = os.path.join(project_root, data_cfg["processed_dir"])
    models_dir = os.path.join(project_root, "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load
    df = load_raw_data(raw_dir, data_cfg["sample_frac"], seed)

    # 2. Clean columns
    df = clean_columns(df)

    # 3. Handle inf/NaN
    df = handle_inf_nan(df, method=pre_cfg["inf_handling"])

    # 4. Drop identifier columns
    df = drop_non_feature_columns(df, pre_cfg["drop_columns"])

    # 5. Encode labels
    df, label_encoder = encode_labels(df, "Label")

    # 6. Separate features and target
    y = df["Label"].values
    X = df.drop(columns=["Label"])

    # Ensure all remaining features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[preprocessing] WARNING: Dropping non-numeric columns "
              f"remaining after cleanup: {non_numeric}")
        X = X.drop(columns=non_numeric)

    feature_names = list(X.columns)
    X = X.values.astype(np.float32)

    print(f"[preprocessing] Final feature count: {len(feature_names)}")
    print(f"[preprocessing] Final dataset shape: {X.shape}")

    # 7. Stratified split
    splits = split_data(
        X, y,
        data_cfg["train_ratio"], data_cfg["val_ratio"],
        data_cfg["test_ratio"], seed
    )

    # 8. Scale (fit on train only)
    splits, scaler = scale_features(splits)

    # 9. Save processed data
    for key, arr in splits.items():
        np.save(os.path.join(processed_dir, f"{key}.npy"), arr)
    print(f"[preprocessing] Saved processed arrays to {processed_dir}")

    # 10. Save artefacts
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.joblib"))
    print(f"[preprocessing] Saved scaler & label encoder to {models_dir}")

    # 11. Update config with runtime info
    class_mapping = {int(i): str(c) for i, c in
                     enumerate(label_encoder.classes_)}
    config["feature_order"] = feature_names
    config["class_mapping"] = class_mapping
    config["n_features"] = len(feature_names)
    config["n_classes"] = len(label_encoder.classes_)
    _save_config(config, config_path)
    print(f"[preprocessing] Updated {config_path} with feature_order "
          f"({len(feature_names)} features) and class_mapping "
          f"({len(class_mapping)} classes)")

    return splits, label_encoder, scaler, feature_names, config


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Determine project root (one level up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_preprocessing(project_root)

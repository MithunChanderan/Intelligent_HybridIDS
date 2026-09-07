"""
evaluation.py - Metrics computation, confusion matrix, and experiment comparison.

For each experiment reports:
  - Precision, Recall, F1 (macro + weighted)
  - Per-class precision / recall / F1
  - Confusion matrix (saved as PNG)
  - False-positive rate (per-class and macro-average)
  - Training time (wall clock seconds)
  - Inference latency (median per-sample, measured over the test set)
  - Model file size on disk
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray,
                n_classes: int) -> dict:
    """
    Compute per-class false-positive rate and macro-average FPR.

    FPR_c = FP_c / (FP_c + TN_c)
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fpr_per_class = {}
    for c in range(n_classes):
        # FP for class c = sum of column c minus the diagonal
        fp = cm[:, c].sum() - cm[c, c]
        # TN for class c = total - (TP + FP + FN)
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr_per_class[int(c)] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    macro_fpr = float(np.mean(list(fpr_per_class.values())))
    return {"per_class": fpr_per_class, "macro": macro_fpr}


def measure_inference_latency(predict_fn, X_test: np.ndarray,
                               n_runs: int = 3) -> float:
    """
    Measure median per-sample inference latency in milliseconds.

    Runs the full test set n_runs times, takes the median total time,
    divides by the number of samples.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X_test)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    median_total = float(np.median(times))
    per_sample_ms = (median_total / len(X_test)) * 1000
    return per_sample_ms


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB."""
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    # For directories (e.g., SavedModel format)
    total = 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list, save_path: str,
                          title: str = "Confusion Matrix"):
    """Save a heatmap confusion matrix as PNG."""
    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)),
                                     max(6, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Confusion matrix saved to {save_path}")


def evaluate_experiment(experiment_name: str,
                        y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: list,
                        train_time: float,
                        predict_fn,
                        X_test: np.ndarray,
                        model_path: str,
                        results_dir: str) -> dict:
    """
    Run full evaluation suite for one experiment.

    Returns a metrics dict and saves confusion matrix PNG + metrics JSON.
    """
    os.makedirs(results_dir, exist_ok=True)
    n_classes = len(class_names)
    all_labels = list(range(n_classes))

    # Core metrics — pass labels explicitly so classes absent from
    # the test split (rare with sampling) are still reported as 0.
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=all_labels,
                           average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=all_labels,
                                      average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=all_labels,
                                average="macro", zero_division=0)

    # FPR
    fpr = compute_fpr(y_true, y_pred, n_classes)

    # Latency
    latency_ms = measure_inference_latency(predict_fn, X_test)

    # Model size
    model_size_mb = get_model_size_mb(model_path)

    metrics = {
        "experiment": experiment_name,
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_fpr": round(fpr["macro"], 6),
        "fpr_per_class": {class_names[k]: round(v, 6)
                          for k, v in fpr["per_class"].items()},
        "training_time_s": round(float(train_time), 2),
        "inference_latency_ms_per_sample": round(float(latency_ms), 6),
        "model_size_mb": round(float(model_size_mb), 3),
        "per_class_report": {
            k: v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "classification_report_summary": {
            "macro_avg": report.get("macro avg", {}),
            "weighted_avg": report.get("weighted avg", {}),
        },
    }

    # Save metrics JSON
    json_path = os.path.join(results_dir, f"metrics_{experiment_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluation] Metrics saved to {json_path}")

    # Save confusion matrix
    cm_path = os.path.join(results_dir,
                            f"confusion_matrix_{experiment_name}.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path,
                          title=f"Confusion Matrix - {experiment_name}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {experiment_name} Results")
    print(f"{'='*60}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  Weighted F1:    {weighted_f1:.4f}")
    print(f"  Macro FPR:      {fpr['macro']:.4f}")
    print(f"  Train time:     {train_time:.2f}s")
    print(f"  Latency:        {latency_ms:.4f} ms/sample")
    print(f"  Model size:     {model_size_mb:.3f} MB")
    print(f"{'='*60}\n")

    return metrics


def build_comparison_table(all_metrics: list[dict],
                            results_dir: str) -> pd.DataFrame:
    """
    Build and save comparison_table.csv ranking experiments by macro-F1.
    """
    rows = []
    for m in all_metrics:
        rows.append({
            "Experiment": m["experiment"],
            "Macro-F1": m["macro_f1"],
            "Weighted-F1": m["weighted_f1"],
            "Macro-Precision": m["macro_precision"],
            "Macro-Recall": m["macro_recall"],
            "Macro-FPR": m["macro_fpr"],
            "Training Time (s)": m["training_time_s"],
            "Latency (ms/sample)": m["inference_latency_ms_per_sample"],
            "Model Size (MB)": m["model_size_mb"],
        })
    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    df = df[["Rank"] + [c for c in df.columns if c != "Rank"]]

    csv_path = os.path.join(results_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"[evaluation] Comparison table saved to {csv_path}")
    print(df.to_string(index=False))

    # Also save aggregate metrics.json
    agg_path = os.path.join(results_dir, "metrics.json")
    with open(agg_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[evaluation] Aggregate metrics saved to {agg_path}")

    return df

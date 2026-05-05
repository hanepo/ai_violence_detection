from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]


def _plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, save_path: Path) -> None:
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["non_violent", "violent"])
    ax.set_yticklabels(["non_violent", "violent"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Validation Threshold)")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def _plot_line(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(x, y, linewidth=2.2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def _plot_precision_recall(recall: np.ndarray, precision: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(recall, precision, linewidth=2.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vision model metrics charts from threshold sweep CSV")
    parser.add_argument("--sweep-csv", default="output/vision_threshold_sweep.csv")
    parser.add_argument("--threshold", type=float, default=0.35, help="Validation threshold to report")
    parser.add_argument("--out-dir", default="output/model_metrics/vision")
    args = parser.parse_args()

    sweep_csv = Path(args.sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {sweep_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sweep_csv)
    required_cols = {"threshold", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "accuracy"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sweep CSV: {sorted(missing)}")

    df = df.sort_values("threshold").reset_index(drop=True)

    # Validation row at nearest threshold.
    idx = int((df["threshold"] - float(args.threshold)).abs().idxmin())
    row = df.iloc[idx]

    # Save a clean metrics table.
    table_csv = out_dir / "vision_threshold_metrics.csv"
    df.to_csv(table_csv, index=False)

    # Save validation summary JSON.
    summary = {
        "selected_threshold": float(row["threshold"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tp": int(row["tp"]),
        "best_f1_threshold": float(df.loc[df["f1"].idxmax(), "threshold"]),
        "best_f1": float(df["f1"].max()),
    }
    (out_dir / "model_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x = df["threshold"].to_numpy(dtype=np.float32)
    precision = df["precision"].to_numpy(dtype=np.float32)
    recall = df["recall"].to_numpy(dtype=np.float32)
    f1 = df["f1"].to_numpy(dtype=np.float32)

    _plot_confusion_matrix(
        tn=int(row["tn"]),
        fp=int(row["fp"]),
        fn=int(row["fn"]),
        tp=int(row["tp"]),
        save_path=out_dir / "confusion_matrix.png",
    )
    _plot_precision_recall(recall=recall, precision=precision, save_path=out_dir / "precision_recall_curve.png")
    _plot_line(x=x, y=f1, title="F1-Confidence Curve", ylabel="F1", save_path=out_dir / "f1_confidence_curve.png")
    _plot_line(
        x=x,
        y=precision,
        title="Precision-Confidence Curve",
        ylabel="Precision",
        save_path=out_dir / "precision_confidence_curve.png",
    )
    _plot_line(
        x=x,
        y=recall,
        title="Recall-Confidence Curve",
        ylabel="Recall",
        save_path=out_dir / "recall_confidence_curve.png",
    )

    print("Generated metrics report:")
    print(f"- table: {table_csv}")
    print(f"- summary: {out_dir / 'model_validation.json'}")
    print(f"- confusion matrix: {out_dir / 'confusion_matrix.png'}")
    print(f"- precision-recall: {out_dir / 'precision_recall_curve.png'}")
    print(f"- f1-confidence: {out_dir / 'f1_confidence_curve.png'}")
    print(f"- precision-confidence: {out_dir / 'precision_confidence_curve.png'}")
    print(f"- recall-confidence: {out_dir / 'recall_confidence_curve.png'}")


if __name__ == "__main__":
    main()

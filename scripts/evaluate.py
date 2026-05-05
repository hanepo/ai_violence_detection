from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


"""
Expected CSV format:
- y_true: 0 or 1
- y_pred: 0 or 1
- latency_ms: end-to-end alert latency in milliseconds (optional)
"""


def evaluate(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    y_true = df["y_true"].astype(int)
    y_pred = df["y_pred"].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print(classification_report(y_true, y_pred, zero_division=0))

    if "latency_ms" in df.columns:
        print("Avg latency (ms):", round(float(df["latency_ms"].mean()), 2))
        print("P95 latency (ms):", round(float(df["latency_ms"].quantile(0.95)), 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detection metrics from CSV logs")
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    args = parser.parse_args()
    evaluate(Path(args.csv))

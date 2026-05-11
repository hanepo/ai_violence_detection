#!/usr/bin/env python3
"""Generate evaluation report: training curves, metrics bars,
threshold analysis, and confusion matrices for vision or audio model."""
from __future__ import annotations

import argparse
import json
import random as rng_mod
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.inference import AudioInference, VisionInference


# ─────────────────────────── Shared plot helpers ─────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(
    df: pd.DataFrame,
    out_path: Path,
    finetune_start_epoch: int | None = None,
) -> None:
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    if "accuracy" in df.columns:
        ax1.plot(x, df["accuracy"].values, label="Training Accuracy")
    if "val_accuracy" in df.columns:
        ax1.plot(x, df["val_accuracy"].values, label="Validation Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.grid(alpha=0.25)

    if "loss" in df.columns:
        ax2.plot(x, df["loss"].values, label="Training Loss")
    if "val_loss" in df.columns:
        ax2.plot(x, df["val_loss"].values, label="Validation Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.grid(alpha=0.25)

    if finetune_start_epoch is not None:
        vline_x = finetune_start_epoch - 0.5
        for ax in (ax1, ax2):
            ax.axvline(
                vline_x, color="red", linestyle="--", linewidth=1.5,
                label="Start Fine Tuning",
            )

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_path)


def plot_metrics_bars(
    metrics: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    labels = ["Train", "Validation", "Test (Held-out)"]
    keys = ["train", "val", "test"]
    metric_names = ["Loss", "Accuracy", "Precision", "Recall"]
    metric_keys = ["loss", "accuracy", "precision", "recall"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ax, mname, mkey in zip(axes, metric_names, metric_keys):
        vals = [metrics.get(k, {}).get(mkey, 0.0) for k in keys]
        bars = ax.bar(labels, vals, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=9, fontweight="bold")
        ax.set_title(mname)
        top = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, top * 1.18)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_threshold_grid(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    out_path: Path,
) -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = np.array(
        [precision_score(y_true, y_scores >= t, zero_division=0) for t in thresholds]
    )
    recalls = np.array(
        [recall_score(y_true, y_scores >= t, zero_division=0) for t in thresholds]
    )
    f1s = np.array(
        [f1_score(y_true, y_scores >= t, zero_division=0) for t in thresholds]
    )
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_scores)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(pr_rec, pr_prec, color="blue")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(thresholds, f1s, color="green")
    ax.set_title("F1 Score vs. Confidence Threshold")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(thresholds, precisions, color="red")
    ax.set_title("Precision vs. Confidence Threshold")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(thresholds, recalls, color="purple")
    ax.set_title("Recall vs. Confidence Threshold")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4, n * 2), max(4, n * 2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.tight_layout()
    _save(fig, out_path)


# ─────────────────────────── Audio pipeline ──────────────────────────────────

def _load_audio_dataset(
    data_dir: Path,
    sample_rate: int = 16000,
    max_seconds: float = 2.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Replicates train_audio.py data loading and split logic exactly."""
    import librosa

    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class folders in {data_dir}")

    max_samp = int(sample_rate * max_seconds)
    x_all: list[np.ndarray] = []
    y_all: list[int] = []

    for ci, cn in enumerate(classes):
        wavs = list((data_dir / cn).rglob("*.wav"))
        print(f"    Loading {cn}: {len(wavs)} files")
        for wav in wavs:
            audio, _ = librosa.load(wav, sr=sample_rate, mono=True)
            audio = audio[:max_samp] if audio.size >= max_samp else np.pad(audio, (0, max_samp - audio.size))
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfcc = mfcc[:, :80]
            if mfcc.shape[1] < 80:
                mfcc = np.pad(mfcc, ((0, 0), (0, 80 - mfcc.shape[1])))
            mu, sigma = mfcc.mean(), mfcc.std()
            mfcc = (mfcc - mu) / max(sigma, 1e-6)
            x_all.append(mfcc.astype(np.float32))
            y_all.append(ci)

    x = np.stack(x_all)
    y = np.array(y_all, dtype=np.int32)
    idx = np.arange(len(x))
    np.random.seed(seed)
    np.random.shuffle(idx)
    x, y = x[idx], y[idx]

    n = len(x)
    t_end = int(n * train_ratio)
    v_end = t_end + int(n * val_ratio)
    return x[:t_end], y[:t_end], x[t_end:v_end], y[t_end:v_end], x[v_end:], y[v_end:], classes


def run_audio_eval(args: argparse.Namespace) -> None:
    data_dir = ROOT_DIR / args.audio_data_dir
    model_path = ROOT_DIR / args.audio_model
    out_dir = ROOT_DIR / (args.out_dir or "models/audio_eval_report")
    hist_csv = ROOT_DIR / args.audio_history_csv
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Audio Evaluation Report")
    print(f"{'='*60}")
    print(f"Data : {data_dir}")
    print(f"Model: {model_path}")
    print(f"Out  : {out_dir}")

    model = AudioInference(str(model_path))
    if not model.runner.available():
        print("  [WARNING] TFLite model not available — using fallback scores")

    print("\nLoading audio dataset (this may take a few minutes)...")
    x_train, y_train, x_val, y_val, x_test, y_test, classes = _load_audio_dataset(
        data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Classes: {classes}")
    print(f"Split  : train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    all_metrics: dict[str, dict[str, float]] = {}
    val_scores: list[float] = []
    val_binary: list[int] = []

    for split_name, xs, ys, label in [
        ("train", x_train, y_train, "Train"),
        ("val",   x_val,   y_val,   "Validation"),
        ("test",  x_test,  y_test,  "Test (Held-out)"),
    ]:
        print(f"\n  Evaluating {label} ({len(xs)} samples)...")
        y_pred_idx: list[int] = []
        y_scores: list[float] = []

        for mfcc, _ in zip(xs, ys):
            result = model.predict(mfcc)
            probs = result["probabilities"]
            if probs:
                pred_label = max(probs, key=probs.get)
                pred_idx = classes.index(pred_label) if pred_label in classes else 0
            else:
                pred_idx = 0
            y_pred_idx.append(pred_idx)
            y_scores.append(model.score(mfcc))

        y_true_list = ys.tolist()
        acc   = accuracy_score(y_true_list, y_pred_idx)
        prec  = precision_score(y_true_list, y_pred_idx, average="macro", zero_division=0)
        rec   = recall_score(y_true_list, y_pred_idx, average="macro", zero_division=0)
        f1    = f1_score(y_true_list, y_pred_idx, average="macro", zero_division=0)
        bin_true = [0 if classes[y] == "neutral" else 1 for y in y_true_list]
        loss  = log_loss(bin_true, np.clip(y_scores, 1e-7, 1 - 1e-7))

        print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  Loss={loss:.4f}")
        all_metrics[split_name] = {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "loss": float(loss),
        }

        cm = confusion_matrix(y_true_list, y_pred_idx, labels=list(range(len(classes))))
        plot_confusion_matrix(
            cm, classes, f"Audio — {label}",
            out_dir / f"confusion_matrix_{split_name}.png",
        )

        if split_name == "val":
            val_scores = y_scores
            val_binary = bin_true

    print("\n  Computing threshold analysis on validation set...")
    plot_threshold_grid(
        np.array(val_binary),
        np.array(val_scores),
        out_dir / "threshold_analysis.png",
    )

    plot_metrics_bars(all_metrics, out_dir / "metrics_bars.png")

    if hist_csv.exists():
        df = pd.read_csv(hist_csv)
        plot_training_curves(
            df,
            out_dir / "training_curves.png",
            finetune_start_epoch=args.finetune_start_epoch,
        )
    else:
        print(f"\n  [INFO] {hist_csv.name} not found — skipping training curves")

    summary = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "classes": classes,
        "split_sizes": {
            "train": int(len(x_train)), "val": int(len(x_val)), "test": int(len(x_test)),
        },
        "metrics": all_metrics,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_dir / 'eval_metrics.json'}")
    print(f"\n{'='*60}")
    print(f"Audio eval report done → {out_dir}")
    print(f"{'='*60}")


# ─────────────────────────── Vision pipeline ─────────────────────────────────

def _collect_vision_items(
    data_dir: Path,
) -> tuple[list[tuple[Path, int]], list[str]]:
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise ValueError(f"Expected ≥2 class folders in {data_dir}, found {class_names}")
    items: list[tuple[Path, int]] = []
    for ci, cn in enumerate(class_names):
        for img in (data_dir / cn).rglob("*"):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((img, ci))
    if not items:
        raise ValueError(f"No images found under {data_dir}")
    return items, class_names


def _split_stratified(
    items: list[tuple[Path, int]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Stratified split — identical to train_vision.py split_items()."""
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)

    rng = rng_mod.Random(seed)
    train_items: list[tuple[Path, int]] = []
    val_items:   list[tuple[Path, int]] = []
    test_items:  list[tuple[Path, int]] = []

    for cls_items in sorted(by_class.values(), key=lambda v: v[0][1]):
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_items.extend(cls_items[:n_train])
        val_items.extend(cls_items[n_train: n_train + n_val])
        test_items.extend(cls_items[n_train + n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)
    return train_items, val_items, test_items


def run_vision_eval(args: argparse.Namespace) -> None:
    import cv2

    data_dir   = ROOT_DIR / args.vision_data_dir
    model_path = ROOT_DIR / args.vision_model
    out_dir    = ROOT_DIR / (args.out_dir or "models/vision_eval_report")
    hist_csv   = ROOT_DIR / args.vision_history_csv
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Vision Evaluation Report")
    print(f"{'='*60}")
    print(f"Data : {data_dir}")
    print(f"Model: {model_path}")
    print(f"Out  : {out_dir}")

    if not data_dir.exists():
        print(f"\n[ERROR] Vision data directory not found: {data_dir}")
        print("Run dataset preparation first:")
        print('  python scripts/prepare_datasets.py --cctv-fights')
        return

    model = VisionInference(str(model_path))

    print("\nCollecting image paths...")
    items, class_names = _collect_vision_items(data_dir)
    print(f"Classes: {class_names} → {len(items)} total images")

    train_items, val_items, test_items = _split_stratified(
        items, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )
    print(f"Split  : train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    violent_idx = class_names.index("violent") if "violent" in class_names else 1

    def _score_items(
        items_list: list[tuple[Path, int]], desc: str
    ) -> tuple[list[int], list[int], list[float]]:
        y_true:   list[int]   = []
        y_pred:   list[int]   = []
        y_scores: list[float] = []
        n = len(items_list)
        for i, (img_path, label) in enumerate(items_list):
            if i % 1000 == 0:
                print(f"    {desc}: {i}/{n}")
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            score = model.score(frame)
            y_true.append(label)
            y_scores.append(score)
            y_pred.append(1 if score >= 0.5 else 0)
        return y_true, y_pred, y_scores

    all_metrics: dict[str, dict[str, float]] = {}
    val_scores:  list[float] = []
    val_binary:  list[int]   = []

    for split_name, items_list, label in [
        ("train", train_items, "Train"),
        ("val",   val_items,   "Validation"),
        ("test",  test_items,  "Test (Held-out)"),
    ]:
        print(f"\n  Evaluating {label} ({len(items_list)} images)...")
        y_true, y_pred, y_scores = _score_items(items_list, label)

        y_true_bin = [1 if y == violent_idx else 0 for y in y_true]
        acc  = accuracy_score(y_true_bin, y_pred)
        prec = precision_score(y_true_bin, y_pred, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred, zero_division=0)
        f1   = f1_score(y_true_bin, y_pred, zero_division=0)
        loss = log_loss(y_true_bin, np.clip(y_scores, 1e-7, 1 - 1e-7))

        print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  Loss={loss:.4f}")
        all_metrics[split_name] = {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "loss": float(loss),
        }

        cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
        plot_confusion_matrix(
            cm, class_names, f"Vision — {label}",
            out_dir / f"confusion_matrix_{split_name}.png",
        )

        if split_name == "val":
            val_scores = y_scores
            val_binary = y_true_bin

    print("\n  Computing threshold analysis on validation set...")
    plot_threshold_grid(
        np.array(val_binary),
        np.array(val_scores),
        out_dir / "threshold_analysis.png",
    )

    plot_metrics_bars(all_metrics, out_dir / "metrics_bars.png")

    if hist_csv.exists():
        df = pd.read_csv(hist_csv)
        plot_training_curves(
            df,
            out_dir / "training_curves.png",
            finetune_start_epoch=args.finetune_start_epoch,
        )
    else:
        print(f"\n  [INFO] {hist_csv.name} not found — will be created after training")

    summary = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "classes": class_names,
        "split_sizes": {
            "train": len(train_items), "val": len(val_items), "test": len(test_items),
        },
        "metrics": all_metrics,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_dir / 'eval_metrics.json'}")
    print(f"\n{'='*60}")
    print(f"Vision eval report done → {out_dir}")
    print(f"{'='*60}")


# ─────────────────────────── Model replacement ───────────────────────────────

def maybe_replace_model(
    new_tflite: Path,
    new_metrics_json: Path,
    current_tflite: Path,
) -> bool:
    """Replace current model if new one has better val accuracy."""
    import shutil
    try:
        new_m = json.loads(new_metrics_json.read_text())
        new_val = float(new_m["metrics"]["val"]["accuracy"])
    except Exception as e:
        print(f"  [ERROR] Could not read new metrics: {e}")
        return False

    # Try to find current model's metrics
    candidates = [
        current_tflite.parent / "audio_training_report" / "metrics.json",
        current_tflite.parent / "vision_training_report" / "metrics.json",
        current_tflite.parent / "audio_eval_report" / "eval_metrics.json",
        current_tflite.parent / "vision_eval_report" / "eval_metrics.json",
    ]
    curr_val = 0.0
    for p in candidates:
        if p.exists():
            try:
                m = json.loads(p.read_text())
                curr_val = float(m["metrics"]["val"]["accuracy"])
                break
            except Exception:
                continue

    print(f"  Current val accuracy: {curr_val:.4f}")
    print(f"  New     val accuracy: {new_val:.4f}")

    if new_val > curr_val:
        shutil.copy2(new_tflite, current_tflite)
        labels_src = new_tflite.with_suffix(".labels.txt")
        if labels_src.exists():
            shutil.copy2(labels_src, current_tflite.with_suffix(".labels.txt"))
        print(f"  Replaced {current_tflite.name} (↑ +{new_val - curr_val:.4f})")
        return True

    print("  Current model is still best — no replacement.")
    return False


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation report for vision or audio model"
    )
    parser.add_argument("--model", required=True, choices=["audio", "vision"])
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: models/{model}_eval_report/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--finetune-start-epoch", type=int, default=None,
                        help="Draw 'Start Fine Tuning' line at this epoch on training curves")

    # Audio
    parser.add_argument("--audio-data-dir", default="dataset/audio")
    parser.add_argument("--audio-model", default="models/audio.tflite")
    parser.add_argument("--audio-history-csv", default="models/audio_training_history.csv")

    # Vision
    parser.add_argument("--vision-data-dir", default="dataset/vision")
    parser.add_argument("--vision-model", default="models/vision.tflite")
    parser.add_argument("--vision-history-csv", default="models/vision_training_history.csv")

    args = parser.parse_args()

    if args.model == "audio":
        run_audio_eval(args)
    else:
        run_vision_eval(args)


if __name__ == "__main__":
    main()

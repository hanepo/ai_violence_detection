from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def load_dataset(
    root: Path,
    sample_rate: int = 16000,
    max_seconds: float = 2.0,
    binary_aggressive: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {root}")

    x_data = []
    y_data = []
    max_samples = int(sample_rate * max_seconds)

    for class_idx, class_name in enumerate(classes):
        class_dir = root / class_name
        for wav_path in class_dir.rglob("*.wav"):
            wav, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
            if wav.size < max_samples:
                wav = np.pad(wav, (0, max_samples - wav.size))
            else:
                wav = wav[:max_samples]

            mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=40)
            mfcc = mfcc[:, :80]
            if mfcc.shape[1] < 80:
                mfcc = np.pad(mfcc, ((0, 0), (0, 80 - mfcc.shape[1])))

            # Per-sample normalization improves generalization across recordings.
            mu = float(mfcc.mean())
            sigma = float(mfcc.std())
            mfcc = (mfcc - mu) / max(sigma, 1e-6)

            x_data.append(mfcc.astype(np.float32))
            if binary_aggressive:
                y_data.append(1 if class_name.lower() == "aggressive" else 0)
            else:
                y_data.append(class_idx)

    x = np.stack(x_data)
    y = np.array(y_data, dtype=np.int32)
    if binary_aggressive:
        return x, y, ["non_aggressive", "aggressive"]
    return x, y, classes


def build_model(n_classes: int, binary_aggressive: bool = False) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(40, 80, 1))
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if binary_aggressive:
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def _collect_audio_preds(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> tuple[list[int], list[int]]:
    probs = model.predict(x, verbose=0)
    probs = np.array(probs)
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        y_pred = (probs.reshape(-1) >= 0.5).astype(np.int32).tolist()
    else:
        y_pred = np.argmax(probs, axis=1).tolist()
    return y.tolist(), y_pred


def _print_cm(cm: np.ndarray, class_names: list[str], title: str) -> None:
    col_w = max(len(n) for n in class_names) + 2
    print(f"\n  Confusion Matrix — {title}")
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in class_names) + "  <- Predicted"
    print(header)
    for i, row_name in enumerate(class_names):
        row = f"{row_name:>{col_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(class_names)))
        print(row)
    print("  ^ Actual")


def _plot_cm(cm: np.ndarray, class_names: list[str], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 2), max(4, len(class_names) * 2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _save_training_history(history: dict[str, list[float]], csv_path: Path, fig_path: Path, title: str) -> None:
    max_len = max((len(v) for v in history.values()), default=0)
    keys = sorted(history.keys())

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for i in range(max_len):
            row = [i + 1]
            for k in keys:
                vals = history.get(k, [])
                row.append(vals[i] if i < len(vals) else "")
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    epochs = np.arange(1, max_len + 1)

    if "loss" in history:
        ax.plot(epochs[: len(history["loss"])], history["loss"], label="loss", linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="val_loss", linewidth=2)
    if "accuracy" in history:
        ax.plot(epochs[: len(history["accuracy"])], history["accuracy"], label="accuracy", linewidth=2)
    if "val_accuracy" in history:
        ax.plot(epochs[: len(history["val_accuracy"])], history["val_accuracy"], label="val_accuracy", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {fig_path}")


def train(
    data_dir: Path,
    out_model: Path,
    epochs: int = 12,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    binary_aggressive: bool = False,
    report_dir: Path | None = None,
) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    x, y, classes = load_dataset(data_dir, binary_aggressive=binary_aggressive)
    x = np.expand_dims(x, axis=-1)

    idx = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    n = len(x)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    print(f"Class names: {classes}")
    print(f"Split sizes -> train: {len(x_train)} val: {len(x_val)} test: {len(x_test)}")

    model = build_model(len(classes), binary_aggressive=binary_aggressive)

    unique, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(k): float(len(y_train) / (len(unique) * v)) for k, v in zip(unique, counts)}

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
    )

    _save_training_history(
        history.history,
        csv_path=out_model.parent / "audio_training_history.csv",
        fig_path=out_model.parent / "audio_training_history.png",
        title="Audio Training Metrics and Loss",
    )

    cm_dir = out_model.parent
    splits = [("train", x_train, y_train, "Train"), ("val", x_val, y_val, "Validation"), ("test", x_test, y_test, "Test (Held-out)")]
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split_name, xs, ys, label in splits:
        yt, yp = _collect_audio_preds(model, xs, ys)
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, average="macro", zero_division=0)
        rec = recall_score(yt, yp, average="macro", zero_division=0)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        cm = confusion_matrix(yt, yp)
        print(f"\n=== Audio {label} Metrics ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")
        _print_cm(cm, classes, f"Audio — {label}")
        _plot_cm(cm, classes, f"Audio — {label}", cm_dir / f"audio_cm_{split_name}.png")
        metrics_by_split[split_name] = {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
        }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    saved_model_dir = out_model.parent / "audio_saved_model"
    model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_model.write_bytes(tflite_model)

    labels_path = out_model.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(classes), encoding="utf-8")
    print(f"Saved TFLite model to: {out_model}")
    print(f"Saved class labels to: {labels_path}")

    rep = report_dir or (out_model.parent / "audio_training_report")
    rep.mkdir(parents=True, exist_ok=True)
    summary = {
        "class_names": classes,
        "n_samples_total": int(n),
        "split_sizes": {"train": int(len(x_train)), "val": int(len(x_val)), "test": int(len(x_test))},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "binary_aggressive": binary_aggressive,
        },
        "metrics": metrics_by_split,
        "artifacts_relative": {
            "tflite": str(out_model.as_posix()),
            "labels": str(labels_path.as_posix()),
            "saved_model_dir": str((out_model.parent / "audio_saved_model").as_posix()),
        },
    }
    (rep / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for fname in [
        "audio_training_history.csv",
        "audio_training_history.png",
        "audio_cm_train.png",
        "audio_cm_val.png",
        "audio_cm_test.png",
    ]:
        src = cm_dir / fname
        if src.exists():
            shutil.copy2(src, rep / fname)
    shutil.copy2(out_model, rep / out_model.name)
    shutil.copy2(labels_path, rep / labels_path.name)
    print(f"  Report directory: {rep.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio aggression classifier from WAV files")
    parser.add_argument("--data-dir", default="dataset/audio")
    parser.add_argument("--out-model", default="models/audio.tflite")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--binary-aggressive", action="store_true", help="Train binary model: aggressive vs non-aggressive")
    parser.add_argument(
        "--report-dir",
        default="models/audio_training_report",
        help="Directory for metrics.json, learning curves, confusion matrices, and model copies",
    )
    args = parser.parse_args()
    train(
        Path(args.data_dir),
        Path(args.out_model),
        args.epochs,
        args.batch_size,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.binary_aggressive,
        report_dir=Path(args.report_dir),
    )

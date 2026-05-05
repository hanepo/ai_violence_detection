from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def collect_labeled_images(data_dir: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 classes under {data_dir}, found: {class_names}")

    items: list[tuple[Path, int]] = []
    for idx, name in enumerate(class_names):
        for img in (data_dir / name).rglob("*"):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((img, idx))
    if not items:
        raise ValueError(f"No images found under {data_dir}")
    return items, class_names


def split_items(
    items: list[tuple[Path, int]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    by_class: dict[int, list[tuple[Path, int]]] = {0: [], 1: []}
    for item in items:
        by_class[item[1]].append(item)

    rng = random.Random(seed)
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    test_items: list[tuple[Path, int]] = []

    for cls, cls_items in by_class.items():
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_items.extend(cls_items[:n_train])
        val_items.extend(cls_items[n_train : n_train + n_val])
        test_items.extend(cls_items[n_train + n_val :])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)
    return train_items, val_items, test_items


def build_dataset(items: list[tuple[Path, int]], image_size: int, batch_size: int, training: bool) -> tf.data.Dataset:
    paths = [str(p) for p, _ in items]
    labels = np.array([y for _, y in items], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=max(512, len(items)), seed=42, reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, label: tf.Tensor):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (image_size, image_size))
        img = tf.cast(img, tf.float32)
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.08)
            img = tf.image.random_contrast(img, lower=0.92, upper=1.08)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img, tf.cast(label, tf.float32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _collect_vision_preds(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for x_batch, y_batch in ds:
        probs = model.predict(x_batch, verbose=0).reshape(-1)
        preds = (probs >= 0.5).astype(np.int32)
        y_true.extend(y_batch.numpy().astype(np.int32).tolist())
        y_pred.extend(preds.tolist())
    return y_true, y_pred


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
    image_size: int = 224,
    epochs: int = 8,
    finetune_epochs: int = 4,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    items, class_names = collect_labeled_images(data_dir)
    train_items, val_items, test_items = split_items(items, seed=42, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    print(f"Class names: {class_names}")
    print(f"Split sizes -> train: {len(train_items)} val: {len(val_items)} test: {len(test_items)}")

    train_ds = build_dataset(train_items, image_size=image_size, batch_size=batch_size, training=True)
    val_ds = build_dataset(val_items, image_size=image_size, batch_size=batch_size, training=False)
    test_ds = build_dataset(test_items, image_size=image_size, batch_size=batch_size, training=False)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    # Mild class weighting to avoid violent-class overprediction on the held-out split.
    class_totals = {0: 0, 1: 0}
    for _, cls in train_items:
        class_totals[int(cls)] += 1
    total = class_totals[0] + class_totals[1]
    class_weight = {
        0: float(total / (2.0 * max(class_totals[0], 1))),
        1: float(total / (2.0 * max(class_totals[1], 1))),
    }

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    hist_stage1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, verbose=2)

    # Fine-tune the upper part of the backbone for better domain adaptation.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    hist_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs + finetune_epochs,
        initial_epoch=epochs,
        class_weight=class_weight,
        verbose=2,
    )

    merged_history: dict[str, list[float]] = {}
    all_keys = set(hist_stage1.history.keys()).union(hist_stage2.history.keys())
    for k in all_keys:
        merged_history[k] = list(hist_stage1.history.get(k, [])) + list(hist_stage2.history.get(k, []))

    _save_training_history(
        merged_history,
        csv_path=out_model.parent / "vision_training_history.csv",
        fig_path=out_model.parent / "vision_training_history.png",
        title="Vision Training Metrics and Loss",
    )

    # Confusion matrices and metrics for all three splits.
    cm_dir = out_model.parent
    eval_train_ds = build_dataset(train_items, image_size=image_size, batch_size=batch_size, training=False)
    splits = [("train", eval_train_ds, "Train"), ("val", val_ds, "Validation"), ("test", test_ds, "Test (Held-out)")]
    for split_name, ds, label in splits:
        yt, yp = _collect_vision_preds(model, ds)
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        cm = confusion_matrix(yt, yp)
        print(f"\n=== Vision {label} Metrics ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")
        _print_cm(cm, class_names, f"Vision — {label}")
        _plot_cm(cm, class_names, f"Vision — {label}", cm_dir / f"vision_cm_{split_name}.png")

    out_model.parent.mkdir(parents=True, exist_ok=True)
    saved_model_dir = out_model.parent / "vision_saved_model"
    model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_model.write_bytes(tflite_model)
    print(f"Saved TFLite model to: {out_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vision model for violent vs non-violent frames")
    parser.add_argument("--data-dir", default="dataset/vision")
    parser.add_argument("--out-model", default="models/vision.tflite")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        out_model=Path(args.out_model),
        image_size=args.image_size,
        epochs=args.epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config import load_settings
from utils.inference import VisionInference


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


def get_positive_index(class_names: list[str]) -> int:
    # Match an explicit violent class name; avoid matching "non_violent".
    preferred = {"violent", "violence", "fight", "aggressive"}
    for i, name in enumerate(class_names):
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        if normalized in preferred:
            return i

    for i, name in enumerate(class_names):
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        if "non" in normalized and "violent" in normalized:
            continue
        if "violent" in normalized:
            return i
    return 1


def score_images(model: VisionInference, items: list[tuple[Path, int]], positive_index: int) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []

    for img_path, class_idx in items:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        score = float(model.score(frame))
        y_true.append(1 if class_idx == positive_index else 0)
        y_score.append(score)

    return np.array(y_true, dtype=np.int32), np.array(y_score, dtype=np.float32)


def evaluate_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    return {
        "threshold": threshold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def pick_best(rows: list[dict], min_recall: float) -> dict:
    candidates = [r for r in rows if r["recall"] >= min_recall]
    if candidates:
        # Prefer the highest precision while preserving minimum recall.
        return sorted(candidates, key=lambda r: (r["precision"], r["f1"], r["accuracy"]), reverse=True)[0]
    # Fallback: maximize F1 if recall constraint cannot be met.
    return sorted(rows, key=lambda r: (r["f1"], r["recall"], r["accuracy"]), reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep thresholds for vision violence classifier")
    parser.add_argument("--data-dir", default="dataset/vision")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--end", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--out-csv", default="output/vision_threshold_sweep.csv")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Limit test samples for quicker calibration (0 = all)")
    args = parser.parse_args()

    settings = load_settings()
    model_path = args.model_path or settings.vision_model_path

    items, class_names = collect_labeled_images(Path(args.data_dir))
    _, _, test_items = split_items(
        items,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if args.max_test_samples and args.max_test_samples > 0:
        test_items = test_items[: args.max_test_samples]

    positive_index = get_positive_index(class_names)
    model = VisionInference(str(model_path))

    y_true, y_score = score_images(model, test_items, positive_index)
    if y_true.size == 0:
        raise RuntimeError("No valid test samples were scored.")

    thresholds = np.arange(args.start, args.end + 1e-9, args.step)
    rows = [evaluate_threshold(y_true, y_score, float(t)) for t in thresholds]
    best = pick_best(rows, min_recall=args.min_recall)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== Vision Threshold Sweep (Held-out Test) ===")
    print(f"Test samples: {y_true.size}")
    print(f"Class names : {class_names} (positive='{class_names[positive_index]}')")
    print(f"Sweep range : {args.start:.2f}..{args.end:.2f} step {args.step:.2f}")
    print(f"CSV saved   : {out_csv}")
    print()
    print("Best threshold (with recall constraint)")
    print(f"threshold : {best['threshold']:.2f}")
    print(f"accuracy  : {best['accuracy']:.4f}")
    print(f"precision : {best['precision']:.4f}")
    print(f"recall    : {best['recall']:.4f}")
    print(f"f1        : {best['f1']:.4f}")
    print(f"TN FP FN TP: {best['tn']} {best['fp']} {best['fn']} {best['tp']}")


if __name__ == "__main__":
    main()

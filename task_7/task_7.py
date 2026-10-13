#!/usr/bin/env python3
"""Reproducible Task 7 experiments: CNN classification and MLP regression.

Run from the repository root with the project conda environment:
    python task_7/task_7.py --data dataset_2026 --output task_7/outputs

The outer data split is class-stratified 75/25. Exact duplicate images are
grouped before splitting, and an additional grouped validation split is made
only from the training partition. Hyperparameters and resize resolution are
selected on validation data; the test partition is evaluated after refitting
on the complete 75% training partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "dataset_2026"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = ["0", "4", "7", "8", "A", "D", "H"]
SEED = 20261013


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels make the experiment reproducible on the same
    # software/hardware stack. Some CUDA ops may still differ across versions.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_digest(path: Path) -> str:
    """Hash decoded grayscale pixels so byte-level PNG metadata is irrelevant."""
    with Image.open(path) as image:
        image = image.convert("L")
        return hashlib.sha256(image.tobytes()).hexdigest()


def discover_dataset(root: Path) -> Tuple[List[Path], np.ndarray, List[str], List[str]]:
    paths: List[Path] = []
    labels: List[int] = []
    digests: List[str] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError("Missing class directory: {}".format(class_dir))
        class_paths = sorted(class_dir.glob("*.png"))
        if not class_paths:
            raise RuntimeError("No PNG images found in {}".format(class_dir))
        for path in class_paths:
            paths.append(path)
            labels.append(class_index)
            digests.append(image_digest(path))
    if len(set(labels)) != len(CLASS_NAMES):
        raise RuntimeError("Expected exactly seven populated classes")
    return paths, np.asarray(labels, dtype=np.int64), digests, list(CLASS_NAMES)


def grouped_subset(
    indices: Sequence[int],
    digests: Sequence[str],
    target_count: int,
    seed: int,
) -> List[int]:
    """Choose a near-exact count while keeping identical images together.

    A subset-sum dynamic program finds the requested count when possible; if
    duplicate group sizes make it impossible, the closest count below target
    is used. Input indices must belong to one class.
    """
    members: Dict[str, List[int]] = {}
    for index in indices:
        members.setdefault(digests[index], []).append(index)
    keys = list(members.keys())
    random.Random(seed).shuffle(keys)
    target_count = max(0, min(int(target_count), len(indices)))
    reachable: List[Optional[Tuple[str, ...]]] = [None] * (target_count + 1)
    reachable[0] = tuple()
    if target_count:
        for key in keys:
            group_size = len(members[key])
            for total in range(target_count, group_size - 1, -1):
                if reachable[total] is None and reachable[total - group_size] is not None:
                    reachable[total] = reachable[total - group_size] + (key,)
            if reachable[target_count] is not None:
                break
    chosen_total = target_count
    while chosen_total > 0 and reachable[chosen_total] is None:
        chosen_total -= 1
    chosen_groups = reachable[chosen_total] or tuple()
    return [index for key in chosen_groups for index in members[key]]


def make_splits(
    labels: np.ndarray, digests: Sequence[str], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation and test indices with duplicate isolation."""
    trainval: List[int] = []
    test: List[int] = []
    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == class_id).tolist()
        n_test = int(round(len(class_indices) * 0.25))
        test.extend(grouped_subset(class_indices, digests, n_test, seed + class_id))
        test_set = set(test)
        # Restrict to this class's rows; no class can share a pixel digest with
        # another class, but keep the filter explicit for auditability.
        class_test = {i for i in test_set if labels[i] == class_id}
        trainval.extend(i for i in class_indices if i not in class_test)

    validation: List[int] = []
    train: List[int] = []
    for class_id in range(len(CLASS_NAMES)):
        class_indices = [i for i in trainval if labels[i] == class_id]
        n_validation = int(round(len(class_indices) * 0.10))
        validation.extend(
            grouped_subset(class_indices, digests, n_validation, seed + 100 + class_id)
        )
        validation_set = set(validation)
        class_validation = {i for i in validation_set if labels[i] == class_id}
        train.extend(i for i in class_indices if i not in class_validation)

    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
        np.asarray(sorted(test), dtype=np.int64),
    )


def audit_splits(
    indices: Sequence[np.ndarray], labels: np.ndarray, digests: Sequence[str]
) -> Dict[str, Any]:
    names = ["train", "validation", "test"]
    report: Dict[str, Any] = {"partitions": {}}
    digest_sets: List[set] = []
    for name, part in zip(names, indices):
        hist = {CLASS_NAMES[c]: int(np.sum(labels[part] == c)) for c in range(len(CLASS_NAMES))}
        report["partitions"][name] = {
            "images": int(len(part)),
            "class_counts": hist,
            "unique_exact_pixel_images": len({digests[i] for i in part}),
        }
        digest_sets.append({digests[i] for i in part})
    report["duplicate_digest_intersections"] = {
        "train_validation": len(digest_sets[0] & digest_sets[1]),
        "train_test": len(digest_sets[0] & digest_sets[2]),
        "validation_test": len(digest_sets[1] & digest_sets[2]),
    }
    if any(report["duplicate_digest_intersections"].values()):
        raise AssertionError("Exact duplicate image leaked across a data partition")
    return report


def load_resized(paths: Sequence[Path], side: int) -> np.ndarray:
    """Load grayscale samples and resize to a fixed square using bilinear."""
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    images = np.empty((len(paths), 1, side, side), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("L").resize((side, side), resampling)
            images[i, 0] = np.asarray(image, dtype=np.float32) / 255.0
    return images


def make_tensor_splits(
    image_array: np.ndarray,
    labels: np.ndarray,
    parts: Sequence[np.ndarray],
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, Dict[str, float]]:
    train_indices = parts[0]
    mean = float(image_array[train_indices].mean())
    std = float(image_array[train_indices].std())
    if std < 1e-6:
        raise RuntimeError("Training image standard deviation is nearly zero")
    normalized = (image_array - mean) / std
    datasets = tuple(
        TensorDataset(
            torch.from_numpy(normalized[part]),
            torch.from_numpy(labels[part].astype(np.int64)),
        )
        for part in parts
    )
    return datasets[0], datasets[1], datasets[2], {"mean": mean, "std": std}


class RegressionMLP(nn.Module):
    """Fully connected one-hot regression model (no convolutional layers)."""

    def __init__(self, side: int, n_classes: int, width: int = 512, dropout: float = 0.25):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(side * side, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.BatchNorm1d(width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CharacterCNN(nn.Module):
    """Compact CNN with convolutional BatchNorm and global pooling."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=n_classes).float()


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
        persistent_workers=workers > 0,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression: bool,
    n_classes: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    truths: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            if regression:
                loss = nn.functional.mse_loss(outputs, one_hot(labels, n_classes))
            else:
                loss = nn.functional.cross_entropy(outputs, labels)
            losses.append(float(loss.item()))
            truths.append(labels.cpu().numpy())
            predictions.append(outputs.cpu().numpy())
    y_true = np.concatenate(truths)
    scores = np.concatenate(predictions)
    y_pred = scores.argmax(axis=1)
    return float(np.mean(losses)), float(np.mean(y_true == y_pred)), y_true, scores


def train_candidate(
    model_kind: str,
    side: int,
    learning_rate: float,
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    model: nn.Module
    regression = model_kind == "mlp"
    model = RegressionMLP(side, len(CLASS_NAMES)) if regression else CharacterCNN(len(CLASS_NAMES))
    model.to(device)
    train_loader = make_loader(train_dataset, args.batch_size, True, seed, args.workers)
    validation_loader = make_loader(validation_dataset, args.batch_size * 2, False, seed, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=learning_rate / 20
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_acc = -1.0
    best_epoch = 0
    best_loss = float("inf")
    stale = 0
    history: List[Dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs_tune + 1):
        model.train()
        train_losses: List[float] = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if regression:
                loss = nn.functional.mse_loss(outputs, one_hot(labels, len(CLASS_NAMES)))
            else:
                loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_acc, _, _ = evaluate_model(
            model, validation_loader, device, regression, len(CLASS_NAMES)
        )
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_acc > best_acc or (math.isclose(val_acc, best_acc) and val_loss < best_loss):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Candidate training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {
        "model_kind": model_kind,
        "run_seed": seed,
        "input_size": side,
        "learning_rate": learning_rate,
        "validation_accuracy": best_acc,
        "validation_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": elapsed,
        "history": history,
        "state_dict": best_state,
    }


def train_final_model(
    model_kind: str,
    side: int,
    learning_rate: float,
    epochs: int,
    train_dataset: TensorDataset,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[nn.Module, float]:
    seed_everything(seed)
    regression = model_kind == "mlp"
    model = RegressionMLP(side, len(CLASS_NAMES)) if regression else CharacterCNN(len(CLASS_NAMES))
    model.to(device)
    loader = make_loader(train_dataset, args.batch_size, True, seed, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _epoch in range(1, epochs + 1):
        model.train()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = (
                nn.functional.mse_loss(outputs, one_hot(labels, len(CLASS_NAMES)))
                if regression
                else nn.functional.cross_entropy(outputs, labels)
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return model, elapsed


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    for c, name in enumerate(CLASS_NAMES):
        tp = int(matrix[c, c])
        precision = tp / max(1, int(matrix[:, c].sum()))
        recall = tp / max(1, int(matrix[c, :].sum()))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[c].sum())}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def predict_scores(
    models: Sequence[nn.Module], dataset: TensorDataset, device: torch.device, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    loader = make_loader(dataset, batch_size, False, SEED, 0)
    all_scores: List[np.ndarray] = []
    all_truths: List[np.ndarray] = []
    for model in models:
        model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            model_scores = [model(images) for model in models]
            mean_scores = torch.stack(model_scores, dim=0).mean(dim=0)
            all_scores.append(mean_scores.cpu().numpy())
            all_truths.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(all_truths), np.concatenate(all_scores), elapsed


def fit_foreground_geometry(paths: Sequence[Path]) -> Dict[str, Any]:
    """Fit a class-agnostic glyph bounding-box template on tuning-train only.

    Dark (<128) pixels define dataset foreground. Median relative bounding-box
    width/height are fixed without inspecting target labels or model scores.
    """
    geometry = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
        ys, xs = np.nonzero(array < 128)
        if len(xs):
            geometry.append(((xs.max() - xs.min() + 1) / array.shape[1],
                             (ys.max() - ys.min() + 1) / array.shape[0]))
    if not geometry:
        raise RuntimeError("No training foreground pixels found")
    width_fraction, height_fraction = np.median(geometry, axis=0)
    return {
        "width_fraction": float(width_fraction),
        "height_fraction": float(height_fraction),
        "fit_partition": "tuning_train",
        "fit_images": len(paths),
        "foreground_threshold": 128,
        "statistic": "class-agnostic median relative foreground bounding-box width/height",
    }


def largest_foreground(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected binary component; discard crop noise."""
    visited = np.zeros(mask.shape, dtype=bool)
    best = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    clean = np.zeros_like(mask, dtype=bool)
    if best:
        rows, cols = zip(*best)
        clean[rows, cols] = True
    return clean


def prepare_target_images(
    crop_paths: Sequence[Path], side: int,
    geometry: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Prepare original grayscale baseline or fixed binary geometry alignment."""
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    images = []
    for path in crop_paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if geometry is None:
            # Original baseline: invert, square-pad, and resize the gray crop.
            array = 255 - array
            height, width = array.shape
            canvas = np.full((max(height, width), max(height, width)), 255, dtype=np.uint8)
            y0 = (canvas.shape[0] - height) // 2
            x0 = (canvas.shape[1] - width) // 2
            canvas[y0:y0 + height, x0:x0 + width] = array
            image = Image.fromarray(canvas).resize((side, side), resampling)
        else:
            # Task 6 binary has bright foreground. Keep one glyph, crop its
            # bbox, align to the train-fitted median bbox, and invert polarity.
            mask = largest_foreground(array > 127)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                raise RuntimeError("Empty target foreground: {}".format(path))
            cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            glyph = Image.fromarray(np.where(cropped, 0, 255).astype(np.uint8))
            width = max(1, min(side, round(side * geometry["width_fraction"])))
            height = max(1, min(side, round(side * geometry["height_fraction"])))
            glyph = glyph.resize((width, height), resampling)
            image = Image.new("L", (side, side), 255)
            image.paste(glyph, ((side - width) // 2, (side - height) // 2))
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(images)[:, None]


def infer_target_crops(
    models: Sequence[nn.Module],
    crop_paths: Sequence[Path],
    side: int,
    normalization: Dict[str, float],
    device: torch.device,
    geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify the ten Task-6 crops used as the Figure-1 target.

    These crops are inference-only.  They are deliberately loaded after the
    held-out experiment and never enter either the training or validation
    split.
    """
    if not crop_paths:
        return {"available": False, "reason": "Task-6 crops were not found"}
    validate_target_crop_sequence(crop_paths, "bin" if geometry else "gray")
    if not models:
        raise ValueError("Target inference requires at least one trained model")
    images = prepare_target_images(crop_paths, side, geometry)
    array = (images - normalization["mean"]) / normalization["std"]
    tensor = torch.from_numpy(array)
    loader = DataLoader(TensorDataset(tensor), batch_size=32, shuffle=False)
    for model in models:
        model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            scores.append(torch.stack([model(batch.to(device)) for model in models]).mean(0).cpu().numpy())
    scores_array = np.concatenate(scores, axis=0)
    if scores_array.shape != (10, len(CLASS_NAMES)) or not np.isfinite(scores_array).all():
        raise RuntimeError("Target score matrix must contain ten finite seven-class rows")
    predicted_ids = scores_array.argmax(axis=1)
    reference_ids = np.asarray([CLASS_NAMES.index(char) for char in "HD44780A00"])
    return {
        "available": True,
        "source": [str(path.relative_to(PROJECT_ROOT)) for path in crop_paths],
        "expected_string": "HD44780A00",
        "predicted_string": "".join(CLASS_NAMES[int(index)] for index in predicted_ids),
        "predicted_labels": [CLASS_NAMES[int(index)] for index in predicted_ids],
        "correct_characters": int(np.sum(predicted_ids == reference_ids)),
        "reference_character_accuracy": float(np.mean(predicted_ids == reference_ids)),
        "scores": scores_array.tolist(),
        "input_size": side,
        "normalization": normalization,
        "preprocessing": "binary-largest-component / train-median bbox / black-on-white" if geometry else "gray-invert / square-pad / resize",
        "foreground_geometry": geometry,
        "note": "Inference only; these crops were not used to fit any parameter.",
    }


def save_confusion_matrix(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 5.7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, shrink=0.82, label="Number of images")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted character")
    axis.set_ylabel("True character")
    axis.set_title(title)
    threshold = max(1, matrix.max()) * 0.55
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center",
                      color="white" if matrix[row, col] > threshold else "black", fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_history_plot(results: Sequence[Dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for row in results:
        label = "{} {}px lr={:g}".format(row["model_kind"].upper(), row["input_size"], row["learning_rate"])
        epochs = [entry["epoch"] for entry in row["history"]]
        axes[0].plot(epochs, [entry["validation_accuracy"] for entry in row["history"]], label=label)
        axes[1].plot(epochs, [entry["validation_loss"] for entry in row["history"]], label=label)
    axes[0].set_title("Validation accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Validation loss")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def validate_target_crop_sequence(crop_paths: Sequence[Path], suffix: str) -> None:
    """Reject partial, extra, duplicate, missing-position, or reordered rows."""
    expected = ["char_{:02d}_{}.png".format(index, suffix) for index in range(1, 11)]
    actual = [path.name for path in crop_paths]
    if actual != expected:
        raise ValueError(
            "Task-6 {} crops must be exactly ten ordered files char_01_{}.png "
            "through char_10_{}.png; found {}: {}".format(suffix, suffix, suffix, len(actual), actual)
        )


def validate_target_crop_pairs(gray_paths: Sequence[Path], binary_paths: Sequence[Path]) -> bool:
    """Both absent is optional; every partial or unmatched crop set is an error."""
    if not gray_paths and not binary_paths:
        return False
    validate_target_crop_sequence(gray_paths, "gray")
    validate_target_crop_sequence(binary_paths, "bin")
    return True


def validate_checkpoint_metadata(
    payload: Dict[str, Any], model_kind: str,
    previous: Optional[Dict[str, Any]], checkpoint: Path,
) -> None:
    """Check score ordering and shared per-ensemble preprocessing before use."""
    if payload.get("model_kind") != model_kind or payload.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint model kind/class ordering mismatch: {}".format(checkpoint))
    side = payload.get("input_size")
    normalization = payload.get("normalization")
    if not isinstance(side, int) or side < 4:
        raise ValueError("Invalid checkpoint input_size: {}".format(checkpoint))
    if not isinstance(normalization, dict) or set(normalization) != {"mean", "std"}:
        raise ValueError("Invalid checkpoint normalization fields: {}".format(checkpoint))
    if not all(isinstance(normalization[key], (float, int)) and math.isfinite(normalization[key]) for key in ("mean", "std")) or normalization["std"] <= 0:
        raise ValueError("Checkpoint normalization must be finite with positive std: {}".format(checkpoint))
    if previous is not None and (side != previous["input_size"] or normalization != previous["normalization"]):
        raise ValueError("Ensemble checkpoint input size/normalization mismatch: {}".format(checkpoint))


def evaluate_target_checkpoints(
    checkpoint_paths: Dict[str, Sequence[Path]],
    geometry: Dict[str, Any],
    device: torch.device,
    output: Path,
    expected_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run both preprocessing paths with fixed, already-trained checkpoints."""
    gray_paths = sorted((PROJECT_ROOT / "task_6" / "outputs" / "chars").glob("char_*_gray.png"))
    binary_paths = sorted((PROJECT_ROOT / "task_6" / "outputs" / "chars").glob("char_*_bin.png"))
    if not validate_target_crop_pairs(gray_paths, binary_paths):
        return {
            kind: {"available": False, "reason": "Task-6 grayscale and binary crops were not found",
                   "baseline_gray_inference": {"available": False, "reason": "Task-6 crops were not found"}}
            for kind in ["cnn", "mlp"]
        }
    target_results: Dict[str, Any] = {}
    for model_kind in ["cnn", "mlp"]:
        models = []
        payload = None
        for checkpoint in checkpoint_paths[model_kind]:
            previous = payload
            payload = torch.load(checkpoint, map_location=device)
            validate_checkpoint_metadata(payload, model_kind, previous, checkpoint)
            if expected_metadata is not None:
                expected = expected_metadata[model_kind]
                if payload["input_size"] != expected["input_size"] or (expected.get("normalization") is not None and payload["normalization"] != expected["normalization"]):
                    raise ValueError("Checkpoint metadata does not match saved experiment: {}".format(checkpoint))
            side = int(payload["input_size"])
            model = CharacterCNN(len(CLASS_NAMES)) if model_kind == "cnn" else RegressionMLP(side, len(CLASS_NAMES))
            model.load_state_dict(payload["state_dict"])
            model.to(device)
            models.append(model)
        if payload is None:
            raise RuntimeError("No checkpoint for {}".format(model_kind))
        baseline = infer_target_crops(models, gray_paths, side, payload["normalization"], device)
        aligned = infer_target_crops(models, binary_paths, side, payload["normalization"], device, geometry)
        aligned["baseline_gray_inference"] = baseline
        aligned["checkpoint_files"] = [str(path.relative_to(output)) for path in checkpoint_paths[model_kind]]
        aligned["checkpoint_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in checkpoint_paths[model_kind]
        }
        aligned["adaptation_policy"] = (
            "Single predefined label-free rule: largest binary component, "
            "median tuning-train bbox width/height, centered white canvas. "
            "No target label, target score, or held-out test measurement selects this rule. "
            "Original gray baseline is retained. This is an exploratory target-domain check."
        )
        target_results[model_kind] = aligned
        del models
    if gray_paths and binary_paths:
        side = 48
        baseline_images = prepare_target_images(gray_paths, side)
        aligned_images = prepare_target_images(binary_paths, side, geometry)
        fig, axes = plt.subplots(2, len(gray_paths), figsize=(12, 3.2), constrained_layout=True)
        for index in range(len(gray_paths)):
            axes[0, index].imshow(baseline_images[index, 0], cmap="gray", vmin=0, vmax=1)
            axes[1, index].imshow(aligned_images[index, 0], cmap="gray", vmin=0, vmax=1)
            axes[0, index].set_title("Position {}".format(index + 1), fontsize=8)
            for row in range(2):
                axes[row, index].axis("off")
        fig.suptitle("Original gray baseline (top) and fixed binary geometry alignment (bottom)", fontsize=11)
        fig.savefig(output / "target_preprocessing.png", dpi=220)
        plt.close(fig)
    return target_results


def run_inference_only(args: argparse.Namespace, device: torch.device) -> None:
    """Refresh target predictions without retraining or overwriting test results."""
    metrics_path = args.output / "task7_metrics.json"
    with metrics_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    seed_everything(int(summary["seed"]))
    with (args.output / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        training_paths = [args.data / row["relative_path"] for row in csv.DictReader(handle) if row["partition"] == "train"]
    geometry = fit_foreground_geometry(training_paths)
    checkpoint_paths = {}
    expected_metadata = {}
    for kind in ["cnn", "mlp"]:
        record = summary["test_results"][kind]
        side = int(record["input_size"])
        expected_metadata[kind] = {
            "input_size": side,
            "normalization": summary.get("target_predictions", {}).get(kind, {}).get("normalization"),
        }
        checkpoint_paths[kind] = [
            args.output / "{}_{}px_seed{}.pt".format(kind, side, summary["seed"] + 7000 + member * 37 + (0 if kind == "cnn" else 1))
            for member in range(int(record["ensemble_members"]))
        ]
    predictions = evaluate_target_checkpoints(checkpoint_paths, geometry, device, args.output, expected_metadata)
    summary["target_predictions"] = predictions
    summary["preprocessing"]["target_foreground_geometry"] = geometry
    for kind in ["cnn", "mlp"]:
        summary["test_results"][kind]["figure1_inference"] = predictions[kind]
        print("{} original={} aligned={}".format(kind, predictions[kind]["baseline_gray_inference"].get("predicted_string"), predictions[kind].get("predicted_string")))
    with (args.output / "target_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs-tune", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run a short one-resolution pipeline check")
    parser.add_argument("--inference-only", action="store_true", help="Refresh Figure-1 predictions from saved selected checkpoints; preserve held-out metrics")
    args = parser.parse_args()
    if args.smoke:
        args.epochs_tune = min(args.epochs_tune, 2)
        args.patience = 2

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if args.inference_only:
        run_inference_only(args, device)
        return
    print("Loading dataset from {}".format(args.data))
    paths, labels, digests, _ = discover_dataset(args.data)
    train_indices, validation_indices, test_indices = make_splits(labels, digests, args.seed)
    parts = (train_indices, validation_indices, test_indices)
    split_audit = audit_splits(parts, labels, digests)
    split_audit["source"] = str(args.data.resolve())
    split_audit["outer_training_fraction"] = float((len(train_indices) + len(validation_indices)) / len(paths))
    split_audit["outer_test_fraction"] = float(len(test_indices) / len(paths))
    split_audit["split_policy"] = "class-stratified grouped 75/25; exact decoded-pixel duplicates never cross partitions"
    split_audit["seed"] = args.seed
    with (args.output / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_audit, handle, indent=2)
    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "class", "partition", "pixel_sha256"])
        partition_by_index = {}
        for name, part in zip(["train", "validation", "test"], parts):
            partition_by_index.update({int(i): name for i in part})
        for index, path in enumerate(paths):
            writer.writerow([str(path.relative_to(args.data)), CLASS_NAMES[int(labels[index])], partition_by_index[index], digests[index]])
    print("Split sizes: train={} validation={} test={}".format(*[len(part) for part in parts]))
    print("Exact duplicate overlap: {}".format(split_audit["duplicate_digest_intersections"]))

    sides = [32] if args.smoke else [32, 48]
    learning_rates = [1e-3] if args.smoke else [1e-3, 3e-4]
    tuning_results: List[Dict[str, Any]] = []
    cached: Dict[int, Tuple[TensorDataset, TensorDataset, TensorDataset, Dict[str, float]]] = {}
    for side in sides:
        print("Preprocessing to {}x{}".format(side, side))
        raw_images = load_resized(paths, side)
        cached[side] = make_tensor_splits(raw_images, labels, parts)
        del raw_images
        train_ds, val_ds, _test_ds, normalization = cached[side]
        for model_kind in ["cnn", "mlp"]:
            for learning_rate in learning_rates:
                run_seed = args.seed + side * 100 + int(learning_rate * 100000) + (0 if model_kind == "cnn" else 1)
                print("Tuning {}: size={} lr={:g}".format(model_kind.upper(), side, learning_rate), flush=True)
                result = train_candidate(
                    model_kind, side, learning_rate, train_ds, val_ds, device, args, run_seed
                )
                result["normalization"] = normalization
                tuning_results.append(result)
                print("  validation accuracy={:.4f}; best epoch={}".format(
                    result["validation_accuracy"], result["best_epoch"]
                ), flush=True)

    public_tuning = [
        {key: value for key, value in row.items() if key not in ("state_dict", "history")}
        for row in tuning_results
    ]
    with (args.output / "tuning_results.json").open("w", encoding="utf-8") as handle:
        json.dump(public_tuning, handle, indent=2)
    write_csv(args.output / "tuning_results.csv", public_tuning)
    save_history_plot(tuning_results, args.output / "tuning_curves.png")

    selected: Dict[str, Dict[str, Any]] = {}
    for model_kind in ["cnn", "mlp"]:
        candidates = [row for row in tuning_results if row["model_kind"] == model_kind]
        selected[model_kind] = max(
            candidates,
            key=lambda row: (row["validation_accuracy"], -row["input_size"], -row["learning_rate"]),
        )

    final_results: Dict[str, Any] = {}
    final_checkpoints: Dict[str, List[Path]] = {}
    for model_kind in ["cnn", "mlp"]:
        choice = selected[model_kind]
        side = int(choice["input_size"])
        train_ds, val_ds, test_ds, normalization = cached[side]
        # After selection, refit on every image in the mandated 75% outer
        # training partition (train + validation). Validation is used only to
        # select the configuration and epoch budget.
        full_train_ds = TensorDataset(
            torch.cat([train_ds.tensors[0], val_ds.tensors[0]], dim=0),
            torch.cat([train_ds.tensors[1], val_ds.tensors[1]], dim=0),
        )
        chosen_epochs = max(1, int(choice["best_epoch"]))
        n_models = 1 if model_kind == "cnn" else (1 if args.smoke else 3)
        models: List[nn.Module] = []
        model_checkpoints: List[Path] = []
        total_training_seconds = 0.0
        for ensemble_index in range(n_models):
            fit_seed = args.seed + 7000 + ensemble_index * 37 + (0 if model_kind == "cnn" else 1)
            print("Final fit {} model {}/{}: {}px, {} epochs".format(
                model_kind.upper(), ensemble_index + 1, n_models, side, chosen_epochs
            ), flush=True)
            model, elapsed = train_final_model(
                model_kind,
                side,
                float(choice["learning_rate"]),
                chosen_epochs,
                full_train_ds,
                device,
                args,
                fit_seed,
            )
            total_training_seconds += elapsed
            models.append(model)
            checkpoint = args.output / "{}_{}px_seed{}.pt".format(model_kind, side, fit_seed)
            torch.save(
                {
                    "model_kind": model_kind,
                    "input_size": side,
                    "class_names": CLASS_NAMES,
                    "normalization": normalization,
                    "state_dict": model.state_dict(),
                    "seed": fit_seed,
                },
                checkpoint,
            )
            model_checkpoints.append(checkpoint)
        y_true, scores, inference_seconds = predict_scores(
            models, test_ds, device, args.batch_size * 2
        )
        y_pred = scores.argmax(axis=1)
        metrics = classification_metrics(y_true, y_pred)
        if model_kind == "mlp":
            target_matrix = np.eye(len(CLASS_NAMES), dtype=np.float32)[y_true]
            metrics["test_mse"] = float(np.mean((scores - target_matrix) ** 2))
        metrics.update(
            {
                "model": "CNN" if model_kind == "cnn" else "MLP one-hot regression",
                "input_size": side,
                "selected_learning_rate": float(choice["learning_rate"]),
                "selected_epochs": chosen_epochs,
                "ensemble_members": n_models,
                "training_seconds_sum": total_training_seconds,
                "test_inference_seconds": inference_seconds,
                "test_inference_ms_per_image": 1000.0 * inference_seconds / len(test_ds),
                "parameter_count_per_model": int(sum(p.numel() for p in models[0].parameters())),
                "test_images": int(len(test_ds)),
            }
        )
        final_results[model_kind] = metrics
        final_checkpoints[model_kind] = model_checkpoints
        save_confusion_matrix(
            np.asarray(metrics["confusion_matrix"]),
            args.output / "{}_confusion_matrix.png".format(model_kind),
            "{} test confusion matrix".format(metrics["model"]),
        )
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_rows = []
    for key in ["cnn", "mlp"]:
        row = final_results[key]
        comparison_rows.append(
            {
                "method": row["model"],
                "input_size": row["input_size"],
                "test_accuracy": row["accuracy"],
                "macro_precision": row["macro_precision"],
                "macro_recall": row["macro_recall"],
                "macro_f1": row["macro_f1"],
                "train_seconds_sum": row["training_seconds_sum"],
                "test_inference_ms_per_image": row["test_inference_ms_per_image"],
                "parameters_per_model": row["parameter_count_per_model"],
                "ensemble_members": row["ensemble_members"],
            }
        )
    write_csv(args.output / "method_comparison.csv", comparison_rows)

    target_geometry = fit_foreground_geometry([paths[int(index)] for index in train_indices])
    target_predictions = evaluate_target_checkpoints(final_checkpoints, target_geometry, device, args.output)
    for kind in ["cnn", "mlp"]:
        final_results[kind]["figure1_inference"] = target_predictions[kind]

    with (args.output / "target_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(target_predictions, handle, indent=2)

    summary = {
        "dataset": str(args.data.resolve()),
        "classes": CLASS_NAMES,
        "device": str(device),
        "seed": args.seed,
        "split": split_audit,
        "preprocessing": {
            "source_mode": "grayscale 8-bit",
            "resize": "bilinear to square; candidate sides {} px".format(sides),
            "normalization": "global training-partition pixel mean/std z-score; identical statistics applied to validation and test",
            "augmentation": "none",
            "target_foreground_geometry": target_geometry,
            "duplicate_control": "identical decoded grayscale pixel arrays are kept in a single partition",
        },
        "tuning_grid": {
            "input_sizes": sides, "learning_rates": learning_rates, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "max_epochs": args.epochs_tune, "early_stopping_patience": args.patience,
            "candidate_seed_rule": "seed + side*100 + int(lr*100000) + (1 for MLP, 0 for CNN)",
            "selection": "maximize validation accuracy; ties: smaller input side, then smaller learning rate",
            "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=initial_lr/20)",
            "final_scheduler": "CosineAnnealingLR(T_max=selected_best_epoch)",
            "optimizer": "AdamW(betas=(0.9,0.999), eps=1e-8); weight_decay recorded above",
        },
        "selected_validation_models": {
            key: {
                "input_size": int(value["input_size"]),
                "learning_rate": float(value["learning_rate"]),
                "validation_accuracy": float(value["validation_accuracy"]),
                "selected_epochs": int(value["best_epoch"]),
                "validation_members": 1,
                "run_seed": int(value["run_seed"]),
            }
            for key, value in selected.items()
        },
        "architectures": {
            "cnn": "Conv(1,32)-BN-ReLU x2-MaxPool; Conv(32,64)-BN-ReLU x2-MaxPool; Conv(64,128)-BN-ReLU; global average pool; linear 7 logits; cross-entropy",
            "mlp": "Flatten-(Linear 512-BN-GELU-Dropout 0.25)-(Linear 256-BN-GELU-Dropout 0.25)-Linear 7 scores; one-hot target MSE regression; class is argmax(score)",
        },
        "test_results": final_results,
        "target_predictions": target_predictions,
        "limitations": [
            "Duplicate grouping uses exact decoded grayscale pixels; visually similar but non-identical files may remain across partitions.",
            "The supplied dataset has balanced labels and centered synthetic glyphs; results may not transfer to differently segmented real chip characters.",
        ],
    }
    with (args.output / "task7_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nHeld-out test results")
    for row in comparison_rows:
        print("{}: accuracy={:.4f}, macro-F1={:.4f}, test inference={:.3f} ms/image".format(
            row["method"], row["test_accuracy"], row["macro_f1"], row["test_inference_ms_per_image"]
        ))
    print("Artifacts: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()

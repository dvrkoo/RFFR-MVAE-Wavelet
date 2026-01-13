import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import ImageFile
from sklearn.metrics import roc_auc_score, roc_curve

ImageFile.LOAD_TRUNCATED_IMAGES = True

from configs.config import config
from models.model_detector import RFFRL
from utils.dataset import Deepfake_Dataset


def load_checkpoint_with_anomaly_data(checkpoint_path):
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    anomaly_mode = checkpoint.get("anomaly_mode", False)
    if not anomaly_mode:
        raise ValueError(
            f"Checkpoint was not trained in anomaly detection mode.\n"
            f"Please use test_model.py for standard supervised testing."
        )

    anomaly_center = checkpoint.get("anomaly_center", None)
    if anomaly_center is None:
        raise ValueError(
            f"Checkpoint missing 'anomaly_center' field.\n"
            f"This checkpoint may have been saved before anomaly detection support was added.\n"
            f"Please retrain with anomaly_detection_mode=True."
        )

    anomaly_center = anomaly_center.cuda()

    model = RFFRL().cuda()
    model.dd.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "aucs": checkpoint.get("aucs", []),
        "best_aucs": checkpoint.get("best_aucs", []),
        "center_loss_weight": checkpoint.get("center_loss_weight", 1.0),
        "anomaly_score_percentile": checkpoint.get("anomaly_score_percentile", 95),
        "path": str(checkpoint_path),
    }

    print(f"Epoch: {metadata['epoch']}")
    print(f"Center loss weight: {metadata['center_loss_weight']}")
    print(f"Anomaly score percentile: {metadata['anomaly_score_percentile']}")

    if metadata["aucs"]:
        dataset_names = ["Mixed", "DF", "F2F", "FSW", "NT", "DFD", "CelebDF"]
        print("Validation AUCs (distance-based):")
        for i, auc in enumerate(metadata["aucs"][:5]):
            name = dataset_names[i] if i < len(dataset_names) else f"Dataset_{i}"
            print(f"  {name}: {auc:.4f}")

    return model, anomaly_center, metadata


def compute_features_and_distances(model, dataloader, anomaly_center):
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            _, features = model(images)

            batch_size = features.size(0)
            center_expanded = anomaly_center.unsqueeze(0).expand(batch_size, -1)
            distances = torch.norm(features - center_expanded, p=2, dim=1)

            all_distances.append(distances.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_distances = np.concatenate(all_distances)
    all_labels = np.concatenate(all_labels)

    return all_distances, all_labels


def compute_threshold_from_real(distances, labels, percentile=95):
    real_mask = labels == 0
    real_distances = distances[real_mask]

    if len(real_distances) == 0:
        raise ValueError("No real samples found in dataset for threshold computation")

    threshold = np.percentile(real_distances, percentile)

    print(f"\nReal sample distance statistics:")
    print(f"  Mean: {real_distances.mean():.4f}")
    print(f"  Std: {real_distances.std():.4f}")
    print(f"  Min: {real_distances.min():.4f}")
    print(f"  Max: {real_distances.max():.4f}")
    print(f"  {percentile}th percentile (threshold): {threshold:.4f}")

    return threshold


def evaluate_anomaly_detection(distances, labels, threshold):
    predictions = (distances > threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    auc = roc_auc_score(labels, distances)

    results = {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": len(labels),
        "threshold": float(threshold),
    }

    return results


def print_results(results, dataset_name="Test"):
    print(f"\n{dataset_name} Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  FPR: {results['fpr']:.4f}")
    print(f"  Threshold: {results['threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {results['tp']}, TN: {results['tn']}")
    print(f"  FP: {results['fp']}, FN: {results['fn']}")


def load_test_data(label_paths, dataset_base):
    test_dict = []

    if isinstance(label_paths, str):
        label_paths = [label_paths]

    for label_path in label_paths:
        path = dataset_base + label_path
        if os.path.exists(path):
            with open(path) as f:
                test_dict += json.load(f)
            print(f"Loaded {len(test_dict)} samples from {label_path}")
        else:
            print(f"Warning: {path} not found, skipping")

    return test_dict


def main():
    parser = argparse.ArgumentParser(description="Test anomaly detection model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint trained with anomaly_detection_mode=True",
    )
    parser.add_argument(
        "--val-label",
        type=str,
        default=None,
        help="Path to validation label (for computing threshold). Uses config.val_label_path if not provided.",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default=None,
        help="Path to test label. Uses config.test_label_path if not provided.",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=None,
        help="Percentile of real distances to use as threshold (default: use checkpoint value)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for testing (default: 32)",
    )

    args = parser.parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    model, anomaly_center, metadata = load_checkpoint_with_anomaly_data(args.checkpoint)

    percentile = (
        args.percentile
        if args.percentile is not None
        else metadata["anomaly_score_percentile"]
    )

    val_label_path = args.val_label if args.val_label else config.val_label_path
    test_label_path = args.test_label if args.test_label else config.test_label_path

    print("\n" + "=" * 80)
    print("STEP 1: Computing threshold from validation set")
    print("=" * 80)

    val_dict = load_test_data(val_label_path, config.dataset_base)
    val_dataset = Deepfake_Dataset(val_dict, train=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    val_distances, val_labels = compute_features_and_distances(
        model, val_dataloader, anomaly_center
    )
    threshold = compute_threshold_from_real(val_distances, val_labels, percentile)

    val_results = evaluate_anomaly_detection(val_distances, val_labels, threshold)
    print_results(val_results, "Validation")

    print("\n" + "=" * 80)
    print("STEP 2: Testing on test set")
    print("=" * 80)

    test_dict = load_test_data(test_label_path, config.dataset_base)
    test_dataset = Deepfake_Dataset(test_dict, train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_distances, test_labels = compute_features_and_distances(
        model, test_dataloader, anomaly_center
    )
    test_results = evaluate_anomaly_detection(test_distances, test_labels, threshold)
    print_results(test_results, "Test")

    fake_mask = test_labels == 1
    if np.sum(fake_mask) > 0:
        fake_distances = test_distances[fake_mask]
        print(f"\nFake sample distance statistics:")
        print(f"  Mean: {fake_distances.mean():.4f}")
        print(f"  Std: {fake_distances.std():.4f}")
        print(f"  Min: {fake_distances.min():.4f}")
        print(f"  Max: {fake_distances.max():.4f}")

    print("\n" + "=" * 80)
    print("Testing complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

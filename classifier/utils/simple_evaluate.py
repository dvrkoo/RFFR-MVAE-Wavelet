from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from configs.config import config


def eval_one_dataset(
    valid_dataloader, model, dataset_name="Unknown", anomaly_center=None
):
    """Evaluate model on a single dataset and return metrics.

    Args:
        valid_dataloader: DataLoader for validation data
        model: Model to evaluate
        dataset_name: Name of dataset for logging
        anomaly_center: Center tensor for anomaly detection mode (if None, uses supervised mode)

    Returns:
        metrics: Dictionary containing:
            - auc: AUC score
            - accuracy: Overall accuracy
            - real_accuracy: Accuracy on real samples (label=0)
            - fake_accuracy: Accuracy on fake samples (label=1)
            - real_count: Number of real samples
            - fake_count: Number of fake samples
        distance_stats: Distance statistics for anomaly detection mode (or None)
    """
    model.eval()

    score_list = []
    label_list = []
    pred_list = []

    real_distances = []
    fake_distances = []

    with torch.no_grad():
        for batch_idx, (input, unnormed, label) in enumerate(
            tqdm(valid_dataloader, desc=f"Eval {dataset_name}", unit="batch")
        ):
            input = Variable(input).cuda()
            unnormed = Variable(unnormed).cuda()

            features, cls_out, _ = model(unnormed, input, test=True)

            if config.anomaly_detection_mode and anomaly_center is not None:
                batch_size = features.size(0)
                center = anomaly_center.unsqueeze(0).expand(batch_size, -1).cuda()
                distances = torch.norm(features - center, p=2, dim=1)

                distances_np = distances.cpu().numpy()
                label_np = label.numpy() if isinstance(label, torch.Tensor) else label

                real_mask = label_np == 0
                fake_mask = label_np == 1
                real_distances.append(distances_np[real_mask])
                fake_distances.append(distances_np[fake_mask])

                scores = distances_np
                # For anomaly detection, higher distance = more likely fake
                preds = (distances_np > np.median(distances_np)).astype(int)
            else:
                prob = F.softmax(cls_out, dim=1).cpu().numpy()
                scores = prob[:, 1]  # Probability of fake class
                preds = np.argmax(prob, axis=1)  # Predicted class (0=real, 1=fake)

            score_list.append(scores)
            pred_list.append(preds)
            label_list.append(
                label.numpy() if isinstance(label, torch.Tensor) else label
            )

    score_list = np.concatenate(score_list)
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    # Calculate metrics
    label_array = label_list
    unique_labels = np.unique(label_array)

    # Initialize metrics
    metrics = {
        "auc": 0.0,
        "accuracy": 0.0,
        "real_accuracy": 0.0,
        "fake_accuracy": 0.0,
        "real_count": 0,
        "fake_count": 0,
    }

    if len(unique_labels) < 2:
        print(
            f"Warning: Only one class present in {dataset_name} (class {unique_labels[0]}). Cannot compute AUC."
        )
        print(f"Label distribution: {np.bincount(label_array.astype(int))}")
    else:
        # AUC Score
        metrics["auc"] = roc_auc_score(label_list, score_list)

        # Overall Accuracy
        metrics["accuracy"] = np.mean(pred_list == label_list)

        # Real samples accuracy (label = 0)
        real_mask = label_list == 0
        if np.sum(real_mask) > 0:
            metrics["real_accuracy"] = np.mean(pred_list[real_mask] == label_list[real_mask])
            metrics["real_count"] = int(np.sum(real_mask))

        # Fake samples accuracy (label = 1)
        fake_mask = label_list == 1
        if np.sum(fake_mask) > 0:
            metrics["fake_accuracy"] = np.mean(pred_list[fake_mask] == label_list[fake_mask])
            metrics["fake_count"] = int(np.sum(fake_mask))

        mode_str = (
            "Anomaly"
            if (config.anomaly_detection_mode and anomaly_center is not None)
            else "Supervised"
        )
        print(f"{dataset_name} Metrics ({mode_str}):")
        print(f"  AUC:            {metrics['auc']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  Real Accuracy:  {metrics['real_accuracy']:.4f} ({metrics['real_count']} samples)")
        print(f"  Fake Accuracy:  {metrics['fake_accuracy']:.4f} ({metrics['fake_count']} samples)")

        if (
            config.anomaly_detection_mode
            and anomaly_center is not None
            and len(real_distances) > 0
            and len(fake_distances) > 0
        ):
            real_distances_flat = (
                np.concatenate(real_distances) if real_distances else np.array([])
            )
            fake_distances_flat = (
                np.concatenate(fake_distances) if fake_distances else np.array([])
            )

            if len(real_distances_flat) > 0 and len(fake_distances_flat) > 0:
                real_mean = np.mean(real_distances_flat)
                fake_mean = np.mean(fake_distances_flat)
                separation = fake_mean - real_mean
                print(
                    f"  Real mean distance: {real_mean:.4f}, Fake mean distance: {fake_mean:.4f}, Separation: {separation:.4f}"
                )

            real_distances = real_distances_flat
            fake_distances = fake_distances_flat

    torch.cuda.empty_cache()

    distance_stats = (
        {"real_distances": real_distances, "fake_distances": fake_distances}
        if (config.anomaly_detection_mode and anomaly_center is not None)
        else None
    )

    return metrics, distance_stats


def eval_multiple_dataset(dataloaders, model, anomaly_center=None):
    """Evaluate model on multiple datasets and return metrics.

    Args:
        dataloaders: List of dataloaders
        model: Model to evaluate
        anomaly_center: Center tensor for anomaly detection mode (if None, uses supervised mode)

    Returns:
        all_metrics: List of metrics dictionaries
        all_distance_stats: List of distance statistics dicts (None for supervised mode)
    """
    all_metrics = []
    all_distance_stats = []
    dataset_names = ["Mixed_Val", "DF", "F2F", "FSW", "NT"]

    for i, dataloader in enumerate(dataloaders):
        dataset_name = dataset_names[i] if i < len(dataset_names) else f"Dataset_{i}"
        metrics, distance_stats = eval_one_dataset(
            dataloader, model, dataset_name, anomaly_center
        )
        all_metrics.append(metrics)
        all_distance_stats.append(distance_stats)

    return all_metrics, all_distance_stats

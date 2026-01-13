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
    """Evaluate model on a single dataset and return AUC score.

    Args:
        valid_dataloader: DataLoader for validation data
        model: Model to evaluate
        dataset_name: Name of dataset for logging
        anomaly_center: Center tensor for anomaly detection mode (if None, uses supervised mode)
    """
    model.eval()

    score_list = []
    label_list = []

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
            else:
                prob = F.softmax(cls_out, dim=1).cpu().numpy()[:, 1]
                scores = prob

            score_list.append(scores)
            label_list.append(
                label.numpy() if isinstance(label, torch.Tensor) else label
            )

    score_list = np.concatenate(score_list)
    label_list = np.concatenate(label_list)
    label_array = label_list
    unique_labels = np.unique(label_array)
    if len(unique_labels) < 2:
        print(
            f"Warning: Only one class present in {dataset_name} (class {unique_labels[0]}). Cannot compute AUC."
        )
        print(f"Label distribution: {np.bincount(label_array.astype(int))}")
        auc_score = 0.0
    else:
        auc_score = roc_auc_score(label_list, score_list)
        mode_str = (
            "Anomaly"
            if (config.anomaly_detection_mode and anomaly_center is not None)
            else "Supervised"
        )
        print(f"{dataset_name} AUC ({mode_str}): {auc_score:.4f}")

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

    return auc_score, distance_stats


def eval_multiple_dataset(dataloaders, model, anomaly_center=None):
    """Evaluate model on multiple datasets and return list of AUC scores.

    Args:
        dataloaders: List of dataloaders
        model: Model to evaluate
        anomaly_center: Center tensor for anomaly detection mode (if None, uses supervised mode)

    Returns:
        aucs: List of AUC scores
        all_distance_stats: List of distance statistics dicts (None for supervised mode)
    """
    aucs = []
    all_distance_stats = []
    dataset_names = ["Mixed_Val", "DF", "F2F", "FSW", "NT"]

    for i, dataloader in enumerate(dataloaders):
        dataset_name = dataset_names[i] if i < len(dataset_names) else f"Dataset_{i}"
        auc, distance_stats = eval_one_dataset(
            dataloader, model, dataset_name, anomaly_center
        )
        aucs.append(auc)
        all_distance_stats.append(distance_stats)

    return aucs, all_distance_stats

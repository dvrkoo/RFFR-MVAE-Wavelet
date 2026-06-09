#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_configuration(config_path=None):
    if config_path:
        from configs.config_loader import load_config

        cfg = load_config(config_path=config_path, flat_compat=True)
    else:
        from configs.config import config as cfg

    import models.model_detector as model_detector
    import models.model_mae as model_mae
    import models.model_mae_vae as model_mae_vae
    import utils.simple_evaluate as simple_evaluate
    import utils.wavelet_utils as wavelet_utils

    model_detector.config = cfg
    model_mae.config = cfg
    model_mae_vae.config = cfg
    simple_evaluate.config = cfg
    wavelet_utils.config = cfg
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single RFFR classifier test entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Classifier checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint/rffr/best_model")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["latest", "highest_auc", "mixed"],
        default="highest_auc",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets from config.test_label_path to evaluate, e.g. DF F2F FSW NT DFD CelebDF Mixed",
    )
    parser.add_argument("--fake-label", type=str, default=None, help="Custom fake label JSON")
    parser.add_argument("--real-label", type=str, default=None, help="Custom real label JSON")
    parser.add_argument("--dataset-name", type=str, default="Custom")
    parser.add_argument("--samples", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./test_results")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--list-checkpoints", action="store_true")
    return parser.parse_args()


def set_reproducibility(cfg, args):
    seed = args.seed if args.seed is not None else cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu else cfg.gpus
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_json(path):
    with open(path) as f:
        return json.load(f)


def resolve_label_path(dataset_base, path):
    label_path = Path(path)
    if label_path.is_absolute():
        return str(label_path)
    return str(Path(dataset_base) / label_path)


def find_checkpoints(checkpoint_dir):
    base = Path(checkpoint_dir)
    if not base.exists():
        return []
    return sorted(base.rglob("*.pth.tar"), key=lambda p: p.stat().st_mtime, reverse=True)


def select_checkpoint(args):
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoints = find_checkpoints(args.checkpoint_dir)
    if args.list_checkpoints:
        for path in checkpoints:
            print(path)
        raise SystemExit(0)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {args.checkpoint_dir}")

    if args.strategy == "latest":
        return checkpoints[0]
    if args.strategy == "mixed":
        mixed = [p for p in checkpoints if "0__AUC" in p.name or "Mixed" in p.name]
        return mixed[0] if mixed else checkpoints[0]

    auc_checkpoints = [p for p in checkpoints if "AUC" in p.name]
    if not auc_checkpoints:
        return checkpoints[0]

    def auc_value(path):
        parts = path.name.replace(".pth.tar", "").split("_")
        for i, part in enumerate(parts):
            if part == "AUC" and i + 1 < len(parts):
                try:
                    return float(parts[i + 1])
                except ValueError:
                    return -1.0
        return -1.0

    return max(auc_checkpoints, key=auc_value)


def load_model(checkpoint_path):
    from models.model_detector import RFFRL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFFRL().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    try:
        model.dd.load_state_dict(state_dict)
    except RuntimeError:
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("backbone_") or key.startswith("classifier"):
                remapped[f"dd.{key}"] = value
            else:
                remapped[key] = value
        model.load_state_dict(remapped, strict=False)

    model.eval()
    return model, checkpoint


def balanced_dataset(fake_items, real_items, max_samples):
    n = min(len(fake_items), len(real_items), max_samples)
    fake_subset = random.sample(fake_items, n)
    real_subset = random.sample(real_items, n)
    items = real_subset + fake_subset
    random.shuffle(items)
    return items, len(real_subset), len(fake_subset)


def config_datasets(cfg, requested=None):
    names = ["DF", "F2F", "FSW", "NT", "FS", "DFD", "CelebDF"]
    entries = []

    real_default = []
    for real_path in cfg.real_test_label_path:
        real_default.extend(load_json(resolve_label_path(cfg.dataset_base, real_path)))

    fake_by_name = []
    for i, fake_path in enumerate(cfg.test_label_path):
        name = names[i] if i < len(names) else f"Dataset_{i + 1}"
        fake_by_name.append((name, fake_path))

    if requested is None or "Mixed" in requested:
        mixed_fake = []
        for _, fake_path in fake_by_name[:4]:
            mixed_fake.extend(load_json(resolve_label_path(cfg.dataset_base, fake_path)))
        entries.append(("Mixed", mixed_fake, real_default))

    for name, fake_path in fake_by_name:
        if requested and name not in requested:
            continue
        real_items = real_default
        specific_attr = f"{name.lower()}_real_test_label_path"
        if hasattr(cfg, specific_attr):
            real_items = load_json(resolve_label_path(cfg.dataset_base, getattr(cfg, specific_attr)))
        fake_items = load_json(resolve_label_path(cfg.dataset_base, fake_path))
        entries.append((name, fake_items, real_items))

    return entries


def custom_dataset(cfg, args):
    if not (args.fake_label and args.real_label):
        return None
    fake_items = load_json(resolve_label_path(cfg.dataset_base, args.fake_label))
    real_items = load_json(resolve_label_path(cfg.dataset_base, args.real_label))
    return [(args.dataset_name, fake_items, real_items)]


def evaluate_dataset(name, fake_items, real_items, model, args):
    from utils.dataset import Deepfake_Dataset
    from utils.simple_evaluate import eval_one_dataset

    items, real_count, fake_count = balanced_dataset(fake_items, real_items, args.samples)
    print(f"{name}: {real_count} real + {fake_count} fake = {len(items)} total")

    dataloader = DataLoader(
        Deepfake_Dataset(items, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metrics, _ = eval_one_dataset(dataloader, model, name)
    return metrics


def main():
    args = parse_args()
    cfg = load_configuration(args.config)
    set_reproducibility(cfg, args)

    checkpoint_path = select_checkpoint(args)
    print(f"Using checkpoint: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path)

    datasets = custom_dataset(cfg, args)
    if datasets is None:
        datasets = config_datasets(cfg, args.datasets)
    if not datasets:
        raise RuntimeError("No datasets selected")

    results = {}
    for name, fake_items, real_items in datasets:
        results[name] = evaluate_dataset(name, fake_items, real_items, model, args)

    print("\nResults")
    print("=" * 80)
    for name, metrics in results.items():
        print(
            f"{name:<12} AUC={metrics['auc']:.4f} "
            f"ACC={metrics['accuracy']:.4f} "
            f"real={metrics['real_count']} fake={metrics['fake_count']}"
        )

    if args.save_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        serializable_results = {
            name: {key: float(value) if isinstance(value, np.floating) else value for key, value in metrics.items()}
            for name, metrics in results.items()
        }
        output = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(checkpoint_path),
            "epoch": checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
            "datasets": list(serializable_results.keys()),
            "results": serializable_results,
        }
        output_path = output_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

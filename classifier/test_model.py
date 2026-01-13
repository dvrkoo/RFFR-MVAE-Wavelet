import os
import sys
import json
import glob
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from configs.config import config
from models.model_detector import RFFRL
from utils.dataset import Deepfake_Dataset
from utils.simple_evaluate import eval_one_dataset


class CheckpointManager:
    def __init__(self, base_dir="./checkpoint/rffr"):
        self.base_dir = Path(base_dir)

    def discover_runs(self):
        if not self.base_dir.exists():
            return []

        runs = []
        for run_dir in sorted(self.base_dir.iterdir()):
            if run_dir.is_dir():
                best_model_path = run_dir / "best_model"
                if best_model_path.exists():
                    runs.append(
                        {
                            "path": run_dir,
                            "name": run_dir.name,
                            "modified": run_dir.stat().st_mtime,
                            "architecture": self._extract_architecture(run_dir.name),
                            "protocol": self._extract_protocol(run_dir.name),
                            "timestamp": self._extract_timestamp(run_dir.name),
                        }
                    )

        return sorted(runs, key=lambda x: x["modified"], reverse=True)

    def _extract_architecture(self, name):
        parts = name.split("_")
        if len(parts) >= 2:
            if parts[0].endswith("branch"):
                return "_".join(parts[:3])
        return "unknown"

    def _extract_protocol(self, name):
        if "F2F_All" in name:
            return "F2F_All"
        elif "DF_All" in name:
            return "DF_All"
        elif "FSW_All" in name:
            return "FSW_All"
        elif "NT_All" in name:
            return "NT_All"
        elif "Mixed_All" in name:
            return "Mixed_All"
        return "unknown"

    def _extract_timestamp(self, name):
        parts = name.split("_")
        for i, part in enumerate(parts):
            if "-" in part and ":" in part:
                if i + 1 < len(parts):
                    return f"{part}_{parts[i+1]}"
        return "unknown"

    def get_latest_run(self):
        runs = self.discover_runs()
        return runs[0] if runs else None

    def get_run_by_name(self, name):
        for run in self.discover_runs():
            if name in run["name"]:
                return run
        return None

    def list_checkpoints(self, run_dir):
        best_model_dir = Path(run_dir) / "best_model"

        if not best_model_dir.exists():
            return {}

        checkpoint_dirs = list(best_model_dir.iterdir())
        if not checkpoint_dirs:
            return {}

        checkpoint_dir = checkpoint_dirs[0]
        checkpoint_files = list(checkpoint_dir.glob("*.pth.tar"))

        checkpoints = {
            "dir": checkpoint_dir,
            "files": [],
            "by_dataset": {},
            "periodic": [],
        }

        for ckpt_file in checkpoint_files:
            info = self._parse_checkpoint_filename(ckpt_file.name)
            info["path"] = ckpt_file
            checkpoints["files"].append(info)

            if info["dataset_id"] == 0:
                checkpoints["periodic"].append(info)
            else:
                dataset_name = info["dataset_name"]
                if dataset_name not in checkpoints["by_dataset"]:
                    checkpoints["by_dataset"][dataset_name] = []
                checkpoints["by_dataset"][dataset_name].append(info)

        for dataset in checkpoints["by_dataset"]:
            checkpoints["by_dataset"][dataset].sort(
                key=lambda x: x["auc"], reverse=True
            )

        checkpoints["periodic"].sort(key=lambda x: x["epoch"], reverse=True)

        return checkpoints

    def _parse_checkpoint_filename(self, filename):
        parts = filename.replace(".pth.tar", "").split("_")

        dataset_id = int(parts[0])
        dataset_names = {0: "Periodic", 1: "Mixed", 2: "F2F", 3: "FSW", 4: "NT"}

        info = {
            "filename": filename,
            "dataset_id": dataset_id,
            "dataset_name": dataset_names.get(dataset_id, "Unknown"),
            "auc": None,
            "epoch": None,
        }

        if dataset_id > 0:
            try:
                info["auc"] = float(parts[2])
                info["epoch"] = int(parts[3])
            except (IndexError, ValueError):
                pass
        else:
            try:
                info["epoch"] = int(parts[2])
            except (IndexError, ValueError):
                pass

        return info

    def get_best_checkpoint(self, run_dir, strategy="highest_auc"):
        checkpoints = self.list_checkpoints(run_dir)

        if not checkpoints or not checkpoints["files"]:
            return None

        if strategy == "highest_auc":
            dataset_ckpts = []
            for dataset, ckpts in checkpoints["by_dataset"].items():
                if ckpts:
                    dataset_ckpts.append(ckpts[0])

            if dataset_ckpts:
                best = max(dataset_ckpts, key=lambda x: x["auc"] or 0)
                return best["path"]

        elif strategy == "latest_epoch":
            all_ckpts = checkpoints["files"]
            if all_ckpts:
                best = max(all_ckpts, key=lambda x: x["epoch"] or 0)
                return best["path"]

        elif strategy == "mixed":
            if "Mixed" in checkpoints["by_dataset"]:
                return checkpoints["by_dataset"]["Mixed"][0]["path"]

        return checkpoints["files"][0]["path"] if checkpoints["files"] else None

    def load_model(self, checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location="cuda", weights_only=False
        )

        model = RFFRL().cuda()
        model.dd.load_state_dict(checkpoint["state_dict"])
        model.eval()

        metadata = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "aucs": checkpoint.get("aucs", []),
            "best_aucs": checkpoint.get("best_aucs", []),
            "path": str(checkpoint_path),
        }

        print(f"Epoch: {metadata['epoch']}")
        if metadata["aucs"]:
            dataset_names = ["Mixed", "DF", "F2F", "FSW", "NT", "DFD", "CelebDF"]
            print("Validation AUCs:")
            for i, auc in enumerate(metadata["aucs"][:5]):
                name = dataset_names[i] if i < len(dataset_names) else f"Dataset_{i}"
                print(f"  {name}: {auc:.4f}")

        return model, metadata


class TestDataManager:
    def __init__(self):
        self.dataset_base = config.dataset_base
        self.test_label_paths = config.test_label_path
        self.real_test_label_paths = config.real_test_label_path
        self.dataset_names = ["DF", "F2F", "FSW", "NT", "FS", "DFD", "CelebDF"]

    def load_real_test_data(self):
        test_real_dict = []
        for real_test_dataset in self.real_test_label_paths:
            path = self.dataset_base + real_test_dataset
            with open(path) as f:
                test_real_dict += json.load(f)

        print(f"Loaded {len(test_real_dict)} real test samples")
        return test_real_dict

    def create_balanced_testset(self, fake_dict, real_dict, max_samples=700):
        num_samples = min(len(fake_dict), len(real_dict), max_samples)

        test_fake_subset = random.sample(fake_dict, num_samples)
        test_real_subset = random.sample(real_dict, num_samples)

        balanced_dict = test_real_subset + test_fake_subset
        random.shuffle(balanced_dict)

        real_count = sum(1 for item in balanced_dict if item["label"] == 0)
        fake_count = sum(1 for item in balanced_dict if item["label"] == 1)

        return balanced_dict, real_count, fake_count

    def create_balanced_testset_by_video(
        self,
        fake_dict,
        real_dict,
        max_samples=700,
        frames_per_video=10,
        uniform_sampling=True,
    ):
        from collections import defaultdict

        fake_videos = defaultdict(list)
        for item in fake_dict:
            video_name = item["path"].split("/")[-2]
            fake_videos[video_name].append(item)

        real_videos = defaultdict(list)
        for item in real_dict:
            video_name = item["path"].split("/")[-2]
            real_videos[video_name].append(item)

        num_videos_needed = max_samples // frames_per_video

        fake_video_names = list(fake_videos.keys())
        real_video_names = list(real_videos.keys())

        num_fake_videos = min(num_videos_needed, len(fake_video_names))
        num_real_videos = min(num_videos_needed, len(real_video_names))

        sampled_fake_videos = random.sample(fake_video_names, num_fake_videos)
        sampled_real_videos = random.sample(real_video_names, num_real_videos)

        def select_frames(video_frames, num_frames, uniform=True):
            if num_frames >= len(video_frames):
                return video_frames

            if uniform:
                sorted_frames = sorted(
                    video_frames,
                    key=lambda x: int(x["path"].split("/")[-1].split("_")[0]),
                )
                indices = np.linspace(0, len(sorted_frames) - 1, num_frames, dtype=int)
                return [sorted_frames[idx] for idx in indices]
            else:
                return random.sample(video_frames, num_frames)

        test_fake_subset = []
        for video_name in sampled_fake_videos:
            video_frames = fake_videos[video_name]
            selected = select_frames(
                video_frames, frames_per_video, uniform=uniform_sampling
            )
            test_fake_subset.extend(selected)

        test_real_subset = []
        for video_name in sampled_real_videos:
            video_frames = real_videos[video_name]
            selected = select_frames(
                video_frames, frames_per_video, uniform=uniform_sampling
            )
            test_real_subset.extend(selected)

        balanced_dict = test_real_subset + test_fake_subset
        random.shuffle(balanced_dict)

        real_count = len(test_real_subset)
        fake_count = len(test_fake_subset)

        sampling_method = "uniform" if uniform_sampling else "random"
        print(
            f"  Sampled {num_real_videos} real videos ({real_count} frames) + "
            f"{num_fake_videos} fake videos ({fake_count} frames) [{sampling_method} frame selection]"
        )

        return balanced_dict, real_count, fake_count

    def get_test_dataloaders(
        self,
        max_samples=700,
        batch_size=64,
        datasets=None,
        sample_by_video=False,
        frames_per_video=10,
        uniform_frames=True,
    ):
        # 1. Load the DEFAULT real data
        test_real_dict = self.load_real_test_data()

        # 2. Load the DFD-SPECIFIC real data
        dfd_real_dict_specific = []
        try:
            dfd_real_path = self.dataset_base + config.dfd_real_test_label_path
            with open(dfd_real_path) as f:
                dfd_real_dict_specific = json.load(f)
            print(
                f"Loaded {len(dfd_real_dict_specific)} specific real samples for DFD."
            )
        except (AttributeError, FileNotFoundError):
            print(
                "WARNING: 'dfd_real_test_label_path' not in config or file not found."
            )
            dfd_real_dict_specific = None

        # 3. Load the CelebDF-SPECIFIC real data (NEW BLOCK)
        celebdf_real_dict_specific = []
        try:
            celebdf_real_path = self.dataset_base + config.celebdf_real_test_label_path
            with open(celebdf_real_path) as f:
                celebdf_real_dict_specific = json.load(f)
            print(
                f"Loaded {len(celebdf_real_dict_specific)} specific real samples for CelebDF."
            )
        except (AttributeError, FileNotFoundError):
            print(
                "WARNING: 'celebdf_real_test_label_path' not in config or file not found."
            )
            celebdf_real_dict_specific = None

        dataloaders = {}

        for i, test_path in enumerate(self.test_label_paths):
            dataset_name = (
                self.dataset_names[i]
                if i < len(self.dataset_names)
                else f"Dataset_{i+1}"
            )

            if datasets and dataset_name not in datasets:
                continue

            with open(self.dataset_base + test_path) as f:
                test_fake_dict = json.load(f)

            # 4. *** KEY CHANGE ***
            # Decide which real dataset to use for balancing
            current_real_dict_to_use = test_real_dict  # Start with the default

            if dataset_name == "DFD" and dfd_real_dict_specific is not None:
                current_real_dict_to_use = dfd_real_dict_specific
                print(f"\nUsing DFD-specific real samples for {dataset_name}")

            # 5. Add elif for CelebDF (NEW BLOCK)
            elif dataset_name == "CelebDF" and celebdf_real_dict_specific is not None:
                current_real_dict_to_use = celebdf_real_dict_specific
                print(f"\nUsing CelebDF-specific real samples for {dataset_name}")

            else:
                print(f"\nUsing default real samples for {dataset_name}")

            if sample_by_video:
                balanced_dict, real_count, fake_count = (
                    self.create_balanced_testset_by_video(
                        test_fake_dict,
                        current_real_dict_to_use,
                        max_samples,
                        frames_per_video,
                        uniform_frames,
                    )
                )
            else:
                balanced_dict, real_count, fake_count = self.create_balanced_testset(
                    test_fake_dict, current_real_dict_to_use, max_samples
                )

            print(
                f"{dataset_name}: {real_count} real + {fake_count} fake = {len(balanced_dict)} total"
            )

            dataloader = DataLoader(
                Deepfake_Dataset(balanced_dict, train=False),
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
            )

            dataloaders[dataset_name] = dataloader

        # Note: The 'Mixed' dataset will still use the DEFAULT real set.
        if datasets is None or "Mixed" in datasets:
            print("\nCreating Mixed dataset (DF + F2F + FSW + NT, excluding FS)...")
            mixed_fake_dict = []
            for i, test_path in enumerate(self.test_label_paths[:4]):
                with open(self.dataset_base + test_path) as f:
                    mixed_fake_dict.extend(json.load(f))

            if sample_by_video:
                balanced_dict, real_count, fake_count = (
                    self.create_balanced_testset_by_video(
                        mixed_fake_dict,
                        test_real_dict,
                        max_samples,
                        frames_per_video,
                        uniform_frames,
                    )
                )
            else:
                balanced_dict, real_count, fake_count = self.create_balanced_testset(
                    mixed_fake_dict, test_real_dict, max_samples
                )

            print(
                f"Mixed: {real_count} real + {fake_count} fake = {len(balanced_dict)} total"
            )

            dataloader = DataLoader(
                Deepfake_Dataset(balanced_dict, train=False),
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
            )

            dataloaders["Mixed"] = dataloader

        return dataloaders


class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_single_dataset(self, model, dataloader, dataset_name):
        auc, _ = eval_one_dataset(dataloader, model, dataset_name)
        return auc

    def evaluate_all_datasets(self, model, dataloaders):
        results = {}

        for dataset_name, dataloader in dataloaders.items():
            auc = self.evaluate_single_dataset(model, dataloader, dataset_name)
            results[dataset_name] = auc

        return results


class ResultLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_to_console(self, results, metadata=None):
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        if metadata:
            print(f"\nCheckpoint: {Path(metadata['path']).name}")
            print(f"Epoch: {metadata['epoch']}")

        print("\nTest AUC Scores:")
        for dataset_name, auc in results.items():
            print(f"  {dataset_name:10s}: {auc:.4f}")

        if len(results) > 1:
            avg_datasets = [
                name for name in results.keys() if name not in ["FS", "Mixed"]
            ]
            if avg_datasets:
                avg_auc = np.mean([results[name] for name in avg_datasets])
                print(f"\n  {'Average (excluding FS)':25s}: {avg_auc:.4f}")

        print("=" * 60 + "\n")

    def log_to_file(self, results, metadata=None, filename="test_results.txt"):
        filepath = self.log_dir / filename

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filepath, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Test Results - {timestamp}\n")
            f.write(f"{'='*60}\n")

            if metadata:
                f.write(f"\nCheckpoint: {Path(metadata['path']).name}\n")
                f.write(f"Full path: {metadata['path']}\n")
                f.write(f"Epoch: {metadata['epoch']}\n")

            f.write("\nTest AUC Scores:\n")
            for dataset_name, auc in results.items():
                f.write(f"  {dataset_name:10s}: {auc:.4f}\n")

            if len(results) > 1:
                avg_datasets = [
                    name for name in results.keys() if name not in ["FS", "Mixed"]
                ]
                if avg_datasets:
                    avg_auc = np.mean([results[name] for name in avg_datasets])
                    f.write(f"\n  {'Average (excluding FS)':25s}: {avg_auc:.4f}\n")

            f.write(f"{'='*60}\n")

        print(f"\nResults appended to: {filepath}")

    def save_json(self, results, metadata=None, test_config=None, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = Path(metadata["path"]).stem if metadata else "unknown"
            filename = f"test_{timestamp}_{checkpoint_name}.json"

        filepath = self.log_dir / filename

        output = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": {
                "path": metadata.get("path", "unknown") if metadata else "unknown",
                "epoch": metadata.get("epoch", "unknown") if metadata else "unknown",
                "validation_aucs": metadata.get("aucs", []) if metadata else [],
                "best_aucs": metadata.get("best_aucs", []) if metadata else [],
            },
            "test_config": test_config or {},
            "results": results,
            "summary": {
                "average_auc": np.mean(
                    [v for k, v in results.items() if k not in ["FS", "Mixed"]]
                ),
                "datasets_tested": list(results.keys()),
                "num_datasets": len(results),
            },
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"JSON results saved to: {filepath}")
        return filepath


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular test script for RFFR deepfake detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest checkpoint automatically
  python test_model.py
  
  # Specify run by name (partial match works)
  python test_model.py --run 3branch_wavelet_residual_F2F_All
  
  # Specify exact checkpoint file
  python test_model.py --checkpoint path/to/checkpoint.pth.tar
  
  # Test specific datasets only
  python test_model.py --datasets DF F2F
  
  # Custom sample size and save results
  python test_model.py --samples 1000 --save-json
  
  # Sample by video with uniform frame distribution (70 videos x 10 frames = 700 frames)
  python test_model.py --sample-by-video --frames-per-video 10
  
  # Sample by video with random frame selection
  python test_model.py --sample-by-video --frames-per-video 10 --random-frames
  
  # Sample by video with fewer frames per video (140 videos x 5 frames = 700 frames)
  python test_model.py --sample-by-video --frames-per-video 5 --samples 700
  
  # List available runs without testing
  python test_model.py --list-runs
        """,
    )

    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory name (supports partial matching)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Specific checkpoint file path"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="highest_auc",
        choices=["highest_auc", "latest_epoch", "mixed"],
        help="Strategy for selecting best checkpoint (default: highest_auc)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=["DF", "F2F", "FSW", "NT", "FS", "Mixed", "DFD", "CelebDF"],
        help="Specific datasets to test (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=14000,
        help="Max samples per dataset (default: 700)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for testing (default: 16)",
    )
    parser.add_argument(
        "--save-json", action="store_true", help="Save results to JSON file"
    )
    parser.add_argument(
        "--list-runs", action="store_true", help="List available runs and exit"
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List checkpoints in selected run and exit",
    )
    parser.add_argument(
        "--sample-by-video",
        action="store_true",
        help="Sample frames uniformly across videos instead of random frame sampling",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=10,
        help="Frames per video when using --sample-by-video (default: 10)",
    )
    parser.add_argument(
        "--uniform-frames",
        action="store_true",
        default=True,
        help="Select frames uniformly across video duration (default: True)",
    )
    parser.add_argument(
        "--random-frames",
        action="store_true",
        help="Select frames randomly instead of uniformly (overrides --uniform-frames)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device ID to use (e.g., '0' or '1'). Overrides config.gpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="Directory to save test results (default: ./test_results)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu else config.gpus
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ckpt_manager = CheckpointManager()

    if args.list_runs:
        runs = ckpt_manager.discover_runs()
        if not runs:
            print("No training runs found in ./checkpoint/rffr/")
            return

        print(f"\nFound {len(runs)} training run(s):\n")
        for i, run in enumerate(runs, 1):
            print(f"{i}. {run['name']}")
            print(f"   Architecture: {run['architecture']}")
            print(f"   Protocol: {run['protocol']}")
            print(
                f"   Modified: {datetime.fromtimestamp(run['modified']).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print()
        return

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return
        run_dir = None
    else:
        if args.run:
            run = ckpt_manager.get_run_by_name(args.run)
            if not run:
                print(f"Error: No run found matching '{args.run}'")
                print("\nAvailable runs:")
                for r in ckpt_manager.discover_runs():
                    print(f"  - {r['name']}")
                return
        else:
            run = ckpt_manager.get_latest_run()
            if not run:
                print("Error: No training runs found in ./checkpoint/rffr/")
                return

        run_dir = run["path"]
        print(f"\nSelected run: {run['name']}")
        print(f"Architecture: {run['architecture']}")
        print(f"Protocol: {run['protocol']}")

        if args.list_checkpoints:
            checkpoints = ckpt_manager.list_checkpoints(run_dir)
            if not checkpoints or not checkpoints["files"]:
                print("No checkpoints found in this run")
                return

            print(f"\nCheckpoints in: {checkpoints['dir']}\n")

            print("Dataset-specific checkpoints:")
            for dataset, ckpts in sorted(checkpoints["by_dataset"].items()):
                print(f"\n  {dataset}:")
                for ckpt in ckpts:
                    print(
                        f"    - {ckpt['filename']} (AUC: {ckpt['auc']:.5f}, Epoch: {ckpt['epoch']})"
                    )

            if checkpoints["periodic"]:
                print(f"\n  Periodic checkpoints:")
                for ckpt in checkpoints["periodic"]:
                    print(f"    - {ckpt['filename']} (Epoch: {ckpt['epoch']})")

            return

        checkpoint_path = ckpt_manager.get_best_checkpoint(
            run_dir, strategy=args.strategy
        )
        if not checkpoint_path:
            print(f"Error: No checkpoint found in run directory")
            return

    print(f"\nUsing checkpoint selection strategy: {args.strategy}")

    model, metadata = ckpt_manager.load_model(checkpoint_path)

    data_manager = TestDataManager()
    print("\nPreparing test datasets...")

    uniform_frames = args.uniform_frames and not args.random_frames

    if args.sample_by_video:
        frame_selection = "uniform" if uniform_frames else "random"
        print(
            f"Using video-based sampling: {args.frames_per_video} frames per video ({frame_selection} selection)"
        )
    else:
        print("Using random frame sampling")

    dataloaders = data_manager.get_test_dataloaders(
        max_samples=args.samples,
        batch_size=args.batch_size,
        datasets=args.datasets,
        sample_by_video=args.sample_by_video,
        frames_per_video=args.frames_per_video,
        uniform_frames=uniform_frames,
    )

    if not dataloaders:
        print("Error: No test dataloaders created")
        return

    evaluator = ModelEvaluator()
    print("\nRunning evaluation...")
    results = evaluator.evaluate_all_datasets(model, dataloaders)

    test_config = {
        "samples": args.samples,
        "batch_size": args.batch_size,
        "datasets": args.datasets if args.datasets else "all",
        "sampling_method": "video_based" if args.sample_by_video else "random_frame",
        "frames_per_video": args.frames_per_video if args.sample_by_video else None,
        "uniform_frames": uniform_frames if args.sample_by_video else None,
        "gpu": args.gpu if args.gpu else config.gpus,
        "checkpoint_strategy": args.strategy,
    }

    logger = ResultLogger(log_dir=args.output_dir)
    logger.log_to_console(results, metadata)
    logger.log_to_file(results, metadata)

    if args.save_json:
        json_path = logger.save_json(results, metadata, test_config)

    print("Testing complete!")


if __name__ == "__main__":
    main()

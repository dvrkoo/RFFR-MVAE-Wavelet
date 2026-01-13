import random
import numpy as np
import argparse
import os
import torch

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

from tqdm import tqdm
from configs.config import config
from models.model_detector import RFFRL

from utils.simple_evaluate import eval_one_dataset, save_bandit_analysis
from utils.visualize_bandit import (
    save_forgery_samples,
    create_summary_visualization_new,
)
from torch.utils.data import DataLoader
from utils.dataset import Deepfake_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test RFFR deepfake detection model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help='Specific model directory to use (e.g., "2025-09-17-16:57:52_5a7a45"). If not provided, uses latest.',
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Specific model file to use. If not provided, uses best AUC model from directory.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualization samples",
    )
    return parser.parse_args()


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = "cuda"

import glob

# This is a stand-alone test-file.
# It loads in the deepfake detector (net.dd) and tests on the test set specified in "test_path".

args = parse_args()
net = RFFRL().cuda()

# Model loading logic with command-line support
best_model_dir = "./checkpoint/rffr/best_model/"
paths = []

if args.model_file:
    # Use specific model file provided
    if os.path.exists(args.model_file):
        paths = [args.model_file]
        print(f"Using specified model file: {args.model_file}")
    else:
        print(f"Error: Specified model file does not exist: {args.model_file}")
        exit(1)
elif args.model_dir:
    # Use specific model directory provided
    target_dir = os.path.join(best_model_dir, args.model_dir)
    if os.path.exists(target_dir):
        model_pattern = os.path.join(target_dir, "*.pth.tar")
        model_files = glob.glob(model_pattern)

        if model_files:
            # Use the Mixed AUC model (typically the best overall)
            mixed_model = [f for f in model_files if "0__AUC" in f]
            if mixed_model:
                paths = [mixed_model[0]]
                print(
                    f"Using Mixed AUC model from specified directory: {mixed_model[0]}"
                )
            else:
                # Fallback to any available model
                paths = [model_files[0]]
                print(
                    f"Using available model from specified directory: {model_files[0]}"
                )
        else:
            print(f"No model files found in specified directory: {target_dir}")
            exit(1)
    else:
        print(f"Error: Specified model directory does not exist: {target_dir}")
        exit(1)
elif os.path.exists(best_model_dir):
    # Automatically find the best model from the most recent training run
    training_dirs = [
        d
        for d in os.listdir(best_model_dir)
        if os.path.isdir(os.path.join(best_model_dir, d))
    ]
    if training_dirs:
        # Sort by modification time to get the most recent
        training_dirs.sort(
            key=lambda x: os.path.getmtime(os.path.join(best_model_dir, x)),
            reverse=True,
        )
        latest_dir = training_dirs[0]
        print(f"Using latest training directory: {latest_dir}")

        # Find all .pth.tar files in the latest directory
        model_pattern = os.path.join(best_model_dir, latest_dir, "*.pth.tar")
        model_files = glob.glob(model_pattern)

        if model_files:
            # Use the Mixed AUC model (typically the best overall)
            mixed_model = [f for f in model_files if "0__AUC" in f]
            if mixed_model:
                paths = [mixed_model[0]]
                print(f"Using best Mixed AUC model: {mixed_model[0]}")
            else:
                # Fallback to any available model
                paths = [model_files[0]]
                print(f"Using available model: {model_files[0]}")
        else:
            print("No model files found in latest training directory")
            paths = []
    else:
        print("No training directories found")
        paths = []
else:
    print("Best model directory not found, using manual path")
    paths = ["./checkpoint/rffr/best_model/2025-09-17-16:57:52_5a7a45/0__AUC_0.pth.tar"]
for modelpath in paths:
    if not modelpath:  # Skip empty paths
        continue

    checkpoint = torch.load(modelpath)
    net.dd.load_state_dict(checkpoint["state_dict"])

    # Load real test data for proper evaluation
    test_real_dict = []
    for real_test_dataset in config.real_test_label_path:
        with open(config.dataset_base + real_test_dataset) as f:
            test_real_dict += json.load(f)

    print(f"Loaded {len(test_real_dict)} real test samples")

    # Create balanced test sets using real test data
    dataset_names = ["DF", "F2F", "FSW", "NT"]
    random.seed(config.seed)  # Ensure reproducibility

    for i, test_path in enumerate(config.test_label_path):
        test_json = open(config.dataset_base + test_path)
        test_fake_dict = json.load(test_json)

        dataset_name = dataset_names[i] if i < len(dataset_names) else f"Dataset_{i+1}"
        print(f"\nTest dataset: {test_path} ({dataset_name})")

        # Create balanced test set using real test data
        num_samples = min(
            len(test_fake_dict), len(test_real_dict), 700
        )  # Limit to reasonable size
        test_fake_subset = random.sample(test_fake_dict, num_samples)
        test_real_subset = random.sample(test_real_dict, num_samples)

        balanced_test_dict = test_real_subset + test_fake_subset
        random.shuffle(balanced_test_dict)

        # Verify balance
        real_count = sum(1 for item in balanced_test_dict if item["label"] == 0)
        fake_count = sum(1 for item in balanced_test_dict if item["label"] == 1)
        print(
            f"Balanced test set: {real_count} real + {fake_count} fake = {len(balanced_test_dict)} total"
        )

        test_dataloader = DataLoader(
            Deepfake_Dataset(balanced_test_dict, train=False),
            batch_size=16,
            shuffle=False,
        )

        # Run evaluation with bandit tracking and visualization
        capture_viz = not args.no_visualizations
        result = eval_one_dataset(
            test_dataloader,
            net,
            dataset_name,
            track_bandit=True,
            capture_visualizations=capture_viz,
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            auc, bandit_info, visualization_data = result
        elif isinstance(result, tuple) and len(result) == 2:
            auc, bandit_info = result
            visualization_data = []
        else:
            auc = result
            bandit_info = None
            visualization_data = []

        print("Model: ", os.path.basename(modelpath))
        print("AUC:", round(float(auc), 4))

        # Save bandit analysis
        if bandit_info:
            analysis_path = f"logs/bandit_analysis_{dataset_name}.json"
            from utils.simple_evaluate import analyze_bandit_patterns

            analysis = analyze_bandit_patterns(bandit_info, dataset_name)

            import json

            os.makedirs("logs", exist_ok=True)
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"Bandit analysis saved to {analysis_path}")

            # Print most important patches
            if analysis.get("most_important_patches"):
                print(
                    f'Most important patches for {dataset_name}: {analysis["most_important_patches"]}'
                )
            if analysis.get("most_selected_blocks"):
                print(
                    f'Most selected blocks for {dataset_name}: {analysis["most_selected_blocks"]}'
                )

        # Save visualization samples
        # if visualization_data and not args.no_visualizations:
        #     print(
        #         f"Creating visualizations for {len(visualization_data)} samples from {dataset_name}"
        #     )
        #     saved_paths = save_forgery_samples(
        #         visualization_data, max_samples_per_dataset=3
        #     )
        #     print(f"Visualizations saved: {len(saved_paths)} files")

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        f = open("logs/evaluate.txt", "a")
        f.write("\nModel: " + os.path.basename(modelpath))
        f.write("\nTest path: " + test_path)
        f.write("\nDataset: " + dataset_name)
        f.write("\nAUC: " + str(round(float(auc), 4)))
        f.write("\n")
        f.close()

    # Create summary visualization after all datasets are processed
    if not args.no_visualizations:
        print("\nCreating summary visualization...")
        all_analyses = {}
        for i, test_path in enumerate(config.test_label_path):
            dataset_name = (
                dataset_names[i] if i < len(dataset_names) else f"Dataset_{i+1}"
            )
            analysis_path = f"logs/bandit_analysis_{dataset_name}.json"

            if os.path.exists(analysis_path):
                with open(analysis_path, "r") as f:
                    all_analyses[dataset_name] = json.load(f)

        if all_analyses:
            summary_path = create_summary_visualization_new(all_analyses)
            print(f"Summary visualization saved to {summary_path}")

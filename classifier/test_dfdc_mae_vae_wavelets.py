#!/usr/bin/env python3
"""
Test MAE-VAE Wavelet models on DFDC dataset.
This script tests all available MAE-VAE models trained on DFD on the DFDC test set.
"""

import random
import numpy as np
import argparse
import os
import torch
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from configs.config import config
from models.model_mae_vae import mae_vae_vit_base_patch16
from torch.utils.data import DataLoader
from utils.dataset import Deepfake_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test MAE-VAE models on DFDC dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Specific model path to test. If not provided, searches for DFD MAE-VAE models.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=860,
        help="Number of samples per class (default: 860, balanced for DFDC real samples)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detailed results to JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for testing (default: 16)",
    )
    return parser.parse_args()


def setup_environment(gpu_id, seed=42):
    """Setup random seeds and GPU environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dfdc_data(num_samples=860):
    """Load and balance DFDC dataset."""
    dataset_base = config.dataset_base

    # Load DFDC fake samples
    dfdc_fake_path = os.path.join(dataset_base, "DFDC/all_test_label.json")
    with open(dfdc_fake_path) as f:
        dfdc_fake = json.load(f)

    # Load DFDC real samples
    dfdc_real_path = os.path.join(dataset_base, "DFDC/real_test_label.json")
    with open(dfdc_real_path) as f:
        dfdc_real = json.load(f)

    print(f"Loaded {len(dfdc_fake)} DFDC fake samples")
    print(f"Loaded {len(dfdc_real)} DFDC real samples")

    # Balance dataset
    num_samples = min(num_samples, len(dfdc_fake), len(dfdc_real))
    fake_subset = random.sample(dfdc_fake, num_samples)
    real_subset = random.sample(dfdc_real, num_samples)

    balanced_data = real_subset + fake_subset
    random.shuffle(balanced_data)

    print(
        f"\nBalanced test set: {num_samples} real + {num_samples} fake = {len(balanced_data)} total"
    )

    return balanced_data


def evaluate_model(model, dataloader, device="cuda"):
    """Evaluate MAE-VAE model on test data."""
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            # Get reconstruction residuals from MAE-VAE
            residuals, _ = model.patch_by_patch_DIFF(
                images,
                test=True,
                return_reconstructed=False,
                return_wavelet_residuals=False,
            )

            # Calculate anomaly scores from residuals
            # Higher residual = more likely fake
            scores = torch.mean(torch.abs(residuals), dim=[1, 2, 3]).cpu().numpy()

            all_labels.extend(labels)
            all_scores.extend(scores)

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_scores)

    return auc, all_labels, all_scores


def find_dfd_mae_vae_models():
    """Find all DFD MAE-VAE model checkpoints."""
    checkpoint_base = (
        "/andromeda/personal/nmarini/RFFR/rffr_generative/checkpoint/mae_vae/"
    )

    # Look for DFD models
    dfd_dir = os.path.join(checkpoint_base, "DFD")

    if not os.path.exists(dfd_dir):
        print(f"Warning: DFD MAE-VAE checkpoint directory not found: {dfd_dir}")
        # Try CDF directory as fallback
        cdf_dir = os.path.join(checkpoint_base, "CDF")
        if os.path.exists(cdf_dir):
            print(f"Using CDF models as fallback: {cdf_dir}")
            dfd_dir = cdf_dir
        else:
            return []

    # Find all .pth.tar files
    import glob

    model_files = glob.glob(os.path.join(dfd_dir, "*.pth.tar"))
    model_files.sort()  # Sort by name

    return model_files


def main():
    args = parse_args()

    # Setup environment
    setup_environment(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load DFDC data
    test_data = load_dfdc_data(num_samples=args.samples)

    # Create dataloader
    test_dataloader = DataLoader(
        Deepfake_Dataset(test_data, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Find models to test
    if args.model_path:
        model_paths = [args.model_path]
        print(f"\nTesting specified model: {args.model_path}")
    else:
        model_paths = find_dfd_mae_vae_models()
        if not model_paths:
            print("Error: No DFD MAE-VAE models found!")
            print("Please specify a model path with --model-path")
            return
        print(f"\nFound {len(model_paths)} DFD MAE-VAE models to test")

    # Test each model
    results = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")

        try:
            # Load model
            model = mae_vae_vit_base_patch16(
                vae_latent_dim=config.vae_latent_dim, freeze_encoder=False
            )

            checkpoint = torch.load(model_path, map_location=device)

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)

            # Evaluate
            auc, labels, scores = evaluate_model(model, test_dataloader, device)

            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  AUC: {auc:.4f}")

            # Save results
            result = {
                "model": model_name,
                "model_path": model_path,
                "auc": float(auc),
                "num_samples": args.samples * 2,
                "num_real": args.samples,
                "num_fake": args.samples,
            }
            results.append(result)

            # Log to file
            os.makedirs("logs", exist_ok=True)
            with open("logs/dfdc_mae_vae_wavelets.txt", "a") as f:
                f.write(f"\nModel: {model_name}\n")
                f.write(f"AUC: {auc:.4f}\n")
                f.write(f"Samples: {args.samples} real + {args.samples} fake\n")
                f.write("-" * 80 + "\n")

        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - DFDC MAE-VAE Wavelet Models")
    print(f"{'='*80}")
    for result in results:
        print(f"{result['model']}: AUC = {result['auc']:.4f}")

    avg_auc = 0.0
    if results:
        avg_auc = np.mean([r["auc"] for r in results])
        print(f"\nAverage AUC: {avg_auc:.4f}")

    # Save JSON results
    if args.save_json:
        os.makedirs("test_results", exist_ok=True)
        json_path = "test_results/dfdc_mae_vae_wavelets_results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "dataset": "DFDC",
                    "model_type": "MAE-VAE Wavelets",
                    "num_samples_per_class": args.samples,
                    "results": results,
                    "average_auc": float(avg_auc),
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()

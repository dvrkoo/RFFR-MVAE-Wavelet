#!/usr/bin/env python3
"""
Test script for MAE-VAE generative model reconstruction evaluation.
Evaluates reconstruction error across all FaceForensics++ forgery types.

Usage:
    python test_mae_vae_reconstruction.py [--checkpoint path/to/checkpoint.pth.tar] [--output_dir ./test_results_mae_vae]
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# --- Project-specific imports ---
# Assuming your dataset and config files are in these locations
from dataset import Deepfake_Dataset
from configs.config import config

# Import your custom MAE-VAE model
# Make sure the file containing HybridMAEVAE is accessible, e.g., in models/model_mae_vae.py
from models.model_mae_vae import mae_vae_vit_base_patch16


def find_best_checkpoint(checkpoint_dir="checkpoint/mae_vae/best/"):
    """Find the best checkpoint based on lowest loss in filename."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = list(checkpoint_dir.glob("best_loss_*.pth.tar"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    best_checkpoint = None
    best_loss = float("inf")

    for checkpoint_file in checkpoint_files:
        filename = checkpoint_file.name
        try:
            loss_str = filename.split("_")[2]
            loss_value = float(loss_str)
            if loss_value < best_loss:
                best_loss = loss_value
                best_checkpoint = checkpoint_file
        except (IndexError, ValueError):
            print(f"Warning: Could not parse loss from filename: {filename}")
            continue

    if best_checkpoint is None:
        raise ValueError("Could not find a valid checkpoint file.")

    print(f"Selected best checkpoint: {best_checkpoint.name} (loss: {best_loss:.6f})")
    return str(best_checkpoint)


def load_test_data():
    """Load test data for all forgery types and real images."""
    data_label_path = Path("../data_label")  # Adjust this path if needed
    test_data = {
        "real": [],
        "df": [],
        "f2f": [],
        "fsw": [],
        "nt": [],
        "fs": [],
    }

    # Load real test data
    real_test_path = data_label_path / "Faceforensics/excludes_hq/real_test_label.json"
    if real_test_path.exists():
        with open(real_test_path, "r") as f:
            test_data["real"] = [item["path"] for item in json.load(f)]

    # Load fake test data
    forgery_files = {
        "df": "df_test_label.json",
        "f2f": "f2f_test_label.json",
        "fsw": "fsw_test_label.json",
        "nt": "nt_test_label.json",
        "fs": "fs_test_label.json",
    }
    for forgery_type, filename in forgery_files.items():
        test_file_path = data_label_path / "Faceforensics/excludes_hq" / filename
        if test_file_path.exists():
            with open(test_file_path, "r") as f:
                test_data[forgery_type] = [item["path"] for item in json.load(f)]
        else:
            print(f"Warning: Test file not found: {test_file_path}")

    print("\nTest data loaded:")
    for data_type, paths in test_data.items():
        print(f"  {data_type.upper()}: {len(paths)} images")

    return test_data


def compute_reconstruction_metrics(original, reconstructed):
    """Compute reconstruction error metrics between two images."""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Ensure range is [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)

    # Compute full-image metrics
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))

    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float("inf")

    return {"mse": mse, "mae": mae, "psnr": psnr}


def test_reconstruction(
    checkpoint_path, output_dir, max_samples_per_type=100, save_visualizations=True
):
    """Test MAE-VAE reconstruction error across all forgery types."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vis_path = None
    if save_visualizations:
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)

    print("Loading MAE-VAE model...")
    # Instantiate your HybridMAEVAE model
    net = mae_vae_vit_base_patch16(
        vae_latent_dim=config.vae_latent_dim, freeze_encoder=config.freeze_mae_encoder
    )
    net = nn.DataParallel(net).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    net = net.module
    net.eval()
    print(f"Model loaded from: {checkpoint_path}")

    test_data = load_test_data()
    results = defaultdict(dict)
    detailed_results = defaultdict(lambda: defaultdict(list))

    for data_type, image_paths in test_data.items():
        if not image_paths:
            print(f"Skipping {data_type} - no data available")
            continue

        print(f"\nTesting {data_type.upper()} ({len(image_paths)} images)...")

        if len(image_paths) > max_samples_per_type:
            image_paths = image_paths[:max_samples_per_type]
            print(f"  Limited to {max_samples_per_type} samples for testing.")

        dataset = Deepfake_Dataset(image_paths, train=False)
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
        )

        with torch.no_grad():
            for i, data_batch in enumerate(
                tqdm(dataloader, desc=f"Processing {data_type}")
            ):
                data_batch = data_batch.to(device)

                # --- Inference using MAE-VAE ---
                # Forward pass with a 0.75 mask ratio to test reconstruction
                # Use deterministic=True for the VAE to get the mean of the latent space
                loss, pred_patches, mask = net(
                    data_batch, mask_ratio=0.75, deterministic=True
                )

                # Reconstruct the full image from patches
                reconstruction = net.unpatchify(pred_patches)
                reconstruction = torch.clamp(reconstruction, 0, 1)

                # Process each image in the batch
                for j in range(data_batch.shape[0]):
                    original_img = data_batch[j]
                    recon_img = reconstruction[j]

                    metrics = compute_reconstruction_metrics(original_img, recon_img)

                    for metric_name, value in metrics.items():
                        if value is not None:
                            detailed_results[data_type][metric_name].append(value)

                    # Save visualization for the first batch of each type
                    if i == 0 and j < 5 and save_visualizations and vis_path:
                        # Create the mask visualization
                        mask_single = mask[j : j + 1]  # Shape: (1, num_patches)
                        patch_size = net.patch_embed.patch_size[0]
                        mask_img = mask_single.unsqueeze(-1).repeat(
                            1, 1, patch_size**2 * 3
                        )
                        mask_img = net.unpatchify(mask_img)

                        # Create a red overlay for masked areas
                        red_overlay = torch.zeros_like(original_img)
                        red_overlay[0, :, :] = 1.0  # Red channel
                        masked_input = original_img * (
                            1 - mask_img * 0.7
                        ) + red_overlay * (mask_img * 0.7)

                        # Calculate error map
                        error_map = (recon_img - original_img).abs()
                        error_map = error_map / (error_map.max() + 1e-8)  # Normalize

                        vis_tensor = torch.stack(
                            [
                                original_img,
                                masked_input.squeeze(0),
                                recon_img,
                                error_map,
                            ]
                        )
                        save_path = vis_path / f"{data_type}_sample_{j:02d}.png"
                        save_image(vis_tensor, save_path, nrow=4, normalize=False)

        # Compute summary statistics for this data type
        if detailed_results[data_type]:
            summary = {}
            for metric_name in detailed_results[data_type].keys():
                values = detailed_results[data_type][metric_name]
                if values:
                    summary[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                    }
            results[data_type] = summary

    return results, detailed_results


def save_results(results, detailed_results, output_dir):
    """Save summary and detailed results to JSON files."""

    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    output_path = Path(output_dir)
    serializable_results = convert_to_serializable(results)
    serializable_detailed = convert_to_serializable(detailed_results)

    with open(output_path / "reconstruction_summary.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    with open(output_path / "reconstruction_detailed.json", "w") as f:
        json.dump(serializable_detailed, f, indent=2)

    print(f"\nResults successfully saved to {output_path}")


def print_summary(results):
    """Print a summary of the reconstruction results to the console."""
    print("\n" + "=" * 80)
    print("RECONSTRUCTION ERROR SUMMARY")
    print("=" * 80)

    for data_type, metrics in results.items():
        print(f"\n{data_type.upper()}")
        print("-" * 40)
        for metric_name, stats in metrics.items():
            print(
                f"  {metric_name.upper():<10}: Mean={stats['mean']:.6f}, Std={stats['std']:.6f}"
            )

    print(f"\n{'RECONSTRUCTION DIFFICULTY RANKING (HIGHER MSE IS WORSE)'}")
    print("-" * 60)
    mse_ranking = sorted(
        [
            (dt, metrics["mse"]["mean"])
            for dt, metrics in results.items()
            if "mse" in metrics
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    for i, (data_type, mse) in enumerate(mse_ranking, 1):
        print(f"  {i}. {data_type.upper():<6}: {mse:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test MAE-VAE reconstruction on FF++ forgeries"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file. Auto-detects if not provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results_mae_vae",
        help="Output directory for results and visualizations.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to test per forgery type.",
    )
    parser.add_argument(
        "--no_vis", action="store_true", help="Disable saving of visualization images."
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        try:
            args.checkpoint = find_best_checkpoint()
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: Could not automatically find checkpoint. {e}")
            print("Please specify the path using the --checkpoint argument.")
            return 1

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return 1

    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Saving results to: {args.output_dir}")

    try:
        results, detailed_results = test_reconstruction(
            args.checkpoint,
            args.output_dir,
            max_samples_per_type=args.max_samples,
            save_visualizations=not args.no_vis,
        )

        if not results:
            print(
                "\nTesting finished, but no results were generated. Check data paths and model."
            )
            return 1

        print_summary(results)
        save_results(results, detailed_results, args.output_dir)

        print("\nReconstruction testing completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

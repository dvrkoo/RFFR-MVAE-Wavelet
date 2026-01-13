#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import Deepfake_Dataset
from configs.config import config
from models.model_mae import mae_vit_base_patch16


def find_best_checkpoint(checkpoint_dir="checkpoint/mae/CDF/"):
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
        raise ValueError("Could not find valid checkpoint file")

    print(f"Selected best checkpoint: {best_checkpoint.name} (loss: {best_loss:.6f})")
    return str(best_checkpoint)


def get_block_position(block_id):
    row = block_id // 4
    col = block_id % 4

    patch_row = 3 * row + 1
    patch_col = 3 * col + 1

    return patch_row, patch_col


def create_block_mask_visualization(block_id, img_size=224, patch_size=16):
    num_patches = img_size // patch_size
    mask = np.zeros((img_size, img_size, 3))

    patch_row, patch_col = get_block_position(block_id)

    magic_offsets = [0, 1, 2, 14, 15, 16, 28, 29, 30]

    for offset in magic_offsets:
        offset_row = offset // 14
        offset_col = offset % 14
        abs_row = patch_row + offset_row
        abs_col = patch_col + offset_col

        if abs_row < num_patches and abs_col < num_patches:
            pixel_row_start = abs_row * patch_size
            pixel_row_end = (abs_row + 1) * patch_size
            pixel_col_start = abs_col * patch_size
            pixel_col_end = (abs_col + 1) * patch_size

            mask[pixel_row_start:pixel_row_end, pixel_col_start:pixel_col_end] = 1

    return mask


def visualize_single_block_reconstruction(model, image, block_id, device):
    model.eval()
    with torch.no_grad():
        image_device = image.to(device)

        loss, pred, mask = model(image_device, block=True, block_id=block_id)

        reconstruction = model.merge_output(pred, mask, image_device)
        reconstruction = torch.clamp(reconstruction, 0, 1)

        patch_size = int(model.patch_embed.patch_size[0])
        mask_img = mask.unsqueeze(-1).repeat(1, 1, patch_size * patch_size * 3)
        mask_img = model.unpatchify(mask_img)

        red_overlay = torch.zeros_like(image_device)
        red_overlay[:, 0, :, :] = 1.0
        masked_input = image_device * (1 - mask_img * 0.5) + red_overlay * (
            mask_img * 0.5
        )

        error_full = torch.abs(reconstruction - image_device)

        masked_region_only = mask_img > 0.5
        if masked_region_only.sum() > 0:
            mse = (
                torch.sum(((reconstruction - image_device) ** 2) * mask_img)
                / mask_img.sum()
            )
            mae = torch.sum(error_full * mask_img) / mask_img.sum()
        else:
            mse = torch.tensor(0.0)
            mae = torch.tensor(0.0)

    return {
        "original": image_device[0].cpu(),
        "masked_input": masked_input[0].cpu(),
        "reconstruction": reconstruction[0].cpu(),
        "error": error_full[0].cpu(),
        "mask": mask_img[0].cpu(),
        "mse": mse.item(),
        "mae": mae.item(),
    }


def visualize_all_blocks_single_image(
    model, image, output_path, device, image_name="sample"
):
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(6, 10, hspace=0.4, wspace=0.3)

    original_np = image[0].permute(1, 2, 0).cpu().numpy()
    original_np = np.clip(original_np, 0, 1)

    ax_orig = fig.add_subplot(gs[0, :2])
    ax_orig.imshow(original_np)
    ax_orig.set_title("Original Image", fontsize=14, fontweight="bold")
    ax_orig.axis("off")

    ax_grid = fig.add_subplot(gs[0, 2:4])
    ax_grid.imshow(original_np)
    patch_size = 16
    num_patches = 14
    for i in range(num_patches + 1):
        ax_grid.axhline(i * patch_size, color="white", alpha=0.5, linewidth=0.8)
        ax_grid.axvline(i * patch_size, color="white", alpha=0.5, linewidth=0.8)

    for block_id in range(16):
        patch_row, patch_col = get_block_position(block_id)
        block_pixel_row = patch_row * patch_size
        block_pixel_col = patch_col * patch_size
        rect = patches.Rectangle(
            (block_pixel_col - patch_size, block_pixel_row - patch_size),
            3 * patch_size,
            3 * patch_size,
            linewidth=1.5,
            edgecolor="red",
            facecolor="red",
            alpha=0.2,
        )
        ax_grid.add_patch(rect)

        center_y = block_pixel_row + patch_size * 0.5
        center_x = block_pixel_col + patch_size * 0.5
        ax_grid.text(
            center_x,
            center_y,
            str(block_id),
            color="yellow",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    ax_grid.set_title("Block Positions (0-15)", fontsize=14, fontweight="bold")
    ax_grid.axis("off")

    block_metrics = []

    for block_id in range(16):
        result = visualize_single_block_reconstruction(model, image, block_id, device)
        block_metrics.append(
            {"block_id": block_id, "mse": result["mse"], "mae": result["mae"]}
        )

        row_offset = 1 + (block_id // 4)
        col_offset = (block_id % 4) * 2

        ax_masked = fig.add_subplot(gs[row_offset, col_offset])
        masked_np = result["masked_input"].permute(1, 2, 0).numpy()
        masked_np = np.clip(masked_np, 0, 1)
        ax_masked.imshow(masked_np)
        ax_masked.set_title(f"Block {block_id} Input", fontsize=10)
        ax_masked.axis("off")

        ax_recon = fig.add_subplot(gs[row_offset, col_offset + 1])
        recon_np = result["reconstruction"].permute(1, 2, 0).numpy()
        recon_np = np.clip(recon_np, 0, 1)
        ax_recon.imshow(recon_np)
        ax_recon.set_title(f'Recon {block_id}\nMSE: {result["mse"]:.5f}', fontsize=10)
        ax_recon.axis("off")

    metrics_mse = [m["mse"] for m in block_metrics]
    metrics_mae = [m["mae"] for m in block_metrics]

    ax_mse = fig.add_subplot(gs[0, 5:7])
    block_ids = list(range(16))
    bars = ax_mse.bar(
        block_ids, metrics_mse, color="steelblue", alpha=0.7, edgecolor="navy"
    )
    ax_mse.set_xlabel("Block ID", fontsize=11)
    ax_mse.set_ylabel("MSE", fontsize=11)
    ax_mse.set_title("Reconstruction MSE per Block", fontweight="bold", fontsize=12)
    ax_mse.grid(True, alpha=0.3)
    ax_mse.set_xticks(range(0, 16, 2))

    for i, (bar, val) in enumerate(zip(bars, metrics_mse)):
        if val == max(metrics_mse):
            bar.set_color("red")
            bar.set_alpha(0.8)

    ax_mae = fig.add_subplot(gs[0, 7:9])
    bars = ax_mae.bar(
        block_ids, metrics_mae, color="coral", alpha=0.7, edgecolor="darkred"
    )
    ax_mae.set_xlabel("Block ID", fontsize=11)
    ax_mae.set_ylabel("MAE", fontsize=11)
    ax_mae.set_title("Reconstruction MAE per Block", fontweight="bold", fontsize=12)
    ax_mae.grid(True, alpha=0.3)
    ax_mae.set_xticks(range(0, 16, 2))

    for i, (bar, val) in enumerate(zip(bars, metrics_mae)):
        if val == max(metrics_mae):
            bar.set_color("darkred")
            bar.set_alpha(0.8)

    mse_grid = np.array(metrics_mse).reshape(4, 4)
    ax_heatmap = fig.add_subplot(gs[1:3, 8:10])
    im = ax_heatmap.imshow(mse_grid, cmap="hot", interpolation="nearest")
    ax_heatmap.set_title("MSE Heatmap\n(by position)", fontweight="bold", fontsize=11)
    ax_heatmap.set_xlabel("Block Column")
    ax_heatmap.set_ylabel("Block Row")
    ax_heatmap.set_xticks(range(4))
    ax_heatmap.set_yticks(range(4))

    for i in range(4):
        for j in range(4):
            val = mse_grid[i, j]
            text = ax_heatmap.text(
                j,
                i,
                f"{val:.4f}",
                ha="center",
                va="center",
                color="white" if val > mse_grid.max() * 0.5 else "black",
                fontsize=8,
            )

    plt.colorbar(im, ax=ax_heatmap, label="MSE", shrink=0.8)

    stats_text = f"Statistics:\n"
    stats_text += f"Mean MSE: {np.mean(metrics_mse):.6f}\n"
    stats_text += f"Std MSE: {np.std(metrics_mse):.6f}\n"
    stats_text += (
        f"Min MSE: {np.min(metrics_mse):.6f} (Block {np.argmin(metrics_mse)})\n"
    )
    stats_text += (
        f"Max MSE: {np.max(metrics_mse):.6f} (Block {np.argmax(metrics_mse)})\n"
    )

    ax_stats = fig.add_subplot(gs[3:5, 8:10])
    ax_stats.text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax_stats.axis("off")

    fig.suptitle(
        f"MAE Block Reconstruction Analysis: {image_name}",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return block_metrics


def create_spatial_heatmap(
    block_metrics_list, output_path, title="Reconstruction Error Heatmap"
):
    mse_grid = np.zeros((4, 4))
    mae_grid = np.zeros((4, 4))

    for sample_metrics in block_metrics_list:
        for metric in sample_metrics:
            block_id = metric["block_id"]
            row = block_id // 4
            col = block_id % 4
            mse_grid[row, col] += metric["mse"]
            mae_grid[row, col] += metric["mae"]

    num_samples = len(block_metrics_list)
    mse_grid /= num_samples
    mae_grid /= num_samples

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(mse_grid, cmap="hot", interpolation="nearest")
    ax1.set_title("Mean MSE by Block Position", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Block Column")
    ax1.set_ylabel("Block Row")
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))

    for i in range(4):
        for j in range(4):
            text = ax1.text(
                j,
                i,
                f"{mse_grid[i, j]:.5f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

    plt.colorbar(im1, ax=ax1, label="MSE")

    im2 = ax2.imshow(mae_grid, cmap="hot", interpolation="nearest")
    ax2.set_title("Mean MAE by Block Position", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Block Column")
    ax2.set_ylabel("Block Row")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))

    for i in range(4):
        for j in range(4):
            text = ax2.text(
                j,
                i,
                f"{mae_grid[i, j]:.5f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

    plt.colorbar(im2, ax=ax2, label="MAE")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return mse_grid, mae_grid


def load_test_samples(data_type="real", num_samples=5):
    data_label_path = Path("../data_label")

    type_to_file = {
        "real": "real_test_label.json",
        "df": "df_test_label.json",
        "f2f": "f2f_test_label.json",
        "fsw": "fsw_test_label.json",
        "nt": "nt_test_label.json",
        "fs": "fs_test_label.json",
    }

    label_file = data_label_path / "Faceforensics/excludes_hq" / type_to_file[data_type]

    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    with open(label_file, "r") as f:
        data = json.load(f)
        image_paths = [item["path"] for item in data[:num_samples]]

    return image_paths


def analyze_data_type(model, data_type, output_dir, device, num_samples=5):
    print(f"\nAnalyzing {data_type.upper()}...")

    type_dir = output_dir / data_type
    type_dir.mkdir(exist_ok=True)

    image_paths = load_test_samples(data_type, num_samples)
    dataset = Deepfake_Dataset(image_paths, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    all_metrics = []

    for idx, image in enumerate(dataloader):
        print(f"  Processing sample {idx + 1}/{num_samples}...")

        image_name = f"{data_type}_sample_{idx:03d}"
        output_path = type_dir / f"{image_name}_blocks.png"

        metrics = visualize_all_blocks_single_image(
            model, image, output_path, device, image_name
        )
        all_metrics.append(metrics)

    heatmap_path = type_dir / f"{data_type}_spatial_heatmap.png"
    create_spatial_heatmap(
        all_metrics,
        heatmap_path,
        title=f"{data_type.upper()} Reconstruction Error Heatmap (n={num_samples})",
    )

    metrics_summary = {
        "block_metrics": all_metrics,
        "mean_mse_per_block": {},
        "mean_mae_per_block": {},
    }

    for block_id in range(16):
        mse_values = [sample[block_id]["mse"] for sample in all_metrics]
        mae_values = [sample[block_id]["mae"] for sample in all_metrics]

        metrics_summary["mean_mse_per_block"][block_id] = {
            "mean": np.mean(mse_values),
            "std": np.std(mse_values),
            "min": np.min(mse_values),
            "max": np.max(mse_values),
        }
        metrics_summary["mean_mae_per_block"][block_id] = {
            "mean": np.mean(mae_values),
            "std": np.std(mae_values),
            "min": np.min(mae_values),
            "max": np.max(mae_values),
        }

    summary_path = type_dir / f"{data_type}_metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def compare_data_types(all_metrics, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Reconstruction Error Comparison Across Data Types",
        fontsize=16,
        fontweight="bold",
    )

    data_types = list(all_metrics.keys())
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(data_types)))

    ax = axes[0, 0]
    for idx, (data_type, metrics) in enumerate(all_metrics.items()):
        mse_means = [
            metrics["mean_mse_per_block"][block_id]["mean"] for block_id in range(16)
        ]
        ax.plot(
            range(16),
            mse_means,
            marker="o",
            label=data_type.upper(),
            color=colors[idx],
            linewidth=2,
        )
    ax.set_xlabel("Block ID")
    ax.set_ylabel("Mean MSE")
    ax.set_title("MSE by Block Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for idx, (data_type, metrics) in enumerate(all_metrics.items()):
        mae_means = [
            metrics["mean_mae_per_block"][block_id]["mean"] for block_id in range(16)
        ]
        ax.plot(
            range(16),
            mae_means,
            marker="s",
            label=data_type.upper(),
            color=colors[idx],
            linewidth=2,
        )
    ax.set_xlabel("Block ID")
    ax.set_ylabel("Mean MAE")
    ax.set_title("MAE by Block Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    overall_mse = []
    for data_type in data_types:
        mse_values = [
            all_metrics[data_type]["mean_mse_per_block"][block_id]["mean"]
            for block_id in range(16)
        ]
        overall_mse.append(np.mean(mse_values))

    ax.bar(data_types, overall_mse, color=colors, alpha=0.7)
    ax.set_ylabel("Mean MSE (all blocks)")
    ax.set_title("Overall Reconstruction MSE")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    overall_mae = []
    for data_type in data_types:
        mae_values = [
            all_metrics[data_type]["mean_mae_per_block"][block_id]["mean"]
            for block_id in range(16)
        ]
        overall_mae.append(np.mean(mae_values))

    ax.bar(data_types, overall_mae, color=colors, alpha=0.7)
    ax.set_ylabel("Mean MAE (all blocks)")
    ax.set_title("Overall Reconstruction MAE")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        output_dir / "comparison_across_types.png", dpi=200, bbox_inches="tight"
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MAE block-based reconstruction"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./block_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--data_types",
        type=str,
        nargs="+",
        default=["real", "df", "f2f", "fsw", "nt"],
        help="Data types to analyze",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples per data type"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        try:
            args.checkpoint = find_best_checkpoint()
        except (FileNotFoundError, ValueError) as e:
            print(f"Error finding checkpoint: {e}")
            print("Please specify checkpoint path with --checkpoint")
            return 1

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.checkpoint}")
    net = mae_vit_base_patch16()
    net = nn.DataParallel(net).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    net = net.module
    net.eval()

    print(f"Output directory: {output_dir}")
    print(
        f"Analyzing {len(args.data_types)} data types with {args.num_samples} samples each"
    )

    all_metrics = {}

    for data_type in args.data_types:
        try:
            metrics = analyze_data_type(
                net, data_type, output_dir, device, args.num_samples
            )
            all_metrics[data_type] = metrics
        except FileNotFoundError as e:
            print(f"Warning: Skipping {data_type} - {e}")
            continue

    if len(all_metrics) > 1:
        print("\nCreating comparison visualizations...")
        compare_data_types(all_metrics, output_dir)

    print(f"\nVisualization complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())

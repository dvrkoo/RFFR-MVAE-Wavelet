import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rffr_generative"))

from rffr_generative.dataset import Deepfake_Dataset
from rffr_generative.configs.config import config
from rffr_generative.models.model_mae import mae_vit_base_patch16
from rffr_generative.models.model_vae import vae_base
from rffr_generative.models.model_wavelet_vae import wavelet_vae_base
from rffr_generative.models.model_multibranch_wavelet_vae import (
    multibranch_wavelet_vae_base,
)
from rffr_generative.models.model_mae_vae import mae_vae_vit_base_patch16


def set_seed(seed=912):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_generator_model(checkpoint_path, device="cuda"):
    print(f"\n{'='*80}")
    print(f"Loading generative model checkpoint from: {checkpoint_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" in ckpt:
        gen_config = ckpt["config"]
        generator_type = gen_config.get("generator_type", "mae")
        vae_latent_dim = gen_config.get("vae_latent_dim", 256)
        vae_base_channels = gen_config.get("vae_base_channels", 64)
        wavelet_vae_wavelet_type = gen_config.get("wavelet_vae_wavelet_type", "db4")
        multibranch_ll_latent = gen_config.get("multibranch_wavelet_ll_latent_dim", 128)
        multibranch_hf_latent = gen_config.get(
            "multibranch_wavelet_high_freq_latent_dim", 384
        )
        multibranch_ll_channels = gen_config.get(
            "multibranch_wavelet_ll_base_channels", 32
        )
        multibranch_hf_channels = gen_config.get(
            "multibranch_wavelet_high_freq_base_channels", 48
        )
        multibranch_hf_weight = gen_config.get(
            "multibranch_wavelet_high_freq_loss_weight", 2.0
        )
        freeze_mae_encoder = gen_config.get("freeze_mae_encoder", False)

        print(f"✓ Checkpoint contains config metadata")
        print(f"  Generator type: {generator_type}")
        if generator_type in ["vae", "wavelet_vae"]:
            print(f"  Latent dim: {vae_latent_dim}")
            print(f"  Base channels: {vae_base_channels}")
            if generator_type == "wavelet_vae":
                print(f"  Wavelet type: {wavelet_vae_wavelet_type}")
        elif generator_type == "multibranch_wavelet_vae":
            print(f"  LL latent dim: {multibranch_ll_latent}")
            print(f"  High-freq latent dim: {multibranch_hf_latent}")
            print(f"  LL base channels: {multibranch_ll_channels}")
            print(f"  High-freq base channels: {multibranch_hf_channels}")
            print(f"  Wavelet type: {wavelet_vae_wavelet_type}")
    else:
        print(f"⚠ Warning: Checkpoint has no config metadata")
        print(f"  Using defaults from config.py")
        generator_type = config.generator_type
        vae_latent_dim = config.vae_latent_dim
        vae_base_channels = config.vae_base_channels
        wavelet_vae_wavelet_type = config.wavelet_vae_wavelet_type
        multibranch_ll_latent = config.multibranch_wavelet_ll_latent_dim
        multibranch_hf_latent = config.multibranch_wavelet_high_freq_latent_dim
        multibranch_ll_channels = config.multibranch_wavelet_ll_base_channels
        multibranch_hf_channels = config.multibranch_wavelet_high_freq_base_channels
        multibranch_hf_weight = config.multibranch_wavelet_high_freq_loss_weight
        freeze_mae_encoder = config.freeze_mae_encoder
        print(f"  Generator type: {generator_type}")

    print(f"\nInitializing model architecture: {generator_type}")

    if generator_type == "vae":
        net = vae_base(latent_dim=vae_latent_dim, base_channels=vae_base_channels)
        print(
            f"✓ Loaded VAE (latent_dim={vae_latent_dim}, base_channels={vae_base_channels})"
        )

    elif generator_type == "wavelet_vae":
        net = wavelet_vae_base(
            latent_dim=vae_latent_dim,
            base_channels=vae_base_channels,
            wavelet_type=wavelet_vae_wavelet_type,
        )
        print(
            f"✓ Loaded WaveletVAE (latent_dim={vae_latent_dim}, base_channels={vae_base_channels}, wavelet={wavelet_vae_wavelet_type})"
        )

    elif generator_type == "multibranch_wavelet_vae":
        net = multibranch_wavelet_vae_base(
            ll_latent_dim=multibranch_ll_latent,
            high_freq_latent_dim=multibranch_hf_latent,
            ll_base_channels=multibranch_ll_channels,
            high_freq_base_channels=multibranch_hf_channels,
            wavelet_type=wavelet_vae_wavelet_type,
            high_freq_loss_weight=multibranch_hf_weight,
        )
        print(f"✓ Loaded MultiBranchWaveletVAE")
        print(
            f"  LL: latent={multibranch_ll_latent}, channels={multibranch_ll_channels}"
        )
        print(
            f"  HF: latent={multibranch_hf_latent}, channels={multibranch_hf_channels}"
        )

    elif generator_type == "mae_vae":
        net = mae_vae_vit_base_patch16(
            vae_latent_dim=vae_latent_dim, freeze_encoder=freeze_mae_encoder
        )
        print(
            f"✓ Loaded MAE-VAE hybrid (latent_dim={vae_latent_dim}, freeze_encoder={freeze_mae_encoder})"
        )

    else:
        net = mae_vit_base_patch16()
        print(f"✓ Loaded standard MAE")

    print(f"\nLoading checkpoint weights...")
    state_dict = ckpt["state_dict"]

    if any(k.startswith("module.") for k in state_dict.keys()):
        print(f"  Detected DataParallel checkpoint, stripping 'module.' prefix...")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠ Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"    - {key}")
        else:
            print(f"  First 5 missing keys:")
            for key in missing_keys[:5]:
                print(f"    - {key}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"    - {key}")
        else:
            print(f"  First 5 unexpected keys:")
            for key in unexpected_keys[:5]:
                print(f"    - {key}")

    if not missing_keys and not unexpected_keys:
        print(f"✓ All weights loaded successfully (strict match)")

    net = net.to(device)
    net.eval()

    print(f"✓ Model loaded and ready for inference\n")
    print(f"{'='*80}\n")

    return net, generator_type


def compute_metrics(original, reconstructed, is_vae=True):
    orig_np = original.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()

    batch_size = orig_np.shape[0]

    mse_loss = np.mean((orig_np - recon_np) ** 2)

    if is_vae:
        residual_raw = orig_np - recon_np
    else:
        residual_raw = orig_np - recon_np

    residual_scaled = residual_raw * 4
    residual_clamped = np.clip(residual_scaled, -1, 1)

    residual_mag_mean = np.mean(np.abs(residual_scaled))
    residual_mag_std = np.std(np.abs(residual_scaled))
    residual_mag_max = np.max(np.abs(residual_scaled))

    clamped_ratio = np.mean(np.abs(residual_scaled) >= 1.0)

    residual_energy = np.mean(residual_scaled**2)

    psnr_vals = []
    ssim_vals = []

    for b in range(batch_size):
        orig_img = np.transpose(orig_np[b], (1, 2, 0))
        recon_img = np.transpose(recon_np[b], (1, 2, 0))

        orig_img = np.clip(orig_img, 0, 1)
        recon_img = np.clip(recon_img, 0, 1)

        try:
            psnr_val = psnr(orig_img, recon_img, data_range=1.0)
            psnr_vals.append(psnr_val)
        except:
            pass

        try:
            ssim_val = ssim(orig_img, recon_img, data_range=1.0, channel_axis=2)
            ssim_vals.append(ssim_val)
        except:
            pass

    avg_psnr = np.mean(psnr_vals) if psnr_vals else 0.0
    avg_ssim = np.mean(ssim_vals) if ssim_vals else 0.0

    return {
        "mse": mse_loss,
        "residual_mag_mean": residual_mag_mean,
        "residual_mag_std": residual_mag_std,
        "residual_mag_max": residual_mag_max,
        "residual_energy": residual_energy,
        "clamped_ratio": clamped_ratio,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }


def test_reconstruction_single(
    net,
    generator_type,
    test_label_path,
    test_set_name,
    output_dir="output_reconstruction",
    num_samples=None,
    batch_size=1,
    deterministic=True,
    device="cuda",
):

    print(f"\n{'='*80}")
    print(f"Testing reconstruction on: {test_set_name}")
    print(f"{'='*80}\n")

    print(f"Loading test dataset from: {test_label_path}")
    with open(test_label_path) as f:
        test_dict = json.load(f)

    test_paths = [item["path"] for item in test_dict]
    test_labels = [item["label"] for item in test_dict]

    # Disable caching for test/evaluation to avoid issues with non-existent cache directories
    dataloader = DataLoader(
        Deepfake_Dataset(test_paths, train=False, precache=False), 
        batch_size=batch_size, 
        shuffle=False
    )

    print(f"Dataset size: {len(dataloader)} batches")

    test_output_dir = os.path.join(output_dir, test_set_name)
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"Output directory: {test_output_dir}\n")

    criterion = nn.MSELoss()

    is_vae = generator_type in ["vae", "wavelet_vae", "multibranch_wavelet_vae"]

    total_metrics = {
        "mse": 0.0,
        "residual_mag_mean": 0.0,
        "residual_mag_std": 0.0,
        "residual_mag_max": 0.0,
        "residual_energy": 0.0,
        "clamped_ratio": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    real_metrics = {
        "mse": 0.0,
        "residual_mag_mean": 0.0,
        "residual_mag_std": 0.0,
        "residual_mag_max": 0.0,
        "residual_energy": 0.0,
        "clamped_ratio": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    fake_metrics = {
        "mse": 0.0,
        "residual_mag_mean": 0.0,
        "residual_mag_std": 0.0,
        "residual_mag_max": 0.0,
        "residual_energy": 0.0,
        "clamped_ratio": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    count = 0
    real_count = 0
    fake_count = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc=f"Processing {test_set_name}")):
            if num_samples is not None and i >= num_samples:
                break

            data = data.to(device)
            label = test_labels[i]

            if is_vae:
                reconstructed, mu, logvar = net(data, deterministic=deterministic)
                loss = criterion(reconstructed, data)
                residual = data - reconstructed

                metrics = compute_metrics(data, reconstructed, is_vae=True)

                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                if label == 0:
                    for key in real_metrics:
                        real_metrics[key] += metrics[key]
                    real_count += 1
                else:
                    for key in fake_metrics:
                        fake_metrics[key] += metrics[key]
                    fake_count += 1

                count += 1

                save_path = os.path.join(
                    test_output_dir,
                    f'{generator_type}_{i:04d}_label{label}_mse{metrics["mse"]:.4f}.png',
                )

                save_image(
                    torch.cat([data, reconstructed, residual]),
                    save_path,
                    nrow=batch_size,
                )
            else:
                loss, pred, mask = net(data, block=True)
                loss = loss.mean()
                reconstructed = net.unpatchify(pred)

                mask_expanded = mask.unsqueeze(2)
                masked_patchified_data = net.patchify(data) * (1 - mask_expanded)
                masked_data = net.unpatchify(masked_patchified_data)
                merge = net.unpatchify(masked_patchified_data + pred * mask_expanded)

                # Full reconstruction without masking (matching classifier eval)
                if hasattr(net, "patch_by_patch_DIFF_progressive"):
                    # Use progressive method for artifact-free visualization
                    full_res, full_recon, _ = net.patch_by_patch_DIFF_progressive(
                        data, block_id=None, test=True, return_reconstructed=True
                    )
                elif hasattr(net, "patch_by_patch_DIFF"):
                    # Fallback to original method for other MAE models
                    full_res, full_recon, _ = net.patch_by_patch_DIFF(
                        data, block_id=None, test=True, return_reconstructed=True
                    )
                else:
                    # Fallback for other models
                    full_loss, full_pred, full_mask = net(data, mask_ratio=0.0)
                    full_recon = net.unpatchify(full_pred)

                metrics = compute_metrics(data, merge, is_vae=False)

                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                if label == 0:
                    for key in real_metrics:
                        real_metrics[key] += metrics[key]
                    real_count += 1
                else:
                    for key in fake_metrics:
                        fake_metrics[key] += metrics[key]
                    fake_count += 1

                count += 1

                residual = torch.abs(merge - data) * 4
                full_residual = torch.abs(full_recon - data) * 4

                save_path = os.path.join(
                    test_output_dir,
                    f'{generator_type}_{i:04d}_label{label}_mse{metrics["mse"]:.4f}.png',
                )

                save_image(
                    torch.cat(
                        [data, masked_data, merge, full_recon, residual, full_residual]
                    ),
                    save_path,
                    nrow=6,
                )

    for key in total_metrics:
        total_metrics[key] /= count if count > 0 else 1

    for key in real_metrics:
        real_metrics[key] /= real_count if real_count > 0 else 1

    for key in fake_metrics:
        fake_metrics[key] /= fake_count if fake_count > 0 else 1

    print(f"\n{test_set_name} - Results:")
    print(f"  Total samples: {count} (Real: {real_count}, Fake: {fake_count})")
    print(f"\n  Overall Metrics:")
    print(f"    MSE: {total_metrics['mse']:.6f}")
    print(f"    PSNR: {total_metrics['psnr']:.2f} dB")
    print(f"    SSIM: {total_metrics['ssim']:.4f}")
    print(
        f"    Residual Magnitude (×4): mean={total_metrics['residual_mag_mean']:.4f}, std={total_metrics['residual_mag_std']:.4f}, max={total_metrics['residual_mag_max']:.4f}"
    )
    print(f"    Residual Energy: {total_metrics['residual_energy']:.6f}")
    print(f"    Clamped Ratio: {total_metrics['clamped_ratio']:.2%}")

    if real_count > 0 and fake_count > 0:
        print(f"\n  Real Images:")
        print(f"    MSE: {real_metrics['mse']:.6f}")
        print(f"    PSNR: {real_metrics['psnr']:.2f} dB")
        print(f"    SSIM: {real_metrics['ssim']:.4f}")
        print(
            f"    Residual Magnitude (×4): mean={real_metrics['residual_mag_mean']:.4f}, std={real_metrics['residual_mag_std']:.4f}"
        )

        print(f"\n  Fake Images:")
        print(f"    MSE: {fake_metrics['mse']:.6f}")
        print(f"    PSNR: {fake_metrics['psnr']:.2f} dB")
        print(f"    SSIM: {fake_metrics['ssim']:.4f}")
        print(
            f"    Residual Magnitude (×4): mean={fake_metrics['residual_mag_mean']:.4f}, std={fake_metrics['residual_mag_std']:.4f}"
        )

        print(f"\n  Fake vs Real Ratio:")
        print(
            f"    MSE: {fake_metrics['mse'] / real_metrics['mse'] if real_metrics['mse'] > 0 else 0:.2f}x"
        )
        print(
            f"    Residual Magnitude: {fake_metrics['residual_mag_mean'] / real_metrics['residual_mag_mean'] if real_metrics['residual_mag_mean'] > 0 else 0:.2f}x"
        )

    return {
        "overall": total_metrics,
        "real": real_metrics if real_count > 0 else None,
        "fake": fake_metrics if fake_count > 0 else None,
        "counts": {"total": count, "real": real_count, "fake": fake_count},
    }


def test_reconstruction_all(
    checkpoint_path,
    test_dir="data_label/Faceforensics/excludes_hq",
    output_dir="output_reconstruction",
    num_samples=None,
    batch_size=1,
    deterministic=True,
):
    set_seed(config.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, generator_type = load_generator_model(checkpoint_path, device)

    test_label_files = sorted(glob.glob(os.path.join(test_dir, "*test_label.json")))

    if not test_label_files:
        print(f"⚠ No test label files found in {test_dir}")
        return {}

    print(f"\nFound {len(test_label_files)} test sets:")
    for f in test_label_files:
        print(f"  - {os.path.basename(f)}")
    print()

    results = {}

    for test_label_path in test_label_files:
        test_set_name = os.path.basename(test_label_path).replace(
            "_test_label.json", ""
        )

        dataset_results = test_reconstruction_single(
            net=net,
            generator_type=generator_type,
            test_label_path=test_label_path,
            test_set_name=test_set_name,
            output_dir=output_dir,
            num_samples=num_samples,
            batch_size=batch_size,
            deterministic=deterministic,
            device=device,
        )

        results[test_set_name] = dataset_results

    print(f"\n{'='*80}")
    print(f"ALL TEST SETS - RECONSTRUCTION SUMMARY")
    print(f"{'='*80}")
    print(f"Generator type: {generator_type}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"\nPer-dataset results:")
    print(f"{'-'*80}")

    total_samples = 0
    total_real = 0
    total_fake = 0

    weighted_overall = {
        "mse": 0.0,
        "residual_mag_mean": 0.0,
        "residual_mag_std": 0.0,
        "residual_mag_max": 0.0,
        "residual_energy": 0.0,
        "clamped_ratio": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    weighted_real = {
        "mse": 0.0,
        "residual_mag_mean": 0.0,
        "residual_mag_std": 0.0,
        "residual_mag_max": 0.0,
        "residual_energy": 0.0,
        "clamped_ratio": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    weighted_fake = {
        "mse": 0.0,
        "residual_mag_mean": 0.0,
        "residual_mag_std": 0.0,
        "residual_mag_max": 0.0,
        "residual_energy": 0.0,
        "clamped_ratio": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    for test_set_name in sorted(results.keys()):
        r = results[test_set_name]
        counts = r["counts"]
        overall = r["overall"]

        print(f"\n  {test_set_name}:")
        print(
            f"    Samples: {counts['total']} (Real: {counts['real']}, Fake: {counts['fake']})"
        )
        print(
            f"    MSE: {overall['mse']:.6f} | PSNR: {overall['psnr']:.2f} dB | SSIM: {overall['ssim']:.4f}"
        )
        print(
            f"    Residual Mag (×4): {overall['residual_mag_mean']:.4f} ± {overall['residual_mag_std']:.4f}"
        )

        total_samples += counts["total"]
        total_real += counts["real"]
        total_fake += counts["fake"]

        for key in weighted_overall:
            weighted_overall[key] += overall[key] * counts["total"]

        if r["real"] is not None:
            for key in weighted_real:
                weighted_real[key] += r["real"][key] * counts["real"]

        if r["fake"] is not None:
            for key in weighted_fake:
                weighted_fake[key] += r["fake"][key] * counts["fake"]

    for key in weighted_overall:
        weighted_overall[key] /= total_samples if total_samples > 0 else 1

    for key in weighted_real:
        weighted_real[key] /= total_real if total_real > 0 else 1

    for key in weighted_fake:
        weighted_fake[key] /= total_fake if total_fake > 0 else 1

    print(f"\n{'-'*80}")
    print(f"  OVERALL AGGREGATE:")
    print(
        f"    Total Samples: {total_samples} (Real: {total_real}, Fake: {total_fake})"
    )
    print(
        f"    MSE: {weighted_overall['mse']:.6f} | PSNR: {weighted_overall['psnr']:.2f} dB | SSIM: {weighted_overall['ssim']:.4f}"
    )
    print(
        f"    Residual Mag (×4): {weighted_overall['residual_mag_mean']:.4f} ± {weighted_overall['residual_mag_std']:.4f}"
    )
    print(f"    Residual Energy: {weighted_overall['residual_energy']:.6f}")
    print(f"    Clamped Ratio: {weighted_overall['clamped_ratio']:.2%}")

    if total_real > 0 and total_fake > 0:
        print(f"\n  REAL vs FAKE COMPARISON:")
        print(
            f"    Real  - MSE: {weighted_real['mse']:.6f}, Residual Mag: {weighted_real['residual_mag_mean']:.4f}"
        )
        print(
            f"    Fake  - MSE: {weighted_fake['mse']:.6f}, Residual Mag: {weighted_fake['residual_mag_mean']:.4f}"
        )
        print(
            f"    Ratio - MSE: {weighted_fake['mse'] / weighted_real['mse'] if weighted_real['mse'] > 0 else 0:.2f}x, Residual Mag: {weighted_fake['residual_mag_mean'] / weighted_real['residual_mag_mean'] if weighted_real['residual_mag_mean'] > 0 else 0:.2f}x"
        )

    print(f"{'='*80}\n")

    results_path = os.path.join(output_dir, "reconstruction_results.json")
    with open(results_path, "w") as f:
        json.dump(
            convert_numpy_types(
                {
                    "checkpoint": checkpoint_path,
                    "generator_type": generator_type,
                    "results": results,
                    "aggregate": {
                        "overall": weighted_overall,
                        "real": weighted_real if total_real > 0 else None,
                        "fake": weighted_fake if total_fake > 0 else None,
                        "counts": {
                            "total": total_samples,
                            "real": total_real,
                            "fake": total_fake,
                        },
                    },
                }
            ),
            f,
            indent=2,
        )

    print(f"Results saved to: {results_path}\n")

    return results


def test_reconstruction(
    checkpoint_path,
    test_label_path,
    output_dir="output_reconstruction",
    num_samples=None,
    batch_size=1,
    deterministic=True,
):
    set_seed(config.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, generator_type = load_generator_model(checkpoint_path, device)

    test_set_name = (
        os.path.basename(test_label_path)
        .replace("_label.json", "")
        .replace(".json", "")
    )

    dataset_results = test_reconstruction_single(
        net=net,
        generator_type=generator_type,
        test_label_path=test_label_path,
        test_set_name=test_set_name,
        output_dir=output_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        deterministic=deterministic,
        device=device,
    )

    print(f"\n{'='*80}")
    print(f"Reconstruction Test Complete")
    print(f"{'='*80}")
    print(f"Generator type: {generator_type}")
    print(f"Samples processed: {dataset_results['counts']['total']}")
    print(f"Overall Metrics:")
    print(f"  MSE: {dataset_results['overall']['mse']:.6f}")
    print(f"  PSNR: {dataset_results['overall']['psnr']:.2f} dB")
    print(f"  SSIM: {dataset_results['overall']['ssim']:.4f}")
    print(
        f"  Residual Magnitude (×4): {dataset_results['overall']['residual_mag_mean']:.4f} ± {dataset_results['overall']['residual_mag_std']:.4f}"
    )
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return dataset_results


def main():
    parser = argparse.ArgumentParser(description="Test generative model reconstruction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth.tar file)",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default=None,
        help="Path to test label JSON file (for single test set)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data_label/Faceforensics/excludes_hq",
        help="Directory containing test label files (for all test sets)",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Test on all test sets found in test-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_reconstruction",
        help="Output directory for reconstructed images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process per test set (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic sampling for VAE models (default: deterministic)",
    )
    parser.add_argument(
        "--seed", type=int, default=912, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    config.seed = args.seed

    if args.all_tests:
        test_reconstruction_all(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            deterministic=not args.stochastic,
        )
    elif args.test_label:
        test_reconstruction(
            checkpoint_path=args.checkpoint,
            test_label_path=args.test_label,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            deterministic=not args.stochastic,
        )
    else:
        print("Error: Either --test-label or --all-tests must be specified")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

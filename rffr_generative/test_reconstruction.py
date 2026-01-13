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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rffr_generative"))

from rffr_generative.dataset import Deepfake_Dataset
from rffr_generative.configs.config import config
from rffr_generative.models.model_mae import mae_vit_base_patch16
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

        print(f"✓ Checkpoint contains config metadata")
        print(f"  Generator type: {generator_type}")
        if generator_type == "mae_vae":
            print(f"  VAE Latent dim: {vae_latent_dim}")
    else:
        print(f"⚠ Warning: Checkpoint has no config metadata")
        print(f"  Using defaults from config.py")
        generator_type = config.generator_type
        vae_latent_dim = config.vae_latent_dim
        print(f"  Generator type: {generator_type}")

    print(f"\nInitializing model architecture: {generator_type}")

    if generator_type == "mae_vae":
        net = mae_vae_vit_base_patch16(
            vae_latent_dim=vae_latent_dim, freeze_encoder=True
        )
        print(f"✓ Loaded MAE-VAE hybrid (latent_dim={vae_latent_dim})")
    elif generator_type == "mae":
        net = mae_vit_base_patch16()
        print(f"✓ Loaded standard MAE")
    else:
        raise ValueError(
            f"Unsupported generator_type: {generator_type}. "
            f"Only 'mae' and 'mae_vae' are supported in RFFR-MVAE."
        )

    print(f"\nLoading checkpoint weights...")
    missing_keys, unexpected_keys = net.load_state_dict(
        ckpt["state_dict"], strict=False
    )

    if missing_keys:
        print(f"⚠ Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"    - {key}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"    - {key}")

    if not missing_keys and not unexpected_keys:
        print(f"✓ All weights loaded successfully (strict match)")

    net = net.to(device)
    net.eval()

    print(f"✓ Model loaded and ready for inference\n")
    print(f"{'='*80}\n")

    return net, generator_type


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

    print(f"Loading test dataset from: {test_label_path}")
    with open(test_label_path) as f:
        test_dict = json.load(f)

    dataloader = DataLoader(
        Deepfake_Dataset(test_dict, train=False), batch_size=batch_size, shuffle=False
    )

    print(f"Dataset size: {len(dataloader)} batches")
    print(f"Output directory: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.MSELoss()

    # MAE-VAE uses a different forward signature than standard MAE
    is_mae_vae = generator_type == "mae_vae"

    total_loss = 0.0
    count = 0

    print(f"Starting reconstruction test...")
    print(f"{'='*80}\n")

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Processing")):
            if num_samples is not None and i >= num_samples:
                break

            data = data.to(device)

            if is_mae_vae:
                # MAE-VAE forward pass
                reconstructed, mu, logvar = net(data, deterministic=deterministic)
                loss = criterion(reconstructed, data)
                residual = data - reconstructed

                total_loss += loss.item()
                count += 1

                save_path = os.path.join(
                    output_dir, f"{generator_type}_{i:04d}_loss{loss.item():.4f}.png"
                )

                save_image(
                    torch.cat([data, reconstructed, residual]),
                    save_path,
                    nrow=batch_size,
                )
            else:
                # Standard MAE forward pass
                loss, pred, mask = net(data, block=True)
                loss = loss.mean()
                reconstructed = net.unpatchify(pred)

                mask_expanded = mask.unsqueeze(2)
                masked_patchified_data = net.patchify(data) * (1 - mask_expanded)
                masked_data = net.unpatchify(masked_patchified_data)
                merge = net.unpatchify(masked_patchified_data + pred * mask_expanded)
                residual = torch.abs(merge - data) * 4

                total_loss += loss.item()
                count += 1

                save_path = os.path.join(
                    output_dir, f"{generator_type}_{i:04d}_loss{loss.item():.4f}.png"
                )

                save_image(
                    torch.cat([data, masked_data, merge, residual]), save_path, nrow=4
                )

    avg_loss = total_loss / count if count > 0 else 0.0

    print(f"\n{'='*80}")
    print(f"Reconstruction Test Complete")
    print(f"{'='*80}")
    print(f"Generator type: {generator_type}")
    print(f"Samples processed: {count}")
    print(f"Average reconstruction loss: {avg_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return avg_loss


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
        default="./data_label/ff_270/test/real_test_label.json",
        help="Path to test label JSON file",
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
        help="Number of samples to process (default: all)",
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

    test_reconstruction(
        checkpoint_path=args.checkpoint,
        test_label_path=args.test_label,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()

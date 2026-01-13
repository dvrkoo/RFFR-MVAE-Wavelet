#!/usr/bin/env python3
"""
Comprehensive test script for MAE (stock) model.
Tests on all datasets: FF++, CelebDF, DFD, DFDC, and FFHQ generators.
"""

import os
import sys
import argparse

# Parse args FIRST to set CUDA_VISIBLE_DEVICES before any torch import
def parse_args_early():
    parser = argparse.ArgumentParser(description='Comprehensive test for MAE model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output', type=str, default='test_mae_comprehensive.json')
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()

args = parse_args_early()

# Set CUDA device BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")

# CRITICAL: Configure for MAE BEFORE importing anything else from RFFR
# Add RFFR paths first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now configure the config singleton BEFORE any model imports
from configs.config import config

# Set MAE specific settings on the config singleton
config.generative_model_type = "mae"
config.mae_path = "/seidenas/users/nmarini/generative_checkpoint/mae/FFHQ_mae_STAGE1_best/best_loss_0.00928_275.pth.tar"
config.use_wavelets = False
config.wavelet_residual_branch = False
config.classifier_uses_wavelets = False
config.use_adaptive_vit = True

print(f"[CONFIG] generative_model_type = {config.generative_model_type}")
print(f"[CONFIG] mae_path = {config.mae_path}")
print(f"[CONFIG] wavelet_residual_branch = {config.wavelet_residual_branch}")
print(f"[CONFIG] use_wavelets = {config.use_wavelets}")

# Now import the rest
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_model(checkpoint_path, device):
    """Load the RFFR model from checkpoint"""
    # Re-verify config before import
    print(f"\n[PRE-IMPORT CHECK] wavelet_residual_branch = {config.wavelet_residual_branch}")
    print(f"[PRE-IMPORT CHECK] generative_model_type = {config.generative_model_type}")
    
    from models.model_detector import RFFRL
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    model = RFFRL().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.dd.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded state_dict (epoch: {epoch})")
    elif 'net' in checkpoint:
        model.dd.load_state_dict(checkpoint['net'])
        print("Loaded from 'net' key")
    else:
        model.dd.load_state_dict(checkpoint)
        print("Loaded directly")
    
    model.eval()
    return model


def evaluate_dataset(model, dataloader, dataset_name, device):
    """Evaluate model on a single dataset"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset_name}", leave=False):
            # Dataset returns (normed, unnormed, label)
            normed, unnormed, labels = batch
            rgb_01 = unnormed  # [0,1] range
            rgb_norm = normed  # Normalized
            rgb_01 = rgb_01.to(device)
            rgb_norm = rgb_norm.to(device)
            labels = labels.to(device)
            
            feature, outputs, _ = model(rgb_01, rgb_norm, test=True)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate AUC
    if len(np.unique(all_labels)) < 2:
        auc = 0.0
        print(f"Warning: {dataset_name} has only one class!")
    else:
        auc = roc_auc_score(all_labels, all_probs)
    
    return auc, len(all_labels)


def load_json_labels(path):
    """Load labels from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def create_balanced_dataset(real_data, fake_data, max_samples=None):
    """Create balanced dataset from real and fake samples"""
    if max_samples:
        real_data = random.sample(real_data, min(len(real_data), max_samples))
        fake_data = random.sample(fake_data, min(len(fake_data), max_samples))
    
    combined = real_data + fake_data
    random.shuffle(combined)
    return combined


def main():
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Config was already set at module level
    print(f"\n[MAIN] Verifying config settings:")
    print(f"  generative_model_type: {config.generative_model_type}")
    print(f"  mae_path: {config.mae_path}")
    print(f"  wavelet_residual_branch: {config.wavelet_residual_branch}")
    
    # Now import dataset (after config is set)
    from utils.dataset import Deepfake_Dataset
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Base paths
    data_label_base = "/andromeda/personal/nmarini/RFFR/data_label/"
    ff_base = data_label_base + "Faceforensics/excludes_hq/"
    ffhq_base = data_label_base + "ffhq_classifier_FFHQ_SG123_STAGE2/train/"
    dfdc_base = data_label_base + "DFDC/"
    
    results = {
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": {
            "generative_model_type": config.generative_model_type,
            "wavelet_residual_branch": config.wavelet_residual_branch,
        },
        "datasets": {}
    }
    
    # ============== FF++ Tests ==============
    print("\n" + "="*60)
    print("Testing on FaceForensics++ datasets")
    print("="*60)
    
    # Load FF++ real test data
    ff_real_test = load_json_labels(ff_base + "real_test_label.json")
    print(f"FF++ Real test samples: {len(ff_real_test)}")
    
    ff_forgeries = {
        "DF": "df_test_label.json",
        "F2F": "f2f_test_label.json",
        "FSW": "fsw_test_label.json",
        "NT": "nt_test_label.json",
        "FS": "fs_test_label.json",
    }
    
    for name, label_file in ff_forgeries.items():
        print(f"\nTesting FF++ {name}...")
        fake_data = load_json_labels(ff_base + label_file)
        print(f"  Fake samples: {len(fake_data)}")
        
        test_data = create_balanced_dataset(ff_real_test, fake_data, args.max_samples)
        dataloader = DataLoader(
            Deepfake_Dataset(test_data, train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        auc, n_samples = evaluate_dataset(model, dataloader, f"FF++_{name}", device)
        results["datasets"][f"FF++_{name}"] = {"auc": round(float(auc), 4), "samples": n_samples}
        print(f"  AUC: {auc:.4f} ({n_samples} samples)")
    
    # ============== CelebDF Test ==============
    print("\n" + "="*60)
    print("Testing on CelebDF")
    print("="*60)
    
    celebdf_real = load_json_labels(ff_base + "celebdf_real_test_label_fixed.json")
    celebdf_fake = load_json_labels(ff_base + "celebdf_test_label_fixed.json")
    print(f"CelebDF Real: {len(celebdf_real)}, Fake: {len(celebdf_fake)}")
    
    celebdf_data = create_balanced_dataset(celebdf_real, celebdf_fake, args.max_samples)
    dataloader = DataLoader(
        Deepfake_Dataset(celebdf_data, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    auc, n_samples = evaluate_dataset(model, dataloader, "CelebDF", device)
    results["datasets"]["CelebDF"] = {"auc": round(float(auc), 4), "samples": n_samples}
    print(f"CelebDF AUC: {auc:.4f} ({n_samples} samples)")
    
    # ============== DFD Test ==============
    print("\n" + "="*60)
    print("Testing on DFD (DeepFake Detection)")
    print("="*60)
    
    dfd_real = load_json_labels(ff_base + "dfd_real_test_label_fixed.json")
    dfd_fake = load_json_labels(ff_base + "dfd_test_label_fixed.json")
    print(f"DFD Real: {len(dfd_real)}, Fake: {len(dfd_fake)}")
    
    dfd_data = create_balanced_dataset(dfd_real, dfd_fake, args.max_samples)
    dataloader = DataLoader(
        Deepfake_Dataset(dfd_data, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    auc, n_samples = evaluate_dataset(model, dataloader, "DFD", device)
    results["datasets"]["DFD"] = {"auc": round(float(auc), 4), "samples": n_samples}
    print(f"DFD AUC: {auc:.4f} ({n_samples} samples)")
    
    # ============== DFDC Test ==============
    print("\n" + "="*60)
    print("Testing on DFDC")
    print("="*60)
    
    dfdc_real = load_json_labels(dfdc_base + "real_test_label.json")
    dfdc_fake = load_json_labels(dfdc_base + "all_test_label.json")
    print(f"DFDC Real: {len(dfdc_real)}, Fake: {len(dfdc_fake)}")
    
    # Use max 7000 samples for DFDC (it's large)
    max_dfdc = args.max_samples if args.max_samples else 7000
    dfdc_data = create_balanced_dataset(dfdc_real, dfdc_fake, max_dfdc)
    dataloader = DataLoader(
        Deepfake_Dataset(dfdc_data, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    auc, n_samples = evaluate_dataset(model, dataloader, "DFDC", device)
    results["datasets"]["DFDC"] = {"auc": round(float(auc), 4), "samples": n_samples}
    print(f"DFDC AUC: {auc:.4f} ({n_samples} samples)")
    
    # ============== FFHQ Generator Tests ==============
    print("\n" + "="*60)
    print("Testing on FFHQ Generators")
    print("="*60)
    
    # Load FFHQ real test data
    ffhq_real_test = load_json_labels(ffhq_base + "real_test_label.json")
    print(f"FFHQ Real test samples: {len(ffhq_real_test)}")
    
    ffhq_generators = {
        "StyleGAN1": "stylegan1_test_label.json",
        "StyleGAN2": "stylegan2_test_label.json", 
        "StyleGAN3": "stylegan3_test_label.json",
        "StyleGAN-XL": "styleganxl_test_label.json",
        "SD_v1.4": "sdv1_4_test_label.json",
        "SD_v2.1": "sdv2_1_test_label.json",
    }
    
    for name, label_file in ffhq_generators.items():
        label_path = ffhq_base + label_file
        if not os.path.exists(label_path):
            print(f"\nSkipping {name} - label file not found: {label_path}")
            continue
            
        print(f"\nTesting FFHQ {name}...")
        fake_data = load_json_labels(label_path)
        print(f"  Fake samples: {len(fake_data)}")
        
        test_data = create_balanced_dataset(ffhq_real_test, fake_data, args.max_samples)
        dataloader = DataLoader(
            Deepfake_Dataset(test_data, train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        auc, n_samples = evaluate_dataset(model, dataloader, f"FFHQ_{name}", device)
        results["datasets"][f"FFHQ_{name}"] = {"auc": round(float(auc), 4), "samples": n_samples}
        print(f"  AUC: {auc:.4f} ({n_samples} samples)")
    
    # ============== Summary ==============
    print("\n" + "="*60)
    print("SUMMARY - All Results")
    print("="*60)
    
    # Group results
    ff_results = {k: v for k, v in results["datasets"].items() if k.startswith("FF++")}
    cross_dataset = {k: v for k, v in results["datasets"].items() if k in ["CelebDF", "DFD", "DFDC"]}
    ffhq_results = {k: v for k, v in results["datasets"].items() if k.startswith("FFHQ_")}
    
    print("\nFF++ Forgeries:")
    for name, data in ff_results.items():
        print(f"  {name}: {data['auc']:.4f}")
    if ff_results:
        avg_ff = np.mean([v['auc'] for v in ff_results.values()])
        print(f"  Average: {avg_ff:.4f}")
    
    print("\nCross-Dataset (OOD):")
    for name, data in cross_dataset.items():
        print(f"  {name}: {data['auc']:.4f}")
    if cross_dataset:
        avg_cross = np.mean([v['auc'] for v in cross_dataset.values()])
        print(f"  Average: {avg_cross:.4f}")
    
    print("\nFFHQ Generators:")
    ffhq_in_dist = {k: v for k, v in ffhq_results.items() if any(x in k for x in ["StyleGAN1", "StyleGAN2", "StyleGAN3"]) and "XL" not in k}
    ffhq_ood = {k: v for k, v in ffhq_results.items() if any(x in k for x in ["XL", "SD_"])}
    
    print("  In-Distribution (SG1/2/3):")
    for name, data in ffhq_in_dist.items():
        print(f"    {name}: {data['auc']:.4f}")
    if ffhq_in_dist:
        avg_in = np.mean([v['auc'] for v in ffhq_in_dist.values()])
        print(f"    Average: {avg_in:.4f}")
    
    print("  OOD (SG-XL, SD):")
    for name, data in ffhq_ood.items():
        print(f"    {name}: {data['auc']:.4f}")
    if ffhq_ood:
        avg_ood = np.mean([v['auc'] for v in ffhq_ood.values()])
        print(f"    Average: {avg_ood:.4f}")
    
    # Save results
    output_path = f"/andromeda/personal/nmarini/RFFR/rffr_classifier/logs/{args.output}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

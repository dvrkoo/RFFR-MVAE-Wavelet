#!/usr/bin/env python3
"""
Test RFFR models on FFHQ dataset with multiple generators

This script tests RFFR models trained with 3 fake frames on the FFHQ benchmark
across multiple generative models:
- StyleGAN1
- StyleGAN2
- StyleGAN3
- StyleGANXL
- Stable Diffusion v1.4
- Stable Diffusion v2.1

Author: OpenCode
Date: 2025-12-20
"""

import sys
import os
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# Add RFFR paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model_detector import RFFRL
from utils.dataset import Deepfake_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test RFFR on FFHQ with multiple generators')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to RFFR checkpoint file (.pth.tar)')
    
    # Dataset paths
    parser.add_argument('--ffhq_root', type=str, default='/oblivion/Datasets/FFHQ',
                        help='Path to FFHQ dataset root')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=666,
                        help='Random seed for reproducibility')
    
    # GPU
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    # Generator selection
    parser.add_argument('--generators', type=str, nargs='+',
                        default=['stylegan1-psi-0.5', 'stylegan2-psi-0.5', 'stylegan3-psi-0.5',
                                'styleganxl-psi-0.5', 'sdv1_4', 'sdv2_1'],
                        help='List of generators to test')
    
    return parser.parse_args()


def load_rffr_model(args):
    """Load trained RFFR model with checkpoint key remapping"""
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = RFFRL().cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Extract state dict from checkpoint
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
        print(f"Loaded from checkpoint['net'] (epoch: {checkpoint.get('epoch', 'unknown')})")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded from checkpoint['model_state_dict']")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Loaded from checkpoint['state_dict'] (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        state_dict = checkpoint
        print(f"Loaded checkpoint directly")
    
    # Fix missing 'dd.' prefix for deepfake detector components
    # The checkpoint stores keys like 'backbone_1.*', 'backbone_2.*', 'backbone_3.*', 'classifier.*'
    # But RFFRL model expects 'dd.backbone_1.*', 'dd.backbone_2.*', 'dd.backbone_3.*', 'dd.classifier.*'
    new_state_dict = {}
    remapped_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('backbone_') or key.startswith('classifier'):
            new_key = 'dd.' + key  # Add dd. prefix for detector components
            new_state_dict[new_key] = value
            remapped_count += 1
        else:
            new_state_dict[key] = value
    
    if remapped_count > 0:
        print(f"Remapped {remapped_count} keys (added 'dd.' prefix for detector components)")
    
    # Load with strict=False to handle potential mismatches gracefully
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys")
        if len(missing_keys) <= 5:
            for k in missing_keys:
                print(f"  - {k}")
        else:
            print(f"  First 5: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys")
        if len(unexpected_keys) <= 5:
            for k in unexpected_keys:
                print(f"  - {k}")
        else:
            print(f"  First 5: {unexpected_keys[:5]}")
    
    if not missing_keys and not unexpected_keys:
        print(f"✓ Model loaded successfully with all keys matched!\n")
    else:
        print(f"✓ Model loaded (with warnings - check if critical components are missing)\n")
    
    model.eval()
    
    return model


def compute_accuracy(labels, predictions):
    """Compute binary accuracy"""
    pred_labels = (predictions > 0.5).astype(int)
    return np.mean(pred_labels == labels)


def create_ffhq_dataset(ffhq_root, generator, split='test_set'):
    """
    Create dataset dict for FFHQ
    
    Args:
        ffhq_root: Path to FFHQ root
        generator: Generator name (e.g., 'stylegan2-psi-0.5')
        split: 'test_set', 'val_set', or 'train_set'
    
    Returns:
        List of dicts with 'path' and 'label' keys
    """
    real_dir = Path(ffhq_root) / 'images1024x1024' / split
    fake_dir = Path(ffhq_root) / 'generated' / generator / 'images1024x1024' / split
    
    if not real_dir.exists():
        raise FileNotFoundError(f"Real images directory not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake images directory not found: {fake_dir}")
    
    # Get all image files
    real_images = sorted(list(real_dir.glob('*.png')) + list(real_dir.glob('*.jpg')))
    fake_images = sorted(list(fake_dir.glob('*.png')) + list(fake_dir.glob('*.jpg')))
    
    # Create dataset dict
    data_dict = []
    
    # Add real images (label=0)
    for img_path in real_images:
        data_dict.append({
            'path': str(img_path),
            'label': 0
        })
    
    # Add fake images (label=1)
    for img_path in fake_images:
        data_dict.append({
            'path': str(img_path),
            'label': 1
        })
    
    print(f"  Real images: {len(real_images)}")
    print(f"  Fake images: {len(fake_images)}")
    print(f"  Total: {len(data_dict)}")
    
    return data_dict


def test_generator(model, args, generator):
    """Test on a specific generator"""
    print(f"\n{'='*80}")
    print(f"Testing {generator}")
    print(f"{'='*80}\n")
    
    try:
        # Create dataset
        data_dict = create_ffhq_dataset(args.ffhq_root, generator, split='test_set')
        
        dataset = Deepfake_Dataset(data_dict, train=False)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        
        # Test
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for normed, unnormed, labels in tqdm(dataloader, desc=f"Testing {generator}"):
                normed = normed.cuda()
                unnormed = unnormed.cuda()
                
                # Forward pass - RFFR returns (logits, features)
                outputs = model(normed, unnormed)
                
                # Get logits (first output)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits).squeeze()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        acc = compute_accuracy(all_labels, all_probs) * 100
        auc = roc_auc_score(all_labels, all_probs) * 100
        
        # Compute per-class accuracy
        real_mask = all_labels == 0
        fake_mask = all_labels == 1
        
        real_acc = compute_accuracy(all_labels[real_mask], all_probs[real_mask]) * 100
        fake_acc = compute_accuracy(all_labels[fake_mask], all_probs[fake_mask]) * 100
        
        print(f"\nResults for {generator}:")
        print(f"  Total samples: {len(all_labels)} (Real: {np.sum(real_mask)}, Fake: {np.sum(fake_mask)})")
        print(f"  Overall ACC: {acc:.2f}%")
        print(f"  Overall AUC: {auc:.2f}%")
        print(f"  Real ACC: {real_acc:.2f}%")
        print(f"  Fake ACC: {fake_acc:.2f}%")
        
        return {
            'acc': acc,
            'auc': auc,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'total_samples': len(all_labels),
            'num_real': int(np.sum(real_mask)),
            'num_fake': int(np.sum(fake_mask))
        }
    
    except Exception as e:
        print(f"Error testing {generator}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary_table(results):
    """Print comprehensive summary table"""
    print(f"\n{'='*100}")
    print("RFFR FFHQ TEST RESULTS (Multiple Generators)")
    print(f"{'='*100}\n")
    
    print(f"{'Generator':<30} {'Samples':<12} {'ACC':<12} {'AUC':<12} {'Real_ACC':<12} {'Fake_ACC':<12}")
    print("-" * 100)
    
    for gen, res in results.items():
        if res:
            print(f"{gen:<30} {res['total_samples']:<12} "
                  f"{res['acc']:>10.2f}%  {res['auc']:>10.2f}%  "
                  f"{res['real_acc']:>10.2f}%  {res['fake_acc']:>10.2f}%")
    
    # Compute average across all generators
    if results:
        valid_results = [r for r in results.values() if r is not None]
        if valid_results:
            avg_acc = np.mean([r['acc'] for r in valid_results])
            avg_auc = np.mean([r['auc'] for r in valid_results])
            avg_real_acc = np.mean([r['real_acc'] for r in valid_results])
            avg_fake_acc = np.mean([r['fake_acc'] for r in valid_results])
            
            print("-" * 100)
            print(f"{'AVERAGE':<30} {'':<12} "
                  f"{avg_acc:>10.2f}%  {avg_auc:>10.2f}%  "
                  f"{avg_real_acc:>10.2f}%  {avg_fake_acc:>10.2f}%")
    
    print(f"{'='*100}\n")


def save_results(results, args):
    """Save results to JSON and TXT files"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Compute average
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        avg_acc = np.mean([r['acc'] for r in valid_results.values()])
        avg_auc = np.mean([r['auc'] for r in valid_results.values()])
        avg_real_acc = np.mean([r['real_acc'] for r in valid_results.values()])
        avg_fake_acc = np.mean([r['fake_acc'] for r in valid_results.values()])
        
        results['AVERAGE'] = {
            'acc': avg_acc,
            'auc': avg_auc,
            'real_acc': avg_real_acc,
            'fake_acc': avg_fake_acc,
            'total_samples': sum(r['total_samples'] for r in valid_results.values()),
            'num_real': sum(r['num_real'] for r in valid_results.values()),
            'num_fake': sum(r['num_fake'] for r in valid_results.values())
        }
    
    # Save JSON
    json_file = output_dir / f'test_ffhq_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'checkpoint': str(args.checkpoint),
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {json_file}")
    
    # Save TXT
    txt_file = output_dir / f'test_ffhq_{timestamp}.txt'
    with open(txt_file, 'w') as f:
        f.write(f"RFFR FFHQ TEST RESULTS (Multiple Generators)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"{'Generator':<30} {'Samples':<12} {'ACC':<12} {'AUC':<12} {'Real_ACC':<12} {'Fake_ACC':<12}\n")
        f.write("-" * 100 + "\n")
        
        for gen, res in results.items():
            if res and gen != 'AVERAGE':
                f.write(f"{gen:<30} {res['total_samples']:<12} "
                        f"{res['acc']:>10.2f}%  {res['auc']:>10.2f}%  "
                        f"{res['real_acc']:>10.2f}%  {res['fake_acc']:>10.2f}%\n")
        
        if 'AVERAGE' in results:
            res = results['AVERAGE']
            f.write("-" * 100 + "\n")
            f.write(f"{'AVERAGE':<30} {'':<12} "
                    f"{res['acc']:>10.2f}%  {res['auc']:>10.2f}%  "
                    f"{res['real_acc']:>10.2f}%  {res['fake_acc']:>10.2f}%\n")
        
        f.write(f"{'='*100}\n")
    
    print(f"Summary saved to: {txt_file}\n")


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Print configuration
    print(f"\n{'='*100}")
    print("RFFR FFHQ TESTING (Multiple Generators)")
    print(f"{'='*100}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FFHQ Root: {args.ffhq_root}")
    print(f"GPU: {args.gpu}")
    print(f"Random Seed: {args.seed}")
    print(f"\nGenerators to test:")
    for gen in args.generators:
        print(f"  - {gen}")
    print(f"{'='*100}\n")
    
    # Load model
    model = load_rffr_model(args)
    
    # Results dictionary
    results = {}
    
    # Test each generator
    for generator in args.generators:
        result = test_generator(model, args, generator)
        if result:
            results[generator] = result
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    save_results(results, args)
    
    print("Testing completed successfully!")


if __name__ == '__main__':
    main()

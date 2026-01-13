#!/usr/bin/env python3
"""
Simple test script for DFDC dataset using RFFR model.
This script tests a single model on the DFDC dataset.
"""

import random
import numpy as np
import argparse
import os
import torch
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from configs.config import config
from models.model_detector import RFFRL
from utils.simple_evaluate import eval_one_dataset
from torch.utils.data import DataLoader
from utils.dataset import Deepfake_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test RFFR on DFDC dataset')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint. If not provided, uses latest model.'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=7000,
        help='Maximum number of samples per class (default: 7000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for testing (default: 64)'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results to JSON file'
    )
    return parser.parse_args()


def get_latest_checkpoint():
    """Get the latest checkpoint from the best_model directory"""
    import glob
    from pathlib import Path
    
    best_model_dir = Path('./checkpoint/rffr/best_model/')
    if not best_model_dir.exists():
        return None
    
    # Find all training run directories
    run_dirs = [d for d in best_model_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Get the most recent run
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    
    # Find checkpoints in the run directory
    checkpoints = list(latest_run.glob('*.pth.tar'))
    if not checkpoints:
        return None
    
    # Prefer AUC checkpoint, otherwise use the most recent
    auc_checkpoints = [c for c in checkpoints if 'AUC' in c.name]
    if auc_checkpoints:
        return str(auc_checkpoints[0])
    
    return str(max(checkpoints, key=lambda x: x.stat().st_mtime))


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    net = RFFRL().to(device)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f'Error: Checkpoint file not found: {checkpoint_path}')
            return
    else:
        checkpoint_path = get_latest_checkpoint()
        if not checkpoint_path:
            print('Error: No checkpoint found. Please specify --checkpoint')
            return
        print(f'Using latest checkpoint: {checkpoint_path}')
    
    # Load checkpoint
    print(f'\nLoading model from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.dd.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    # Load DFDC data
    print('\nLoading DFDC dataset...')
    dfdc_fake_path = config.dataset_base + 'DFDC/all_test_label.json'
    dfdc_real_path = config.dataset_base + 'DFDC/real_test_label.json'
    
    with open(dfdc_fake_path) as f:
        dfdc_fake_dict = json.load(f)
    
    with open(dfdc_real_path) as f:
        dfdc_real_dict = json.load(f)
    
    print(f'Loaded {len(dfdc_fake_dict)} fake samples')
    print(f'Loaded {len(dfdc_real_dict)} real samples')
    
    # Create balanced test set
    num_samples = min(len(dfdc_fake_dict), len(dfdc_real_dict), args.samples)
    test_fake_subset = random.sample(dfdc_fake_dict, num_samples)
    test_real_subset = random.sample(dfdc_real_dict, num_samples)
    
    balanced_test_dict = test_real_subset + test_fake_subset
    random.shuffle(balanced_test_dict)
    
    real_count = sum(1 for item in balanced_test_dict if item['label'] == 0)
    fake_count = sum(1 for item in balanced_test_dict if item['label'] == 1)
    print(f'\nBalanced test set: {real_count} real + {fake_count} fake = {len(balanced_test_dict)} total')
    
    # Create data loader
    test_dataloader = DataLoader(
        Deepfake_Dataset(balanced_test_dict, train=False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Run evaluation
    print('\nRunning evaluation on DFDC...')
    result = eval_one_dataset(
        test_dataloader,
        net,
        'DFDC',
        track_bandit=True,
        capture_visualizations=False
    )
    
    # Handle different return formats
    if isinstance(result, tuple) and len(result) >= 2:
        auc = result[0]
    else:
        auc = result
    
    print(f'\n' + '='*80)
    print(f'DFDC Test Results')
    print('='*80)
    print(f'Model: {os.path.basename(checkpoint_path)}')
    print(f'AUC: {round(float(auc), 4)}')
    print('='*80)
    
    # Save results if requested
    if args.save_json:
        os.makedirs('logs', exist_ok=True)
        results = {
            'checkpoint': checkpoint_path,
            'dataset': 'DFDC',
            'auc': float(auc),
            'samples': {
                'real': real_count,
                'fake': fake_count,
                'total': len(balanced_test_dict)
            }
        }
        
        output_file = 'logs/test_dfdc_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {output_file}')
    
    # Append to evaluate log
    os.makedirs('logs', exist_ok=True)
    with open('logs/evaluate.txt', 'a') as f:
        f.write('\nModel: ' + os.path.basename(checkpoint_path))
        f.write('\nDataset: DFDC')
        f.write('\nAUC: ' + str(round(float(auc), 4)))
        f.write('\n')


if __name__ == '__main__':
    main()

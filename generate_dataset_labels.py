#!/usr/bin/env python3
"""
Unified Label Generation Script for RFFR-MVAE

This script replaces all individual label generation scripts (create_labels.py,
create_ffhq_labels.py, create_fs_labels.py, create_reduced_labels.py) with a
single, unified, well-tested tool.

Supports:
- FaceForensics++ (FF++) dataset
- FFHQ dataset  
- Custom image directories
- Both generative (real-only) and classifier (real+fake) training
- Frame sampling strategies (uniform, middle-frame, all frames)
- Built-in validation and statistics
- Integration with rffr_generative dataset configuration

Usage Examples:
    # FaceForensics++: Generate all splits and types
    python generate_dataset_labels.py --dataset ff++ \\
      --ff-root /path/to/FF++ --output-dir ./data_label
    
    # FFHQ: Generate Stage 1 labels (generative training, real only)
    python generate_dataset_labels.py --dataset ffhq \\
      --ffhq-root /path/to/FFHQ --stage 1 --output-dir ./data_label
    
    # Custom: Generate from arbitrary image directory
    python generate_dataset_labels.py --dataset custom \\
      --image-dir /path/to/images --dataset-name my_dataset \\
      --split train --type real --label 0
    
    # Validate existing labels
    python generate_dataset_labels.py --validate \\
      --label-file ./data_label/ff_270/train/real_train_label.json
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import sys

# Seed for reproducibility
random.seed(912)

# ============================================================================
# FACEFORENSICS++ SPECIFIC LOGIC
# ============================================================================

def get_ff_video_splits() -> Tuple[List[str], List[str], List[str]]:
    """
    FaceForensics++ official train/val/test splits.
    Returns video IDs for train (720), val (140), test (140) sets.
    """
    all_videos = [f"{i:03d}" for i in range(1000)]
    train_videos = all_videos[:720]    # 000-719
    val_videos = all_videos[720:860]   # 720-859
    test_videos = all_videos[860:]     # 860-999
    return train_videos, val_videos, test_videos


def collect_ff_images(
    root_dir: str,
    split_videos: List[str],
    is_real: bool = True,
    manipulation_type: Optional[str] = None,
    frames_per_video: int = 10,
    single_frame_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Collect image paths from FaceForensics++ dataset.
    
    Args:
        root_dir: Root directory of FF++ (contains original_sequences/, manipulated_sequences/)
        split_videos: List of video IDs to include (e.g., ['000', '001', ...])
        is_real: True for real images, False for fake
        manipulation_type: For fake images: 'Deepfakes', 'Face2Face', 'FaceSwap', 
                          'FaceShifter', 'NeuralTextures'
        frames_per_video: Number of frames to sample per video
        single_frame_mode: If True, only take middle frame (overrides frames_per_video)
    
    Returns:
        List of {'path': str, 'label': int} dictionaries
    """
    data = []
    label = 0 if is_real else 1
    
    if single_frame_mode:
        frames_per_video = 1
    
    if is_real:
        # Real images from original_sequences/youtube/c23/images/{video_id}/*.png
        base_path = Path(root_dir) / "original_sequences" / "youtube" / "c23" / "images"
        
        for video_id in split_videos:
            video_dir = base_path / video_id
            if video_dir.exists():
                all_images = sorted(video_dir.glob("*.png"))
                selected = _sample_frames(all_images, frames_per_video, single_frame_mode)
                
                for img_file in selected:
                    data.append({"path": str(img_file.resolve()), "label": label})
    else:
        # Fake images from manipulated_sequences/{type}/c23/images/{video_pair}/*.png
        if manipulation_type is None:
            raise ValueError("manipulation_type required for fake images")
        
        base_path = Path(root_dir) / "manipulated_sequences" / manipulation_type / "c23" / "images"
        
        if base_path.exists():
            for video_pair_dir in sorted(base_path.iterdir()):
                if video_pair_dir.is_dir():
                    # Extract source video ID (e.g., "000_003" -> "000")
                    source_video = video_pair_dir.name.split("_")[0]
                    
                    if source_video in split_videos:
                        all_images = sorted(video_pair_dir.glob("*.png"))
                        selected = _sample_frames(all_images, frames_per_video, single_frame_mode)
                        
                        for img_file in selected:
                            data.append({"path": str(img_file.resolve()), "label": label})
    
    return data


def _sample_frames(all_images: List[Path], frames_per_video: int, single_frame: bool) -> List[Path]:
    """Sample frames from video uniformly or take middle frame."""
    if len(all_images) == 0:
        return []
    
    if single_frame:
        # Take middle frame
        middle_idx = len(all_images) // 2
        return [all_images[middle_idx]]
    elif len(all_images) <= frames_per_video:
        # Take all frames if fewer than requested
        return all_images
    else:
        # Uniformly sample frames_per_video frames
        step = len(all_images) / frames_per_video
        indices = [int(i * step) for i in range(frames_per_video)]
        return [all_images[i] for i in indices]


def generate_ff_labels(
    ff_root: str,
    output_dir: str,
    frames_per_video: int = 10,
    single_frame_val_test: bool = False,
    reduced_fake_frames: Optional[int] = None,
):
    """
    Generate all FaceForensics++ label files.
    
    Creates:
    - data_label/ff_270/train/ - Training labels (720 videos)
    - data_label/Faceforensics/excludes_hq/ - Val/test labels (140 each)
    - data_label/FN/train/ - Copy of real training labels for generative training
    
    Args:
        ff_root: Root directory of FF++ dataset
        output_dir: Output directory for label files
        frames_per_video: Frames to sample per video for training
        single_frame_val_test: Use only 1 frame per video for val/test
        reduced_fake_frames: If set, use different frame count for fake images
    """
    print("\n" + "="*80)
    print("GENERATING FACEFORENSICS++ LABELS")
    print("="*80)
    
    root_path = Path(ff_root)
    if not root_path.exists():
        raise FileNotFoundError(f"FF++ root not found: {ff_root}")
    
    output_path = Path(output_dir)
    
    # Create output directories
    ff_train_dir = output_path / "ff_270" / "train"
    ff_val_test_dir = output_path / "Faceforensics" / "excludes_hq"
    fn_train_dir = output_path / "FN" / "train"
    
    for dir_path in [ff_train_dir, ff_val_test_dir, fn_train_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get splits
    train_videos, val_videos, test_videos = get_ff_video_splits()
    
    print(f"\nSplits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    print(f"Training frames per video: {frames_per_video}")
    if single_frame_val_test:
        print(f"Val/Test: Single frame mode (middle frame only)")
    else:
        print(f"Val/Test frames per video: {frames_per_video}")
    if reduced_fake_frames:
        print(f"Fake images: Using {reduced_fake_frames} frames per video")
    
    # Manipulation types
    manipulations = {
        "df": "Deepfakes",
        "f2f": "Face2Face",
        "fs": "FaceSwap",
        "fsw": "FaceShifter",
        "nt": "NeuralTextures",
    }
    
    # === TRAINING SPLIT ===
    print("\n" + "-"*80)
    print("GENERATING TRAINING LABELS")
    print("-"*80)
    
    # Real training
    print("\n[1/6] Collecting real training images...")
    real_train = collect_ff_images(
        ff_root, train_videos, is_real=True, frames_per_video=frames_per_video
    )
    print(f"  Found: {len(real_train)} images")
    
    output_file = ff_train_dir / "real_train_label.json"
    with open(output_file, 'w') as f:
        json.dump(real_train, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Fake training (all manipulation types)
    all_fake_train = []
    for idx, (short_name, full_name) in enumerate(manipulations.items(), start=2):
        print(f"\n[{idx}/6] Collecting {full_name} training images...")
        
        fake_frames = reduced_fake_frames if reduced_fake_frames else frames_per_video
        
        fake_train = collect_ff_images(
            ff_root,
            train_videos,
            is_real=False,
            manipulation_type=full_name,
            frames_per_video=fake_frames,
        )
        print(f"  Found: {len(fake_train)} images")
        
        # Save individual manipulation type
        output_file = ff_train_dir / f"{short_name}_train_label.json"
        with open(output_file, 'w') as f:
            json.dump(fake_train, f, indent=2)
        print(f"  Saved: {output_file}")
        
        all_fake_train.extend(fake_train)
    
    # Save mixed fake training
    print(f"\nCombining all fake manipulations: {len(all_fake_train)} images")
    output_file = ff_train_dir / "mixed_fake_train_label.json"
    with open(output_file, 'w') as f:
        json.dump(all_fake_train, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # === VALIDATION SPLIT ===
    print("\n" + "-"*80)
    print("GENERATING VALIDATION LABELS")
    print("-"*80)
    
    # Real validation
    print("\nCollecting real validation images...")
    real_val = collect_ff_images(
        ff_root,
        val_videos,
        is_real=True,
        frames_per_video=frames_per_video,
        single_frame_mode=single_frame_val_test,
    )
    print(f"  Found: {len(real_val)} images")
    
    output_file = ff_val_test_dir / "real_val_label.json"
    with open(output_file, 'w') as f:
        json.dump(real_val, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Fake validation
    for short_name, full_name in manipulations.items():
        print(f"\nCollecting {full_name} validation images...")
        
        val_data = collect_ff_images(
            ff_root,
            val_videos,
            is_real=False,
            manipulation_type=full_name,
            frames_per_video=frames_per_video,
            single_frame_mode=single_frame_val_test,
        )
        print(f"  Found: {len(val_data)} images")
        
        output_file = ff_val_test_dir / f"{short_name}_val_label.json"
        with open(output_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"  Saved: {output_file}")
    
    # === TEST SPLIT ===
    print("\n" + "-"*80)
    print("GENERATING TEST LABELS")
    print("-"*80)
    
    # Real test
    print("\nCollecting real test images...")
    real_test = collect_ff_images(
        ff_root,
        test_videos,
        is_real=True,
        frames_per_video=frames_per_video,
        single_frame_mode=single_frame_val_test,
    )
    print(f"  Found: {len(real_test)} images")
    
    output_file = ff_val_test_dir / "real_test_label.json"
    with open(output_file, 'w') as f:
        json.dump(real_test, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Fake test
    for short_name, full_name in manipulations.items():
        print(f"\nCollecting {full_name} test images...")
        
        test_data = collect_ff_images(
            ff_root,
            test_videos,
            is_real=False,
            manipulation_type=full_name,
            frames_per_video=frames_per_video,
            single_frame_mode=single_frame_val_test,
        )
        print(f"  Found: {len(test_data)} images")
        
        output_file = ff_val_test_dir / f"{short_name}_test_label.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"  Saved: {output_file}")
    
    # === GENERATIVE TRAINING (Copy of real training) ===
    print("\n" + "-"*80)
    print("GENERATING GENERATIVE TRAINING LABELS (FN/train/)")
    print("-"*80)
    
    output_file = fn_train_dir / "real_train_label.json"
    with open(output_file, 'w') as f:
        json.dump(real_train, f, indent=2)
    print(f"  Saved: {output_file} ({len(real_train)} images)")
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("FF++ LABEL GENERATION COMPLETE!")
    print("="*80)
    print(f"\nTraining split ({len(train_videos)} videos):")
    print(f"  Real: {len(real_train)} images")
    print(f"  Fake (mixed): {len(all_fake_train)} images")
    print(f"\nValidation split ({len(val_videos)} videos):")
    print(f"  Real: {len(real_val)} images")
    print(f"\nTest split ({len(test_videos)} videos):")
    print(f"  Real: {len(real_test)} images")
    print("\nFiles created in:")
    print(f"  {ff_train_dir}/")
    print(f"  {ff_val_test_dir}/")
    print(f"  {fn_train_dir}/")
    print("="*80)


# ============================================================================
# FFHQ SPECIFIC LOGIC
# ============================================================================

def generate_ffhq_labels(
    ffhq_root: str,
    output_dir: str,
    stage: int = 1,
):
    """
    Generate FFHQ dataset labels.
    
    Stage 1: Generative training (real images only)
        - Creates data_label/ffhq_mae_vae_STAGE1/train/real_train_label.json
        - 50,000 training images, 10,000 validation images
    
    Stage 2: Classifier training (real + fake from StyleGAN 1/2/3)
        - Creates data_label/ffhq_classifier_SG123_STAGE2/train/
        - Real + fake labels with balanced sampling
    
    Args:
        ffhq_root: Root directory of FFHQ (contains images1024x1024/)
        output_dir: Output directory for label files
        stage: 1 for generative training, 2 for classifier training
    """
    print("\n" + "="*80)
    print(f"GENERATING FFHQ STAGE {stage} LABELS")
    print("="*80)
    
    ffhq_path = Path(ffhq_root)
    if not (ffhq_path / "images1024x1024").exists():
        raise FileNotFoundError(f"FFHQ images1024x1024 not found in: {ffhq_root}")
    
    output_path = Path(output_dir)
    
    if stage == 1:
        _generate_ffhq_stage1(ffhq_path, output_path)
    elif stage == 2:
        _generate_ffhq_stage2(ffhq_path, output_path)
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2")


def _generate_ffhq_stage1(ffhq_path: Path, output_path: Path):
    """Stage 1: Real images only for generative training."""
    print("\nStage 1: Generative Training (Real Images Only)")
    
    stage1_dir = output_path / "ffhq_mae_vae_STAGE1" / "train"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    
    # Training: 50,000 real images
    print("\n[1/2] Creating real_train_label.json (50,000 images)...")
    train_real = []
    train_path = ffhq_path / "images1024x1024" / "train_set"
    
    for i in range(50000):
        img_path = train_path / f"{i:05d}.png"
        if img_path.exists():
            train_real.append({"path": str(img_path.resolve()), "label": 0})
    
    print(f"  Found: {len(train_real)} images")
    output_file = stage1_dir / "real_train_label.json"
    with open(output_file, 'w') as f:
        json.dump(train_real, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Validation: 10,000 real images
    print("\n[2/2] Creating real_val_label.json (10,000 images)...")
    val_real = []
    val_path = ffhq_path / "images1024x1024" / "val_set"
    
    for i in range(10000):
        img_path = val_path / f"{i:05d}.png"
        if img_path.exists():
            val_real.append({"path": str(img_path.resolve()), "label": 0})
    
    print(f"  Found: {len(val_real)} images")
    output_file = stage1_dir / "real_val_label.json"
    with open(output_file, 'w') as f:
        json.dump(val_real, f, indent=2)
    print(f"  Saved: {output_file}")
    
    print("\n" + "="*80)
    print("FFHQ STAGE 1 COMPLETE!")
    print(f"  Training: {len(train_real)} real images")
    print(f"  Validation: {len(val_real)} real images")
    print(f"  Output: {stage1_dir}/")
    print("="*80)


def _generate_ffhq_stage2(ffhq_path: Path, output_path: Path):
    """Stage 2: Real + fake for classifier training."""
    print("\nStage 2: Classifier Training (Real + Fake from StyleGAN 1/2/3)")
    print("Sampling: Random, balanced across generators")
    
    stage2_dir = output_path / "ffhq_classifier_SG123_STAGE2" / "train"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for generated images
    generated_path = ffhq_path / "generated"
    if not generated_path.exists():
        raise FileNotFoundError(f"Generated images not found: {generated_path}")
    
    train_generators = ['stylegan1-psi-0.5', 'stylegan2-psi-0.5', 'stylegan3-psi-0.5']
    
    # Real training
    print("\n[1/4] Creating real_train_label.json (50,000 images)...")
    train_real = []
    for i in range(50000):
        img_path = ffhq_path / "images1024x1024" / "train_set" / f"{i:05d}.png"
        if img_path.exists():
            train_real.append({"path": str(img_path.resolve()), "label": 0})
    
    print(f"  Found: {len(train_real)} images")
    output_file = stage2_dir / "real_train_label.json"
    with open(output_file, 'w') as f:
        json.dump(train_real, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Fake training (balanced sampling from 3 generators)
    print("\n[2/4] Creating fake_train_sg123_label.json (50,000 fake, balanced)...")
    train_fake = []
    images_per_gen = 16667  # ~16,667 per generator = ~50K total
    
    for idx, gen in enumerate(train_generators):
        print(f"  Processing {gen}...")
        gen_train_path = generated_path / gen / "images1024x1024" / "train_set"
        
        if not gen_train_path.exists():
            print(f"    ⚠ Warning: {gen_train_path} not found, skipping")
            continue
        
        all_images = sorted(gen_train_path.glob("*.png"))
        print(f"    Available: {len(all_images)} images")
        
        # Adjust last generator to reach exactly 50K
        if idx == len(train_generators) - 1:
            images_per_gen = 50000 - len(train_fake)
        
        # Random sampling
        sampled = random.sample(all_images, min(images_per_gen, len(all_images)))
        print(f"    Sampling: {len(sampled)} images (random)")
        
        for img_path in sampled:
            train_fake.append({"path": str(img_path.resolve()), "label": 1})
    
    # Shuffle to mix generators
    random.shuffle(train_fake)
    
    print(f"  Total fake: {len(train_fake)} images")
    output_file = stage2_dir / "fake_train_sg123_label.json"
    with open(output_file, 'w') as f:
        json.dump(train_fake, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Real validation
    print("\n[3/4] Creating real_val_label.json (10,000 images)...")
    val_real = []
    for i in range(10000):
        img_path = ffhq_path / "images1024x1024" / "val_set" / f"{i:05d}.png"
        if img_path.exists():
            val_real.append({"path": str(img_path.resolve()), "label": 0})
    
    print(f"  Found: {len(val_real)} images")
    output_file = stage2_dir / "real_val_label.json"
    with open(output_file, 'w') as f:
        json.dump(val_real, f, indent=2)
    print(f"  Saved: {output_file}")
    
    # Fake validation
    print("\n[4/4] Creating fake_val_sg123_label.json (10,000 fake, balanced)...")
    val_fake = []
    images_per_gen_val = 3333  # ~3,333 per generator = ~10K total
    
    for idx, gen in enumerate(train_generators):
        print(f"  Processing {gen}...")
        gen_val_path = generated_path / gen / "images1024x1024" / "val_set"
        
        if not gen_val_path.exists():
            print(f"    ⚠ Warning: {gen_val_path} not found, skipping")
            continue
        
        all_images = sorted(gen_val_path.glob("*.png"))
        print(f"    Available: {len(all_images)} images")
        
        # Adjust last generator to reach exactly 10K
        if idx == len(train_generators) - 1:
            images_per_gen_val = 10000 - len(val_fake)
        
        # Random sampling
        sampled = random.sample(all_images, min(images_per_gen_val, len(all_images)))
        print(f"    Sampling: {len(sampled)} images (random)")
        
        for img_path in sampled:
            val_fake.append({"path": str(img_path.resolve()), "label": 1})
    
    # Shuffle to mix generators
    random.shuffle(val_fake)
    
    print(f"  Total fake: {len(val_fake)} images")
    output_file = stage2_dir / "fake_val_sg123_label.json"
    with open(output_file, 'w') as f:
        json.dump(val_fake, f, indent=2)
    print(f"  Saved: {output_file}")
    
    print("\n" + "="*80)
    print("FFHQ STAGE 2 COMPLETE!")
    print(f"  Training: {len(train_real)} real + {len(train_fake)} fake")
    print(f"  Validation: {len(val_real)} real + {len(val_fake)} fake")
    print(f"  Output: {stage2_dir}/")
    print("="*80)


# ============================================================================
# CUSTOM DIRECTORY LOGIC
# ============================================================================

def generate_custom_labels(
    image_dir: str,
    dataset_name: str,
    split: str,
    data_type: str,
    label_value: int,
    output_dir: str,
    recursive: bool = True,
):
    """
    Generate labels from custom image directory.
    
    Args:
        image_dir: Directory containing images
        dataset_name: Dataset name for output structure
        split: train/val/test
        data_type: real/fake/df/etc.
        label_value: 0=real, 1=fake
        output_dir: Output directory root
        recursive: Search subdirectories
    """
    print("\n" + "="*80)
    print("GENERATING CUSTOM LABELS")
    print("="*80)
    print(f"Image directory: {image_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Type: {data_type}")
    print(f"Label: {label_value}")
    print(f"Recursive: {recursive}")
    
    img_path = Path(image_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find images
    print("\nSearching for images...")
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    images = []
    
    pattern = "**/*" if recursive else "*"
    for ext in image_extensions:
        images.extend(img_path.glob(f"{pattern}{ext}"))
        images.extend(img_path.glob(f"{pattern}{ext.upper()}"))
    
    images = sorted(set(images))  # Remove duplicates
    print(f"Found: {len(images)} images")
    
    if len(images) == 0:
        print("⚠ No images found!")
        return
    
    # Show sample
    print("\nSample paths:")
    for img in images[:5]:
        print(f"  {img}")
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more")
    
    # Create labels
    print("\nGenerating labels...")
    labels = []
    for img_file in images:
        labels.append({"path": str(img_file.resolve()), "label": label_value})
    
    # Save
    output_path = Path(output_dir) / dataset_name / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{data_type}_{split}_label.json"
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"✓ Total: {len(labels)} images")
    
    # Validate sample
    print("\nValidating sample...")
    sample = random.sample(labels, min(10, len(labels)))
    missing = 0
    for item in sample:
        if not Path(item['path']).exists():
            print(f"  ⚠ Missing: {item['path']}")
            missing += 1
    
    if missing == 0:
        print("  ✓ All sampled paths exist")
    else:
        print(f"  ⚠ {missing}/{len(sample)} sampled paths missing")
    
    print("\n" + "="*80)
    print("CUSTOM LABEL GENERATION COMPLETE!")
    print("="*80)


# ============================================================================
# VALIDATION LOGIC
# ============================================================================

def validate_labels(label_file: str, check_paths: bool = True):
    """
    Validate a label JSON file.
    
    Checks:
    - File exists and is valid JSON
    - Structure is correct (list of dicts with 'path' and 'label')
    - Optionally checks if image paths exist
    - Reports statistics
    """
    print("\n" + "="*80)
    print(f"VALIDATING LABELS: {Path(label_file).name}")
    print("="*80)
    
    # Load file
    label_path = Path(label_file)
    if not label_path.exists():
        print(f"❌ File not found: {label_file}")
        return False
    
    try:
        with open(label_path, 'r') as f:
            labels = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    
    # Check structure
    if not isinstance(labels, list):
        print(f"❌ Labels must be a list, got {type(labels)}")
        return False
    
    if len(labels) == 0:
        print("❌ Label list is empty")
        return False
    
    print(f"✓ Structure: Valid list with {len(labels):,} entries")
    
    # Check entries
    issues = []
    for i, entry in enumerate(labels[:100]):  # Check first 100
        if not isinstance(entry, dict):
            issues.append(f"Entry {i}: not a dictionary")
            continue
        
        if 'path' not in entry:
            issues.append(f"Entry {i}: missing 'path' field")
        elif not isinstance(entry['path'], str):
            issues.append(f"Entry {i}: 'path' is not string")
        
        if 'label' not in entry:
            issues.append(f"Entry {i}: missing 'label' field")
        elif not isinstance(entry['label'], int):
            issues.append(f"Entry {i}: 'label' is not integer")
    
    if issues:
        print(f"⚠ Entry validation issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"✓ Entries: All checked entries have valid structure")
    
    # Statistics
    print("\n" + "-"*80)
    print("STATISTICS")
    print("-"*80)
    
    label_counts = Counter(entry.get('label') for entry in labels)
    print(f"Total samples: {len(labels):,}")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = "real" if label == 0 else "fake" if label == 1 else f"label_{label}"
        percentage = (count / len(labels)) * 100
        print(f"  {label_name} (label={label}): {count:>8,} ({percentage:>5.1f}%)")
    
    # File extensions
    extensions = Counter(Path(entry['path']).suffix.lower() for entry in labels if 'path' in entry)
    print(f"\nFile extensions:")
    for ext, count in extensions.most_common(5):
        print(f"  {ext:>8}: {count:>8,}")
    
    # Check for duplicates
    path_counts = Counter(entry.get('path') for entry in labels)
    duplicates = {p: c for p, c in path_counts.items() if c > 1}
    if duplicates:
        print(f"\n⚠ Duplicates: {len(duplicates)} paths appear multiple times")
    else:
        print(f"\n✓ No duplicate paths")
    
    # Check paths exist
    if check_paths:
        print("\n" + "-"*80)
        print("PATH VALIDATION (sampling 100 paths)")
        print("-"*80)
        
        sample = random.sample(labels, min(100, len(labels)))
        missing = 0
        for entry in sample:
            if 'path' in entry and not Path(entry['path']).exists():
                missing += 1
        
        if missing == 0:
            print(f"✓ All {len(sample)} sampled paths exist")
        else:
            estimated_missing = int((missing / len(sample)) * len(labels))
            print(f"⚠ {missing}/{len(sample)} sampled paths missing")
            print(f"  Estimated {estimated_missing:,}/{len(labels):,} total missing")
    
    # Final verdict
    print("\n" + "="*80)
    if len(issues) == 0:
        print("✓ VALIDATION PASSED")
        print("="*80)
        return True
    else:
        print(f"⚠ VALIDATION COMPLETED WITH {len(issues)} ISSUES")
        print("="*80)
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Label Generation for RFFR-MVAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FaceForensics++: Generate all labels
  %(prog)s --dataset ff++ --ff-root /path/to/FF++ --output-dir ./data_label
  
  # FFHQ Stage 1: Generative training (real only)
  %(prog)s --dataset ffhq --ffhq-root /path/to/FFHQ --stage 1
  
  # FFHQ Stage 2: Classifier training (real + fake)
  %(prog)s --dataset ffhq --ffhq-root /path/to/FFHQ --stage 2
  
  # Custom directory
  %(prog)s --dataset custom --image-dir /path/to/images \\
           --dataset-name my_data --split train --type real --label 0
  
  # Validate existing labels
  %(prog)s --validate --label-file ./data_label/ff_270/train/real_train_label.json
        """
    )
    
    parser.add_argument('--dataset', choices=['ff++', 'ffhq', 'custom'],
                        help='Dataset type')
    parser.add_argument('--output-dir', default='./data_label',
                        help='Output directory for label files')
    
    # FaceForensics++ options
    ff_group = parser.add_argument_group('FaceForensics++ Options')
    ff_group.add_argument('--ff-root', help='Root directory of FF++ dataset')
    ff_group.add_argument('--frames-per-video', type=int, default=10,
                          help='Frames to sample per video (default: 10)')
    ff_group.add_argument('--single-frame-val-test', action='store_true',
                          help='Use only 1 frame (middle) for val/test')
    ff_group.add_argument('--reduced-fake-frames', type=int,
                          help='Use different frame count for fake images')
    
    # FFHQ options
    ffhq_group = parser.add_argument_group('FFHQ Options')
    ffhq_group.add_argument('--ffhq-root', help='Root directory of FFHQ dataset')
    ffhq_group.add_argument('--stage', type=int, choices=[1, 2],
                            help='1=generative (real only), 2=classifier (real+fake)')
    
    # Custom options
    custom_group = parser.add_argument_group('Custom Directory Options')
    custom_group.add_argument('--image-dir', help='Directory containing images')
    custom_group.add_argument('--dataset-name', help='Dataset name for output structure')
    custom_group.add_argument('--split', choices=['train', 'val', 'test'],
                              help='Data split')
    custom_group.add_argument('--type', help='Data type (real/fake/df/f2f/etc.)')
    custom_group.add_argument('--label', type=int, help='Label value (0=real, 1=fake)')
    custom_group.add_argument('--no-recursive', action='store_true',
                              help="Don't search subdirectories")
    
    # Validation options
    val_group = parser.add_argument_group('Validation Options')
    val_group.add_argument('--validate', action='store_true',
                           help='Validate existing label file')
    val_group.add_argument('--label-file', help='Label file to validate')
    val_group.add_argument('--no-path-check', action='store_true',
                           help='Skip checking if paths exist')
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        if not args.label_file:
            parser.error("--validate requires --label-file")
        validate_labels(args.label_file, check_paths=not args.no_path_check)
        return
    
    # Generation mode
    if not args.dataset:
        parser.error("--dataset required (or use --validate)")
    
    if args.dataset == 'ff++':
        if not args.ff_root:
            parser.error("--ff-root required for FF++ dataset")
        generate_ff_labels(
            args.ff_root,
            args.output_dir,
            frames_per_video=args.frames_per_video,
            single_frame_val_test=args.single_frame_val_test,
            reduced_fake_frames=args.reduced_fake_frames,
        )
    
    elif args.dataset == 'ffhq':
        if not args.ffhq_root:
            parser.error("--ffhq-root required for FFHQ dataset")
        if not args.stage:
            parser.error("--stage required for FFHQ dataset (1 or 2)")
        generate_ffhq_labels(
            args.ffhq_root,
            args.output_dir,
            stage=args.stage,
        )
    
    elif args.dataset == 'custom':
        required = ['image_dir', 'dataset_name', 'split', 'type', 'label']
        missing = [arg for arg in required if getattr(args, arg.replace('-', '_')) is None]
        if missing:
            parser.error(f"Custom dataset requires: {', '.join('--' + m for m in missing)}")
        
        generate_custom_labels(
            args.image_dir,
            args.dataset_name,
            args.split,
            args.type,
            args.label,
            args.output_dir,
            recursive=not args.no_recursive,
        )


if __name__ == "__main__":
    main()

"""
Example configurations for RFFR-MVAE training

This file demonstrates how to configure different training scenarios using the
improved dataset selection system.

To use an example:
1. Copy the desired configuration to config.py
2. Modify paths and hyperparameters as needed
3. Run train.py
"""

# ============================================================================
# EXAMPLE 1: Training MAE on FaceForensics++ Real Images
# ============================================================================
class FF270_MAE_Config:
    """Standard MAE training on FF++ 270 videos (real images only)"""
    
    # Experiment
    seed = 912
    comment = "MAE training on FaceForensics++ 270 real images"
    experiment_name = "ff270_mae"
    
    # Model
    generator_type = "mae"
    mae_mask_ratio = 0.75
    
    # Dataset - Simple selection
    dataset_name = "ff270"          # FaceForensics++ 270 videos
    dataset_split = "train"         # Use training split
    dataset_type = "real"           # Real images only
    use_forgerynet = False          # No ForgeryNet augmentation
    
    # Training
    batch_size = 180
    lr = 1.5e-4 * (batch_size / 256)
    max_iter = 100000
    
    # Paths (will be auto-configured)
    real_label_path = None  # Auto: ../data_label/ff_270/train/real_train_label.json


# ============================================================================
# EXAMPLE 2: Training MAE-VAE on FFHQ with ForgeryNet Augmentation
# ============================================================================
class FFHQ_MAE_VAE_ForgeryNet_Config:
    """MAE-VAE with VAE bottleneck, trained on FFHQ + ForgeryNet"""
    
    # Experiment
    seed = 912
    comment = "MAE-VAE on FFHQ with ForgeryNet augmentation"
    experiment_name = "ffhq_mae_vae_fn"
    
    # Model
    generator_type = "mae_vae"
    vae_latent_dim = 768
    vae_beta = 0.0001
    vae_kl_warmup_steps = 5000
    freeze_mae_encoder = False
    
    # Dataset - Primary + ForgeryNet
    dataset_name = "ffhq"
    dataset_split = "train"
    dataset_type = "real"
    
    # ForgeryNet augmentation
    use_forgerynet = True
    forgerynet_num_videos = 150000
    forgerynet_frames_per_video = 1
    forgerynet_rotate_videos = True
    forgerynet_categories = ["16", "17", "18", "19"]  # All manipulation types
    
    # Training
    batch_size = 180
    lr = 1.5e-4 * (batch_size / 256)
    max_iter = 1000000
    
    # Dataset caching (faster loading)
    use_dataset_cache = True
    dataset_cache_file = "./cache/ffhq_train_cache.pkl"
    dataset_cache_workers = 32


# ============================================================================
# EXAMPLE 3: Training on Deepfakes Only (Single Manipulation Type)
# ============================================================================
class FF270_Deepfakes_Config:
    """Train on only Deepfakes manipulation type from FF++"""
    
    # Experiment
    seed = 912
    comment = "MAE on Deepfakes manipulation only"
    experiment_name = "ff270_df_only"
    
    # Model
    generator_type = "mae"
    
    # Dataset - Select specific manipulation type
    dataset_name = "ff270_fake100"
    dataset_split = "train"
    dataset_type = "df"  # Deepfakes only (other options: f2f, fs, fsw, nt)
    use_forgerynet = False
    
    # Training
    batch_size = 128
    lr = 1.5e-4 * (batch_size / 256)
    max_iter = 50000


# ============================================================================
# EXAMPLE 4: Training on Multiple Manipulation Types (Mixed Dataset)
# ============================================================================
class FF270_MixedFake_Config:
    """Train on mixed fake images (all manipulation types combined)"""
    
    # Experiment
    seed = 912
    comment = "MAE on mixed fake images from FF++"
    experiment_name = "ff270_mixed_fake"
    
    # Model
    generator_type = "mae"
    
    # Dataset
    dataset_name = "ff270_fake100"
    dataset_split = "train"
    dataset_type = "mixed_fake"  # All manipulation types combined
    use_forgerynet = False
    
    # Training
    batch_size = 180
    lr = 1.5e-4 * (batch_size / 256)
    max_iter = 100000


# ============================================================================
# EXAMPLE 5: Fine-tuning on DFD (DeepFake Detection Dataset)
# ============================================================================
class DFD_FineTune_Config:
    """Fine-tune on DFD dataset"""
    
    # Experiment
    seed = 912
    comment = "Fine-tuning MAE-VAE on DFD dataset"
    experiment_name = "dfd_finetune"
    
    # Model
    generator_type = "mae_vae"
    vae_latent_dim = 768
    
    # Dataset
    dataset_name = "dfd"
    dataset_split = "train"
    dataset_type = "real"
    use_forgerynet = False
    
    # Training (lower LR for fine-tuning)
    batch_size = 64
    lr = 1e-5  # Lower learning rate for fine-tuning
    max_iter = 10000
    
    # Load pretrained checkpoint
    my_pretrained = "/path/to/pretrained/checkpoint.pth.tar"


# ============================================================================
# EXAMPLE 6: Custom Dataset Path
# ============================================================================
class CustomDataset_Config:
    """Use a custom dataset JSON file"""
    
    # Experiment
    seed = 912
    comment = "Training on custom dataset"
    experiment_name = "custom_dataset"
    
    # Model
    generator_type = "mae"
    
    # Dataset - Custom path
    dataset_name = "custom"
    custom_dataset_path = "/path/to/my/custom/dataset_label.json"
    use_forgerynet = False
    
    # Training
    batch_size = 128
    lr = 1.5e-4 * (batch_size / 256)
    max_iter = 50000


# ============================================================================
# EXAMPLE 7: Validation/Testing Configuration
# ============================================================================
class ValidationConfig:
    """Configuration for validation or testing"""
    
    # Use validation split instead of training
    dataset_name = "faceforensics_hq_fake100"
    dataset_split = "val"  # or "test"
    dataset_type = "real"
    
    # Smaller batch size for evaluation
    batch_size = 32


# ============================================================================
# QUICK REFERENCE: Available Datasets
# ============================================================================
"""
Run this command to see all available datasets:
    python dataset_utils.py

Common dataset_name options:
    - "ff270": FaceForensics++ 270 videos
    - "ff270_fake100": FaceForensics++ 270 videos, 100 fake for training
    - "ffhq": High-quality faces (FFHQ)
    - "forgerynet": ForgeryNet dataset
    - "dfd": DeepFake Detection dataset
    - "faceforensics_hq": FaceForensics high-quality
    - "faceforensics_hq_fake100": FaceForensics HQ with 100 fake videos
    - "custom": Use custom_dataset_path

Common dataset_type options:
    - "real": Real/pristine images
    - "fake": Generic fake images
    - "mixed_fake": All manipulation types combined
    - "df": Deepfakes
    - "f2f": Face2Face
    - "fs": FaceSwap
    - "fsw": FaceShifter
    - "nt": NeuralTextures
    - "dfd_real": DFD real images
    - "dfd_fake": DFD fake images
    - "celebdf_real": CelebDF real images
    - "celebdf_fake": CelebDF fake images

dataset_split options:
    - "train": Training split
    - "val": Validation split
    - "test": Test split
"""

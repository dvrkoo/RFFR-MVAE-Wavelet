class DefaultConfigs(object):
    """
    Comprehensive configuration for RFFR-MVAE Generator

    This configuration manages all aspects of the generative model training including:
    - Model architecture (MAE, MAE-VAE)
    - Training hyperparameters
    - Dataset configuration
    - Checkpoint and logging paths
    """

    # ============================================================================
    # EXPERIMENT METADATA
    # ============================================================================
    seed = 912
    comment = "RFFR-MVAE: Generative modeling on FFHQ real images (MAE-VAE Stage 1)"
    experiment_name = "FFHQ_mae_vae_STAGE1"

    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    # Generator type: "mae" or "mae_vae"
    generator_type = "mae"

    # MAE (Masked Autoencoder) Configuration
    mae_encoder_embed_dim = 768
    mae_encoder_depth = 12
    mae_encoder_num_heads = 12
    mae_decoder_embed_dim = 512
    mae_decoder_depth = 8
    mae_decoder_num_heads = 16
    mae_patch_size = 16
    mae_mask_ratio = 0.75
    freeze_mae_encoder = False

    # VAE Configuration (only used when generator_type="mae_vae")
    vae_bottleneck_type = "simple"  # "simple" or "sophisticated"
    vae_latent_dim = 768  # Dimensionality of VAE latent space
    vae_beta = 0.0001  # Beta parameter for KL divergence weighting
    vae_kl_warmup_steps = 5000  # Steps to warmup KL loss from 0 to vae_beta

    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    batch_size = 40
    gradient_accumulation_steps = 10  # Number of steps to accumulate gradients (effective_batch_size = batch_size * gradient_accumulation_steps)
    lr = 1.5e-4 * (
        (batch_size * gradient_accumulation_steps) / 256
    )  # Learning rate scaled by effective batch size (linear scaling rule)
    weight_decay = 0.05
    beta1 = 0.9  # Adam beta1
    beta2 = 0.95  # Adam beta2

    # Learning Rate Scheduling
    use_lr_scheduling = True
    lr_warmup = 5000  # Warmup iterations for learning rate

    # Training Duration
    max_iter = 1000000
    iter_per_epoch = 312

    # GPU Configuration
    gpus = "0,1"  # Comma-separated GPU IDs

    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================

    # Primary Dataset Selection
    # Available options:
    #   - "ff270": FaceForensics++ with 270 videos per manipulation type
    #   - "ff270_fake100": FaceForensics++ 270 videos, but only 100 fake videos for training
    #   - "ffhq": High-quality face dataset (FFHQ)
    #   - "forgerynet": Large-scale ForgeryNet dataset
    #   - "dfd": DeepFake Detection dataset
    #   - "celebdf": CelebDF dataset
    #   - "custom": Use custom path specified in 'custom_dataset_path'
    dataset_name = "ff270"  # Primary dataset to use for training

    # Dataset Split Selection
    # Options: "train", "val", "test"
    dataset_split = "train"  # Which split to use

    # Dataset Type Selection
    # Options:
    #   - "real": Only real/pristine images (for generative model training)
    #   - "fake": Only manipulated/fake images
    #   - "mixed": Combination of real and fake
    #   - "df": Deepfakes manipulation type
    #   - "f2f": Face2Face manipulation type
    #   - "fs": FaceSwap manipulation type
    #   - "fsw": FaceShifter manipulation type
    #   - "nt": NeuralTextures manipulation type
    dataset_type = "real"  # Type of data to load

    # Custom dataset path (only used when dataset_name="custom")
    custom_dataset_path = None

    # Data Quality and Augmentation
    lq = False  # Use low quality images (compression artifacts)
    aug = False  # Enable data augmentation

    # ForgeryNet Integration (can be combined with primary dataset)
    # When enabled, creates a MixedDataset combining primary dataset with ForgeryNet
    use_forgerynet = False  # Add ForgeryNet data to training
    forgerynet_num_videos = 150000  # Number of videos to sample per epoch
    forgerynet_frames_per_video = 1  # Frames to extract per video
    forgerynet_rotate_videos = False  # Rotate video selection each epoch
    # ForgeryNet manipulation categories:
    #   "16": Face swap
    #   "17": Face reenactment
    #   "18": Entire face synthesis
    #   "19": Attribute manipulation
    forgerynet_categories = ["16", "17", "18", "19"]

    # Dataset Caching (for faster loading)
    use_dataset_cache = False  # Toggle to enable/disable image precaching to RAM
    dataset_cache_file = None  # Path to cache file (None = no disk caching, RAM only)
    dataset_cache_workers = 32  # Number of threads for parallel image loading

    # ============================================================================
    # DATA PATHS (Auto-configured based on dataset_name)
    # ============================================================================
    # Dataset paths are auto-generated based on dataset_name, dataset_split, and dataset_type
    # You can override by setting these manually:
    real_label_path = None  # Will be auto-configured if None
    forgerynet_index_path = "../data_label/FN/train/forgerynet_video_index.json"

    # ============================================================================
    # CHECKPOINT AND MODEL PATHS
    # ============================================================================
    my_pretrained = None  # Path to resume from custom checkpoint
    in_pretrained = None  # Path to pretrained ImageNet weights

    # checkpoint_path = "/seidenas/users/nmarini/generative_checkpoint/mae_vae/FFHQ_mae_vae_STAGE1_current/"
    # best_model_path = "/seidenas/users/nmarini/generative_checkpoint/mae_vae/FFHQ_mae_vae_STAGE1_best/"
    checkpoint_path = f"./checkpoints/{generator_type}_{dataset_name}"
    best_model_path = f"./checkpoints/{generator_type}_{dataset_name}_best"

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    enable_tensorboard = True
    logs = "./logs/"

    # WandB Configuration
    enable_wandb = True  # Enable Weights & Biases logging
    wandb_project = "RFFR-MVAE"  # WandB project name
    wandb_entity = None  # WandB team/username (None = default user)
    wandb_run_name = None  # Custom run name (None = auto-generated)
    wandb_tags = []  # Tags for organizing runs (e.g., ["mae_vae", "stage1"])
    wandb_notes = ""  # Additional notes for this run

    # Files to save in history for reproducibility
    save_code = [
        "configs/config.py",
        "train.py",
        "models/model_mae.py",
        "models/model_mae_vae.py",
        "dataset.py",
        "dataset_utils.py",
        "utils.py",
    ]


config = DefaultConfigs()


# ============================================================================
# DATASET PATH AUTO-CONFIGURATION
# ============================================================================
def get_dataset_path(dataset_name=None, dataset_split=None, dataset_type=None):
    """
    Auto-generate dataset path based on configuration.

    Args:
        dataset_name: Dataset name (ff270, ffhq, etc.). Uses config.dataset_name if None
        dataset_split: Split (train/val/test). Uses config.dataset_split if None
        dataset_type: Type (real/fake/df/f2f/etc.). Uses config.dataset_type if None

    Returns:
        str: Path to dataset JSON file

    Examples:
        >>> get_dataset_path("ff270", "train", "real")
        "../data_label/ff_270/train/real_train_label.json"

        >>> get_dataset_path("ff270_fake100", "train", "df")
        "../data_label/ff_270_fake100/train/df_train_label.json"
    """
    if dataset_name is None:
        dataset_name = config.dataset_name
    if dataset_split is None:
        dataset_split = config.dataset_split
    if dataset_type is None:
        dataset_type = config.dataset_type

    # Map dataset names to directory names
    dataset_dir_map = {
        "ff270": "ff_270",
        "ff270_fake100": "ff_270_fake100",
        "ffhq": "ffhq_mae_vae_STAGE1",  # Adjust as needed
        "forgerynet": "FN",
        "dfd": "Faceforensics/excludes_hq",
        "celebdf": "Faceforensics/excludes_hq",
        "faceforensics_hq": "Faceforensics/excludes_hq",
        "faceforensics_hq_fake100": "Faceforensics/excludes_hq_fake100",
        "custom": None,  # Use custom_dataset_path instead
    }

    if dataset_name == "custom":
        if config.custom_dataset_path is None:
            raise ValueError(
                "dataset_name='custom' requires custom_dataset_path to be set"
            )
        return config.custom_dataset_path

    if dataset_name not in dataset_dir_map:
        raise ValueError(
            f"Unknown dataset_name: '{dataset_name}'. "
            f"Available options: {list(dataset_dir_map.keys())}"
        )

    dataset_dir = dataset_dir_map[dataset_name]

    # Construct path: ../data_label/{dataset_dir}/{split}/{type}_{split}_label.json
    path = f"../data_label/{dataset_dir}/{dataset_split}/{dataset_type}_{dataset_split}_label.json"

    return path


# Auto-configure real_label_path if not manually set
if config.real_label_path is None:
    config.real_label_path = get_dataset_path()
    print(f"[Config] Auto-configured dataset path: {config.real_label_path}")

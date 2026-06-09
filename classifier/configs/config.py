from .label_path import get_label_path


class DefaultConfigs(object):

    generative_model_type = "mae_vae"  # Options: "mae", "mae_vae"
    vae_latent_dim = 256
    vae_base_channels = 64  # Kept for checkpoint/config metadata compatibility

    use_iterative_block_masking = True

    # Setting
    seed = 42
    comment = "FFHQ Stage 2: MAE 2-branch, Train SG123, Test OOD"
    # Logging
    use_wandb = True
    wandb_project = "rffr-classifier"
    wandb_entity = None  # Set to your wandb username if needed

    num_workers = 4  # Number of data loading workers
    pin_memory = True  # Pin memory for DataLoader
    # Models
    lr = 2e-5
    batch_size = 4
    gpus = "0"
    model = "rffr"
    if generative_model_type == "mae":
        # mae_path = "/seidenas/users/nmarini/generative_checkpoint/mae/FF_FN_best/best_loss_0.0032_200.pth.tar"
        # mae_path = "/andromeda/personal/nmarini/RFFR/rffr_generative/checkpoint/mae/best/best_loss_0.00103_100.pth.tar"
        # mae_path = "/andromeda/personal/nmarini/RFFR/rffr_generative/checkpoint/mae/best/FF_FN_best_loss_0.00355_100.pth.tar"
        mae_path = "/seidenas/users/nmarini/generative_checkpoint/mae/FFHQ_mae_STAGE1_best/best_loss_0.00928_275.pth.tar"  # actually performs better
        # mae_path = "/andromeda/personal/nmarini/RFFR/rffr_generative/checkpoint/mae/CDF/best_loss_0.00113_100.pth.tar"
    elif generative_model_type == "mae_vae":
        # mae_path = "/seidenas/users/nmarini/generative_checkpoint/mae_vae/FFHQ_mae_vae_STAGE1_best/best_loss_0.01325_800.pth.tar"
        mae_path = "/home/nick/GitHub/RFFR/rffr_generative/checkpoint/checkpoint/mae_vae/CDF/best_loss_0.03285_100.pth.tar"

    pretrained_weights = "../pretrain/jx_vit_base_p16_224-80ecf9dd.pth"
    # Training Options
    use_gradient_accumulation = (
        False  # Enable gradient accumulation for effective larger batch sizes
    )
    gradient_accumulation_steps = (
        4  # Number of micro-batches to accumulate before optimizer step
    )
    # When gradient accumulation is enabled:
    # - effective_batch_size = batch_size * gradient_accumulation_steps
    # - May reduce training stability due to BatchNorm statistics computed on micro-batches
    # - Slower training but lower memory usage

    # Data
    protocol = "F2F_All"
    dataset_base = "../data_label/"
    (
        real_label_path,
        fake_label_path,
        val_label_path,
        test_label_path,
        metrics,
        real_test_label_path,
    ) = get_label_path(protocol)
    # Schedule
    max_iter = (
        15000  # 600 epochs * 50 iters per epoch (increased for better convergence)
    )
    iter_per_epoch = 50
    # Video subset limiting (for data scaling experiments)
    max_fake_frames = 100  # None = use all frames, or set to 90, 180, 360, etc.

    # Video Subset Selection (for training data experiments)
    use_video_subset = False  # Enable video-level subsetting for FAKE videos only
    video_subset_count = None  # Number of fake videos to use
    video_subset_start_idx = 0  # Starting video index

    # Multi-Forgery Mixing
    use_mixed_forgeries = False  # Enable dynamic multi-forgery mixing at training time
    forgery_mix_types = ["f2f", "df"]  # Which forgeries to mix
    forgery_mix_ratios = [0.5, 0.5]  # Ratios for each forgery (must sum to 1.0)
    forgery_mix_base_dir = (
        "ff_270_fake1"  # Base directory containing forgery label files
    )
    total_fake_samples = 720  # Total fake samples to use

    # Dynamic validation: every 10 epochs (0-100), then every 5 epochs (100-600)
    # Will be calculated dynamically in train.py

    # Learning Rate Scheduling
    use_warmup = False  # Enable linear warmup for stable training
    warmup_steps = 2000  # Linear warmup for first 2000 iterations (40 epochs)
    warmup_start_lr = 1e-6  # Starting LR for warmup (10% of target LR)

    use_scheduler = False  # Enable LR scheduling after warmup
    scheduler_type = "cosine"  # Options: "cosine", "multistep", "exponential"

    # Cosine Annealing settings
    cosine_min_lr = 1e-7  # Minimum LR for cosine annealing
    cosine_T_max = None  # Will be set to (max_iter - warmup_steps) if None

    # MultiStep LR settings (if scheduler_type == "multistep")
    multistep_milestones = [10000, 20000, 25000]  # Iteration milestones for LR drops
    multistep_gamma = 0.5  # LR multiplication factor at milestones

    # Exponential LR settings (if scheduler_type == "exponential")
    exponential_gamma = 0.99995  # LR decay factor per iteration

    # Optimizer settings
    use_adamw = False  # Use AdamW instead of Adam for better generalization
    weight_decay = 1e-4  # L2 regularization for AdamW

    # Wavelet Analysis Options

    # Anomaly Detection Mode
    anomaly_detection_mode = (
        False  # Train only on real samples, detect fakes as anomalies
    )
    center_loss_weight = 2.0  # Weight for center loss in anomaly detection
    center_margin = 5.0  # Margin for separating real samples from decision boundary
    repulsion_weight = 15.0  # Weight for pushing fake samples away from center
    anomaly_score_percentile = (
        95  # Percentile of real sample distances to use as threshold
    )

    # Hybrid Multi-Task Loss (Anomaly Detection)
    use_hybrid_loss = (
        True  # Combine compactness loss with cross-entropy in anomaly mode
    )
    compactness_weight = 0.1  # Weight for compactness loss
    classification_weight = 1.0  # Weight for cross-entropy loss

    use_wavelets = True
    wavelet_type = "haar"  # Wavelet type (db4, haar, bior2.2, coif2)
    wavelet_levels = (
        1  # Number of decomposition levels (optimal for deepfake detection)
    )
    wavelet_high_freq_weight = 1.0  # Weight multiplier for high-frequency components

    generator_outputs_wavelets = False
    classifier_uses_wavelets = True

    # Architecture Options
    # False: 2-branch architecture (RGB + spatial residuals)
    # True:  3-branch architecture (RGB + spatial residuals + wavelet residuals)
    wavelet_residual_branch = True

    # Pretraining Options for Wavelet Branches
    use_imagenet_pretrain_for_wavelets = (
        False  # Use random initialization for frequency domain (recommended)
    )
    # When True: All branches use ImageNet pretraining
    # When False: Only RGB branch uses ImageNet pretraining, wavelet branches use random initialization
    # ImageNet features (edges, textures, objects) don't transfer well to deepfake frequency artifacts

    use_adaptive_vit = True  # The 3-branch wavelet residual branch is 12-channel AdaptiveViT

    # Deprecated architecture flags kept false for older scripts/config snapshots.
    separate_wavelet_branch = False
    four_branch_wavelet = False
    wavelet_only_mode = False
    wavelet_dual_branch_mode = False

    # paths information
    checkpoint_path = "./checkpoint/" + model + "/current_model/"
    best_model_path = "./checkpoint/" + model + "/best_model/"
    logs = "./logs/"

    def set_run_paths(self, run_identifier):
        """Update checkpoint paths to be run-specific"""

        self.checkpoint_path = f"/home/nick/GitHub/RFFR-MVAE-Wavelet/classifier/checkopint/{self.generative_model_type}/{run_identifier}/current_model/"
        self.best_model_path = f"/home/nick/GitHub/RFFR-MVAE-Wavelet/classifier/checkopint/{self.generative_model_type}/{run_identifier}/best_model/"

    # Code Saver
    save_code = [
        "configs/config.py",
        "train.py",
        "utils/simple_evaluate.py",
        "utils/dataset.py",
        "utils/wavelet_utils.py",
        "models/model_mae.py",
        "models/model_detector.py",
    ]


config = DefaultConfigs()


def list_available_protocols():
    """List all available training protocols and their descriptions."""
    protocols = {
        "F2F_All": "Train on Face2Face only",
        "DF_All": "Train on Deepfakes only",
        "FSW_All": "Train on FaceSwap only",
        "NT_All": "Train on NeuralTextures only",
        "FS_All": "Train on FaceShifter only (if available)",
        "Mixed_All": "Train on all forgeries combined (recommended)",
        "F2F_All_Fake5": "Train on Face2Face with reduced fake frames (5 per video)",
        "DF_All_Fake5": "Train on Deepfakes with reduced fake frames (5 per video)",
        "Mixed_All_Fake5": "Train on all forgeries with reduced fake frames (5 per video)",
        "F2F_All_Fake3": "Train on Face2Face with reduced fake frames (3 per video)",
        "DF_All_Fake3": "Train on Deepfakes with reduced fake frames (3 per video)",
        "Mixed_All_Fake3": "Train on all forgeries with reduced fake frames (3 per video)",
    }

    print("Available training protocols:")
    for protocol, description in protocols.items():
        current = " (CURRENT)" if protocol == config.protocol else ""
        print(f"  {protocol}: {description}{current}")

    print(
        "\nNote: Protocols with _FakeN suffix use reduced frames for fake videos only."
    )
    print("Real videos always use the original frame count (10 frames per video).")

    return protocols


def switch_protocol(new_protocol):
    """Switch to a different training protocol."""
    available_protocols = [
        "F2F_All",
        "DF_All",
        "FSW_All",
        "NT_All",
        "FS_All",
        "Mixed_All",
        "F2F_All_Fake5",
        "DF_All_Fake5",
        "FSW_All_Fake5",
        "NT_All_Fake5",
        "FS_All_Fake5",
        "Mixed_All_Fake5",
        "F2F_All_Fake3",
        "DF_All_Fake3",
        "FSW_All_Fake3",
        "NT_All_Fake3",
        "FS_All_Fake3",
        "Mixed_All_Fake3",
        "F2F_All_Fake7",
        "DF_All_Fake7",
        "FSW_All_Fake7",
        "NT_All_Fake7",
        "FS_All_Fake7",
        "Mixed_All_Fake7",
        "F2F_All_Fake100",
    ]

    if new_protocol not in available_protocols:
        print(f"Error: Protocol '{new_protocol}' not available.")
        print("Available protocols:", ", ".join(available_protocols))
        return False

    config.protocol = new_protocol

    (
        config.real_label_path,
        config.fake_label_path,
        config.val_label_path,
        config.test_label_path,
        config.metrics,
        config.real_test_label_path,
    ) = get_label_path(new_protocol)

    print(f"Protocol switched to: {new_protocol}")
    return True

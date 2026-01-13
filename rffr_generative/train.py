import os

from utils import save_checkpoint, AverageMeter, Logger
from utils import mkdirs, time_to_str, save_code, test_checkpoint_save

from dataset import Deepfake_Dataset, ForgeryNetRotatingDataset, MixedDataset
from dataset_utils import validate_dataset_path, print_available_datasets
from torch.utils.data import DataLoader

import json
import random
import hashlib
import numpy as np
from configs.config import config
from datetime import datetime
import time
from timeit import default_timer as timer

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import timm.optim.optim_factory as optim_factory

from models.model_mae import mae_vit_base_patch16
from models.model_mae_vae import mae_vae_vit_base_patch16

from tensorboardX import SummaryWriter

# WandB Integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] WandB not installed. Install with: pip install wandb")

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train():

    mkdirs()

    print("\n[Pre-flight] Checking training environment...")
    try:
        import shutil

        stat = shutil.disk_usage(config.checkpoint_path)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        print(f"[Pre-flight] Checkpoint directory: {config.checkpoint_path}")
        print(
            f"[Pre-flight] Disk space - Total: {total_gb:.2f} GB, Used: {used_gb:.2f} GB, Free: {free_gb:.2f} GB"
        )

        if free_gb < 10:
            print(
                f"[WARNING] Low disk space! Only {free_gb:.2f} GB free. Recommend at least 10 GB for safe training."
            )
        else:
            print(f"[Pre-flight] Disk space check passed ({free_gb:.2f} GB available)")
    except Exception as e:
        print(f"[WARNING] Could not check disk space: {e}")

    test_checkpoint_save()

    timenow = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    runhash = hashlib.sha1()
    runhash.update(timenow.encode("utf-8"))
    runhash.update(config.comment.encode("utf-8"))
    runhash = runhash.hexdigest()[:6]

    save_code(timenow, runhash)

    # Validate dataset path
    print("\n[Dataset] Validating dataset configuration...")
    print(f"[Dataset] Dataset name: {config.dataset_name}")
    print(f"[Dataset] Dataset split: {config.dataset_split}")
    print(f"[Dataset] Dataset type: {config.dataset_type}")
    print(f"[Dataset] Dataset path: {config.real_label_path}")
    
    is_valid, error_msg, sample_count = validate_dataset_path(config.real_label_path)
    
    if not is_valid:
        print(f"\n❌ [ERROR] Dataset validation failed: {error_msg}")
        print("\n[Dataset] Available datasets:")
        print_available_datasets()
        print("\n[Dataset] To fix this issue:")
        print("  1. Check your config.py settings for dataset_name, dataset_split, and dataset_type")
        print("  2. Verify the dataset JSON file exists at the path above")
        print("  3. Or set config.custom_dataset_path to a valid JSON file")
        raise FileNotFoundError(f"Invalid dataset configuration: {error_msg}")
    
    print(f"✓ [Dataset] Validation passed: {sample_count:,} samples found")

    with open(config.real_label_path, "r") as f:
        train_dict = json.load(f)
    train_dict = [item["path"] for item in train_dict]
    length = len(train_dict)
    train_data = train_dict[: -int(0.001 * length)]
    test_data = train_dict[-int(0.001 * length) :]
    # Reserve GPU memory during data caching to prevent others from using it
    print("[GPU Lock] Reserving GPU memory during data caching...")
    gpu_lock_tensors = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            # Reserve ~90% of each GPU (adjust size as needed)
            free_mem = torch.cuda.get_device_properties(i).total_memory
            reserve_size = int(free_mem * 0.9 / 4)  # float32 = 4 bytes
            gpu_lock_tensors.append(torch.zeros(reserve_size, device=f"cuda:{i}"))
    print(f"[GPU Lock] Reserved memory on {len(gpu_lock_tensors)} GPUs")


    if config.use_forgerynet:
        ff_dataset = Deepfake_Dataset(train_data, train=True)
        fn_dataset = ForgeryNetRotatingDataset(
            config.forgerynet_index_path,
            num_videos_per_epoch=config.forgerynet_num_videos,
            frames_per_video=config.forgerynet_frames_per_video,
            seed=config.seed,
            categories=config.forgerynet_categories,
            rotate=config.forgerynet_rotate_videos,
        )
        train_dataset = MixedDataset(ff_dataset, fn_dataset)

        print(f"FF++ dataset size: {len(ff_dataset)} frames")
        print(f"ForgeryNet dataset size: {len(fn_dataset)} frames")
        print(f"Total dataset size: {len(train_dataset)} frames")

        real_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True,
        )
    else:
        ff_dataset = Deepfake_Dataset(train_data, train=True)
        print(f"FF++ dataset size: {len(ff_dataset)} frames (ForgeryNet disabled)")

        real_dataloader = DataLoader(
            ff_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True,
        )

    test_dataloader = DataLoader(
        Deepfake_Dataset(test_data, train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=16,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        Deepfake_Dataset(test_data, train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=16,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        drop_last=True,
    )


    # Release GPU memory lock now that data is cached
    print("[GPU Lock] Releasing GPU memory reservation...")
    del gpu_lock_tensors
    torch.cuda.empty_cache()
    print("[GPU Lock] GPU memory released")

    best_loss = np.inf
    epoch = 0
    start_iter = 0

    loss_logger = AverageMeter()
    eval_loss_logger = AverageMeter()
    kl_loss_logger = AverageMeter()

    if config.generator_type == "mae_vae":
        net = mae_vae_vit_base_patch16(
            vae_latent_dim=config.vae_latent_dim,
            freeze_encoder=config.freeze_mae_encoder,
            vae_bottleneck_type=config.vae_bottleneck_type,
        ).cuda()
        print(
            f"Initialized MAE-VAE with latent_dim={config.vae_latent_dim}, bottleneck={config.vae_bottleneck_type}"
        )
    elif config.generator_type == "mae":
        net = mae_vit_base_patch16().cuda()
        print("Initialized MAE")
    else:
        raise ValueError(f"Unknown generator_type: {config.generator_type}. Supported types: 'mae', 'mae_vae'")

    if config.in_pretrained is not None:
        checkpoint = torch.load(config.in_pretrained)
        net.load_state_dict(checkpoint["model"], strict=True)
    if len(config.gpus) > 1:
        net = torch.nn.DataParallel(net).cuda()

    # net = torch.compile(net)
    # torch.set_float32_matmul_precision('high')
    
    # Print training configuration
    print(f"\n[Training Config]")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.lr:.2e}")
    print(f"  Weight decay: {config.weight_decay}")

    param_groups = optim_factory.param_groups_weight_decay(net, config.weight_decay)
    optimizer = optim.AdamW(
        param_groups, lr=config.lr, betas=(config.beta1, config.beta2)
    )

    if config.in_pretrained is None and config.my_pretrained is not None:
        checkpoint = torch.load(config.my_pretrained, weights_only=False)
        missing_keys = net.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing_keys.missing_keys:
            print(
                f"Loading pretrained weights with missing keys (new layers will train from scratch): {missing_keys.missing_keys}"
            )
            epoch = 0
            start_iter = 0
        else:
            optimizer.load_state_dict(checkpoint["adam_dict"])
            epoch = checkpoint["epoch"]
            start_iter = epoch * config.iter_per_epoch

    if config.enable_tensorboard:
        tblogger = SummaryWriter(comment=config.comment)
    
    # Initialize WandB
    if config.enable_wandb and WANDB_AVAILABLE:
        # Prepare wandb config
        wandb_config = {
            # Model architecture
            "generator_type": config.generator_type,
            "mae_mask_ratio": config.mae_mask_ratio,
            "mae_patch_size": config.mae_patch_size,
            
            # Training hyperparameters
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "max_iter": config.max_iter,
            "iter_per_epoch": config.iter_per_epoch,
            "use_lr_scheduling": config.use_lr_scheduling,
            "lr_warmup": config.lr_warmup,
            
            # Dataset
            "dataset_name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "dataset_type": config.dataset_type,
            "use_forgerynet": config.use_forgerynet,
            
            # Random seed
            "seed": config.seed,
        }
        
        # Add VAE-specific config if using MAE-VAE
        if config.generator_type == "mae_vae":
            wandb_config.update({
                "vae_latent_dim": config.vae_latent_dim,
                "vae_beta": config.vae_beta,
                "vae_kl_warmup_steps": config.vae_kl_warmup_steps,
                "vae_bottleneck_type": config.vae_bottleneck_type,
                "freeze_mae_encoder": config.freeze_mae_encoder,
            })
        
        # Auto-generate run name if not specified
        run_name = config.wandb_run_name or f"{config.generator_type}_{config.dataset_name}_{timenow}_{runhash}"
        
        # Initialize WandB
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=wandb_config,
            tags=config.wandb_tags or [config.generator_type, config.dataset_name],
            notes=config.wandb_notes or config.comment,
            resume="allow",  # Allow resuming if run exists
        )
        
        # Watch model for gradients and parameters
        wandb.watch(net, log="gradients", log_freq=config.iter_per_epoch)
        
        print(f"✓ [WandB] Initialized - Project: {config.wandb_project}, Run: {run_name}")
    elif config.enable_wandb and not WANDB_AVAILABLE:
        print("[Warning] WandB logging enabled in config but wandb not installed. Skipping WandB logging.")

    log = Logger()
    log.open(
        config.logs + timenow + "_" + runhash + "_" + config.comment + ".txt", mode="a"
    )
    log.write(
        "\n-------------- [START %s] %s\n\n"
        % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-------------------")
    )
    log.write("Random seed: %d\n" % config.seed)
    log.write("Comment: %s\n" % config.comment)
    log.write("** start training target model! **\n")
    log.write("--------|----- Train -----|---- Best ----|---- Test ----|\n")
    log.write("  iter  |      loss       |     loss     |     loss     |\n")
    log.write("--------------------------------------------------------|\n")
    start = timer()
    criterion = {
        "mse": nn.MSELoss().cuda(),
        "l1": nn.L1Loss().cuda(),
        "softmax": nn.CrossEntropyLoss().cuda(),
    }

    iter_per_epoch = config.iter_per_epoch  # iters that the model need to be tested
    max_iter = config.max_iter

    train_real_iter = iter(real_dataloader)
    train_real_iters_per_epoch = len(train_real_iter)

    for iter_num in range(start_iter, max_iter + 1):
        if iter_num % train_real_iters_per_epoch == 0:
            train_real_iter = iter(real_dataloader)
            if config.use_forgerynet and hasattr(real_dataloader.dataset, "set_epoch"):
                real_dataloader.dataset.set_epoch(epoch)
        if iter_num != 0 and iter_num % iter_per_epoch == 0:
            epoch = epoch + 1
            loss_logger.reset()

        # Learning rate schedule
        if config.use_lr_scheduling:
            if iter_num < config.lr_warmup:
                lr = config.lr * ((iter_num + 1) / config.lr_warmup)
            else:
                lr = (
                    config.lr
                    * 0.5
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (iter_num - config.lr_warmup)
                            / (config.max_iter - config.lr_warmup)
                        )
                    )
                )
        else:
            lr = config.lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        net.train(True)

        img_real = next(train_real_iter)
        img_real = img_real.cuda()
        input_data = img_real

        if config.generator_type == "mae_vae":
            if iter_num < config.vae_kl_warmup_steps:
                beta = config.vae_beta * (iter_num / config.vae_kl_warmup_steps)
            else:
                beta = config.vae_beta

            rec_loss, pred, mask, recon_component, kl_component = net(input_data, beta=beta)
            rec_loss = rec_loss.mean()
        elif config.generator_type == "mae":
            rec_loss, pred, mask = net(input_data)
            rec_loss = rec_loss.mean()
        else:
            raise ValueError(f"Unknown generator_type: {config.generator_type}")

        # Scale loss by gradient accumulation steps
        scaled_loss = rec_loss / config.gradient_accumulation_steps
        
        # Backward pass (accumulate gradients)
        scaled_loss.backward()
        
        # Only update weights every gradient_accumulation_steps iterations
        if (iter_num + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # Only log loss when optimizer actually updates (for fair comparison)
            loss_logger.update(rec_loss.item())
            
            # Log to WandB immediately after optimizer step (when loss is fresh)
            if config.enable_wandb and WANDB_AVAILABLE:
                wandb_metrics = {
                    "train/loss": rec_loss.item(),
                    "train/loss_avg": loss_logger.avg,
                    "train/learning_rate": lr,
                    "train/epoch": epoch + (iter_num % iter_per_epoch) / iter_per_epoch,
                    "train/iteration": iter_num,
                }
                
                # Add VAE-specific metrics
                if config.generator_type == "mae_vae":
                    wandb_metrics["train/recon_loss"] = recon_component.mean().item()
                    wandb_metrics["train/kl_loss"] = kl_component.mean().item()
                    wandb_metrics["train/beta"] = beta
                
                wandb.log(wandb_metrics, step=iter_num)

        # For MAE-VAE models, show individual loss components
        if config.generator_type == "mae_vae":
            print("\r", end="", flush=True)
            print(
                "  %4.1f | T:%6.4f | R:%6.4f | KL:%8.2f | b:%.4f | best:%6.4f | %s"
                % (
                    epoch + (iter_num % iter_per_epoch) / iter_per_epoch,
                    loss_logger.avg,
                    recon_component.mean().item(),
                    kl_component.mean().item(),
                    beta,
                    best_loss,
                    time_to_str(timer() - start, "min"),
                ),
                end="",
                flush=True,
            )
        else:
            print("\r", end="", flush=True)
            print(
                "  %4.1f |     %6.4f      |     %6.4f    |     %6.4f    |    %s"
                % (
                    epoch + (iter_num % iter_per_epoch) / iter_per_epoch,
                    loss_logger.avg,
                    best_loss,
                    eval_loss_logger.avg,
                    time_to_str(timer() - start, "min"),
                ),
                end="",
                flush=True,
            )

        if iter_num != 0 and (iter_num + 1) % config.iter_per_epoch == 0:

            if loss_logger.avg < best_loss:
                is_best_loss = True
                best_loss = loss_logger.avg
                current_loss = loss_logger.avg

            eval_loss_logger.reset()

            for test_data in test_dataloader:
                test_data = test_data.cuda()
                rec_loss, _, _ = net(test_data)
                rec_loss = rec_loss.mean().cpu().detach().numpy()
                eval_loss_logger.update(rec_loss)

            if (epoch + 1) % 25 == 0:
                save_list = [epoch + 1, current_loss]
                model_config = {
                    "generator_type": config.generator_type,
                }
                # Only save VAE-specific config if using MAE-VAE
                if config.generator_type == "mae_vae":
                    model_config["vae_latent_dim"] = config.vae_latent_dim
                    model_config["vae_bottleneck_type"] = config.vae_bottleneck_type
                save_checkpoint(save_list, net, optimizer, model_config=model_config)

            print("\r", end="", flush=True)
            log.write(
                "  %4.1f |     %6.4f      |     %6.4f    |     %6.4f    |   %s"
                % (
                    epoch + 1,
                    loss_logger.avg,
                    best_loss,
                    eval_loss_logger.avg,
                    time_to_str(timer() - start, "min"),
                )
            )
            log.write("\n")

            if config.enable_tensorboard:
                info = {
                    "Loss_train": loss_logger.avg,
                    "lowest_loss": best_loss,
                    "Test_Loss": eval_loss_logger.avg,
                }
                for tag, value in info.items():
                    tblogger.add_scalar(tag, value, epoch)
            
            # WandB epoch-level logging
            if config.enable_wandb and WANDB_AVAILABLE:
                epoch_metrics = {
                    "epoch/train_loss": loss_logger.avg,
                    "epoch/test_loss": eval_loss_logger.avg,
                    "epoch/best_loss": best_loss,
                    "epoch/epoch_num": epoch + 1,
                }
                wandb.log(epoch_metrics, step=iter_num)


if __name__ == "__main__":
    try:
        train()
    finally:
        # Clean up WandB
        if config.enable_wandb and WANDB_AVAILABLE:
            wandb.finish()
            print("\n[WandB] Run finished and logged successfully")

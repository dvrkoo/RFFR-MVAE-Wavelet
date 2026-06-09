#!/usr/bin/env python3
"""
YAML Configuration Loader for RFFR Classifier

Provides functionality to:
- Load YAML configuration files
- Merge base and experiment configs
- Override values from command line
- Resolve dataset paths using label_path.py
- Validate configuration
- Save config snapshots for reproducibility
- Support backward compatibility with Python config.py
"""

import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy


class ConfigDict(dict):
    """Dictionary that allows attribute-style access (config.key instead of config['key'])"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def dict_to_configdict(d: dict) -> ConfigDict:
    """
    Recursively convert nested dictionaries to ConfigDict for attribute access.

    Args:
        d: Dictionary to convert

    Returns:
        ConfigDict with nested ConfigDicts
    """
    config = ConfigDict()
    for key, value in d.items():
        if isinstance(value, dict):
            config[key] = dict_to_configdict(value)
        else:
            config[key] = value
    return config


def load_yaml(filepath: Union[str, Path]) -> dict:
    """
    Load YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary with config values
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def resolve_paths(config: ConfigDict, config_dir: Path) -> ConfigDict:
    """
    Resolve relative paths to absolute paths based on config directory.

    Args:
        config: Configuration dictionary
        config_dir: Directory containing the config file

    Returns:
        Config with resolved paths
    """
    def resolve_value(value, base_dir):
        if isinstance(value, str) and value.startswith('./'):
            # Relative path - resolve relative to config directory
            return str((base_dir / value).resolve())
        elif isinstance(value, str) and value.startswith('../'):
            # Relative path - resolve relative to config directory
            return str((base_dir / value).resolve())
        return value

    def resolve_dict(d, base_dir):
        if isinstance(d, ConfigDict):
            for key, value in d.items():
                if isinstance(value, ConfigDict):
                    d[key] = resolve_dict(value, base_dir)
                else:
                    d[key] = resolve_value(value, base_dir)
        return d

    return resolve_dict(config, config_dir)


def interpolate_strings(config: ConfigDict) -> ConfigDict:
    """
    Interpolate ${var.subvar} style placeholders in string values.

    Args:
        config: Configuration dictionary

    Returns:
        Config with interpolated strings
    """
    import re

    def get_nested_value(d, path):
        """Get value from nested dict using dot notation (e.g., 'model.name')"""
        keys = path.split('.')
        value = d
        for key in keys:
            if isinstance(value, ConfigDict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def interpolate_value(value, config_root):
        if isinstance(value, str):
            # Find all ${...} patterns
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)

            for match in matches:
                replacement = get_nested_value(config_root, match)
                if replacement is not None:
                    value = value.replace(f'${{{match}}}', str(replacement))

        return value

    def interpolate_dict(d, config_root):
        if isinstance(d, ConfigDict):
            for key, value in d.items():
                if isinstance(value, ConfigDict):
                    d[key] = interpolate_dict(value, config_root)
                else:
                    d[key] = interpolate_value(value, config_root)
        return d

    return interpolate_dict(config, config)


def compute_label_paths(config: ConfigDict) -> ConfigDict:
    """
    Compute dataset label paths using label_path.py.

    Args:
        config: Configuration dictionary

    Returns:
        Config with computed label paths
    """
    # Import label_path module
    try:
        from .label_path import get_label_path
    except ImportError:
        from label_path import get_label_path

    if 'dataset' in config and 'protocol' in config.dataset:
        protocol = config.dataset.protocol

        (
            real_label_path,
            fake_label_path,
            val_label_path,
            test_label_path,
            metrics,
            real_test_label_path,
        ) = get_label_path(protocol)

        config.dataset.real_label_path = real_label_path
        config.dataset.fake_label_path = fake_label_path
        config.dataset.val_label_path = val_label_path
        config.dataset.test_label_path = test_label_path
        config.dataset.metrics = metrics
        config.dataset.real_test_label_path = real_test_label_path

    return config


def compute_derived_values(config: ConfigDict) -> ConfigDict:
    """
    Compute derived configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        Config with computed values
    """
    # Compute effective batch size
    if 'training' in config:
        if config.training.use_gradient_accumulation:
            config.training.effective_batch_size = (
                config.training.batch_size * config.training.gradient_accumulation_steps
            )
        else:
            config.training.effective_batch_size = config.training.batch_size

        # Set cosine_T_max if not specified
        if (config.training.lr_scheduling.use_scheduler and
            config.training.lr_scheduling.scheduler_type == "cosine" and
            config.training.lr_scheduling.cosine_T_max is None):

            if config.training.lr_scheduling.use_warmup:
                config.training.lr_scheduling.cosine_T_max = (
                    config.training.max_iter - config.training.lr_scheduling.warmup_steps
                )
            else:
                config.training.lr_scheduling.cosine_T_max = config.training.max_iter

    # Compute label paths from protocol
    config = compute_label_paths(config)

    return config


def parse_cli_overrides(args: list) -> dict:
    """
    Parse command line overrides in the format --key value or --nested.key value.

    Args:
        args: List of command line arguments

    Returns:
        Dictionary with override values
    """
    # Keys that should always be kept as strings (no type conversion)
    STRING_KEYS = {
        'gpu.devices',
        'model.mae_path',
        'model.pretrained_weights',
        'model.architecture.wavelet_type',
        'dataset.protocol',
        'dataset.base_dir',
        'dataset.mixing.forgery_mix_base_dir',
        'logging.wandb.entity',
        'logging.wandb.project',
        'logging.wandb.run_name',
        'logging.logs_dir',
        'paths.checkpoint',
        'paths.best_model',
        'metadata.comment',
        'metadata.experiment_name',
    }

    overrides = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith('--') and i + 1 < len(args):
            key = arg[2:]  # Remove '--'
            value = args[i + 1]

            # Skip config file argument
            if key == 'config':
                i += 2
                continue

            # Check if this key should remain as string
            if key in STRING_KEYS:
                # Keep as string, no type conversion
                pass
            # Try to convert value to appropriate type
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none' or value.lower() == 'null':
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string

            # Handle nested keys (e.g., model.vae.latent_dim -> {"model": {"vae": {"latent_dim": value}}})
            if '.' in key:
                keys = key.split('.')
                current = overrides
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                overrides[key] = value

            i += 2
        else:
            i += 1

    return overrides


def flatten_config(config: ConfigDict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten nested config into flat dictionary with dot notation keys.
    Used for backward compatibility with Python config.py.

    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_flat_config_object(config: ConfigDict) -> object:
    """
    Create a flat config object for backward compatibility.
    This allows accessing config.lr instead of config.training.lr.

    Args:
        config: Nested configuration dictionary

    Returns:
        Object with flat attributes
    """
    class FlatConfig:
        pass

    flat = FlatConfig()
    flat_dict = flatten_config(config)

    # Set attributes from flattened dict
    for key, value in flat_dict.items():
        # Replace dots with underscores for valid Python identifiers
        attr_name = key.replace('.', '_')
        setattr(flat, attr_name, value)

    # Also keep nested access
    for key, value in config.items():
        setattr(flat, key, value)

    # Add special attributes for common patterns
    if 'training' in config:
        flat.lr = config.training.lr
        flat.batch_size = config.training.batch_size
        flat.max_iter = config.training.max_iter
        flat.iter_per_epoch = config.training.iter_per_epoch
        flat.use_gradient_accumulation = config.training.use_gradient_accumulation
        flat.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        flat.use_adamw = config.training.use_adamw
        flat.weight_decay = config.training.weight_decay

        if 'lr_scheduling' in config.training:
            flat.use_warmup = config.training.lr_scheduling.use_warmup
            flat.warmup_steps = config.training.lr_scheduling.warmup_steps
            flat.warmup_start_lr = config.training.lr_scheduling.warmup_start_lr
            flat.use_scheduler = config.training.lr_scheduling.use_scheduler
            flat.scheduler_type = config.training.lr_scheduling.scheduler_type
            flat.cosine_min_lr = config.training.lr_scheduling.cosine_min_lr
            flat.cosine_T_max = config.training.lr_scheduling.cosine_T_max
            flat.multistep_milestones = config.training.lr_scheduling.multistep_milestones
            flat.multistep_gamma = config.training.lr_scheduling.multistep_gamma
            flat.exponential_gamma = config.training.lr_scheduling.exponential_gamma

        if 'loss' in config.training:
            flat.anomaly_detection_mode = config.training.loss.anomaly_detection_mode
            flat.center_loss_weight = config.training.loss.center_loss_weight
            flat.center_margin = config.training.loss.center_margin
            flat.repulsion_weight = config.training.loss.repulsion_weight
            flat.anomaly_score_percentile = config.training.loss.anomaly_score_percentile
            flat.use_hybrid_loss = config.training.loss.use_hybrid_loss
            flat.compactness_weight = config.training.loss.compactness_weight
            flat.classification_weight = config.training.loss.classification_weight

    if 'model' in config:
        flat.model = config.model.name
        flat.generative_model_type = config.model.generative_model_type
        flat.mae_path = config.model.mae_path
        flat.pretrained_weights = config.model.pretrained_weights
        flat.use_iterative_block_masking = config.model.use_iterative_block_masking

        if 'vae' in config.model:
            flat.vae_latent_dim = config.model.vae.latent_dim
            flat.vae_base_channels = config.model.vae.base_channels

        if 'architecture' in config.model:
            flat.use_wavelets = config.model.architecture.use_wavelets
            flat.wavelet_type = config.model.architecture.wavelet_type
            flat.wavelet_levels = config.model.architecture.wavelet_levels
            flat.wavelet_high_freq_weight = config.model.architecture.wavelet_high_freq_weight
            flat.generator_outputs_wavelets = config.model.architecture.generator_outputs_wavelets
            flat.classifier_uses_wavelets = config.model.architecture.classifier_uses_wavelets
            flat.separate_wavelet_branch = getattr(config.model.architecture, 'separate_wavelet_branch', False)
            flat.four_branch_wavelet = getattr(config.model.architecture, 'four_branch_wavelet', False)
            flat.use_imagenet_pretrain_for_wavelets = config.model.architecture.use_imagenet_pretrain_for_wavelets
            flat.use_adaptive_vit = getattr(config.model.architecture, 'use_adaptive_vit', True)
            flat.wavelet_only_mode = getattr(config.model.architecture, 'wavelet_only_mode', False)
            flat.wavelet_dual_branch_mode = getattr(config.model.architecture, 'wavelet_dual_branch_mode', False)
            flat.wavelet_residual_branch = config.model.architecture.wavelet_residual_branch

    if 'dataset' in config:
        flat.protocol = config.dataset.protocol
        flat.dataset_base = config.dataset.base_dir
        flat.num_workers = config.dataset.num_workers
        flat.pin_memory = config.dataset.pin_memory
        flat.max_fake_frames = config.dataset.max_fake_frames
        flat.use_video_subset = config.dataset.use_video_subset
        flat.video_subset_count = config.dataset.video_subset_count
        flat.video_subset_start_idx = config.dataset.video_subset_start_idx

        # Label paths (computed from protocol)
        if 'real_label_path' in config.dataset:
            flat.real_label_path = config.dataset.real_label_path
            flat.fake_label_path = config.dataset.fake_label_path
            flat.val_label_path = config.dataset.val_label_path
            flat.test_label_path = config.dataset.test_label_path
            flat.metrics = config.dataset.metrics
            flat.real_test_label_path = config.dataset.real_test_label_path

        if 'mixing' in config.dataset:
            flat.use_mixed_forgeries = config.dataset.mixing.use_mixed_forgeries
            flat.forgery_mix_types = config.dataset.mixing.forgery_mix_types
            flat.forgery_mix_ratios = config.dataset.mixing.forgery_mix_ratios
            flat.forgery_mix_base_dir = config.dataset.mixing.forgery_mix_base_dir
            flat.total_fake_samples = config.dataset.mixing.total_fake_samples

    if 'logging' in config:
        if 'wandb' in config.logging:
            flat.use_wandb = config.logging.wandb.enabled
            flat.wandb_project = config.logging.wandb.project
            flat.wandb_entity = config.logging.wandb.entity

        flat.logs = config.logging.logs_dir

    if 'paths' in config:
        flat.checkpoint_path = config.paths.checkpoint
        flat.best_model_path = config.paths.best_model
        flat.save_code = config.paths.save_code

    if 'gpu' in config:
        flat.gpus = config.gpu.devices

    if 'metadata' in config:
        flat.seed = config.metadata.seed
        flat.comment = config.metadata.comment

    return flat


def load_config(config_path: Optional[str] = None,
                base_config: str = 'base.yaml',
                cli_args: Optional[list] = None,
                flat_compat: bool = True) -> Union[ConfigDict, object]:
    """
    Load configuration from YAML files with optional CLI overrides.

    Args:
        config_path: Path to experiment config (if None, only loads base config)
        base_config: Path to base config file (relative to configs dir)
        cli_args: Command line arguments for overrides
        flat_compat: Return flat config object for backward compatibility

    Returns:
        Loaded and processed configuration (nested or flat based on flat_compat)
    """
    # Determine config directory
    config_dir = Path(__file__).parent

    # Load base config
    base_path = config_dir / base_config
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    config_dict = load_yaml(base_path)

    # Load experiment config if provided
    if config_path is not None:
        exp_path = Path(config_path)

        # Handle different path formats
        if not exp_path.is_absolute():
            config_path_str = str(config_path)
            if config_path_str.startswith('configs/'):
                config_path_str = config_path_str[8:]
                exp_path = config_dir / config_path_str
            else:
                exp_path = config_dir / config_path

        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")

        exp_config = load_yaml(exp_path)
        config_dict = deep_merge(config_dict, exp_config)

    # Convert to ConfigDict for attribute access
    config = dict_to_configdict(config_dict)

    # Apply CLI overrides
    if cli_args is not None:
        cli_overrides = parse_cli_overrides(cli_args)
        if cli_overrides:
            config = dict_to_configdict(deep_merge(dict(config), cli_overrides))

    # Resolve paths
    config = resolve_paths(config, config_dir.parent)

    # Interpolate string placeholders
    config = interpolate_strings(config)

    # Compute derived values
    config = compute_derived_values(config)

    # Return flat config for backward compatibility if requested
    if flat_compat:
        return create_flat_config_object(config)

    return config


def save_config_snapshot(config: Union[ConfigDict, object], output_path: Union[str, Path]):
    """
    Save configuration snapshot for reproducibility.

    Args:
        config: Configuration to save
        output_path: Path to save config snapshot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to dict
    if isinstance(config, ConfigDict):
        def configdict_to_dict(c):
            if isinstance(c, ConfigDict):
                return {k: configdict_to_dict(v) for k, v in c.items()}
            return c
        config_dict = configdict_to_dict(config)
    else:
        # Flat config object - get attributes
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if not callable(value):
                    config_dict[attr] = value

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def print_config(config: Union[ConfigDict, object], indent: int = 0):
    """
    Pretty print configuration.

    Args:
        config: Configuration to print
        indent: Current indentation level
    """
    if isinstance(config, ConfigDict):
        for key, value in config.items():
            if isinstance(value, ConfigDict):
                print('  ' * indent + f"{key}:")
                print_config(value, indent + 1)
            else:
                print('  ' * indent + f"{key}: {value}")
    else:
        # Flat config object
        for attr in sorted(dir(config)):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if not callable(value):
                    print('  ' * indent + f"{attr}: {value}")


if __name__ == '__main__':
    # Example usage
    import sys

    parser = argparse.ArgumentParser(description='Test config loader')
    parser.add_argument('--config', type=str, help='Path to experiment config')
    parser.add_argument('--flat', action='store_true', help='Use flat compatibility mode')
    args, unknown = parser.parse_known_args()

    # Load config with CLI overrides
    config = load_config(
        config_path=args.config,
        cli_args=sys.argv[1:],
        flat_compat=args.flat
    )

    print("="*80)
    print("LOADED CONFIGURATION")
    print("="*80)

    if args.flat:
        print(f"Type: Flat Config Object (backward compatible)")
        print(f"Example access: config.lr = {config.lr}")
    else:
        print(f"Type: Nested ConfigDict")
        print(f"Example access: config.training.lr = {config.training.lr}")

    print("="*80)
    print_config(config)

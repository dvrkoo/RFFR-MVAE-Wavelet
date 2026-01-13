#!/usr/bin/env python3
"""
YAML Configuration Loader for RFFR-MVAE

Provides functionality to:
- Load YAML configuration files
- Merge base and experiment configs
- Override values from command line
- Compute derived values (LR scaling, effective batch size)
- Resolve dataset paths
- Validate configuration
- Save config snapshots for reproducibility
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
        """Get value from nested dict using dot notation (e.g., 'model.generator_type')"""
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


def compute_derived_values(config: ConfigDict) -> ConfigDict:
    """
    Compute derived configuration values.
    
    Computes:
    - Effective batch size
    - Learning rate with linear scaling
    - Dataset paths
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Config with computed values
    """
    # Compute effective batch size
    if 'training' in config:
        config.training.effective_batch_size = (
            config.training.batch_size * config.training.gradient_accumulation_steps
        )
        
        # Compute learning rate with linear scaling
        if hasattr(config.training, 'base_lr'):
            config.training.lr = config.training.base_lr * (
                config.training.effective_batch_size / 256.0
            )
    
    # Compute dataset path if not custom
    if 'dataset' in config and config.dataset.name != 'custom':
        dataset_path = get_dataset_path(
            config.dataset.name,
            config.dataset.split,
            config.dataset.type,
            config.paths.data_label_base if 'paths' in config else '../data_label'
        )
        config.dataset.label_path = dataset_path
    elif 'dataset' in config and config.dataset.name == 'custom':
        if config.dataset.custom_path is not None:
            config.dataset.label_path = config.dataset.custom_path
        else:
            raise ValueError("dataset.name='custom' requires dataset.custom_path to be set")
    
    return config


def get_dataset_path(dataset_name: str, split: str, data_type: str, base_dir: str = '../data_label') -> str:
    """
    Generate dataset path from name, split, and type.
    
    Args:
        dataset_name: Dataset name (ff270, ffhq, etc.)
        split: Data split (train/val/test)
        data_type: Data type (real/fake/df/f2f/etc.)
        base_dir: Base directory for data labels
    
    Returns:
        Path to dataset JSON file
    """
    # Map dataset names to directory names
    dataset_dir_map = {
        "ff270": "ff_270",
        "ff270_fake100": "ff_270_fake100",
        "ffhq": "ffhq_mae_vae_STAGE1",
        "forgerynet": "FN",
        "dfd": "Faceforensics/excludes_hq",
        "celebdf": "Faceforensics/excludes_hq",
        "faceforensics_hq": "Faceforensics/excludes_hq",
        "faceforensics_hq_fake100": "Faceforensics/excludes_hq_fake100",
    }
    
    if dataset_name not in dataset_dir_map:
        raise ValueError(
            f"Unknown dataset_name: '{dataset_name}'. "
            f"Available options: {list(dataset_dir_map.keys())}"
        )
    
    dataset_dir = dataset_dir_map[dataset_name]
    path = f"{base_dir}/{dataset_dir}/{split}/{data_type}_{split}_label.json"
    
    return path


def parse_cli_overrides(args: list) -> dict:
    """
    Parse command line overrides in the format --key value or --nested.key value.
    
    Args:
        args: List of command line arguments
    
    Returns:
        Dictionary with override values
    """
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
            
            # Try to convert value to appropriate type
            if value.lower() == 'true':
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
            
            # Handle nested keys (e.g., model.vae.beta -> {"model": {"vae": {"beta": value}}})
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


def load_config(config_path: Optional[str] = None, 
                base_config: str = 'base.yaml',
                cli_args: Optional[list] = None) -> ConfigDict:
    """
    Load configuration from YAML files with optional CLI overrides.
    
    Args:
        config_path: Path to experiment config (if None, only loads base config)
        base_config: Path to base config file (relative to configs dir)
        cli_args: Command line arguments for overrides
    
    Returns:
        Loaded and processed configuration
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
        
        # Handle different path formats:
        # 1. Absolute path: /full/path/to/config.yaml
        # 2. Relative to config dir: experiments/mae_ff270.yaml
        # 3. Relative to working dir with configs/ prefix: configs/experiments/mae_ff270.yaml
        if not exp_path.is_absolute():
            # If path starts with 'configs/', remove it since we're already in configs/
            config_path_str = str(config_path)
            if config_path_str.startswith('configs/'):
                config_path_str = config_path_str[8:]  # Remove 'configs/' prefix
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
            cli_config = dict_to_configdict(cli_overrides)
            config = dict_to_configdict(deep_merge(config_dict, cli_overrides))
    
    # Resolve paths
    config = resolve_paths(config, config_dir.parent)
    
    # Interpolate string placeholders
    config = interpolate_strings(config)
    
    # Compute derived values
    config = compute_derived_values(config)
    
    return config


def save_config_snapshot(config: ConfigDict, output_path: Union[str, Path]):
    """
    Save configuration snapshot for reproducibility.
    
    Args:
        config: Configuration to save
        output_path: Path to save config snapshot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert ConfigDict back to regular dict for YAML serialization
    def configdict_to_dict(c):
        if isinstance(c, ConfigDict):
            return {k: configdict_to_dict(v) for k, v in c.items()}
        return c
    
    config_dict = configdict_to_dict(config)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def print_config(config: ConfigDict, indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration to print
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, ConfigDict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")


if __name__ == '__main__':
    # Example usage
    import sys
    
    parser = argparse.ArgumentParser(description='Test config loader')
    parser.add_argument('--config', type=str, help='Path to experiment config')
    args, unknown = parser.parse_known_args()
    
    # Load config with CLI overrides
    config = load_config(
        config_path=args.config,
        cli_args=sys.argv[1:]
    )
    
    print("="*80)
    print("LOADED CONFIGURATION")
    print("="*80)
    print_config(config)

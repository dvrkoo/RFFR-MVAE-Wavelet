#!/usr/bin/env python3
"""
Test script to validate YAML config migration
Tests config loader without requiring full dependencies
"""

import sys
from configs.config_loader import load_config

def test_config_loading():
    """Test that config loads without errors"""
    print("=" * 80)
    print("Testing YAML Config Migration")
    print("=" * 80)
    
    # Test 1: Load base config
    print("\n[Test 1] Loading base config...")
    try:
        config = load_config(config_path=None)
        print("✓ Base config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load base config: {e}")
        return False
    
    # Test 2: Load experiment config (without configs/ prefix)
    print("\n[Test 2a] Loading mae_ff270 experiment config (without prefix)...")
    try:
        config = load_config(config_path="experiments/mae_ff270.yaml")
        print("✓ Experiment config loaded successfully")
        print(f"  - Experiment name: {config.metadata.experiment_name}")
        print(f"  - Generator type: {config.model.generator_type}")
        print(f"  - Dataset: {config.dataset.name}")
    except Exception as e:
        print(f"✗ Failed to load experiment config: {e}")
        return False
    
    # Test 2b: Load experiment config (with configs/ prefix)
    print("\n[Test 2b] Loading mae_ff270 experiment config (with configs/ prefix)...")
    try:
        config = load_config(config_path="configs/experiments/mae_ff270.yaml")
        print("✓ Experiment config loaded successfully (both formats work)")
    except Exception as e:
        print(f"✗ Failed to load experiment config with prefix: {e}")
        return False
    
    # Test 3: Test CLI overrides
    print("\n[Test 3] Testing CLI overrides...")
    try:
        config = load_config(
            config_path="experiments/mae_ff270.yaml",
            cli_args=["--config", "experiments/mae_ff270.yaml", 
                      "--training.batch_size", "64",
                      "--model.mae.mask_ratio", "0.8"]
        )
        assert config.training.batch_size == 64, f"Expected batch_size=64, got {config.training.batch_size}"
        assert config.model.mae.mask_ratio == 0.8, f"Expected mask_ratio=0.8, got {config.model.mae.mask_ratio}"
        print("✓ CLI overrides working correctly")
        print(f"  - Batch size: {config.training.batch_size}")
        print(f"  - Mask ratio: {config.model.mae.mask_ratio}")
    except Exception as e:
        print(f"✗ CLI overrides failed: {e}")
        return False
    
    # Test 4: Test derived values
    print("\n[Test 4] Testing derived value computation...")
    try:
        config = load_config(config_path="experiments/mae_ff270.yaml")
        expected_effective_bs = config.training.batch_size * config.training.gradient_accumulation_steps
        assert config.training.effective_batch_size == expected_effective_bs, \
            f"Expected effective_batch_size={expected_effective_bs}, got {config.training.effective_batch_size}"
        
        expected_lr = config.training.base_lr * (config.training.effective_batch_size / 256)
        assert abs(config.training.lr - expected_lr) < 1e-9, \
            f"Expected lr={expected_lr}, got {config.training.lr}"
        
        print("✓ Derived values computed correctly")
        print(f"  - Effective batch size: {config.training.effective_batch_size}")
        print(f"  - Learning rate: {config.training.lr:.6f}")
    except Exception as e:
        print(f"✗ Derived value computation failed: {e}")
        return False
    
    # Test 5: Test path interpolation
    print("\n[Test 5] Testing path interpolation...")
    try:
        config = load_config(config_path="experiments/mae_ff270.yaml")
        assert "mae_ff270" in config.paths.checkpoint, \
            f"Expected 'mae_ff270' in checkpoint path, got {config.paths.checkpoint}"
        print("✓ Path interpolation working")
        print(f"  - Checkpoint path: {config.paths.checkpoint}")
    except Exception as e:
        print(f"✗ Path interpolation failed: {e}")
        return False
    
    # Test 6: Test MAE-VAE config
    print("\n[Test 6] Loading mae_vae_ffhq_stage1 config...")
    try:
        config = load_config(config_path="experiments/mae_vae_ffhq_stage1.yaml")
        assert config.model.generator_type == "mae_vae", \
            f"Expected generator_type='mae_vae', got {config.model.generator_type}"
        assert hasattr(config.model.vae, 'beta'), "VAE config missing beta parameter"
        print("✓ MAE-VAE config loaded successfully")
        print(f"  - Generator: {config.model.generator_type}")
        print(f"  - VAE latent dim: {config.model.vae.latent_dim}")
        print(f"  - VAE beta: {config.model.vae.beta}")
    except Exception as e:
        print(f"✗ Failed to load MAE-VAE config: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! YAML config migration successful")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)

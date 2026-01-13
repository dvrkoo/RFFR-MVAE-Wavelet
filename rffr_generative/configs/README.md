# YAML Configuration System

The RFFR-MVAE generative training now uses YAML-based configuration for flexibility and reproducibility.

## Quick Start

### Basic Usage

```bash
# Use base configuration (default)
python train.py

# Use experiment configuration
python train.py --config experiments/mae_ff270.yaml

# Override specific parameters
python train.py --config experiments/mae_ff270.yaml \
    --training.batch_size 64 \
    --model.mae.mask_ratio 0.8 \
    --training.gradient_accumulation_steps 5
```

### Available Experiment Configs

- `experiments/mae_ff270.yaml` - MAE training on FaceForensics++ (270 videos)
- `experiments/mae_vae_ffhq_stage1.yaml` - MAE-VAE stage 1 on FFHQ

## Configuration Structure

### Main Sections

```yaml
metadata:         # Experiment metadata (seed, name, comments)
model:           # Model architecture (MAE, VAE settings)
training:        # Training hyperparameters (batch size, LR, etc.)
dataset:         # Dataset selection and configuration
paths:           # File paths (checkpoints, data, logs)
logging:         # TensorBoard and WandB settings
gpu:             # GPU device configuration
```

### Creating New Experiments

1. **Copy an existing experiment:**
   ```bash
   cp experiments/mae_ff270.yaml experiments/my_experiment.yaml
   ```

2. **Edit only the values you want to change:**
   ```yaml
   metadata:
     experiment_name: "my_custom_experiment"
     comment: "Testing new hyperparameters"
   
   training:
     batch_size: 64
     gradient_accumulation_steps: 5
   
   dataset:
     name: "ffhq"
   ```

3. **Run your experiment:**
   ```bash
   python train.py --config experiments/my_experiment.yaml
   ```

## Advanced Features

### CLI Overrides

Use dot notation to override any nested parameter:

```bash
python train.py \
    --config experiments/mae_ff270.yaml \
    --training.batch_size 32 \
    --training.base_lr 0.0002 \
    --model.vae.beta 0.0005 \
    --dataset.use_forgerynet true
```

### Derived Values

Some values are computed automatically:

- **Effective batch size**: `batch_size × gradient_accumulation_steps`
- **Learning rate**: `base_lr × (effective_batch_size / 256)` (linear scaling)
- **Dataset path**: Resolved from `(name, split, type)` triplet

### Path Interpolation

Use `${section.variable}` syntax for dynamic paths:

```yaml
paths:
  checkpoint: "./checkpoints/${model.generator_type}_${dataset.name}"
  # Resolves to: ./checkpoints/mae_ff270
```

### Dataset Selection

Supported datasets:
- `ff270` - FaceForensics++ (270 videos, original split)
- `ff270_fake100` - FF++ with 100 fake videos
- `ffhq` - FFHQ faces
- `forgerynet` - ForgeryNet dataset
- `dfd` - DeepFake Detection dataset
- `celebdf` - CelebDF dataset
- `custom` - Custom dataset (set `dataset.custom_path`)

Dataset types: `real`, `fake`, `mixed`, `df`, `f2f`, `fs`, `fsw`, `nt`

### WandB Integration

Configure Weights & Biases logging:

```yaml
logging:
  wandb:
    enabled: true
    project: "RFFR-MVAE"
    entity: "your-username"
    run_name: "mae_ff270_exp1"
    tags: ["mae", "ff270", "experiment1"]
    notes: "Testing new hyperparameters"
```

## Config Validation

Test your configuration without training:

```bash
# Validate config loading
python configs/config_loader.py --config experiments/mae_ff270.yaml

# Run all config tests
python test_config.py
```

## Migration Notes

### For Users of Old Python Configs

The old Python-based config (`configs/config.py`) is **deprecated** but still present. The new system offers:

- ✅ **Easier experimentation** - Just edit YAML files
- ✅ **Better reproducibility** - Config snapshots saved with checkpoints
- ✅ **CLI overrides** - No code editing needed
- ✅ **Cleaner structure** - Nested organization vs flat attributes
- ✅ **Type safety** - YAML parsing with validation

### Key Differences

| Old (Python) | New (YAML) |
|-------------|-----------|
| `config.batch_size` | `config.training.batch_size` |
| `config.generator_type` | `config.model.generator_type` |
| `config.enable_wandb` | `config.logging.wandb.enabled` |
| `config.gpus` | `config.gpu.devices` |
| `config.seed` | `config.metadata.seed` |

## Examples

### Example 1: Quick Hyperparameter Search

```bash
# Test different batch sizes
for bs in 32 64 128; do
    python train.py --config experiments/mae_ff270.yaml \
        --training.batch_size $bs \
        --metadata.experiment_name "mae_ff270_bs${bs}"
done
```

### Example 2: MAE-VAE Two-Stage Training

```bash
# Stage 1: Train MAE encoder
python train.py --config experiments/mae_vae_ffhq_stage1.yaml

# Stage 2: Fine-tune with VAE
python train.py --config experiments/mae_vae_ffhq_stage2.yaml \
    --paths.pretrained "./checkpoints/mae_vae_ffhq_stage1/best_model.pth"
```

### Example 3: Multi-Dataset Training

```bash
# Train on ForgeryNet + FF++
python train.py --config experiments/mae_ff270.yaml \
    --dataset.use_forgerynet true \
    --dataset.forgerynet.num_videos 150000 \
    --metadata.experiment_name "mae_ff270_fn150k"
```

## Troubleshooting

### Config not found
```
Error: Config file not found: experiments/my_config.yaml
```
→ Check the path is relative to `configs/` directory

### Invalid parameter override
```
Error: Key 'training.invalid_param' not found in config
```
→ Check parameter exists in base.yaml and use correct nesting

### Dataset validation failed
```
❌ [ERROR] Dataset validation failed: File not found
```
→ Check `dataset.name`, `dataset.split`, `dataset.type` are correct
→ Run `print_available_datasets()` to see valid options

## Files

- `base.yaml` - Default configuration for all parameters
- `config_loader.py` - YAML loading and merging logic
- `experiments/*.yaml` - Experiment-specific overrides
- `README.md` - This file
- `../test_config.py` - Configuration validation tests

## Support

For issues or questions:
1. Check `base.yaml` for available parameters
2. Run `python test_config.py` to validate your setup
3. See `train.py` for usage in code

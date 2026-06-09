# RFFR-MVAE-Wavelet

Working repository for our PatchRec/RFFR deepfake detection code before transplanting it into:

```text
https://github.com/dvrkoo/PatchRecDeepFakeDetection
```

This repo is the local paper-code staging area. The current classifier supports:

- MAE generator
- MAE-VAE generator
- 2-branch classifier: RGB + spatial residual
- 3-branch classifier: RGB + spatial residual + wavelet residual

The classifier test entry point is:

```bash
python classifier/test.py
```

## Layout

```text
classifier/              Deepfake classifier
classifier/configs/      Python and YAML classifier configs
classifier/models/       Classifier and generator wrapper models
classifier/utils/        Dataset, evaluation, and wavelet utilities
generative/              MAE / MAE-VAE training code
requirements.txt         Python dependencies
```

## Environment

Install the Python requirements from the repo root:

```bash
pip install -r requirements.txt
```

The active local environment also needs:

```bash
pip install timm scikit-learn
```

`pytorch_wavelets` is optional for the current path because the code falls back to the PyWavelets implementation.

## Local Paths

Local label root on this machine:

```text
/home/nick/.local/share/Trash/files/data_label
```

The FF++ labels under that directory point to images under:

```text
/home/nick/FF++
```

The CelebDF labels under that directory point to images under:

```text
/home/nick/GitHub/tools/celebdf
```

Main MAE-VAE generator checkpoint used by the classifier config:

```text
/home/nick/GitHub/RFFR/rffr_generative/checkpoint/checkpoint/mae_vae/CDF/best_loss_0.03285_100.pth.tar
```

Main 3-branch MAE-VAE classifier checkpoint:

```text
/home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar
```

Other local classifier checkpoints from the previous RFFR repo:

```text
/home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/3branch_wavelet_residual_F2F_All_Fake1_2025-11-20-17:09:17_8a88dc/best_model/2025-11-20-17:09:17_8a88dc/1__AUC_0.72589_285.pth.tar
/home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/3branch_wavelet_residual_F2F_All_2025-11-24-09:36:31_99f735/best_model/2025-11-24-09:36:31_99f735/1__AUC_0.79474_90.pth.tar
/home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae/2branch_standard_F2F_All_Fake100_2025-11-25-15:55:37_560fa9/best_model/2025-11-25-15:55:37_560fa9/1__AUC_0.81892_90.pth.tar
```

## Test Commands

Run a small FF++ Face2Face test:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/f2f_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/real_test_label.json \
  --dataset-name F2F \
  --samples 140 \
  --batch-size 16 \
  --num-workers 4
```

Run FF++ Deepfakes:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/df_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/real_test_label.json \
  --dataset-name DF \
  --samples 140 \
  --batch-size 16 \
  --num-workers 4
```

Run FF++ FaceSwap:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/fsw_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/real_test_label.json \
  --dataset-name FSW \
  --samples 140 \
  --batch-size 16 \
  --num-workers 4
```

Run FF++ NeuralTextures:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/nt_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/real_test_label.json \
  --dataset-name NT \
  --samples 140 \
  --batch-size 16 \
  --num-workers 4
```

Run DFD:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/dfd_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/dfd_real_test_label.json \
  --dataset-name DFD \
  --samples 700 \
  --batch-size 16 \
  --num-workers 4
```

Run CelebDF:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/celebdf_fake_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/celebdf_real_test_label.json \
  --dataset-name CelebDF \
  --samples 700 \
  --batch-size 16 \
  --num-workers 4
```

Save metrics JSON:

```bash
python classifier/test.py \
  --checkpoint /home/nick/GitHub/RFFR/rffr_classifier/checkpoint/checkpoint/FF_FN_mae_vae_wave/best_model/2025-11-26-13:56:57_d59139/1__AUC_0.80015_255.pth.tar \
  --fake-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/f2f_test_label.json \
  --real-label /home/nick/.local/share/Trash/files/data_label/Faceforensics/excludes_hq/real_test_label.json \
  --dataset-name F2F \
  --samples 140 \
  --batch-size 16 \
  --num-workers 4 \
  --save-json \
  --output-dir classifier/test_results
```

## Configs

Classifier YAML examples live in:

```text
classifier/configs/experiments/
```

The important switch is:

```yaml
model:
  generative_model_type: "mae_vae"
  architecture:
    wavelet_residual_branch: true
```

Use `wavelet_residual_branch: false` for the 2-branch classifier.

Use `generative_model_type: "mae"` or `generative_model_type: "mae_vae"` for the generator.

## Training

Train the generator:

```bash
cd generative
python train.py
```

Train the classifier:

```bash
cd classifier
python train.py
```

Use YAML configs where possible:

```bash
cd classifier
python train.py --config configs/experiments/f2f_mae_vae_3branch_wavelet.yaml
```

## Notes For Transplant

Keep the transplant focused on:

- `classifier/test.py`
- `classifier/train.py`
- `classifier/models/model_detector.py`
- `classifier/models/model_mae.py`
- `classifier/models/model_mae_vae.py`
- `classifier/utils/simple_evaluate.py`
- `classifier/utils/dataset.py`
- `classifier/utils/wavelet_utils.py`
- `classifier/configs/`
- `generative/models/model_mae.py`
- `generative/models/model_mae_vae.py`

Do not transplant local artifacts, old duplicate tests, old reconstruction scripts, old result JSONs, or backup files.

# RFFR-MVAE-Wavelet

Paper-code staging repo for the MAE-VAE wavelet residual classifier before transplant to:

```text
https://github.com/dvrkoo/PatchRecDeepFakeDetection
```

Scope is intentionally narrow:

- MAE-VAE generator only
- 3-branch classifier only: RGB + spatial residual + wavelet residual
- Face2Face fake-frame baselines: 1, 3, 5, 7, 100 fake frames per fake video

## Setup

```bash
pip install -r requirements.txt
pip install timm scikit-learn
```

Set paths before training or testing:

```bash
export RFFR_DATA_LABEL_DIR=/path/to/data_label
export RFFR_MAE_VAE_CKPT=/path/to/mae_vae_checkpoint.pth.tar
```

## data_label Contract

`RFFR_DATA_LABEL_DIR` points to the directory containing `ff_270/`, `ff_270_fake*/`, and `Faceforensics/`.

Each label file is a JSON list:

```json
[
  {"path": "/absolute/path/to/real_frame.png", "label": 0},
  {"path": "/absolute/path/to/fake_frame.png", "label": 1}
]
```

Rules:

- `path` must point to a readable cropped face image.
- `label` is `0` for real, `1` for fake.
- Absolute paths are preferred.
- Images are frames/crops, not raw videos.

Expected layout:

```text
data_label/
  ff_270/
    train/
      real_train_label.json

  ff_270_fake1/train/f2f_train_label.json
  ff_270_fake3/train/f2f_train_label.json
  ff_270_fake5/train/f2f_train_label.json
  ff_270_fake7/train/f2f_train_label.json
  ff_270_fake100/train/f2f_train_label.json

  Faceforensics/
    excludes_hq/
      real_val_label.json
      real_test_label.json
      df_val_label.json
      df_test_label.json
      f2f_val_label.json
      f2f_test_label.json
      fsw_val_label.json
      fsw_test_label.json
      nt_val_label.json
      nt_test_label.json
      fs_test_label.json
      dfd_test_label.json
      dfd_real_test_label.json
      celebdf_fake_test_label.json
      celebdf_real_test_label.json
```

`ff_270_fakeN` means fake training labels use `N` frames per fake video. Real training labels stay in `ff_270/train/real_train_label.json`.

Generate FF++ labels from cropped frame folders:

```bash
python tools/create_ffpp_labels.py \
  --ffpp-root /path/to/FF++ \
  --out "$RFFR_DATA_LABEL_DIR" \
  --fake-frames 1,3,5,7,100
```

Check labels before running:

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RFFR_DATA_LABEL_DIR"])
for rel in [
    "Faceforensics/excludes_hq/real_test_label.json",
    "Faceforensics/excludes_hq/f2f_test_label.json",
    "ff_270_fake5/train/f2f_train_label.json",
]:
    data = json.loads((root / rel).read_text())
    missing = [x["path"] for x in data[:20] if not Path(x["path"]).exists()]
    print(rel, "ok" if not missing else f"missing {len(missing)}/20")
PY
```

## Training Baselines

```bash
cd classifier
python train.py --config configs/experiments/f2f_mae_vae_3branch_fake1.yaml
python train.py --config configs/experiments/f2f_mae_vae_3branch_fake3.yaml
python train.py --config configs/experiments/f2f_mae_vae_3branch_fake5.yaml
python train.py --config configs/experiments/f2f_mae_vae_3branch_fake7.yaml
python train.py --config configs/experiments/f2f_mae_vae_3branch_fake100.yaml
```

## Testing

Use `classifier/test.py` with a trained classifier checkpoint.

Set the classifier checkpoint path:

```bash
export RFFR_CLASSIFIER_CKPT=/path/to/classifier_checkpoint.pth.tar
```

Face2Face:

```bash
python classifier/test.py \
  --config experiments/f2f_mae_vae_3branch_fake100.yaml \
  --checkpoint "$RFFR_CLASSIFIER_CKPT" \
  --fake-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/f2f_test_label.json" \
  --real-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/real_test_label.json" \
  --dataset-name F2F \
  --samples 140 \
  --batch-size 16
```

Deepfakes, FaceSwap, NeuralTextures:

```bash
python classifier/test.py --config experiments/f2f_mae_vae_3branch_fake100.yaml --checkpoint "$RFFR_CLASSIFIER_CKPT" --fake-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/df_test_label.json" --real-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/real_test_label.json" --dataset-name DF --samples 140 --batch-size 16
python classifier/test.py --config experiments/f2f_mae_vae_3branch_fake100.yaml --checkpoint "$RFFR_CLASSIFIER_CKPT" --fake-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/fsw_test_label.json" --real-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/real_test_label.json" --dataset-name FSW --samples 140 --batch-size 16
python classifier/test.py --config experiments/f2f_mae_vae_3branch_fake100.yaml --checkpoint "$RFFR_CLASSIFIER_CKPT" --fake-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/nt_test_label.json" --real-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/real_test_label.json" --dataset-name NT --samples 140 --batch-size 16
```

DFD and CelebDF:

```bash
python classifier/test.py --config experiments/f2f_mae_vae_3branch_fake100.yaml --checkpoint "$RFFR_CLASSIFIER_CKPT" --fake-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/dfd_test_label.json" --real-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/dfd_real_test_label.json" --dataset-name DFD --samples 700 --batch-size 16
python classifier/test.py --config experiments/f2f_mae_vae_3branch_fake100.yaml --checkpoint "$RFFR_CLASSIFIER_CKPT" --fake-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/celebdf_fake_test_label.json" --real-label "$RFFR_DATA_LABEL_DIR/Faceforensics/excludes_hq/celebdf_real_test_label.json" --dataset-name CelebDF --samples 700 --batch-size 16
```

## Files To Transplant

- `classifier/`
- `generative/models/model_mae_vae.py`
- `requirements.txt`
- `README.md`

Do not transplant local artifacts, old duplicate tests, reconstruction scripts, result JSONs, or backup files.

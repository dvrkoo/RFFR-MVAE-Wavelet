#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


MANIPS = {
    "df": "Deepfakes",
    "f2f": "Face2Face",
    "fsw": "FaceSwap",
    "nt": "NeuralTextures",
    "fs": "FaceShifter",
}


def videos():
    ids = [f"{i:03d}" for i in range(1000)]
    return ids[:720], ids[720:860], ids[860:]


def sample(files, n):
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def real(root, split, n):
    base = root / "original_sequences" / "youtube" / "c23" / "images"
    rows = []
    for video in split:
        rows.extend({"path": str(p), "label": 0} for p in sample(sorted((base / video).glob("*.png")), n))
    return rows


def fake(root, split, manip, n):
    base = root / "manipulated_sequences" / manip / "c23" / "images"
    rows = []
    if not base.exists():
        return rows
    for pair in sorted(base.iterdir()):
        if pair.is_dir() and pair.name.split("_")[0] in split:
            rows.extend({"path": str(p), "label": 1} for p in sample(sorted(pair.glob("*.png")), n))
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(f"{path}: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description="Create FF++ labels for PM-VAE 3-branch baselines")
    parser.add_argument("--ffpp-root", required=True, help="FF++ root with original_sequences/ and manipulated_sequences/")
    parser.add_argument("--out", required=True, help="Output data_label directory")
    parser.add_argument("--fake-frames", default="1,3,5,7,100")
    parser.add_argument("--real-frames", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.ffpp_root)
    out = Path(args.out)
    train, val, test = videos()

    write(out / "ff_270" / "train" / "real_train_label.json", real(root, train, args.real_frames))
    write(out / "Faceforensics" / "excludes_hq" / "real_val_label.json", real(root, val, args.real_frames))
    write(out / "Faceforensics" / "excludes_hq" / "real_test_label.json", real(root, test, args.real_frames))

    for short, manip in MANIPS.items():
        write(out / "Faceforensics" / "excludes_hq" / f"{short}_val_label.json", fake(root, val, manip, args.real_frames))
        write(out / "Faceforensics" / "excludes_hq" / f"{short}_test_label.json", fake(root, test, manip, args.real_frames))

    for n in [int(x.strip()) for x in args.fake_frames.split(",") if x.strip()]:
        write(out / f"ff_270_fake{n}" / "train" / "f2f_train_label.json", fake(root, train, "Face2Face", n))


if __name__ == "__main__":
    main()

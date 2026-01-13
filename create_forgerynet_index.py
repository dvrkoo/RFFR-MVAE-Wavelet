import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def count_frames_in_video(video_dir):
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
    return len(frame_files)


def get_evenly_spaced_frames(total_frames, num_frames=10):
    if total_frames <= num_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return indices.tolist()


def verify_frames_exist(video_dir, frame_indices, frame_pattern):
    valid_indices = []
    for idx in frame_indices:
        frame_path = os.path.join(video_dir, frame_pattern.format(idx))
        if os.path.isfile(frame_path) and os.path.getsize(frame_path) > 0:
            valid_indices.append(idx)
    return valid_indices


def create_forgerynet_index(
    forgerynet_base_path, output_json_path, frames_per_video=10
):
    forgerynet_base = Path(forgerynet_base_path)

    if not forgerynet_base.exists():
        print(f"Error: ForgeryNet path does not exist: {forgerynet_base_path}")
        return

    video_index = []
    skipped_videos = 0
    corrupted_frames = 0

    categories = sorted([d for d in forgerynet_base.iterdir() if d.is_dir()])

    print(f"Found {len(categories)} categories: {[c.name for c in categories]}")

    for category_dir in categories:
        print(f"\nProcessing category: {category_dir.name}")

        video_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])

        for video_dir in tqdm(video_dirs, desc=f"Category {category_dir.name}"):
            try:
                total_frames = count_frames_in_video(video_dir)

                if total_frames < 10:
                    skipped_videos += 1
                    continue

                frame_indices = get_evenly_spaced_frames(total_frames, frames_per_video)

                valid_indices = verify_frames_exist(
                    video_dir, frame_indices, "face_{:04d}.png"
                )

                if len(valid_indices) < frames_per_video * 0.8:
                    skipped_videos += 1
                    corrupted_frames += len(frame_indices) - len(valid_indices)
                    continue

                if len(valid_indices) < len(frame_indices):
                    corrupted_frames += len(frame_indices) - len(valid_indices)

                video_entry = {
                    "video_dir": str(video_dir),
                    "category": category_dir.name,
                    "total_frames": total_frames,
                    "frame_indices": valid_indices,
                    "frame_pattern": "face_{:04d}.png",
                }

                video_index.append(video_entry)

            except Exception as e:
                skipped_videos += 1
                continue

    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos indexed: {len(video_index)}")
    print(f"Skipped videos (< 10 frames or >20% corrupted): {skipped_videos}")
    print(f"Corrupted/missing frames detected: {corrupted_frames}")

    print(f"\nVideos per category:")
    for cat in sorted(set(v["category"] for v in video_index)):
        cat_count = sum(1 for v in video_index if v["category"] == cat)
        print(f"  Category {cat}: {cat_count} videos")

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(video_index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ForgeryNet index saved to: {output_path}")

    total_frames = sum(len(v["frame_indices"]) for v in video_index)
    print(f"Total valid frames: {total_frames}")
    print(f"{'='*60}")


if __name__ == "__main__":
    forgerynet_path = "/seidenas/datasets/ForgeryNet/Real/"
    output_json = "./data_label/FN/train/forgerynet_video_index.json"

    print("=" * 60)
    print("ForgeryNet Video Index Generator")
    print("=" * 60)
    print(f"ForgeryNet path: {forgerynet_path}")
    print(f"Output JSON: {output_json}")
    print(f"Frames per video: 10")
    print("=" * 60)

    create_forgerynet_index(forgerynet_path, output_json, frames_per_video=10)

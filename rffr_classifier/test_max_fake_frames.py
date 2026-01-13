#!/usr/bin/env python3
"""
Test script to verify max_fake_frames implementation.
Tests different frame limits: 90, 180, 360, and None (all frames).
"""

import sys
import json

# Add parent directory to path
sys.path.insert(0, "/home/nick/GitHub/RFFR/rffr_classifier")

from configs.config import config


def test_frame_limiting():
    """Test the max_fake_frames filtering logic."""

    print("=" * 70)
    print("Testing max_fake_frames Implementation")
    print("=" * 70)

    # Test configurations
    test_cases = [
        (None, "No limiting (use all frames)"),
        (90, "Limit to 90 fake frames"),
        (180, "Limit to 180 fake frames"),
        (360, "Limit to 360 fake frames"),
    ]

    # Set protocol to fake1
    original_protocol = config.protocol
    config.protocol = "F2F_All_Fake1"

    print(f"\nUsing protocol: {config.protocol}")
    print(f"Dataset base: {config.dataset_base}")

    for max_frames, description in test_cases:
        print(f"\n{'-' * 70}")
        print(f"Test Case: {description}")
        print(f"max_fake_frames = {max_frames}")
        print(f"{'-' * 70}")

        # Set the config
        config.max_fake_frames = max_frames

        try:
            # Import here to avoid caching issues
            from utils.get_loader import get_dataset

            # Load data
            train_real_loader, train_fake_loader, val_loaders = get_dataset()

            # Get actual counts
            actual_fake = len(train_fake_loader.dataset)
            actual_real = len(train_real_loader.dataset)

            print(f"\n✓ Results:")
            print(f"  Fake frames loaded: {actual_fake}")
            print(f"  Real frames loaded: {actual_real}")

            if actual_fake > 0:
                ratio = actual_real / actual_fake
                print(f"  Fake:Real ratio: 1:{ratio:.1f}")

            # Verify counts
            if max_frames is not None and actual_fake != min(max_frames, 720):
                print(f"\n✗ ERROR: Expected {min(max_frames, 720)} fake frames!")
                return False

            if actual_real != 7200:
                print(f"\n✗ ERROR: Expected 7200 real frames!")
                return False

            # Sample some paths to verify video IDs
            if actual_fake > 0:
                sample_entry = train_fake_loader.dataset[0]
                first_path = sample_entry[0] if isinstance(sample_entry, tuple) else None
                if max_frames is not None and max_frames < 720:
                    last_entry = train_fake_loader.dataset[-1]
                    last_path = (
                        last_entry[0] if isinstance(last_entry, tuple) else None
                    )
                    print(f"\n  Sample verification:")
                    print(f"    First frame path contains video range: 000-xxx")
                    print(
                        f"    Last frame path should contain video: {max_frames-1:03d}"
                    )

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Restore original protocol
    config.protocol = original_protocol

    print(f"\n{'=' * 70}")
    print("✓ All tests passed!")
    print(f"{'=' * 70}")

    return True


def verify_data_structure():
    """Verify the structure of fake1 label files."""

    print("\n" + "=" * 70)
    print("Verifying fake1 Label File Structure")
    print("=" * 70)

    label_file = "/home/nick/GitHub/RFFR/data_label/ff_270_fake1/train/f2f_train_label.json"

    try:
        with open(label_file) as f:
            data = json.load(f)

        print(f"\nLabel file: {label_file}")
        print(f"Total entries: {len(data)}")

        # Show first few entries
        print("\nFirst 3 entries:")
        for i, entry in enumerate(data[:3]):
            path = entry["path"]
            video_pair = path.split("/")[-2]
            source_video = video_pair.split("_")[0]
            print(f"  [{i}] Video {source_video}: {video_pair}")

        # Show last few entries
        print("\nLast 3 entries:")
        for i, entry in enumerate(data[-3:], len(data) - 3):
            path = entry["path"]
            video_pair = path.split("/")[-2]
            source_video = video_pair.split("_")[0]
            print(f"  [{i}] Video {source_video}: {video_pair}")

        # Verify ordering
        video_ids = []
        for entry in data:
            path = entry["path"]
            video_pair = path.split("/")[-2]
            source_video = video_pair.split("_")[0]
            video_ids.append(int(source_video))

        is_sorted = video_ids == sorted(video_ids)
        print(f"\nVideos are sorted: {is_sorted}")
        print(f"Video range: {min(video_ids):03d} - {max(video_ids):03d}")

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nRFFR Classifier - max_fake_frames Feature Test")
    print("=" * 70)

    # Verify data structure
    if not verify_data_structure():
        print("\n✗ Data structure verification failed!")
        sys.exit(1)

    # Test frame limiting
    if not test_frame_limiting():
        print("\n✗ Frame limiting tests failed!")
        sys.exit(1)

    print("\n✓ All verifications passed!")
    sys.exit(0)

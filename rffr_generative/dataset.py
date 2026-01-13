import cv2
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import ImageFile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys

sys.path.append("..")
from configs.config import config
import json
import random


def load_single_image(args):
    """Load and preprocess a single image for parallel caching"""
    idx, img_path = args
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return idx, img


class Deepfake_Dataset(Dataset):
    def __init__(self, paths, transforms=None, train=True, precache=None, cache_workers=None, 
                 cache_file=None):
        self.train = train
        self.paths = paths
        
        # Use config defaults if not specified
        if precache is None:
            precache = config.use_dataset_cache
        if cache_workers is None:
            cache_workers = config.dataset_cache_workers
        if cache_file is None:
            cache_file = config.dataset_cache_file
            
        self.precache = precache
        self.cache = {}
        self.cache_file = cache_file

        self.transforms = T.Compose(
            [
                T.ToTensor()
            ]
        )

        if self.precache:
            # Only attempt to load/save cache if cache_file is specified
            if self.cache_file and os.path.exists(self.cache_file):
                print(f"Loading cached images from {self.cache_file}...")
                try:
                    with open(self.cache_file, "rb") as f:
                        self.cache = pickle.load(f)
                    print(f"Loaded {len(self.cache)} images from cache")
                except Exception as e:
                    print(f"Failed to load cache: {e}, rebuilding...")
                    self._build_cache(cache_workers)
            else:
                self._build_cache(cache_workers)
    
    def _build_cache(self, cache_workers):
        print(f"Precaching {len(self.paths)} images to RAM using {cache_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=cache_workers) as executor:
            futures = {executor.submit(load_single_image, (i, p)): i for i, p in enumerate(self.paths)}
            
            for future in tqdm(as_completed(futures), total=len(self.paths), desc="Loading images"):
                idx, img = future.result()
                self.cache[idx] = img
        
        print(f"Precaching complete. Using ~{len(self.cache) * 224 * 224 * 3 / 1e9:.1f} GB RAM")
        
        # Only save cache to disk if cache_file is specified
        if self.cache_file:
            print(f"Saving cache to {self.cache_file}...")
            try:
                # Create directory if it doesn't exist
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                    
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self.cache, f)
                print("Cache saved successfully")
            except Exception as e:
                print(f"Failed to save cache: {e}")
        else:
            print("Cache file not specified, skipping disk save (RAM cache only)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        if self.precache and item in self.cache:
            img = self.cache[item]
        else:
            img_path = self.paths[item]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(img)
        return img


class ForgeryNetRotatingDataset(Dataset):
    def __init__(
        self,
        index_json_path,
        num_videos_per_epoch=720,
        frames_per_video=None,
        seed=None,
        categories=["16", "17", "18", "19"],
        rotate=True,
    ):
        with open(index_json_path, "r") as f:
            self.master_index = json.load(f)

        self.categories = categories
        self.category_videos = {cat: [] for cat in categories}
        for video_info in self.master_index:
            cat = video_info["category"]
            if cat in self.category_videos:
                self.category_videos[cat].append(video_info)

        self.num_videos_per_epoch = num_videos_per_epoch
        self.frames_per_video = frames_per_video
        self.videos_per_category = num_videos_per_epoch // len(categories)
        self.seed = seed
        self.rotate = rotate
        self.epoch_seed = 0
        self.current_epoch_videos = []
        self.current_epoch_frames = []

        self.transforms = T.Compose([T.ToTensor()])

        self._sample_epoch_videos()

    def _sample_epoch_videos(self):
        rng = random.Random(
            self.seed + self.epoch_seed if self.seed is not None else None
        )

        self.current_epoch_videos = []
        for cat in self.categories:
            cat_videos = self.category_videos[cat]
            num_to_sample = min(self.videos_per_category, len(cat_videos))
            sampled = rng.sample(cat_videos, num_to_sample)
            self.current_epoch_videos.extend(sampled)

        self.current_epoch_frames = []
        for video_info in self.current_epoch_videos:
            video_dir = video_info["video_dir"]
            frame_pattern = video_info["frame_pattern"]
            frame_indices = video_info["frame_indices"]
            if self.frames_per_video is not None:
                frame_indices = frame_indices[
                    : self.frames_per_video
                ]
            for frame_idx in frame_indices:
                frame_path = os.path.join(video_dir, frame_pattern.format(frame_idx))
                self.current_epoch_frames.append(frame_path)

    def set_epoch(self, epoch):
        if self.rotate:
            self.epoch_seed = epoch
            self._sample_epoch_videos()

    def __len__(self):
        return len(self.current_epoch_frames)

    def __getitem__(self, item):
        img_path = self.current_epoch_frames[item]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise IOError(f"cv2.imread returned None for {img_path}")

            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transforms(img)
            return img

        except Exception as e:
            print(
                f"Warning: Failed to load {img_path} (Error: {e}). Loading a random image instead."
            )
            new_item = random.randint(0, len(self) - 1)
            return self.__getitem__(new_item)


class MixedDataset(Dataset):
    def __init__(self, ff_dataset, fn_dataset):
        self.ff_dataset = ff_dataset
        self.fn_dataset = fn_dataset

        self.ff_len = len(ff_dataset)
        self.fn_len = len(fn_dataset)

        self.total_len = self.ff_len + self.fn_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        if item < self.ff_len:
            return self.ff_dataset[item]
        else:
            return self.fn_dataset[item - self.ff_len]

    def set_epoch(self, epoch):
        if hasattr(self.fn_dataset, "set_epoch"):
            self.fn_dataset.set_epoch(epoch)

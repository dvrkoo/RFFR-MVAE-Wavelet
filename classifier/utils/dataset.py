import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import ImageFile
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append("..")


class Deepfake_Dataset(Dataset):
    def __init__(self, data_dict, train=True, cache_in_memory=False, num_cache_workers=8):
        """
        Dataset for deepfake detection
        
        Args:
            data_dict: List of dicts with 'path' and 'label' keys
            train: Whether this is training mode
            cache_in_memory: If True, preload and cache all images in RAM
            num_cache_workers: Number of threads for parallel caching
        """
        if train:
            self.photo_path = [dicti["path"] for dicti in data_dict]
        else:
            self.photo_path = [dicti["path"] for dicti in data_dict]

        self.photo_label = [dicti["label"] for dicti in data_dict]
        self.cache_in_memory = cache_in_memory
        self.num_cache_workers = num_cache_workers
        self.image_cache = {}  # Dictionary to store cached images

        self.transform_1 = T.ToTensor()
        self.transform_2 = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        # Preload images into RAM if caching is enabled
        if self.cache_in_memory:
            print(f"\n{'='*80}")
            print("Preloading images into RAM for faster testing...")
            print(f"Using {self.num_cache_workers} threads for parallel loading")
            print(f"{'='*80}")
            self._preload_images_parallel()

    def _load_single_image(self, idx):
        """Load and preprocess a single image"""
        img_path = self.photo_path[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return idx, img

    def _preload_images_parallel(self):
        """Preload all images into RAM using parallel threads"""
        print(f"Loading {len(self.photo_path)} images into memory...")
        
        # Use ThreadPoolExecutor for parallel I/O
        with ThreadPoolExecutor(max_workers=self.num_cache_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self._load_single_image, idx): idx 
                      for idx in range(len(self.photo_path))}
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching images"):
                idx, img = future.result()
                self.image_cache[idx] = img
        
        print(f"✓ Successfully cached {len(self.image_cache)} images in RAM")
        
        # Calculate memory usage
        mem_mb = len(self.image_cache) * 224 * 224 * 3 / (1024**2)
        print(f"✓ Estimated memory usage: {mem_mb:.1f} MB")
        print(f"{'='*80}\n")

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        label = self.photo_label[item]
        
        # Get image from cache or load from disk
        if self.cache_in_memory and item in self.image_cache:
            img = self.image_cache[item]
        else:
            img_path = self.photo_path[item]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        unnormed = self.transform_1(img)
        normed = self.transform_2(img)

        return normed, unnormed, label

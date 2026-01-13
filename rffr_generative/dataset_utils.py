"""
Dataset utilities for RFFR-MVAE

This module provides helper functions for:
- Dataset path validation
- Dataset discovery
- Dataset configuration helpers
"""

import os
import json
from pathlib import Path


def validate_dataset_path(path):
    """
    Validate that a dataset JSON file exists and is readable.
    
    Args:
        path: Path to dataset JSON file
    
    Returns:
        tuple: (is_valid, error_message, num_samples)
    
    Example:
        >>> is_valid, error, count = validate_dataset_path("../data_label/ff_270/train/real_train_label.json")
        >>> if is_valid:
        >>>     print(f"Dataset has {count} samples")
    """
    if not os.path.exists(path):
        return False, f"Dataset file not found: {path}", 0
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return False, f"Dataset JSON must be a list, got {type(data)}", 0
        
        if len(data) == 0:
            return False, "Dataset is empty", 0
        
        # Check first item has expected structure
        if len(data) > 0:
            first_item = data[0]
            if not isinstance(first_item, dict):
                return False, f"Dataset items must be dicts, got {type(first_item)}", 0
            if 'path' not in first_item:
                return False, "Dataset items must have 'path' field", 0
        
        return True, None, len(data)
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", 0
    except Exception as e:
        return False, f"Error reading dataset: {e}", 0


def discover_available_datasets(base_path="../data_label"):
    """
    Discover all available dataset JSON files in the data_label directory.
    
    Args:
        base_path: Base directory containing dataset labels
    
    Returns:
        dict: Nested dictionary organized by dataset/split/type
    
    Example:
        >>> datasets = discover_available_datasets()
        >>> print(datasets['ff_270']['train'].keys())
        dict_keys(['real', 'df', 'f2f', 'fs', 'fsw', 'nt'])
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        return {}
    
    datasets = {}
    
    # Scan for JSON files
    for json_file in base_path.rglob("*.json"):
        # Skip files in nested data_label directories (duplicates)
        if "data_label/data_label" in str(json_file):
            continue
        
        # Parse path structure: data_label/{dataset}/{split}/{type}_{split}_label.json
        try:
            relative_path = json_file.relative_to(base_path)
            parts = relative_path.parts
            
            if len(parts) >= 3:
                dataset = parts[0] if len(parts) == 3 else "/".join(parts[:-2])
                split = parts[-2]
                filename = parts[-1]
                
                # Extract type from filename (e.g., "real_train_label.json" -> "real")
                if filename.endswith("_label.json"):
                    type_and_split = filename[:-11]  # Remove "_label.json"
                    # Remove split suffix (e.g., "real_train" -> "real")
                    if f"_{split}" in type_and_split:
                        data_type = type_and_split.replace(f"_{split}", "")
                    else:
                        data_type = type_and_split
                    
                    # Validate the dataset
                    is_valid, error, count = validate_dataset_path(str(json_file))
                    
                    if is_valid:
                        if dataset not in datasets:
                            datasets[dataset] = {}
                        if split not in datasets[dataset]:
                            datasets[dataset][split] = {}
                        
                        datasets[dataset][split][data_type] = {
                            'path': str(json_file),
                            'count': count
                        }
        except Exception as e:
            # Skip files that don't match expected structure
            continue
    
    return datasets


def print_available_datasets(base_path="../data_label"):
    """
    Print a formatted list of all available datasets.
    
    Args:
        base_path: Base directory containing dataset labels
    """
    datasets = discover_available_datasets(base_path)
    
    if not datasets:
        print(f"No datasets found in {base_path}")
        return
    
    print("=" * 80)
    print("AVAILABLE DATASETS")
    print("=" * 80)
    
    for dataset_name in sorted(datasets.keys()):
        print(f"\n📁 {dataset_name}")
        
        for split_name in sorted(datasets[dataset_name].keys()):
            print(f"  ├── {split_name}/")
            
            types = datasets[dataset_name][split_name]
            for i, (type_name, info) in enumerate(sorted(types.items())):
                is_last = (i == len(types) - 1)
                prefix = "  │   └──" if is_last else "  │   ├──"
                count_str = f"({info['count']:,} samples)"
                print(f"{prefix} {type_name} {count_str}")
    
    print("\n" + "=" * 80)


def get_dataset_info(dataset_name, split, data_type, base_path="../data_label"):
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., "ff_270")
        split: Split name (e.g., "train")
        data_type: Type of data (e.g., "real")
        base_path: Base directory containing dataset labels
    
    Returns:
        dict: Dataset info with 'path' and 'count', or None if not found
    """
    datasets = discover_available_datasets(base_path)
    
    try:
        return datasets[dataset_name][split][data_type]
    except KeyError:
        return None


if __name__ == "__main__":
    # Run discovery when executed directly
    print_available_datasets()

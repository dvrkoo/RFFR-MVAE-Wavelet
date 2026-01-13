#!/usr/bin/env python3
"""Quick verification that DFDC dataset is properly configured"""

import json
import sys
sys.path.insert(0, '.')

from configs.config import config

print("="*80)
print("DFDC Dataset Verification")
print("="*80)

# Check config
print("\n1. Checking config.py settings:")
print(f"   dataset_base: {config.dataset_base}")
print(f"   test_label_path length: {len(config.test_label_path)}")
print(f"   Last test path (should be DFDC): {config.test_label_path[-1]}")

try:
    print(f"   dfdc_real_test_label_path: {config.dfdc_real_test_label_path}")
except AttributeError:
    print("   WARNING: dfdc_real_test_label_path not found in config!")

# Check files exist
print("\n2. Checking label files:")
dfdc_fake_path = config.dataset_base + "DFDC/all_test_label.json"
dfdc_real_path = config.dataset_base + "DFDC/real_test_label.json"

try:
    with open(dfdc_fake_path) as f:
        fake_data = json.load(f)
    print(f"   ✓ Fake labels found: {len(fake_data)} samples")
    print(f"     First sample: {fake_data[0]['path'][:80]}...")
    print(f"     Label: {fake_data[0]['label']}")
except Exception as e:
    print(f"   ✗ Error loading fake labels: {e}")

try:
    with open(dfdc_real_path) as f:
        real_data = json.load(f)
    print(f"   ✓ Real labels found: {len(real_data)} samples")
    print(f"     First sample: {real_data[0]['path'][:80]}...")
    print(f"     Label: {real_data[0]['label']}")
except Exception as e:
    print(f"   ✗ Error loading real labels: {e}")

# Check dataset_names in test_model.py
print("\n3. Checking test_model.py integration:")
try:
    from test_model import TestDataManager
    tdm = TestDataManager()
    print(f"   dataset_names: {tdm.dataset_names}")
    if "DFDC" in tdm.dataset_names:
        print("   ✓ DFDC is in dataset_names list")
    else:
        print("   ✗ DFDC NOT in dataset_names list!")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*80)
print("Verification complete!")
print("="*80)

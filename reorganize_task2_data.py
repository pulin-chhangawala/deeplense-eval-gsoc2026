#!/usr/bin/env python3
"""Reorganize task2 dataset into train/val/test structure."""
import os
import shutil
from pathlib import Path
import random

def reorganize_task2():
    base_dir = Path("task2_data")
    
    # Move test data
    print("Organizing test split...")
    test_lenses = list((base_dir / "test_lenses").glob("*.npy"))
    test_nonlenses = list((base_dir / "test_nonlenses").glob("*.npy"))
    
    for f in test_lenses:
        shutil.move(str(f), str(base_dir / "test" / "lens" / f.name))
    for f in test_nonlenses:
        shutil.move(str(f), str(base_dir / "test" / "no_lens" / f.name))
    
    print(f"✓ Test: {len(test_lenses)} lenses, {len(test_nonlenses)} non-lenses")
    
    # Split train/val from training data (80/20)
    print("Splitting train/val from training data...")
    train_lenses = list((base_dir / "train_lenses").glob("*.npy"))
    train_nonlenses = list((base_dir / "train_nonlenses").glob("*.npy"))
    
    random.seed(42)  # For reproducibility
    random.shuffle(train_lenses)
    random.shuffle(train_nonlenses)
    
    # 80/20 split
    split_lens = int(0.8 * len(train_lenses))
    split_nonlens = int(0.8 * len(train_nonlenses))
    
    train_lenses_split = train_lenses[:split_lens]
    val_lenses_split = train_lenses[split_lens:]
    train_nonlenses_split = train_nonlenses[:split_nonlens]
    val_nonlenses_split = train_nonlenses[split_nonlens:]
    
    # Move files
    for f in train_lenses_split:
        shutil.move(str(f), str(base_dir / "train" / "lens" / f.name))
    for f in val_lenses_split:
        shutil.move(str(f), str(base_dir / "val" / "lens" / f.name))
    for f in train_nonlenses_split:
        shutil.move(str(f), str(base_dir / "train" / "no_lens" / f.name))
    for f in val_nonlenses_split:
        shutil.move(str(f), str(base_dir / "val" / "no_lens" / f.name))
    
    print(f"✓ Train: {len(train_lenses_split)} lenses, {len(train_nonlenses_split)} non-lenses")
    print(f"✓ Val:   {len(val_lenses_split)} lenses, {len(val_nonlenses_split)} non-lenses")
    
    # Clean up old directories
    print("Cleaning up old directories...")
    shutil.rmtree(base_dir / "train_lenses", ignore_errors=True)
    shutil.rmtree(base_dir / "train_nonlenses", ignore_errors=True)
    shutil.rmtree(base_dir / "test_lenses", ignore_errors=True)
    shutil.rmtree(base_dir / "test_nonlenses", ignore_errors=True)
    shutil.rmtree(base_dir / "__MACOSX", ignore_errors=True)
    
    print("\n=== VERIFICATION ===")
    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        lens_count = len(list((split_dir / "lens").glob("*.npy")))
        no_lens_count = len(list((split_dir / "no_lens").glob("*.npy")))
        total = lens_count + no_lens_count
        ratio = no_lens_count / max(1, lens_count)
        print(f"{split}: {no_lens_count} no-lens | {lens_count} lens (ratio: {ratio:.1f}:1) | total: {total}")

if __name__ == "__main__":
    reorganize_task2()

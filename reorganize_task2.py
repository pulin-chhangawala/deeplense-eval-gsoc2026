"""Reorganize Task 2 dataset structure."""
from pathlib import Path
import shutil

def reorganize_task2():
    task2_dir = Path("task2_data")
    
    # Create expected directory structure
    for split in ["train", "val", "test"]:
        for cls_name in ["lens", "no_lens"]:
            cls_dir = task2_dir / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files to correct locations
    # Train split
    if (task2_dir / "train_lenses").exists():
        for f in (task2_dir / "train_lenses").glob("*"):
            if f.is_file():
                shutil.move(str(f), str(task2_dir / "train" / "lens" / f.name))
        (task2_dir / "train_lenses").rmdir()
    
    if (task2_dir / "train_nonlenses").exists():
        for f in (task2_dir / "train_nonlenses").glob("*"):
            if f.is_file():
                shutil.move(str(f), str(task2_dir / "train" / "no_lens" / f.name))
        (task2_dir / "train_nonlenses").rmdir()
    
    # Test split
    if (task2_dir / "test_lenses").exists():
        for f in (task2_dir / "test_lenses").glob("*"):
            if f.is_file():
                shutil.move(str(f), str(task2_dir / "test" / "lens" / f.name))
        (task2_dir / "test_lenses").rmdir()
    
    if (task2_dir / "test_nonlenses").exists():
        for f in (task2_dir / "test_nonlenses").glob("*"):
            if f.is_file():
                shutil.move(str(f), str(task2_dir / "test" / "no_lens" / f.name))
        (task2_dir / "test_nonlenses").rmdir()
    
    # Create val split from train (80/20 split)
    import os
    all_lens = list((task2_dir / "train" / "lens").glob("*"))
    all_nolens = list((task2_dir / "train" / "no_lens").glob("*"))
    
    split_idx_lens = int(len(all_lens) * 0.2)
    split_idx_nolens = int(len(all_nolens) * 0.2)
    
    for f in all_lens[:split_idx_lens]:
        shutil.move(str(f), str(task2_dir / "val" / "lens" / f.name))
    
    for f in all_nolens[:split_idx_nolens]:
        shutil.move(str(f), str(task2_dir / "val" / "no_lens" / f.name))
    
    # Clean up macOS files
    if (task2_dir / "__MACOSX").exists():
        shutil.rmtree(task2_dir / "__MACOSX")
    
    print("✓ Task 2 data reorganized successfully!")
    print(f"  Train: {len(list((task2_dir / 'train' / 'lens').glob('*')))} lens, "
          f"{len(list((task2_dir / 'train' / 'no_lens').glob('*')))} no-lens")
    print(f"  Val:   {len(list((task2_dir / 'val' / 'lens').glob('*')))} lens, "
          f"{len(list((task2_dir / 'val' / 'no_lens').glob('*')))} no-lens")
    print(f"  Test:  {len(list((task2_dir / 'test' / 'lens').glob('*')))} lens, "
          f"{len(list((task2_dir / 'test' / 'no_lens').glob('*')))} no-lens")

if __name__ == "__main__":
    reorganize_task2()

"""Create train/val/test split files for SUIM dataset."""
import os
from pathlib import Path
import random

def create_splits(raw_dir="raw_suim", images_dir="data/images", 
                  output_dir="data", train_ratio=0.8, seed=42):
    """
    Create train.txt, val.txt, test.txt using original SUIM structure:
    - train_val/ folder -> split into train (80%) and val (20%)
    - TEST/ folder -> test set (do not alter)
    """
    raw_dir = Path(raw_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Get IDs from original folders
    train_val_ids = sorted([p.stem for p in (raw_dir / "train_val" / "images").glob("*.jpg")])
    test_ids = sorted([p.stem for p in (raw_dir / "TEST" / "images").glob("*.jpg")])
    
    print(f"Found {len(train_val_ids) + len(test_ids)} images total")
    print(f"  - {len(train_val_ids)} from train_val/ folder")
    print(f"  - {len(test_ids)} from TEST/ folder")
    
    # Split train_val into train and val (80/20)
    random.seed(seed)
    random.shuffle(train_val_ids)
    
    split_idx = int(len(train_val_ids) * train_ratio)
    train_ids = train_val_ids[:split_idx]
    val_ids = train_val_ids[split_idx:]
    
    # Write split files
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    test_file = output_dir / "test.txt"
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_ids))
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_ids))
    
    print(f"\n✓ Split files created:")
    print(f"  - train.txt: {len(train_ids)} images ({train_ratio*100:.0f}%)")
    print(f"  - val.txt:   {len(val_ids)} images ({(1-train_ratio)*100:.0f}%)")
    print(f"  - test.txt:  {len(test_ids)} images")
    print(f"\nFiles saved to: {output_dir}/")

if __name__ == "__main__":
    create_splits()

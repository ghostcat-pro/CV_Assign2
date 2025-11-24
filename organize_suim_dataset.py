import os
import shutil
from pathlib import Path

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def copy_pairs(src_images, src_masks, dst_images, dst_masks):
    """Copy image/mask pairs ensuring matching basenames."""
    src_images = Path(src_images)
    src_masks = Path(src_masks)

    count = 0
    for img in sorted(src_images.glob("*")):
        stem = img.stem
        mask = src_masks / f"{stem}.bmp"
        if not mask.exists():
            print(f"[WARN] Mask missing for {stem}, skipping...")
            continue

        shutil.copy(img, dst_images / img.name)
        shutil.copy(mask, dst_masks / mask.name)
        count += 1
    print(f"Copied {count} image/mask pairs → {dst_images.parent}")

def organize_suim_dataset(raw_dir="raw_suim", out_dir="data"):
    """
    raw_dir: folder where you extracted the official SUIM ZIP
    out_dir: folder where you want the organized dataset to live (will create images/ and masks/ subdirs)
    """

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    # Official structure
    train_val_images = raw_dir / "train_val" / "images"
    train_val_masks  = raw_dir / "train_val" / "masks"
    test_images      = raw_dir / "TEST" / "images"
    test_masks       = raw_dir / "TEST" / "masks"     # combined RGB
    test_binary_root = raw_dir / "TEST"               # may contain folders

    if not train_val_images.exists() or not test_images.exists():
        raise RuntimeError("❌ The extracted SUIM dataset does not match expected structure.\n"
                           "Ensure you downloaded from https://irvlab.cs.umn.edu/resources/suim-dataset")

    # Create output structure - flat images/ and masks/ folders
    images_out = ensure_dir(out_dir / "images")
    masks_out  = ensure_dir(out_dir / "masks")

    # Organize TRAIN_VAL
    print("Processing train_val/ ...")
    copy_pairs(train_val_images, train_val_masks, images_out, masks_out)

    # Organize TEST
    print("Processing TEST/ ...")
    copy_pairs(test_images, test_masks, images_out, masks_out)

    print("\n=== DONE ===")
    print(f"Dataset organized under: {out_dir}/")
    print(f"  - {len(list(images_out.glob('*')))} images in images/")
    print(f"  - {len(list(masks_out.glob('*')))} masks in masks/")
    print("\nNext steps:")
    print("  1. Create train/val split files (e.g., train.txt, val.txt)")
    print("  2. Run training from notebook or train.py")

if __name__ == "__main__":
    organize_suim_dataset()

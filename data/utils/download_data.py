import os
from pathlib import Path

SUIM_URL = "https://irvlab.cs.umn.edu/resources/suim-dataset"

def download_suim(out_dir="data"):
    """
    SUIM is hosted on a university page with multiple archives.
    Download manually and unzip so you have:

      data/images/*.jpg
      data/masks/*.bmp

    This project will then work out-of-the-box.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("SUIM download page:", SUIM_URL)
    print("Download the dataset archive and unzip into data/images and data/masks.")

def create_splits(images_dir="data/images", out_dir="data/splits",
                  train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits and save ids to txt files."""
    import random
    os.makedirs(out_dir, exist_ok=True)
    images = sorted([p for p in Path(images_dir).glob("*.jpg")])
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    ids = [p.stem for p in images]
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train+n_val]
    test_ids = ids[n_train+n_val:]

    for name, split in [("train.txt", train_ids), ("val.txt", val_ids), ("test.txt", test_ids)]:
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("\n".join(split))

    print(f"Splits created: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

if __name__ == "__main__":
    download_suim()
    create_splits()

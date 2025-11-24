import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

# SUIM palette: 3-bit RGB colors -> 8 classes
PALETTE = {
    (0, 0, 0): 0,        # background/waterbody
    (128, 0, 0): 1,      # diver
    (0, 128, 0): 2,      # plant/flora
    (128, 128, 0): 3,    # wreck/ruins
    (0, 0, 128): 4,      # robot/instrument
    (128, 0, 128): 5,    # reef/invertebrate
    (0, 128, 128): 6,    # fish/vertebrate
    (128, 128, 128): 7,  # sea-floor/rocks
}

CLASS_NAMES = [
    "background",
    "diver",
    "plant",
    "wreck",
    "robot",
    "reef_invertebrate",
    "fish_vertebrate",
    "sea_floor_rock",
]

def mask_rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB SUIM mask to class indices."""
    h, w, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    mapped = np.zeros((uniq.shape[0],), dtype=np.uint8)
    for i, c in enumerate(uniq):
        tup = tuple(int(x) for x in c)
        mapped[i] = PALETTE.get(tup, 0)
    return mapped[inv].reshape(h, w)

class SUIMDataset(Dataset):
    def __init__(self, split_file, images_dir="data/images", masks_dir="data/masks", transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.images_dir / f"{img_id}.jpg"
        mask_path = self.masks_dir / f"{img_id}.bmp"

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path))
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask_rgb_to_class(mask)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"].long()

        return img, mask

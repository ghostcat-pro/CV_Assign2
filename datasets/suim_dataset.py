import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

# SUIM palette: 3-bit RGB colors (0 or 255) -> 8 classes
PALETTE = {
    (0, 0, 0): 0,        # background/waterbody
    (255, 0, 0): 1,      # diver (changed from 128 to 255)
    (0, 255, 0): 2,      # plant/flora
    (255, 255, 0): 3,    # wreck/ruins
    (0, 0, 255): 4,      # robot/instrument
    (255, 0, 255): 5,    # reef/invertebrate
    (0, 255, 255): 6,    # fish/vertebrate
    (255, 255, 255): 7,  # sea-floor/rocks
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

CLASS_NAMES_MERGED = [
    "background_plant_seafloor",  # merged: background(0), plant(2), sea_floor_rock(7)
    "diver",                       # originally 1
    "wreck",                       # originally 3
    "robot",                       # originally 4
    "reef_invertebrate",          # originally 5
    "fish_vertebrate",            # originally 6
]

def mask_rgb_to_class(mask_rgb: np.ndarray, merge_classes: bool = False) -> np.ndarray:
    """Convert RGB SUIM mask to class indices.
    
    Args:
        mask_rgb: RGB mask image (H, W, 3)
        merge_classes: If True, merge background(0), plant(2), and sea_floor_rock(7) 
                      into class 0, reducing total classes from 8 to 6.
    
    Returns:
        Class indices mask (H, W)
    """
    h, w, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    mapped = np.zeros((uniq.shape[0],), dtype=np.uint8)
    for i, c in enumerate(uniq):
        tup = tuple(int(x) for x in c)
        mapped[i] = PALETTE.get(tup, 0)
    
    result = mapped[inv].reshape(h, w)
    
    if merge_classes:
        # Merge background(0), plant(2), and sea_floor_rock(7) -> 0
        # Remap remaining classes: 1->1, 3->2, 4->3, 5->4, 6->5
        remapping = {0: 0, 1: 1, 2: 0, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0}
        result_merged = np.zeros_like(result)
        for old_cls, new_cls in remapping.items():
            result_merged[result == old_cls] = new_cls
        result = result_merged
    
    return result

class SUIMDataset(Dataset):
    def __init__(self, split_file, images_dir="data/images", masks_dir="data/masks", 
                 transform=None, merge_classes=False):
        """SUIM Dataset loader.
        
        Args:
            split_file: Path to train/val/test split file
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            transform: Albumentations transform pipeline
            merge_classes: If True, reduce from 8 to 6 classes by merging 
                          background, plant, and sea_floor_rock into one class
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.merge_classes = merge_classes
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
        
        # Resize mask to match image dimensions if they differ
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask = mask_rgb_to_class(mask, merge_classes=self.merge_classes)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"].long()

        return img, mask

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(384, 384),  # Match model input size (384x384 for ResNet-based models)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    # Using Affine instead of deprecated ShiftScaleRotate
    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),
    A.Affine(
        translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # shift
        scale=(0.9, 1.1),  # scale
        rotate=(-15, 15),  # rotate
        p=0.5
    ),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(384, 384),  # Match model input size (384x384 for ResNet-based models)
    A.Normalize(),
    ToTensorV2()
])

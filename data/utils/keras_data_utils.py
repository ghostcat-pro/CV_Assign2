"""
Keras data generator for SUIM-Net training
Adapted from the paper's training pipeline
"""
import os
import numpy as np
from glob import glob
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def adjustData(img, mask, n_classes=5):
    """
    Adjust image and mask data for SUIM-Net training
    Args:
        img: RGB image array (H, W, 3) in range [0, 255]
        mask: RGB mask array (H, W, 3) 
        n_classes: Number of classes (5 for original SUIM paper)
    Returns:
        img: Normalized image in range [0, 1]
        mask: Binary mask array (H, W, n_classes)
    """
    # Normalize image to [0, 1]
    img = img / 255.0
    
    # Convert RGB mask to class indices based on SUIM color mapping
    # SUIM uses specific RGB colors for each class:
    # Background (BW): (0, 0, 0)
    # Human divers (HD): (0, 0, 255) - Blue
    # Aquatic plants and sea-grass (PF): (0, 255, 0) - Green  
    # Wrecks/ruins (WR): (0, 255, 255) - Cyan
    # Robots/instruments (RO): (255, 0, 0) - Red
    # Reefs and invertebrates (RI): (255, 0, 255) - Magenta
    # Fish and vertebrates (FV): (255, 255, 0) - Yellow
    # Sea-floor and rocks (SR): (255, 255, 255) - White
    
    h, w = mask.shape[:2]
    new_mask = np.zeros((h, w, n_classes), dtype=np.float32)
    
    # SUIM paper uses 5 classes by merging some categories
    # Class 0: Background + Sea-floor + Rocks
    # Class 1: Human divers  
    # Class 2: Aquatic plants
    # Class 3: Wrecks
    # Class 4: Robots + Reefs + Fish (all other objects)
    
    if n_classes == 5:
        # Background (black) + Sea-floor (white)
        new_mask[:, :, 0] = ((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)) | \
                             ((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))
        # Human divers (blue)
        new_mask[:, :, 1] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)
        # Aquatic plants (green)
        new_mask[:, :, 2] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)
        # Wrecks (cyan)
        new_mask[:, :, 3] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)
        # Other objects: Robots (red) + Reefs (magenta) + Fish (yellow)
        new_mask[:, :, 4] = ((mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)) | \
                             ((mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)) | \
                             ((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0))
    elif n_classes == 8:
        # All 8 original SUIM classes
        # Class 0: Background (black)
        new_mask[:, :, 0] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
        # Class 1: Human divers (blue)
        new_mask[:, :, 1] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)
        # Class 2: Aquatic plants (green)
        new_mask[:, :, 2] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)
        # Class 3: Wrecks (cyan)
        new_mask[:, :, 3] = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)
        # Class 4: Robots (red)
        new_mask[:, :, 4] = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
        # Class 5: Reefs (magenta)
        new_mask[:, :, 5] = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)
        # Class 6: Sea-floor (white)
        new_mask[:, :, 6] = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)
        # Class 7: Fish (yellow)
        new_mask[:, :, 7] = (mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)
    
    return img, new_mask


def trainDataGenerator(batch_size, train_path, image_folder, mask_folder, 
                       aug_dict, image_color_mode="rgb", mask_color_mode="rgb",
                       target_size=(240, 320), n_classes=5, seed=1):
    """
    Generate batches of augmented image and mask data for training.
    
    Args:
        batch_size: Number of samples per batch
        train_path: Path to training data directory
        image_folder: Subfolder name containing images
        mask_folder: Subfolder name containing masks
        aug_dict: Dictionary of augmentation parameters for ImageDataGenerator
        image_color_mode: Color mode for images ("rgb" or "grayscale")
        mask_color_mode: Color mode for masks ("rgb" or "grayscale")
        target_size: Target size (height, width) for resizing
        n_classes: Number of output classes
        seed: Random seed for augmentation synchronization
    
    Yields:
        Tuple of (image_batch, mask_batch) as numpy arrays
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        # Handle variable batch sizes (last batch may be smaller)
        actual_batch_size = img.shape[0]
        img_batch = np.zeros((actual_batch_size, target_size[0], target_size[1], 3), dtype=np.float32)
        mask_batch = np.zeros((actual_batch_size, target_size[0], target_size[1], n_classes), dtype=np.float32)
        
        for i in range(actual_batch_size):
            img_adj, mask_adj = adjustData(img[i], mask[i], n_classes=n_classes)
            img_batch[i] = img_adj
            mask_batch[i] = mask_adj
        
        yield (img_batch, mask_batch)


def valDataGenerator(batch_size, val_path, image_folder, mask_folder,
                     image_color_mode="rgb", mask_color_mode="rgb",
                     target_size=(240, 320), n_classes=5):
    """
    Generate batches of validation data without augmentation.
    
    Args:
        batch_size: Number of samples per batch
        val_path: Path to validation data directory
        image_folder: Subfolder name containing images
        mask_folder: Subfolder name containing masks
        image_color_mode: Color mode for images ("rgb" or "grayscale")
        mask_color_mode: Color mode for masks ("rgb" or "grayscale")
        target_size: Target size (height, width) for resizing
        n_classes: Number of output classes
    
    Yields:
        Tuple of (image_batch, mask_batch) as numpy arrays
    """
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size)
    
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size)
    
    val_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in val_generator:
        # Handle variable batch sizes (last batch may be smaller)
        actual_batch_size = img.shape[0]
        img_batch = np.zeros((actual_batch_size, target_size[0], target_size[1], 3), dtype=np.float32)
        mask_batch = np.zeros((actual_batch_size, target_size[0], target_size[1], n_classes), dtype=np.float32)
        
        for i in range(actual_batch_size):
            img_adj, mask_adj = adjustData(img[i], mask[i], n_classes=n_classes)
            img_batch[i] = img_adj
            mask_batch[i] = mask_adj
        
        yield (img_batch, mask_batch)

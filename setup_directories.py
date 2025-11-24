"""
Setup script to create all necessary directories for the project.
Run this after cloning the repository to ensure all folders exist.
"""
import os
from pathlib import Path

def create_directory(path, description):
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✓ {description}: {path}")

def main():
    print("=" * 70)
    print("Creating project directory structure...")
    print("=" * 70)
    
    # Data directories
    create_directory("data/images", "Images directory")
    create_directory("data/masks", "Masks directory")
    
    # Checkpoint directory
    create_directory("checkpoints", "Checkpoints directory")
    
    # Raw data directory (for dataset extraction)
    create_directory("raw_suim", "Raw SUIM dataset directory")
    
    print("\n" + "=" * 70)
    print("✓ All directories created successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Extract SUIM dataset to raw_suim/ folder")
    print("  2. Run: python organize_suim_dataset.py")
    print("  3. Run: python create_splits.py")
    print("  4. Start training!")
    print("=" * 70)

if __name__ == "__main__":
    main()

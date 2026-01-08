# utils/split_data.py (CORRECTED VERSION)

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_data(raw_dir, processed_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split raw data into train/val/test sets (NO OVERLAP)
    
    Args:
        raw_dir: Path to raw data directory
        processed_dir: Path to processed data directory
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    # Verify ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # IMPORTANT: Clean processed directory to prevent duplicates
    if processed_path.exists():
        print(f"⚠️  Removing existing processed directory to prevent duplicates...")
        shutil.rmtree(processed_path)
    
    # Create directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for class_name in ['0', '1']:
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Starting Data Split (No Overlap Guaranteed)")
    print("="*60)
    
    # Process each class
    for class_name in ['0', '1']:
        class_path = raw_path / class_name
        
        if not class_path.exists():
            print(f"⚠️  Class folder '{class_name}' not found")
            continue
        
        # Get all images (YOUR APPROACH)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        images = set()
        for ext in image_extensions:
            images.update(class_path.glob(f'*{ext}'))

        images = list(images)  # convert back to list

        
        if len(images) == 0:
            print(f"⚠️  No images in class '{class_name}'")
            continue
        
        print(f"\nClass '{class_name}': Found {len(images)} images")
        
        # Shuffle ONCE
        random.shuffle(images)
        
        # Calculate split indices (YOUR APPROACH)
        total = len(images)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        
        # NON-OVERLAPPING SPLIT (YOUR APPROACH)
        split_data = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        print(f"  Train: {len(split_data['train'])} images")
        print(f"  Val:   {len(split_data['val'])} images")
        print(f"  Test:  {len(split_data['test'])} images")
        
        # Copy files (with progress bar)
        for split, split_images in split_data.items():
            dest_dir = processed_path / split / class_name
            
            for img_path in tqdm(split_images, desc=f"  Copying to {split}", ncols=70):
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
    
    print("\n" + "="*60)
    print("✅ Data split completed!")
    print("="*60)
    
    # Verify no overlap
    print("\n🔍 Verifying split integrity...")
    verify_split_integrity(processed_path)

def verify_split_integrity(processed_path):
    """Verify that splits don't overlap (class-aware check)"""
    
    splits = ['train', 'val', 'test']
    all_files = {}

    for split in splits:
        files = set()
        for class_name in ['0', '1']:
            split_path = processed_path / split / class_name
            if split_path.exists():
                for f in split_path.glob('*'):
                    # Include class name to avoid false overlap
                    files.add(f"{class_name}/{f.name}")
        all_files[split] = files

    # Check overlaps
    train_val = all_files['train'] & all_files['val']
    train_test = all_files['train'] & all_files['test']
    val_test = all_files['val'] & all_files['test']

    if train_val or train_test or val_test:
        print("❌ ERROR: Found overlapping files between splits!")
        if train_val:
            print(f"   Train-Val overlap: {len(train_val)} files")
        if train_test:
            print(f"   Train-Test overlap: {len(train_test)} files")
        if val_test:
            print(f"   Val-Test overlap: {len(val_test)} files")
    else:
        print("✅ Verified: No overlap between splits")
        print(f"   Total unique images: "
              f"{len(all_files['train']) + len(all_files['val']) + len(all_files['test'])}")


if __name__ == "__main__":
    split_data(
        raw_dir="data/raw",
        processed_dir="data/processed",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
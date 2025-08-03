#!/usr/bin/env python3
"""
Generate dummy label files for images that don't have corresponding label files.
This is for testing purposes only.
"""
import os
from pathlib import Path

def generate_dummy_labels():
    """Generate dummy label files for images without labels."""
    base_dir = Path("road_defects_dataset")
    
    for split in ['train', 'val', 'test']:
        img_dir = base_dir / 'images' / split
        label_dir = base_dir / 'labels' / split
        
        if not img_dir.exists() or not label_dir.exists():
            continue
            
        # Create label directory if it doesn't exist
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files (without extension)
        img_files = {f.stem for f in img_dir.glob('*.jpg')} | \
                   {f.stem for f in img_dir.glob('*.jpeg')} | \
                   {f.stem for f in img_dir.glob('*.png')}
        
        # Get all existing label files (without extension)
        existing_labels = {f.stem for f in label_dir.glob('*.txt')}
        
        # Find images without labels
        missing_labels = img_files - existing_labels
        
        print(f"Found {len(missing_labels)} images without labels in {split}")
        
        # Create dummy label files
        for img_stem in missing_labels:
            label_file = label_dir / f"{img_stem}.txt"
            
            # Create a dummy label (class 0 with a small box in the center)
            # Format: class x_center y_center width height (all normalized to [0,1])
            dummy_label = "0 0.5 0.5 0.1 0.1\n"  # Small box in the center
            
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write(dummy_label)
            
            print(f"Created dummy label: {label_file}")

if __name__ == "__main__":
    print("Generating dummy label files for testing...")
    generate_dummy_labels()
    print("Done!")

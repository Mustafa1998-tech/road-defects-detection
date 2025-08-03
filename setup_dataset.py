#!/usr/bin/env python3
"""
Setup the road defects dataset for YOLOv5 training.
This script will:
1. Create the required directory structure
2. Copy images to the correct locations
3. Generate dummy label files
4. Create train/val/test splits
"""
import os
import shutil
import random
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "road_defects_dataset"
SOURCE_IMAGES = [
    "C:/Users/ahmed/Downloads/لتدهور والتآكل - Raveling.jpg",
    "C:/Users/ahmed/Downloads/لركام المصقول (بري الركام) - POLISHED AGGREGATE.jpg",
    "C:/Users/ahmed/Downloads/لشروخ الحرارية - THERMAL CRACKS.jpg",
    "C:/Users/ahmed/Downloads/التفكيك - DELAMINATION.jpg",
    "C:/Users/ahmed/Downloads/التموجات - CORRUGATIONS.jpg",
    "C:/Users/ahmed/Downloads/الحفر - POTHOLES.jpg",
    "C:/Users/ahmed/Downloads/الشروخ الانعكاسية- REFLECTIVE CRACKING.jpg",
    "C:/Users/ahmed/Downloads/النزيف أو النضح - BLEEDING.jpg",
    "C:/Users/ahmed/Downloads/تجريف أو أخاديد - RUTTING.jpg",
    "C:/Users/ahmed/Downloads/تشققات (الكلال) التعب وتشققات التعب على الحافة - FATIGUE CRACKS AND EDGE FATIGUE CRACKS.jpg",
    "C:/Users/ahmed/Downloads/تشققات الحواف - EDGE CRACKS.jpg",
    "C:/Users/ahmed/Downloads/تشققات المفاصل الطولية - LONGITUDINAL JOINT CRACKS.jpg",
    "C:/Users/ahmed/Downloads/شروخ انزلاقية - SLIPPAGE CRACK.jpg",
    "C:/Users/ahmed/Downloads/شروخ كتلية - BLOCK CRACKING.jpg"
]

# Class mapping
CLASS_NAMES = [
    "التدهور والتآكل",
    "بري الركام",
    "الشروخ الحرارية",
    "التفكيك",
    "التموجات",
    "الحفر",
    "الشروخ الانعكاسية",
    "النزيف",
    "الأخاديد",
    "تشققات التعب",
    "تشققات الحواف",
    "تشققات المفاصل",
    "الشروخ الانزلاقية",
    "الشروخ الكتلية"
]

def create_directory_structure():
    """Create the required directory structure."""
    dirs = [
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "val",
        DATA_DIR / "images" / "test",
        DATA_DIR / "labels" / "train",
        DATA_DIR / "labels" / "val",
        DATA_DIR / "labels" / "test",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created directory structure")

def copy_images():
    """Copy source images to the dataset directory."""
    # Create a mapping of defect names to class IDs
    defect_to_class = {}
    for i, class_name in enumerate(CLASS_NAMES):
        # Find the source image that matches this class
        for src_path in SOURCE_IMAGES:
            if class_name in src_path:
                defect_to_class[src_path] = i
                break
    
    # Copy images to the training directory (we'll split them later)
    for src_path in defect_to_class.keys():
        if not os.path.exists(src_path):
            print(f"⚠️ Source image not found: {src_path}")
            continue
            
        # Create a safe filename (remove special characters)
        safe_name = os.path.basename(src_path).replace(" ", "_").replace("-", "_")
        dst_path = DATA_DIR / "images" / "train" / safe_name
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copied {src_path} to {dst_path}")
        except Exception as e:
            print(f"❌ Error copying {src_path}: {e}")
    
    return defect_to_class

def create_data_yaml():
    """Create the data.yaml configuration file."""
    yaml_content = f"""# Road Defects Dataset Configuration
train: {str(DATA_DIR / 'images/train')}
val: {str(DATA_DIR / 'images/val')}
test: {str(DATA_DIR / 'images/test')}

# Number of classes
nc: {len(CLASS_NAMES)}

# Class names
names: {CLASS_NAMES}
"""
    yaml_path = BASE_DIR / "data" / "road_defects.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Created data configuration at {yaml_path}")
    return yaml_path

def generate_dummy_labels(defect_to_class):
    """Generate dummy label files for training."""
    images_dir = DATA_DIR / "images" / "train"
    labels_dir = DATA_DIR / "labels" / "train"
    
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        # Find the class ID for this image
        class_id = None
        for src_path, cid in defect_to_class.items():
            if img_path.name.replace("_", " ") in src_path:
                class_id = cid
                break
        
        if class_id is None:
            print(f"⚠️ Could not determine class for {img_path}")
            continue
        
        # Create a dummy bounding box (center of image, 80% of width/height)
        label_content = f"{class_id} 0.5 0.5 0.8 0.8"
        
        # Write the label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_content)
        
        print(f"✅ Created label: {label_path}")

def split_dataset():
    """Split the dataset into train/val/test sets."""
    # This is a simple split - in a real scenario, you might want to ensure
    # that all classes are represented in each split
    images_dir = DATA_DIR / "images" / "train"
    labels_dir = DATA_DIR / "labels" / "train"
    
    # Get all image files
    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Split into train (70%), val (20%), test (10%)
    n = len(image_files)
    train_end = int(0.7 * n)
    val_end = int(0.9 * n)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Move files to their respective directories
    for split, files in splits.items():
        print(f"\n📊 {split.upper()} split: {len(files)} images")
        
        for img_path in files:
            # Move image
            dst_img_dir = DATA_DIR / "images" / split
            dst_img_path = dst_img_dir / img_path.name
            shutil.move(str(img_path), str(dst_img_path))
            
            # Move corresponding label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label_dir = DATA_DIR / "labels" / split
                dst_label_dir.mkdir(parents=True, exist_ok=True)
                dst_label_path = dst_label_dir / label_path.name
                shutil.move(str(label_path), str(dst_label_path))
    
    print("\n✅ Dataset split completed")

def main():
    print("🚀 Setting up road defects dataset for YOLOv5 training...\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Copy images and get class mappings
    print("\n📂 Copying images...")
    defect_to_class = copy_images()
    
    # Create data.yaml
    print("\n📄 Creating data configuration...")
    yaml_path = create_data_yaml()
    
    # Generate dummy labels
    print("\n🏷️  Generating dummy labels...")
    generate_dummy_labels(defect_to_class)
    
    # Split dataset
    print("\n✂️  Splitting dataset into train/val/test...")
    split_dataset()
    
    print("\n✨ Dataset setup completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Review the dataset in: {DATA_DIR}")
    print(f"2. Check the configuration in: {yaml_path}")
    print("3. Start training with: python train.py --img 640 --batch 16 --epochs 50 --data data/road_defects.yaml --weights yolov5s.pt --name road_defects_model")

if __name__ == "__main__":
    main()

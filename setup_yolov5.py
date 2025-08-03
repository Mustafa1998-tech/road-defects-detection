import os
import sys
import subprocess
import shutil
import torch
from pathlib import Path

def setup_yolov5():
    print("🚀 Starting YOLOv5 setup...")
    
    # Create yolov5 directory if it doesn't exist
    yolov5_dir = Path("yolov5")
    if not yolov5_dir.exists():
        print("📥 Downloading YOLOv5...")
        # Clone YOLOv5 repository
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"], check=True)
        
        # Install requirements
        print("📦 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"], check=True)
    else:
        print("✅ YOLOv5 directory already exists")
    
    # Download pre-trained model if it doesn't exist
    model_path = Path("model.pt")
    if not model_path.exists():
        print("⬇️  Downloading pre-trained model...")
        try:
            # Load YOLOv5s model (smallest version)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # Save the model
            torch.save(model.state_dict(), 'model.pt')
            print("✅ Model saved as model.pt")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("Please check your internet connection and try again.")
            return False
    else:
        print("✅ Model already exists")
    
    # Create necessary directories
    data_dir = yolov5_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create data.yaml file if it doesn't exist
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        print("📄 Creating data.yaml...")
        with open(data_yaml, "w", encoding="utf-8") as f:
            f.write("""# YOLOv5 dataset configuration
# Paths
train: ../road_defects_dataset/images/train
val: ../road_defects_dataset/images/val
test: ../road_defects_dataset/images/test

# Number of classes
nc: 14

# Class names
names: [
    'التدهور والتآكل',
    'بري الركام',
    'الشروخ الحرارية',
    'التفكيك',
    'التموجات',
    'الحفر',
    'الشروخ الانعكاسية',
    'النزيف',
    'الأخاديد',
    'تشققات التعب',
    'تشققات الحواف',
    'تشققات المفاصل',
    'الشروخ الانزلاقية',
    'الشروخ الكتلية'
]
""")
        print("✅ Created data.yaml configuration")
    else:
        print("✅ data.yaml already exists")
    
    print("\n✨ Setup completed successfully!")
    print("You can now run the application using: streamlit run road_defect_system.py")
    return True

if __name__ == "__main__":
    setup_yolov5()

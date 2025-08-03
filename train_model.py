# train_model.py
import torch
from pathlib import Path
import subprocess
import yaml

def train_yolov5():
    # Create data.yaml configuration
    data = {
        'train': 'road_defects_dataset/images/train',
        'val': 'road_defects_dataset/images/val',
        'nc': 14,  # number of classes
        'names': [
            'التدهور والتآكل', 'بري الركام', 'الشروخ الحرارية', 'التفكيك',
            'التموجات', 'الحفر', 'الشروخ الانعكاسية', 'النزيف',
            'الأخاديد', 'تشققات التعب', 'تشققات الحواف', 'تشققات المفاصل',
            'الشروخ الانزلاقية', 'الشروخ الكتلية'
        ]
    }
    
    # Save data.yaml
    with open('data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)
    
    # Train the model
    cmd = [
        'python', 'yolov5/train.py',
        '--img', '640',
        '--batch', '16',
        '--epochs', '50',
        '--data', 'data.yaml',
        '--weights', 'yolov5s.pt',
        '--name', 'road_defects_model'
    ]
    
    subprocess.run(cmd)

if __name__ == '__main__':
    train_yolov5()
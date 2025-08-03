import os
import unittest
import yaml
from pathlib import Path

class TestDataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        cls.data_dir = Path("road_defects_dataset")
        cls.data_yaml = Path("data/road_defects.yaml")
        
    def test_dataset_structure(self):
        """Test if the dataset has the correct directory structure."""
        required_dirs = ["images/train", "images/val", "images/test",
                       "labels/train", "labels/val", "labels/test"]
        
        for dir_path in required_dirs:
            full_path = self.data_dir / dir_path
            self.assertTrue(full_path.exists(), f"Directory does not exist: {full_path}")
            self.assertTrue(full_path.is_dir(), f"Path is not a directory: {full_path}")
    
    def test_data_yaml_exists(self):
        """Test if the data YAML file exists and is valid."""
        self.assertTrue(self.data_yaml.exists(), f"Data YAML file not found: {self.data_yaml}")
        
        # Test if YAML is valid
        with open(self.data_yaml, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                self.assertIn('train', data, "'train' key missing in YAML")
                self.assertIn('val', data, "'val' key missing in YAML")
                self.assertIn('nc', data, "'nc' (number of classes) key missing in YAML")
                self.assertIn('names', data, "'names' (class names) key missing in YAML")
            except yaml.YAMLError as e:
                self.fail(f"YAML file is invalid: {e}")
    
    def test_image_files_exist(self):
        """Test if there are image files in the dataset."""
        for split in ['train', 'val', 'test']:
            img_dir = self.data_dir / 'images' / split
            if img_dir.exists():
                img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                self.assertGreater(len(img_files), 0, f"No image files found in {img_dir}")
    
    def test_label_files_match_images(self):
        """Test if each image has a corresponding label file."""
        for split in ['train', 'val', 'test']:
            img_dir = self.data_dir / 'images' / split
            label_dir = self.data_dir / 'labels' / split
            
            if not img_dir.exists() or not label_dir.exists():
                continue
                
            # Get all image files (without extension)
            img_files = {f.stem for f in img_dir.glob('*.jpg')} | \
                       {f.stem for f in img_dir.glob('*.png')}
                        
            # Get all label files (without extension)
            label_files = {f.stem for f in label_dir.glob('*.txt')}
            
            # Check for images without labels
            missing_labels = img_files - label_files
            self.assertEqual(len(missing_labels), 0, 
                          f"Found {len(missing_labels)} images without corresponding labels in {split}")

if __name__ == '__main__':
    unittest.main()

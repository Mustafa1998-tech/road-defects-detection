#!/usr/bin/env python3
"""
Test runner for the road defect detection system.

This script discovers and runs all tests in the 'tests' directory.
"""
import unittest
import sys
import os

def run_tests():
    """Run all tests in the tests directory."""
    # Add the project root to the Python path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Create the data directory if it doesn't exist
    data_dir = os.path.join('yolov5', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a minimal data.yaml if it doesn't exist
    data_yaml = os.path.join(data_dir, 'road_defects.yaml')
    if not os.path.exists(data_yaml):
        with open(data_yaml, 'w', encoding='utf-8') as f:
            f.write("""# Road defects dataset
train: ../road_defects_dataset/images/train
val: ../road_defects_dataset/images/val
test: ../road_defects_dataset/images/test

# Number of classes
nc: 14

# Class names
names: [
    'pothole',
    'crack',
    'patch',
    'rutting',
    'edge_crack',
    'longitudinal_crack',
    'transverse_crack',
    'alligator_crack',
    'block_crack',
    'reflective_crack',
    'slippage_crack',
    'delamination',
    'bleeding',
    'corrugation'
]
""")
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    print("Running road defect detection system tests...\n")
    run_tests()

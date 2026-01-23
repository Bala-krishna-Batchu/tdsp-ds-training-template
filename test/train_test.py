"""
Tests for Computer Vision Model Training
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

# Add parent directory to path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_directory)

from src.train import ImageDataset, get_model, load_data_from_directory


class TestImageDataset(unittest.TestCase):
    """Tests for ImageDataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_paths = []
        for i in range(5):
            img_path = os.path.join(self.temp_dir, f"image_{i}.jpg")
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(img_path)
            self.image_paths.append(img_path)
        self.labels = [0, 0, 1, 1, 0]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        dataset = ImageDataset(self.image_paths, self.labels)
        self.assertEqual(len(dataset), 5)

    def test_dataset_getitem(self):
        """Test dataset returns correct items."""
        dataset = ImageDataset(self.image_paths, self.labels)
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape[0], 3)  # 3 channels
        self.assertEqual(label, 0)


class TestGetModel(unittest.TestCase):
    """Tests for get_model function."""

    def test_resnet18(self):
        """Test resnet18 model creation."""
        model = get_model(num_classes=2, architecture="resnet18")
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.fc.out_features, 2)

    def test_resnet50(self):
        """Test resnet50 model creation."""
        model = get_model(num_classes=5, architecture="resnet50")
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.fc.out_features, 5)


class TestLoadDataFromDirectory(unittest.TestCase):
    """Tests for load_data_from_directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        for class_name in ["no_rust", "rust"]:
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir)
            for i in range(3):
                img = Image.new("RGB", (50, 50))
                img.save(os.path.join(class_dir, f"img_{i}.jpg"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_data(self):
        """Test loading data from directory structure."""
        images, labels, class_names = load_data_from_directory(self.temp_dir)
        self.assertEqual(len(images), 6)
        self.assertEqual(len(labels), 6)
        self.assertEqual(class_names, ["no_rust", "rust"])


if __name__ == "__main__":
    unittest.main()

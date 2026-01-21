"""
Tests for Computer Vision Model Training

Unit tests for CV model training, data loading, and Clarify artifact generation.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add parent directory to path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_directory)

from src.train_cv import (
    CVModelTrainer,
    ImageClassificationDataset,
    DEFAULT_CONFIG,
)
from src.utilities.cv_data_utils import (
    CVDataLoader,
    ClarifyDataFormatter,
    ImagePreprocessor,
)
from src.utilities.clarify_integration import (
    ClarifyArtifactGenerator,
    ClarifyBiasAnalyzer,
    BiasMonitoringAutomation,
)


class TestImageClassificationDataset(unittest.TestCase):
    """Tests for ImageClassificationDataset."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock image files
        self.image_paths = []
        for i in range(5):
            img_path = os.path.join(self.temp_dir, f"image_{i}.jpg")
            # Create a small test image
            from PIL import Image
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(img_path)
            self.image_paths.append(img_path)
        
        self.labels = [0, 0, 1, 1, 0]
        self.metadata = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M"],
            "age_group": ["young", "old", "young", "old", "young"],
        })

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_length(self):
        """Test that dataset returns correct length."""
        dataset = ImageClassificationDataset(
            self.image_paths, self.labels, self.metadata
        )
        self.assertEqual(len(dataset), 5)

    def test_dataset_getitem(self):
        """Test that dataset returns correct items."""
        dataset = ImageClassificationDataset(
            self.image_paths, self.labels, self.metadata
        )
        
        image, label, meta = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape[0], 3)  # 3 channels
        self.assertEqual(label, 0)
        self.assertIsInstance(meta, dict)

    def test_dataset_without_metadata(self):
        """Test dataset without metadata."""
        dataset = ImageClassificationDataset(
            self.image_paths, self.labels, metadata=None
        )
        
        image, label, meta = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(meta, {})


class TestCVModelTrainer(unittest.TestCase):
    """Tests for CVModelTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "Model": {
                "ProjectName": "Test CV Model",
                "ModelName": "test-cv-model",
            },
            "TeamDetails": {
                "PodName": "test-pod",
            },
        }
        self.trainer = CVModelTrainer(self.config, "dev")

    def test_get_model_resnet18(self):
        """Test model creation with resnet18."""
        model = self.trainer._get_model(num_classes=2, architecture="resnet18")
        
        self.assertIsInstance(model, nn.Module)
        # Check final layer output size
        self.assertEqual(model.fc.out_features, 2)

    def test_get_model_efficientnet(self):
        """Test model creation with efficientnet_b0."""
        model = self.trainer._get_model(num_classes=5, architecture="efficientnet_b0")
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.classifier[1].out_features, 5)

    def test_get_model_invalid_architecture(self):
        """Test fallback for invalid architecture."""
        model = self.trainer._get_model(num_classes=2, architecture="invalid_arch")
        
        # Should fall back to resnet18
        self.assertIsInstance(model, nn.Module)

    def test_get_transforms_training(self):
        """Test training transforms include augmentation."""
        transform = self.trainer._get_transforms(224, is_training=True)
        
        # Training transforms should have augmentation
        self.assertIsNotNone(transform)

    def test_get_transforms_evaluation(self):
        """Test evaluation transforms are simpler."""
        transform = self.trainer._get_transforms(224, is_training=False)
        
        self.assertIsNotNone(transform)


class TestClarifyArtifactGenerator(unittest.TestCase):
    """Tests for ClarifyArtifactGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ClarifyArtifactGenerator(output_dir=self.temp_dir)
        
        # Test data
        self.predictions = [0, 1, 1, 0, 1]
        self.ground_truth = [0, 1, 0, 0, 1]
        self.probabilities = [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.85, 0.15],
            [0.1, 0.9],
        ]
        self.class_names = ["negative", "positive"]

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_predictions_file(self):
        """Test predictions file creation."""
        output_path = self.generator.create_predictions_file(
            self.predictions,
            self.ground_truth,
            self.probabilities,
            self.class_names,
        )
        
        self.assertTrue(os.path.exists(output_path))
        
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 5)
        self.assertIn("prediction", df.columns)
        self.assertIn("ground_truth", df.columns)
        self.assertIn("confidence", df.columns)

    def test_create_metadata_file(self):
        """Test metadata file creation."""
        metadata = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M"],
            "age_group": ["young", "old", "young", "old", "young"],
        })
        
        output_path = self.generator.create_metadata_file(
            metadata, self.predictions, self.ground_truth
        )
        
        self.assertTrue(os.path.exists(output_path))
        
        df = pd.read_csv(output_path)
        self.assertIn("gender", df.columns)
        self.assertIn("prediction", df.columns)
        self.assertIn("ground_truth", df.columns)

    def test_create_clarify_config(self):
        """Test Clarify config creation."""
        sensitive_attrs = ["gender", "age_group"]
        output_path = self.generator.create_clarify_config(sensitive_attrs)
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, "r") as f:
            config = json.load(f)
        
        self.assertEqual(config["facet_columns"], sensitive_attrs)
        self.assertIn("post_training_bias", config["methods"])

    def test_create_metrics_file(self):
        """Test metrics file creation."""
        output_path = self.generator.create_metrics_file(
            self.predictions, self.ground_truth, self.class_names
        )
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, "r") as f:
            metrics = json.load(f)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_generate_all_artifacts(self):
        """Test generation of all artifacts."""
        metadata = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M"],
        })
        
        artifacts = self.generator.generate_all_artifacts(
            self.predictions,
            self.ground_truth,
            self.probabilities,
            self.class_names,
            metadata=metadata,
        )
        
        self.assertIn("predictions", artifacts)
        self.assertIn("metadata", artifacts)
        self.assertIn("model_info", artifacts)
        self.assertIn("clarify_config", artifacts)
        self.assertIn("metrics", artifacts)


class TestClarifyBiasAnalyzer(unittest.TestCase):
    """Tests for ClarifyBiasAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ClarifyBiasAnalyzer()

    def test_check_bias_threshold_pass(self):
        """Test threshold check when metrics pass."""
        metrics = {
            "DPPL": 0.05,
            "DI": 0.08,
            "AD": 0.03,
        }
        
        results = self.analyzer.check_bias_threshold(metrics, threshold=0.1)
        
        self.assertTrue(results["pass"])
        self.assertEqual(len(results["violations"]), 0)

    def test_check_bias_threshold_fail(self):
        """Test threshold check when metrics fail."""
        metrics = {
            "DPPL": 0.15,  # Exceeds threshold
            "DI": 0.08,
            "AD": 0.12,   # Exceeds threshold
        }
        
        results = self.analyzer.check_bias_threshold(metrics, threshold=0.1)
        
        self.assertFalse(results["pass"])
        self.assertEqual(len(results["violations"]), 2)


class TestBiasMonitoringAutomation(unittest.TestCase):
    """Tests for BiasMonitoringAutomation."""

    def setUp(self):
        """Set up test fixtures."""
        self.automation = BiasMonitoringAutomation()

    def test_track_bias_drift_no_drift(self):
        """Test drift tracking with no significant drift."""
        current = {"DPPL": 0.05, "DI": 0.08}
        baseline = {"DPPL": 0.04, "DI": 0.07}
        
        results = self.automation.track_bias_drift(
            current, baseline, drift_threshold=0.05
        )
        
        self.assertFalse(results["drift_detected"])

    def test_track_bias_drift_with_drift(self):
        """Test drift tracking with significant drift."""
        current = {"DPPL": 0.15, "DI": 0.08}
        baseline = {"DPPL": 0.05, "DI": 0.07}
        
        results = self.automation.track_bias_drift(
            current, baseline, drift_threshold=0.05
        )
        
        self.assertTrue(results["drift_detected"])
        self.assertTrue(results["metrics"]["DPPL"]["significant"])

    def test_generate_automation_config(self):
        """Test automation config generation."""
        config = self.automation.generate_automation_config(
            model_name="test-model",
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
            sensitive_attributes=["gender", "age"],
            schedule="rate(1 day)",
            alert_emails=["test@example.com"],
        )
        
        self.assertEqual(config["model_name"], "test-model")
        self.assertIn("data_config", config)
        self.assertIn("analysis_config", config)
        self.assertIn("schedule_config", config)
        self.assertIn("alert_config", config)


class TestCVDataLoader(unittest.TestCase):
    """Tests for CVDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = CVDataLoader()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_from_directory(self):
        """Test loading from directory structure."""
        # Create class directories
        for class_name in ["class_0", "class_1"]:
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir)
            
            # Create dummy images
            from PIL import Image
            for i in range(3):
                img = Image.new("RGB", (50, 50))
                img.save(os.path.join(class_dir, f"img_{i}.jpg"))
        
        images, labels, metadata = self.loader.load_dataset_from_local(self.temp_dir)
        
        self.assertEqual(len(images), 6)
        self.assertEqual(len(labels), 6)
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_load_from_manifest(self):
        """Test loading from manifest file."""
        # Create images
        from PIL import Image
        image_paths = []
        for i in range(5):
            img_path = os.path.join(self.temp_dir, f"img_{i}.jpg")
            img = Image.new("RGB", (50, 50))
            img.save(img_path)
            image_paths.append(img_path)
        
        # Create manifest
        manifest_df = pd.DataFrame({
            "image_path": image_paths,
            "label": [0, 1, 0, 1, 0],
            "gender": ["M", "F", "M", "F", "M"],
        })
        manifest_path = os.path.join(self.temp_dir, "manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)
        
        images, labels, metadata = self.loader.load_dataset_from_local(self.temp_dir)
        
        self.assertEqual(len(images), 5)
        self.assertEqual(len(labels), 5)
        self.assertIsNotNone(metadata)
        self.assertIn("gender", metadata.columns)


class TestImagePreprocessor(unittest.TestCase):
    """Tests for ImagePreprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_image_valid(self):
        """Test validation of valid image."""
        from PIL import Image
        img_path = os.path.join(self.temp_dir, "valid.jpg")
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        
        self.assertTrue(ImagePreprocessor.validate_image(img_path))

    def test_validate_image_invalid(self):
        """Test validation of invalid image."""
        invalid_path = os.path.join(self.temp_dir, "invalid.jpg")
        with open(invalid_path, "w") as f:
            f.write("not an image")
        
        self.assertFalse(ImagePreprocessor.validate_image(invalid_path))

    def test_get_image_stats(self):
        """Test getting image statistics."""
        from PIL import Image
        
        image_paths = []
        for i, size in enumerate([(100, 100), (200, 150), (50, 75)]):
            img_path = os.path.join(self.temp_dir, f"img_{i}.jpg")
            img = Image.new("RGB", size)
            img.save(img_path)
            image_paths.append(img_path)
        
        stats = ImagePreprocessor.get_image_stats(image_paths)
        
        self.assertEqual(stats["count"], 3)
        self.assertIn("avg_width", stats)
        self.assertIn("avg_height", stats)


if __name__ == "__main__":
    unittest.main()

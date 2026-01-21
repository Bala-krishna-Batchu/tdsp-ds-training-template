"""
Computer Vision Model Training Script

This script trains a computer vision model and saves artifacts for Clarify bias monitoring.
It replaces the churn prediction model while maintaining the existing structure.

Key features:
- Transfer learning with pre-trained models (ResNet, EfficientNet)
- S3 artifact storage for model, metrics, and predictions
- Clarify-compatible output format for bias monitoring
- Support for multi-class image classification
"""

import ast
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "image_size": 224,
    "num_classes": 2,
    "model_architecture": "resnet18",
    "pretrained": True,
    "test_size": 0.2,
    "random_state": 42,
}


class ImageClassificationDataset(Dataset):
    """Custom dataset for image classification with metadata support."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        metadata: Optional[pd.DataFrame] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of integer labels
            metadata: DataFrame with sensitive attributes for bias analysis
            transform: Image transformations to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.metadata = metadata
        self.transform = transform or self._default_transform()

    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and transform image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a blank image on error
            image = torch.zeros(3, 224, 224)

        # Get metadata if available
        meta = {}
        if self.metadata is not None:
            meta = self.metadata.iloc[idx].to_dict()

        return image, label, meta


class CVModelTrainer:
    """Computer Vision Model Trainer with Clarify-compatible outputs."""

    def __init__(self, config: Dict[str, Any], env: str):
        self.config = config
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.class_names = None
        
        # S3 client for artifact storage
        self.s3_client = boto3.client("s3")
        
        logger.info(f"Using device: {self.device}")

    def _get_model(self, num_classes: int, architecture: str = "resnet18", pretrained: bool = True) -> nn.Module:
        """Get a pre-trained model for transfer learning."""
        
        model_factory = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "efficientnet_b0": models.efficientnet_b0,
            "mobilenet_v2": models.mobilenet_v2,
        }
        
        if architecture not in model_factory:
            logger.warning(f"Unknown architecture {architecture}, defaulting to resnet18")
            architecture = "resnet18"
        
        # Load pre-trained model
        weights = "IMAGENET1K_V1" if pretrained else None
        model = model_factory[architecture](weights=weights)
        
        # Modify final layer for our classification task
        if "resnet" in architecture:
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        elif "efficientnet" in architecture:
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        elif "mobilenet" in architecture:
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        
        return model.to(self.device)

    def _get_transforms(self, image_size: int, is_training: bool = True) -> transforms.Compose:
        """Get image transformations."""
        
        if is_training:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def load_data(
        self,
        train_data_path: str,
        test_data_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """Load training and test data.
        
        Expected data format:
        - Images organized in class folders: train_data_path/class_name/image.jpg
        - Or a manifest file (CSV/JSON) with image paths and labels
        """
        
        hyperparams = self._get_hyperparameters()
        batch_size = int(hyperparams.get("batch_size", DEFAULT_CONFIG["batch_size"]))
        image_size = int(hyperparams.get("image_size", DEFAULT_CONFIG["image_size"]))
        
        # Load metadata for bias analysis if provided
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with columns: {metadata.columns.tolist()}")
        
        # Check for manifest file first
        manifest_path = os.path.join(train_data_path, "manifest.csv")
        if os.path.exists(manifest_path):
            train_images, train_labels, train_meta = self._load_from_manifest(
                manifest_path, metadata
            )
        else:
            train_images, train_labels, train_meta = self._load_from_directory(
                train_data_path, metadata
            )
        
        # Split or load test data
        if test_data_path and os.path.exists(test_data_path):
            test_manifest = os.path.join(test_data_path, "manifest.csv")
            if os.path.exists(test_manifest):
                test_images, test_labels, test_meta = self._load_from_manifest(
                    test_manifest, metadata
                )
            else:
                test_images, test_labels, test_meta = self._load_from_directory(
                    test_data_path, metadata
                )
        else:
            # Split training data
            from sklearn.model_selection import train_test_split
            test_size = float(hyperparams.get("test_size", DEFAULT_CONFIG["test_size"]))
            random_state = int(hyperparams.get("random_state", DEFAULT_CONFIG["random_state"]))
            
            train_images, test_images, train_labels, test_labels = train_test_split(
                train_images, train_labels, test_size=test_size, random_state=random_state, stratify=train_labels
            )
            train_meta, test_meta = None, None
            if train_meta is not None:
                train_meta, test_meta = train_test_split(
                    train_meta, test_size=test_size, random_state=random_state
                )
        
        # Create datasets
        train_transform = self._get_transforms(image_size, is_training=True)
        test_transform = self._get_transforms(image_size, is_training=False)
        
        train_dataset = ImageClassificationDataset(
            train_images, train_labels, train_meta, train_transform
        )
        test_dataset = ImageClassificationDataset(
            test_images, test_labels, test_meta, test_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        logger.info(f"Loaded {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    def _load_from_directory(
        self, data_path: str, metadata: Optional[pd.DataFrame] = None
    ) -> Tuple[List[str], List[int], Optional[pd.DataFrame]]:
        """Load images from directory structure (class folders)."""
        
        image_paths = []
        labels = []
        
        # Get class names from subdirectories
        class_dirs = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])
        
        if not class_dirs:
            # Single directory with images, need manifest
            raise ValueError(f"No class subdirectories found in {data_path}. "
                           "Please organize images in class folders or provide a manifest.csv")
        
        self.class_names = class_dirs
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        
        for class_name in class_dirs:
            class_dir = os.path.join(data_path, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_to_idx[class_name])
        
        logger.info(f"Found {len(image_paths)} images in {len(class_dirs)} classes")
        return image_paths, labels, metadata

    def _load_from_manifest(
        self, manifest_path: str, metadata: Optional[pd.DataFrame] = None
    ) -> Tuple[List[str], List[int], Optional[pd.DataFrame]]:
        """Load images from manifest CSV file.
        
        Expected CSV format:
        image_path,label[,sensitive_attribute1,sensitive_attribute2,...]
        """
        
        df = pd.read_csv(manifest_path)
        
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("Manifest must contain 'image_path' and 'label' columns")
        
        image_paths = df["image_path"].tolist()
        
        # Handle string labels
        if df["label"].dtype == object:
            self.class_names = sorted(df["label"].unique().tolist())
            label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            labels = [label_to_idx[l] for l in df["label"]]
        else:
            labels = df["label"].tolist()
            self.class_names = [str(i) for i in range(max(labels) + 1)]
        
        # Extract metadata columns (excluding image_path and label)
        meta_columns = [c for c in df.columns if c not in ["image_path", "label"]]
        if meta_columns:
            metadata = df[meta_columns]
        
        return image_paths, labels, metadata

    def train(self) -> Dict[str, float]:
        """Train the model."""
        
        hyperparams = self._get_hyperparameters()
        num_epochs = int(hyperparams.get("num_epochs", DEFAULT_CONFIG["num_epochs"]))
        learning_rate = float(hyperparams.get("learning_rate", DEFAULT_CONFIG["learning_rate"]))
        num_classes = int(hyperparams.get("num_classes", DEFAULT_CONFIG["num_classes"]))
        architecture = hyperparams.get("model_architecture", DEFAULT_CONFIG["model_architecture"])
        pretrained = hyperparams.get("pretrained", DEFAULT_CONFIG["pretrained"])
        
        if isinstance(pretrained, str):
            pretrained = pretrained.lower() == "true"
        
        # Initialize model
        self.model = self._get_model(num_classes, architecture, pretrained)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_accuracy = 0.0
        training_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100.0 * correct / total
            
            # Validation phase
            val_loss, val_accuracy = self._evaluate()
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            training_history.append(epoch_metrics)
            
            # Log to wandb
            wandb.log(epoch_metrics)
            
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self._save_checkpoint("best_model.pth")
        
        return {
            "final_train_accuracy": train_accuracy,
            "final_val_accuracy": val_accuracy,
            "best_val_accuracy": best_accuracy,
            "training_history": training_history,
        }

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on test data."""
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.test_loader)
        val_accuracy = 100.0 * correct / total
        
        return val_loss, val_accuracy

    def generate_clarify_artifacts(self, output_path: str) -> Dict[str, str]:
        """Generate artifacts for Clarify bias monitoring.
        
        Creates:
        1. Model predictions with confidence scores
        2. Ground truth labels
        3. Metadata with sensitive attributes
        4. Model explanation data (feature importance)
        
        Returns:
            Dictionary with paths to generated artifacts
        """
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_metadata = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, metadata) in enumerate(self.test_loader):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probabilities.extend(probabilities.cpu().numpy().tolist())
                
                # Collect metadata
                if metadata:
                    for i in range(len(labels)):
                        meta_dict = {k: v[i] if hasattr(v, '__getitem__') else v 
                                   for k, v in metadata.items()}
                        all_metadata.append(meta_dict)

        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Predictions file (Clarify-compatible format)
        predictions_df = pd.DataFrame({
            "prediction": all_predictions,
            "ground_truth": all_labels,
            "prediction_probability": [max(p) for p in all_probabilities],
        })
        
        # Add per-class probabilities
        for i, class_name in enumerate(self.class_names or []):
            predictions_df[f"prob_class_{class_name}"] = [p[i] for p in all_probabilities]
        
        predictions_path = os.path.join(output_path, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        
        # 2. Metadata file (for bias analysis)
        if all_metadata:
            metadata_df = pd.DataFrame(all_metadata)
            metadata_df["prediction"] = all_predictions
            metadata_df["ground_truth"] = all_labels
            metadata_path = os.path.join(output_path, "metadata_with_predictions.csv")
            metadata_df.to_csv(metadata_path, index=False)
        else:
            metadata_path = None
        
        # 3. Model info file
        model_info = {
            "model_type": "computer_vision_classifier",
            "architecture": self.config.get("model_architecture", "resnet18"),
            "num_classes": len(self.class_names) if self.class_names else 2,
            "class_names": self.class_names,
            "input_shape": [3, 224, 224],
            "timestamp": datetime.now().isoformat(),
            "framework": "pytorch",
            "clarify_compatible": True,
        }
        model_info_path = os.path.join(output_path, "model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # 4. Metrics summary
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "precision_macro": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
            "classification_report": classification_report(all_labels, all_predictions, output_dict=True),
        }
        
        # Calculate per-class metrics for bias analysis
        for i, class_name in enumerate(self.class_names or []):
            class_mask = [l == i for l in all_labels]
            class_preds = [p for p, m in zip(all_predictions, class_mask) if m]
            class_labels = [l for l, m in zip(all_labels, class_mask) if m]
            if class_labels:
                metrics[f"class_{class_name}_accuracy"] = accuracy_score(class_labels, class_preds)
        
        metrics_path = os.path.join(output_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # 5. Clarify analysis config template
        clarify_config = {
            "version": "1.0",
            "analysis_type": "post_training_bias",
            "dataset_type": "computer_vision",
            "label_column": "ground_truth",
            "predicted_label_column": "prediction",
            "probability_threshold": 0.5,
            "facet_columns": [],  # To be filled with sensitive attributes
            "metrics": [
                "DPL",  # Difference in Positive Proportions in Labels
                "DI",   # Disparate Impact
                "DPPL", # Difference in Positive Proportions in Predicted Labels
                "AD",   # Accuracy Difference
                "RD",   # Recall Difference
                "DAR",  # Difference in Acceptance Rates
                "DRR",  # Difference in Rejection Rates
            ],
        }
        clarify_config_path = os.path.join(output_path, "clarify_config.json")
        with open(clarify_config_path, "w") as f:
            json.dump(clarify_config, f, indent=2)
        
        logger.info(f"Generated Clarify artifacts at {output_path}")
        
        return {
            "predictions": predictions_path,
            "metadata": metadata_path,
            "model_info": model_info_path,
            "metrics": metrics_path,
            "clarify_config": clarify_config_path,
        }

    def _save_checkpoint(self, filename: str) -> str:
        """Save model checkpoint."""
        
        checkpoint_dir = "/opt/ml/model"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path

    def save_to_s3(self, artifacts: Dict[str, str], s3_bucket: str, s3_prefix: str) -> Dict[str, str]:
        """Upload artifacts to S3."""
        
        s3_paths = {}
        
        for artifact_name, local_path in artifacts.items():
            if local_path and os.path.exists(local_path):
                s3_key = f"{s3_prefix}/{os.path.basename(local_path)}"
                
                try:
                    self.s3_client.upload_file(local_path, s3_bucket, s3_key)
                    s3_path = f"s3://{s3_bucket}/{s3_key}"
                    s3_paths[artifact_name] = s3_path
                    logger.info(f"Uploaded {artifact_name} to {s3_path}")
                except Exception as e:
                    logger.error(f"Failed to upload {artifact_name}: {e}")
        
        # Upload model file
        model_path = "/opt/ml/model/best_model.pth"
        if os.path.exists(model_path):
            s3_key = f"{s3_prefix}/model/best_model.pth"
            try:
                self.s3_client.upload_file(model_path, s3_bucket, s3_key)
                s3_paths["model"] = f"s3://{s3_bucket}/{s3_key}"
                logger.info(f"Uploaded model to s3://{s3_bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload model: {e}")
        
        return s3_paths

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters from SageMaker config."""
        
        hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
        
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, "r") as f:
                hyperparameters = json.load(f)
            logger.info(f"Loaded hyperparameters: {hyperparameters}")
            return hyperparameters
        
        logger.info("Using default hyperparameters")
        return DEFAULT_CONFIG


def read_config(env: str) -> Dict[str, Any]:
    """Read configuration file."""
    
    file_path = f"/opt/ml/code/deploy/{env}/tdspds-config.yml"
    
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configurations = {config}")
    return config


def train_cv_model(env: str) -> None:
    """Main training function for CV model."""
    
    config = read_config(env)
    project_name = config.get("Model", {}).get("ProjectName", "CV Model")
    model_name = config.get("Model", {}).get("ModelName", "cv_model")
    pod_name = config.get("TeamDetails", {}).get("PodName", "tdspds")
    
    # Initialize wandb
    wandb.init(
        entity=pod_name,
        project=project_name,
        notes="Computer Vision Model Training for Clarify Bias Monitoring",
        name=f"cv-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    
    # Initialize trainer
    trainer = CVModelTrainer(config, env)
    
    # Load data
    training_data_path = "/opt/ml/input/data/train"
    test_data_path = "/opt/ml/input/data/test"
    metadata_path = "/opt/ml/input/data/metadata/metadata.csv"
    
    trainer.load_data(
        train_data_path=training_data_path,
        test_data_path=test_data_path if os.path.exists(test_data_path) else None,
        metadata_path=metadata_path if os.path.exists(metadata_path) else None,
    )
    
    # Train model
    logger.info("Starting model training...")
    training_results = trainer.train()
    
    # Log final metrics to wandb
    wandb.log({
        "final_train_accuracy": training_results["final_train_accuracy"],
        "final_val_accuracy": training_results["final_val_accuracy"],
        "best_val_accuracy": training_results["best_val_accuracy"],
    })
    
    # Generate Clarify artifacts
    clarify_output_path = "/opt/ml/output/clarify"
    artifacts = trainer.generate_clarify_artifacts(clarify_output_path)
    
    # Upload to S3
    s3_config = config.get("S3Output", {})
    s3_bucket = s3_config.get("Bucket", "tdsp-ml-products-dev")
    s3_prefix = s3_config.get("Prefix", f"{pod_name}/cv-model/{model_name}")
    
    s3_paths = trainer.save_to_s3(artifacts, s3_bucket, s3_prefix)
    
    # Log S3 paths
    logger.info("Artifacts uploaded to S3:")
    for name, path in s3_paths.items():
        logger.info(f"  {name}: {path}")
    
    # Register model artifact in wandb
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file("/opt/ml/model/best_model.pth")
    
    # Add Clarify artifacts
    for artifact_name, local_path in artifacts.items():
        if local_path and os.path.exists(local_path):
            artifact.add_file(local_path)
    
    wandb.log_artifact(artifact)
    wandb.run.link_artifact(artifact, f"{pod_name}/{project_name}/{model_name}")
    
    wandb.run.finish()
    logger.info("Training complete. Model and artifacts saved.")


if __name__ == "__main__":
    env = os.getenv("ENV", "dev")
    train_cv_model(env)
